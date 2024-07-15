from typing import Optional, List, Union, Tuple

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.opt.modeling_opt import OPTDecoder, OPTForCausalLM, OPTModel, \
    OPTLearnedPositionalEmbedding
from transformers.utils import logging

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, PropagationMode
import torch
from torch import nn

from src.models.adalas_opt.modelling_adalas_opt_modules import AdalasOPTDecoderLayer

logger = logging.get_logger(__name__)


class AdalasOPTDecoder(OPTDecoder):

    def __init__(self, config: AdalasOPTConfig):
        super(OPTDecoder, self).__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([AdalasOPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.prop_config = self.config.propagation_config
        # Initialize weights and apply final processing
        self.post_init()



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        separation_token = 2
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            should_skip_layer = False
            if self.prop_config.propagation_mode == PropagationMode.STATIC_SKIP:
                should_skip_layer = idx in self.prop_config.skip_layers
            elif self.prop_config.propagation_mode == PropagationMode.STOCHASTIC_DROPOUT:
                should_skip_layer = bool(torch.bernoulli(torch.tensor(self.prop_config.skip_probs[idx])).item())
            if should_skip_layer and past_key_values is not None: # if past key values is passed, it means we are dealing with generation of new tok
                if use_cache:
                    past_key_value = past_key_values[idx] if past_key_values is not None else None # take past values for all previous tokens at this layer
                    layer_outputs = decoder_layer.forward(hidden_states,
                                                                   attention_mask=causal_attention_mask,
                                                                   layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                                                                   past_key_value=past_key_value,
                                                                   output_attentions=output_attentions,
                                                                   use_cache=use_cache,
                                                                   propagate_kv_cache_only=True
                                                                   )
                    next_decoder_cache += (layer_outputs[0],) # only populate the cache.
                continue

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            
            if should_skip_layer:
                label_mask = (torch.cumsum(input_ids == separation_token, 1) > 0)[:, :, None] # 1 where labels are
                hidden_states = layer_outputs[0] * torch.logical_not(label_mask) + hidden_states * label_mask # for the label part, keep the hidden states as before. For the prompt part, update
            else:
                hidden_states = layer_outputs[0] 

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                if should_skip_layer:
                    label_mask = (torch.cumsum(input_ids == separation_token, 1) > 0)[:, None, :, None] # 1 where label is
                    previous_self_attn = all_self_attns[-1]
                    current_self_attn = layer_outputs[1] * torch.logical_not(label_mask) + previous_self_attn * label_mask
                    all_self_attns += (current_self_attn,)
                else:
                    all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class AdalasOPTModel(OPTModel):
    def __init__(self, config: AdalasOPTConfig):
        super().__init__(config)
        self.decoder = AdalasOPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()


class AdalasOPTForCausalLM(OPTForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super(OPTForCausalLM, self).__init__(config)
        self.model = AdalasOPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = torch.nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()




