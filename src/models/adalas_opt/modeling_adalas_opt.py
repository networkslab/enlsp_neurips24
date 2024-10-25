from typing import Optional, List, Union, Tuple

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.opt.modeling_opt import OPTDecoder, OPTForCausalLM, OPTModel, \
    OPTLearnedPositionalEmbedding
from transformers.utils import logging
import numpy as np

from src.models.adalas_opt.config_adalas_opt import AdalasOPTConfig, PropagationMode, DynamicPropagationConfig
import torch
from torch import nn

from src.models.adalas_opt.modelling_adalas_opt_modules import AdalasOPTDecoderLayer
from src.models.controllers.controller_types import ControllerType, ControllerInputType
from src.models.controllers.mlp_gumbel_softmax_controller import MLPGumbelSoftmaxController
from src.models.controllers.probabilistic_controller import ProbabilisticController
from src.models.controllers.static_controller import StaticController
from src.utils.utils import freeze_network

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
        self.hidden_size = config.hidden_size
        self.prop_config = self.config.propagation_config
        self.init_controllers()
        self.with_metrics = config.with_metrics
        if self.with_metrics:
            self._init_train_val_metrics()
            self._init_generation_metrics()
        # Initialize weights and apply final processing
        self.post_init()
        self.with_cost_aware_loss = config.with_cost_aware_loss

    def _get_model_device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _init_train_val_metrics(self):
        self.metrics = {'train': self._init_train_val_metric_dict_for_phase(),
                        'eval': self._init_train_val_metric_dict_for_phase()}
        
    def _init_generation_metrics(self):
        self.generation_metrics = {
            'skip_count': None,
            'token_count': None
        }

    def _init_train_val_metric_dict_for_phase(self):
        return {'percentage_skip': [None for _ in range(len(self.layers))]}

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
        separation_token = self.config.sep_token_id
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
        layer_costs = []
        if self.prop_config.propagation_mode == PropagationMode.RANDOM_FOR_BUDGET:
            # TODO: This will generate a new route for every forward call, this is problematic for generation where forward is called
            # multiple times for a single generation. One fix is to select a random route in the trainer code
            random_route = self._set_random_route() # use the returned value for logging if desired.
            # print(f"Executing route {random_route}")
        for idx, decoder_layer in enumerate(self.layers):
            controller = self.controllers[idx]
            controller_input = self.prepare_controller_input(hidden_states, pos_embeds, inputs_embeds)
            controller_out = controller(controller_input) # [execute_bool, skip_bool]
            gumbel_skip = controller_out[:, :, 1]
            gumbel_keep = controller_out[:, :, 0]
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
            input_contains_prompt = torch.any(input_ids == separation_token).item()
            if input_contains_prompt:
                label_mask = (torch.cumsum(input_ids == separation_token, 1) > 1)  # 1 where labels are, assuming prompt is present. Includes the last SEP token
                label_mask = torch.where(input_ids == separation_token, 0, label_mask) #removes last SEP token from the label mask
                input_contains_prompt_and_label = torch.any(label_mask[:, :]).item() #will return false during the first forward pass of generation, and true during training
            else:
                label_mask = torch.ones_like(input_ids) # during autoregressive gen, all tokens are labels
                input_contains_prompt_and_label = False # if there is no prompt then there is no prompt and label
            update_mask = 1 - (label_mask * (1 - gumbel_keep)) # De Morgan's to keep things differential. 1 where we update, 0 where we skip.
            skip_mask = label_mask * gumbel_skip # Should be complement of previous line
            hidden_states = layer_outputs[0] * update_mask[:, :, None] + hidden_states * skip_mask[:, :, None]
            train_eval_phase = 'train' if self.training else 'eval'
            generation_lengths = torch.sum(label_mask, dim = -1)
            num_skips_on_generation = torch.sum(skip_mask, dim = -1)
            if self.with_cost_aware_loss and input_contains_prompt_and_label: # need to compute how many skips on generation
                updates_on_generation = generation_lengths - num_skips_on_generation
                layer_cost_per_seq = updates_on_generation / generation_lengths
                layer_cost_for_batch = torch.mean(layer_cost_per_seq)
                layer_costs.append(layer_cost_for_batch)
            if self.with_metrics and input_contains_prompt_and_label:
                # compute number of skips on label
                with torch.no_grad():
                    percentage_skips = num_skips_on_generation / generation_lengths
                    if self.metrics[train_eval_phase]['percentage_skip'][idx] is None:
                        self.metrics[train_eval_phase]['percentage_skip'][idx] = percentage_skips
                    else:
                        self.metrics[train_eval_phase]['percentage_skip'][idx] = torch.cat((self.metrics[train_eval_phase]['percentage_skip'][idx], percentage_skips))
            elif not input_contains_prompt:
                if self.generation_metrics['skip_count'] is None: #initiliaze metrics on the correct device
                    self.generation_metrics['skip_count'] = torch.zeros(len(self.layers)).to(skip_mask.device)
                    self.generation_metrics['token_count'] = torch.tensor(0).to(skip_mask.device)
                self.generation_metrics['skip_count'][idx] += skip_mask[:, -1].sum()
                if idx == len(self.layers) - 1: # Only increment the token count at the last layer (Once per generated token)
                    self.generation_metrics['token_count'] += input_ids.shape[0] # As many tokens as batch size
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                # TODO implement this following above logic if needed with layer skipping.
                # if should_skip_layer:
                #     label_mask = (torch.cumsum(input_ids == separation_token, 1) > 1)[:, None, :, None] # 1 where label is
                #     previous_self_attn = all_self_attns[-1]
                #     current_self_attn = layer_outputs[1] * torch.logical_not(label_mask) + previous_self_attn * label_mask
                #     all_self_attns += (current_self_attn,)
                # else:
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

        if self.with_cost_aware_loss: # return a tuple where the second entry contains the skips
            return (BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            ), layer_costs)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def init_controllers(self):
        if not hasattr(self.prop_config, 'controller_input_size') or self.prop_config.controller_input_size is None:
            controller_input_dim = self.hidden_size
        else:
            controller_input_dim = self.prop_config.controller_input_size
        if self.prop_config.propagation_mode == PropagationMode.STATIC_EE:
            self.controllers = nn.ModuleList(
                [StaticController(i > self.prop_config.early_exit_layer) for i in range(len(self.layers))]
            )
            return
        if self.prop_config.propagation_mode == PropagationMode.RANDOM_FOR_BUDGET:
            self.controllers = nn.ModuleList(
                [StaticController(skip=False) for _ in range(len(self.layers))]
            )
            return
        if self.prop_config.controller_type == ControllerType.MLP_GUMBEL:
            self.controllers = nn.ModuleList([])
            for layer in range(len(self.layers)):
                if layer in self.prop_config.controller_layers:
                    self.controllers.append(MLPGumbelSoftmaxController(controller_input_dim,
                                                                       tau=self.prop_config.gumbel_temperature,
                                                                       with_fixed_input=self.prop_config.with_fixed_input,
                                                                       layers=self.prop_config.layers, divisor=self.prop_config.divisor))
                else:
                    self.controllers.append(StaticController(False))
        elif self.prop_config.controller_type == ControllerType.STATIC:
            self.controllers = []
            if self.prop_config.propagation_mode == PropagationMode.STATIC_SKIP:
                for layer in range(len(self.layers)):
                    self.controllers.append(StaticController(layer in self.prop_config.skip_layers))
            elif self.prop_config.propagation_mode == PropagationMode.FULL:
                for layer in range(len(self.layers)):
                    self.controllers.append(StaticController(False))
        elif self.prop_config.controller_type == ControllerType.PROBABILISTIC:
            self.controllers = []
            for layer in range(len(self.layers)):
                self.controllers.append(ProbabilisticController(skip_prob=self.prop_config.skip_probs[layer]))
        else:
            raise Exception('Unimplemented controller type')
        

    def prepare_controller_input(self, hidden_states, pos_embeds, inputs_embeds):
        if hasattr(self.prop_config,'controller_input_type') and self.prop_config.controller_input_type is not None:
            if self.prop_config.controller_input_type == ControllerInputType.HIDDEN_STATES:
                return hidden_states.detach()
            elif self.prop_config.controller_input_type == ControllerInputType.POS_EMBEDS:
                #standardize pos_embeds to std 1
                pos_embeds_cpy = 10*pos_embeds.detach().clone()
                pos_embeds_cpy = (pos_embeds_cpy - pos_embeds_cpy.mean()) / pos_embeds.std()
                return pos_embeds_cpy
            elif self.prop_config.controller_input_type == ControllerInputType.INPUTS_EMBEDS:
                #detached inputs_embeds, and standardize pos_embeds
                inputs_embeds_cpy = 10*inputs_embeds.detach().clone()
                inputs_embeds_cpy = (inputs_embeds_cpy - inputs_embeds_cpy.mean()) / inputs_embeds.std()
                return inputs_embeds_cpy
            elif self.prop_config.controller_input_type == ControllerInputType.INITIAL_STATE:
                intial_state = (inputs_embeds + pos_embeds)
                intial_state = 10*(intial_state - intial_state.mean()) / intial_state.std()
                return intial_state.detach()
            else:
                raise Exception('Unimplemented controller input type')
        else:
            return hidden_states.detach()

    def freeze_backbone(self, freeze_head = False):
        freeze_network(self, ['controllers'] if freeze_head else ['controllers', 'embed_tokens'])

    def flush_train_val_metrics(self, phase = None):
        ''' should typically be called after every logging step in the callback'''
        if phase is None:
            self._init_train_val_metrics()
        else:
            self.metrics[phase] = self._init_train_val_metric_dict_for_phase()

    def _set_random_route(self):
        total_num_layers = len(self.layers)
        random_route_for_budget = self.prop_config.generate_random_route(total_num_layers) # 1 is a select, 0 is a skip
        for controller_idx, controller in enumerate(self.controllers):
            controller.skip = np.logical_not(random_route_for_budget[controller_idx])
        return random_route_for_budget

class AdalasOPTModel(OPTModel):
    def __init__(self, config: AdalasOPTConfig):
        super().__init__(config)
        self.decoder = AdalasOPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def freeze(self, freeze_head = False):
        self.decoder.freeze_backbone(freeze_head)


class AdalasOPTForCausalLM(OPTForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: AdalasOPTConfig):
        super(OPTForCausalLM, self).__init__(config)
        self.model = AdalasOPTModel(config)
        self.config = config

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = torch.nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        '''Pretty much the same method as the original super but overriding for loss computation'''
        with_cost_aware_loss = self.config.with_cost_aware_loss
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if not with_cost_aware_loss:
            outputs = self.model.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            outputs, layer_perc_execution = self.model.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )


        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            if self.model.decoder.with_cost_aware_loss:
                layer_perc_execution = torch.stack(layer_perc_execution)
                layer_cost_multipliers = torch.ones_like(layer_perc_execution) # modify based on actual weight of each layer.
                loss += torch.mean(layer_cost_multipliers * layer_perc_execution) * self.config.alpha

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def freeze_backbone(self, freeze_head = False):
        trainable_parameters_before = filter(lambda p: p.requires_grad,
                                      self.parameters())
        num_trainable_params_before = sum(
            [np.prod(p.size()) for p in trainable_parameters_before])

        self.model.freeze(freeze_head)
        named_trainable_params = map(lambda t: t[0], filter(lambda p: p[1].requires_grad,
                                     self.named_parameters()))
        if not freeze_head:
            assert self.lm_head.weight.requires_grad, 'Make sure the head is unfrozen. This may be due to it being tied with embedding'

        trainable_parameters_after = filter(lambda p: p.requires_grad,
                                             self.parameters())
        num_trainable_params_after = sum(
            [np.prod(p.size()) for p in trainable_parameters_after])
        print('Successfully froze network: from {} to {} trainable params.'.format(
            num_trainable_params_before, num_trainable_params_after))





