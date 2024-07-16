import json
from typing import List
from transformers.models.opt import OPTConfig
from enum import Enum


class PropagationMode(Enum):
    FULL = 'full'
    STATIC_SKIP = 'static_skip'
    STOCHASTIC_DROPOUT = 'stochastic_dropout'


class PropagationConfig:
    def __init__(self, propagation_mode = PropagationMode.FULL):
        self.propagation_mode = propagation_mode

    def to_dict(self):
        return {'propagation_mode': self.propagation_mode.value}


class StaticSkipPropagationConfig(PropagationConfig):
    def __init__(self, skip_layers: List[int]):
        super().__init__(PropagationMode.STATIC_SKIP)
        self.skip_layers = skip_layers

    def to_dict(self):
        return {'propagation_mode': self.propagation_mode.value, 'skip_layers': self.skip_layers}

class StochasticDropoutPropagationConfig(PropagationConfig):
    def __init__(self, skip_probs: List[float]):
        super().__init__(PropagationMode.STOCHASTIC_DROPOUT)
        self.skip_probs = skip_probs

    def to_dict(self):
        return {'propagation_mode': self.propagation_mode.value, 'skip_probs': self.skip_probs}


class AdalasOPTConfig(OPTConfig):
    model_type = 'adalas_opt'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self,
                 vocab_size=50272,
                 hidden_size=768,
                 num_hidden_layers=12,
                 ffn_dim=3072,
                 max_position_embeddings=2048,
                 do_layer_norm_before=True,
                 _remove_final_layer_norm=False,
                 word_embed_proj_dim=None,
                 dropout=0.1,
                 attention_dropout=0.0,
                 num_attention_heads=12,
                 activation_function="relu",
                 layerdrop=0.0,
                 init_std=0.02,
                 use_cache=True,
                 pad_token_id=1,
                 bos_token_id=2,
                 eos_token_id=2,
                 enable_bias=True,
                 layer_norm_elementwise_affine=True,
                 propagation_config: PropagationConfig = PropagationConfig(),
                 skip_prompt=False,
                 **kwargs):
        super().__init__(vocab_size,
                         hidden_size,
                         num_hidden_layers,
                         ffn_dim,
                         max_position_embeddings,
                         do_layer_norm_before,
                         _remove_final_layer_norm,
                         word_embed_proj_dim,
                         dropout,
                         attention_dropout,
                         num_attention_heads,
                         activation_function,
                         layerdrop,
                         init_std,
                         use_cache,
                         pad_token_id,
                         bos_token_id,
                         eos_token_id,
                         enable_bias,
                         layer_norm_elementwise_affine,
                         **kwargs)
        self.propagation_config = propagation_config
        self.skip_prompt = skip_prompt

    def to_json_string(self, use_diff: bool = True) -> str:
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        config_dict['propagation_config'] = self.propagation_config.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
