import json
from typing import List
from transformers.models.opt import OPTConfig
from enum import Enum
import numpy as np

from src.models.controllers.controller_types import ControllerType, ControllerInputType


class PropagationMode(Enum):
    FULL = 'full'
    STATIC_SKIP = 'static_skip'
    STOCHASTIC_DROPOUT = 'stochastic_dropout'
    DYNAMIC = 'dynamic'
    STATIC_EE = 'static_ee'
    RANDOM_FOR_BUDGET = 'random_for_budget'


class PropagationConfig:
    def __init__(self, propagation_mode = PropagationMode.FULL):
        self.propagation_mode = propagation_mode
        self.controller_type = ControllerType.STATIC

    def to_dict(self):
        return {'propagation_mode': self.propagation_mode.value}
    
    @staticmethod
    def from_dict(propagation_dict):
        for mode in PropagationMode:
            if propagation_dict['propagation_mode'] == mode.value:
                del propagation_dict['propagation_mode']
                return MAP_PROPAGATION_MODES[mode.value](**propagation_dict)


class StaticSkipPropagationConfig(PropagationConfig):
    def __init__(self, skip_layers: List[int], freeze_skipped = False):
        super().__init__(PropagationMode.STATIC_SKIP)
        self.skip_layers = skip_layers
        self.freeze_skipped = freeze_skipped
        self.controller_type = ControllerType.STATIC

    def to_dict(self):
        return {'propagation_mode': self.propagation_mode.value, 'skip_layers': self.skip_layers, 'freeze_skipped': self.freeze_skipped}

class StaticEEPropagationConfig(PropagationConfig):
    def __init__(self, early_exit_layer: int, freeze_subsequent = True):
        super().__init__(PropagationMode.STATIC_EE)
        self.early_exit_layer = early_exit_layer
        self.freeze_subsequent = freeze_subsequent
        self.controller_type = ControllerType.STATIC # use a bunch of static controllers based on where the exit layer is.

    def to_dict(self):
        return {'propagation_mode': self.propagation_mode.value,
                'ee_layer': self.early_exit_layer,
                'freeze_subsequent': self.freeze_subsequent}
    
class RandomForBudgetPropagationConfig(PropagationConfig):
    def __init__(self, budget: int, enforce_layer_1=False):
        super().__init__(PropagationMode.RANDOM_FOR_BUDGET)
        self.budget = budget
        self.enforce_layer_1 = enforce_layer_1
        self.controller_type = ControllerType.STATIC # use a bunch of static controllers based on the selected layers.

    def generate_random_route(self, total_num_layers: int) -> list[bool]:
        '''generates a random route for the given budget. 1 is a selected layer, 0 is a skipped layer.'''
        assert self.budget <= total_num_layers, 'Budget should be greater or equal to total number of layers'
        if self.enforce_layer_1:
            executed_layers = np.ones(self.budget-1) #reserve one executed layer for layer 1
            skipped_layers = np.zeros(total_num_layers - self.budget)
            concat_layers = np.concatenate([executed_layers, skipped_layers])
            permuted_layers = np.random.permutation(concat_layers)
            permuted_layers = np.concatenate([np.ones(1), permuted_layers])
        else:
            executed_layers = np.ones(self.budget)
            skipped_layers = np.zeros(total_num_layers - self.budget)
            concat_layers = np.concatenate([executed_layers, skipped_layers])
            permuted_layers = np.random.permutation(concat_layers)
        return permuted_layers

    def to_dict(self):
        return {'propagation_mode': self.propagation_mode.value,
                'budget': self.budget,
               'enforce_layer_1': self.enforce_layer_1}

class StochasticDropoutPropagationConfig(PropagationConfig):
    def __init__(self, skip_probs: List[float]):
        super().__init__(propagation_mode=PropagationMode.STOCHASTIC_DROPOUT)
        self.controller_type = ControllerType.PROBABILISTIC
        self.skip_probs = skip_probs

    def to_dict(self):
        return {'propagation_mode': self.propagation_mode.value, 'skip_probs': self.skip_probs}

class DynamicPropagationConfig(PropagationConfig):
    '''Uses trainable gates to determine which layers to skip or execute'''
    def __init__(self, controller_layers,
                 controller_input_size = None,
                 gumbel_temperature = 1.2,
                 controller_type = ControllerType.MLP_GUMBEL,
                 controller_input_type = ControllerInputType.HIDDEN_STATES,
                 with_fixed_input = False,
                 layers = 1,
                 divisor = 8):
        super().__init__(PropagationMode.DYNAMIC)
        self.gumbel_temperature = gumbel_temperature # tau parameter
        self.controller_input_size = controller_input_size
        self.controller_type = ControllerType(controller_type)
        self.controller_input_type = ControllerInputType(controller_input_type) #Enum contructor accepts str or enum itself
        self.controller_layers = controller_layers
        self.with_fixed_input = with_fixed_input
        self.layers = layers
        self.divisor = divisor
        

    def to_dict(self):
        return {
            'propagation_mode': self.propagation_mode.value,
            'controller_layers': self.controller_layers, 
            'controller_input_size': self.controller_input_size,
            'gumbel_temperature': self.gumbel_temperature,
            'controller_type': self.controller_type.value,
            'controller_input_type': self.controller_input_type.value,
            'with_fixed_input': self.with_fixed_input,
            'layers': self.layers,
            'divisor': self.divisor,
        }

MAP_PROPAGATION_MODES = {
    'full': PropagationConfig,
    'static_skip': StaticSkipPropagationConfig,
    'stochastic_dropout': StochasticDropoutPropagationConfig,
    'dynamic': DynamicPropagationConfig,
    'static_ee': StaticEEPropagationConfig,
    'random_for_budget': RandomForBudgetPropagationConfig
}

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
                 sep_token_id=50273,
                 enable_bias=True,
                 layer_norm_elementwise_affine=True,
                 propagation_config: PropagationConfig = PropagationConfig(),
                 skip_prompt=False,
                 with_metrics=True,
                 with_cost_aware_loss = False,
                 alpha = 0.0, # controls how much to emphasize computational cost.
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
        if isinstance(propagation_config,dict):
            self.propagation_config = PropagationConfig.from_dict(propagation_config)
        else:
            self.propagation_config = propagation_config
        self.skip_prompt = skip_prompt
        self.sep_token_id = sep_token_id
        self.with_metrics = with_metrics
        self.with_cost_aware_loss = with_cost_aware_loss
        self.alpha = alpha

    def to_json_string(self, use_diff: bool = True) -> str:
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        config_dict['propagation_config'] = self.propagation_config.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    