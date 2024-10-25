from enum import Enum


class ControllerType(Enum):
    MLP_GUMBEL = 'mlp_gumbel'
    RNN = 'rnn'
    STATIC = 'static' # always returns 1 or 0
    PROBABILISTIC = 'probabilistic' # skips tokens to a specified probability, without any knowledge.

class ControllerInputType(Enum):
    HIDDEN_STATES = 'hidden_states'
    POS_EMBEDS = 'pos_embeds'
    INPUTS_EMBEDS = 'inputs_embeds'
    INITIAL_STATE = 'initial_state'