from enum import Enum


class ControllerType(Enum):
    MLP_GUMBEL = 'mlp_gumbel'
    RNN = 'rnn'
    STATIC = 'static' # always returns 1 or 0
    PROBABILISTIC = 'probabilistic' # skips tokens to a specified probability, without any knowledge.