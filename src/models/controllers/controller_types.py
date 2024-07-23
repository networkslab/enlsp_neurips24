from enum import Enum


class ControllerType(Enum):
    MLP_GUMBEL = 'mlp_gumbel'
    RNN = 'rnn'