import torch

from src.models.controllers.controller_types import ControllerType


class MLPGumbelSoftmaxController(torch.nn.Module):
    def __init__(self, input_dim, tau, with_fixed_input=False) -> None:
        super().__init__()
        self.type = ControllerType.MLP_GUMBEL
        self.linear1 = torch.nn.Linear(input_dim, 2)
        # self.linear2 = torch.nn.Linear(input_dim // 8, 2)
        self.tau = tau
        self.with_fixed_input = with_fixed_input

    def forward(self, X):
        '''Returns (one_hot)
        one_hot: Bernoulli 0/1 where 1 denotes a skip.
        '''
        # self.linear1(X.view(X.shape[0] * X.shape[1], -1))
        if self.with_fixed_input:
            X = torch.ones_like(X)
        X = self.linear1(X)
        # X = torch.nn.functional.relu(X) # add some non linearity
        # X = self.linear2(X)
        one_hot = torch.nn.functional.gumbel_softmax(X, tau=1, hard=True) # c'est tres dur.
        return one_hot