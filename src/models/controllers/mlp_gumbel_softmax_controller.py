import torch

from src.models.controllers.controller_types import ControllerType


class MLPGumbelSoftmaxController(torch.nn.Module):
    def __init__(self, input_dim, tau, with_fixed_input=False, layers=1,divisor=8) -> None:
        super().__init__()
        self.type = ControllerType.MLP_GUMBEL
        for i in range(1,layers+1):
            if i == layers:
                self.add_module(f'linear{i}', torch.nn.Linear(input_dim, 2))
            else:
                if input_dim // divisor < 2:
                    raise ValueError('divisor too large in MLPGumbelSoftmaxController, causing output dim to be less than 2')
                self.add_module(f'linear{i}', torch.nn.Linear(input_dim, input_dim // (divisor)))
                input_dim = input_dim // divisor
                
        self.tau = tau
        self.with_fixed_input = with_fixed_input

    def forward(self, X):
        '''Returns (one_hot)
        one_hot: Bernoulli 0/1 where 1 denotes a skip.
        '''
        # self.linear1(X.view(X.shape[0] * X.shape[1], -1))
        if self.with_fixed_input:
            X = torch.ones_like(X)
        for i in range(1,len(self._modules)+1):
            X = self._modules[f'linear{i}'](X)
            if i != len(self._modules):
                X = torch.nn.functional.relu(X) 
       
        one_hot = torch.nn.functional.gumbel_softmax(X, tau=1, hard=True) # c'est tres dur.
        return one_hot