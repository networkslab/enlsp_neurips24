import torch

from src.models.controllers.controller_types import ControllerType


class ProbabilisticController(torch.nn.Module):
    '''Skips with a given probability, regardless of the input'''
    def __init__(self, skip_prob: float) -> None:
        super().__init__()
        self.type = ControllerType.PROBABILISTIC
        self.skip_prob = skip_prob

    def forward(self, X):
        skip_tensor = torch.bernoulli(torch.zeros((X.shape[0], X.shape[1])) + self.skip_prob)
        keep_tensor = torch.logical_not(skip_tensor)
        return torch.stack([keep_tensor, skip_tensor], dim=-1).to(X.device)