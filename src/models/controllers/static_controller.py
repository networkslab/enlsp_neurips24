import torch

from src.models.controllers.controller_types import ControllerType


class StaticController(torch.nn.Module):
    '''Consistently skips or does not skip based on the skip boolean.
    This is used for the STATIC SKIP propagation mode as well as the full prop mode (where all controllers never skip)'''
    def __init__(self, skip: bool) -> None:
        super().__init__()
        self.type = ControllerType.STATIC
        self.skip = skip

    def forward(self, X):
        if self.skip:
            skip_tensor = torch.ones((X.shape[0], X.shape[1])) # B x SEQ_LEN, we discard the size of the hidden state
            keep_tensor = torch.zeros((X.shape[0], X.shape[1]))
        else:
            skip_tensor = torch.zeros((X.shape[0], X.shape[1]))
            keep_tensor = torch.ones((X.shape[0], X.shape[1]))
        return torch.stack([keep_tensor, skip_tensor], dim=-1).to(X.device)