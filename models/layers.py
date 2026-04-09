"""Reusable custom layers 
"""
import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    """
    Inverted dropout — NO use of nn.Dropout or F.dropout internally.
    At train time: zero each element with prob p, scale surviving by 1/(1-p).
    At eval  time: identity (no scaling needed thanks to inverted scheme).
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f'p must be in [0,1), got {p}')
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * mask / keep

    def extra_repr(self): return f'p={self.p}'
