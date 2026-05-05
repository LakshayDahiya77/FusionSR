import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss is a smooth L1 variant.
    L = mean( sqrt( (pred - target)^2 + eps^2 ) )

    Better than L1: differentiable everywhere, smoother gradients near zero.
    Better than L2: less sensitive to outliers, doesn't over-penalize large errors.
    eps=1e-3 is the standard value from EDSR/RCAN papers.

    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps**2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps2))
        return loss
