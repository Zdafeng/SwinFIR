import torch
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss
from basicsr.losses.losses import _reduction_modes


@weighted_loss
def charbonnier_loss_color(pred, target, eps=1e-6):
    diff = torch.add(pred, -target)
    diff_sq = diff * diff
    diff_sq_color = torch.mean(diff_sq, 1, True)
    error = torch.sqrt(diff_sq_color + eps)
    loss = torch.mean(error)
    return loss


@LOSS_REGISTRY.register()
class CharbonnierLossColor(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-6):
        super(CharbonnierLossColor, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss_color(pred, target, weight, eps=self.eps, reduction=self.reduction)
