"""Combined MSE + IoU loss for localization."""

import torch.nn as nn


class LocalizationLoss(nn.Module):
    """
    MSE (normalized by image_size²) + IoU loss.
    Both terms are in [0, 1] so equal weighting is meaningful.
    """

    def __init__(self, mse_weight: float = 0.5, iou_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.iou_weight = iou_weight
        self.mse_loss   = nn.MSELoss()
        from losses.iou_loss import IoULoss
        self.iou_loss   = IoULoss(reduction='mean')

    def forward(self, pred, target, image_size: int = 224):
        mse = self.mse_loss(pred, target) / (image_size ** 2)
        iou = self.iou_loss(pred, target)
        return self.mse_weight * mse + self.iou_weight * iou