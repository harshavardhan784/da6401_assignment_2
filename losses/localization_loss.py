"""Combined MSE + IoU loss for localization.

Both pred and target are in pixel space [0..image_size].
  - MSE is normalized by image_size² so it lives in [0, ~1].
  - IoU loss is already in [0, 1].
Equal weighting is therefore meaningful.
"""

import torch.nn as nn


class LocalizationLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.5, iou_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.iou_weight = iou_weight
        self.mse_loss   = nn.MSELoss()
        from losses.iou_loss import IoULoss
        self.iou_loss   = IoULoss(reduction="mean")

    def forward(self, pred, target, image_size: int = 224):
        # Both pred and target are in pixel space → normalize MSE
        mse = self.mse_loss(pred, target) / (image_size ** 2)
        iou = self.iou_loss(pred, target)
        return self.mse_weight * mse + self.iou_weight * iou