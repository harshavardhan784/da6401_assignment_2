"""Custom IoU loss 
"""

import torch
import torch.nn as nn

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred, target: (N, 4) -> [cx, cy, w, h]
        """

        # ---- Convert to corners ----
        px, py, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        tx, ty, tw, th = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        px1 = px - pw / 2
        py1 = py - ph / 2
        px2 = px + pw / 2
        py2 = py + ph / 2

        tx1 = tx - tw / 2
        ty1 = ty - th / 2
        tx2 = tx + tw / 2
        ty2 = ty + th / 2

        # ---- Intersection ----
        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)

        inter_area = inter_w * inter_h

        # ---- Areas ----
        pred_area = torch.clamp(pw, min=0) * torch.clamp(ph, min=0)
        target_area = torch.clamp(tw, min=0) * torch.clamp(th, min=0)

        union = pred_area + target_area - inter_area

        # ---- IoU ----
        eps = 1e-7
        iou = inter_area / (union + eps)

        loss = 1 - iou

        # ---- Reduction ----
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss