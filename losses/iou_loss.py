"""
IoU Loss for bounding box regression
"""
import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """
    Custom IoU Loss for bounding box regression.
    Input: [cx, cy, w, h] in pixel space (not normalized)
    Output: 1 - IoU, range [0, 1]
    Supports reduction: 'mean', 'sum', 'none'
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ('mean', 'sum', 'none'), \
            f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
        self.reduction = reduction

    @staticmethod
    def _to_xyxy(boxes):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], -1)

    def forward(self, pred, target):
        """
        Args:
            pred: (N, 4) predicted boxes [cx, cy, w, h]
            target: (N, 4) target boxes [cx, cy, w, h]
        Returns:
            loss: (N,) or scalar depending on reduction
        """
        pred_xyxy = self._to_xyxy(pred)
        target_xyxy = self._to_xyxy(target)
        
        # Intersection coordinates
        x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
        y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
        x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
        y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
        
        # Intersection area
        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter = inter_w * inter_h
        
        # Predicted area
        pred_w = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0)
        pred_h = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
        pred_area = pred_w * pred_h
        
        # Target area
        target_w = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0)
        target_h = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=0)
        target_area = target_w * target_h
        
        # Union area
        union = pred_area + target_area - inter + 1e-7
        
        # IoU and loss
        iou = inter / union
        loss = 1.0 - iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss