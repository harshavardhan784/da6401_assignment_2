"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        assert reduction in ('mean', 'sum', 'none'), \
            f"reduction must be 'mean', 'sum', or 'none', got {reduction}"

    @staticmethod
    def _to_xyxy(boxes: torch.Tensor):
        """Convert boxes from (cx, cy, w, h) to (x_min, y_min, x_max, y_max) format."""
        cx, cy, w, h = boxes.unbind(dim=-1)
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        # TODO: implement IoU loss.
        raise NotImplementedError("Implement IoULoss.forward")
    
    
        pred_xyxy = self._to_xyxy(pred_boxes)
        target_xyxy = self._to_xyxy(target_boxes)
        
        # Compute intersection
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