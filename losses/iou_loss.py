"""Custom IoU loss for bounding box regression
"""

import torch
import torch.nn as nn

# ─── Cell 6: IoU Loss ─────────────────────────────────────────────────────────
class IoULoss(nn.Module):
    """
    1 - IoU  →  range [0,1].  Input: [cx, cy, w, h] in pixel space.
    reduction: 'mean' (default) | 'sum' | 'none'
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ('mean','sum','none')
        self.reduction = reduction

    @staticmethod
    def _to_xyxy(b):
        cx,cy,w,h = b.unbind(-1)
        return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)

    def forward(self, pred, target):
        p = self._to_xyxy(pred); t = self._to_xyxy(target)
        ix1 = torch.max(p[:,0],t[:,0]); iy1 = torch.max(p[:,1],t[:,1])
        ix2 = torch.min(p[:,2],t[:,2]); iy2 = torch.min(p[:,3],t[:,3])
        inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
        ap    = (p[:,2]-p[:,0]).clamp(0) * (p[:,3]-p[:,1]).clamp(0)
        at    = (t[:,2]-t[:,0]).clamp(0) * (t[:,3]-t[:,1]).clamp(0)
        iou   = inter / (ap + at - inter + 1e-7)
        loss  = 1.0 - iou
        if self.reduction=='mean': return loss.mean()
        if self.reduction=='sum':  return loss.sum()
        return loss
