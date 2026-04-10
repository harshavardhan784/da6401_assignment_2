"""
Localization model for DA6401 Assignment 2
"""
import os
import torch
import torch.nn as nn
from .vgg11 import VGG11, init_weights
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """
    VGG11-based object localizer.
    Output: [cx, cy, w, h] in pixel space (0 to IMAGE_SIZE)
    """
    def __init__(self, in_channels=3, use_batch_norm=True, image_size=224, freeze_backbone=False):
        super().__init__()
        self.image_size = image_size
        self.backbone = VGG11(in_channels, use_batch_norm)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # Output in [0, 1] range
        )
        init_weights(self.regressor)

    def load_backbone_weights(self, checkpoint_path):
        """Load pretrained backbone weights from classifier checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Filter backbone weights
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                backbone_state[k.replace('backbone.', '')] = v
            elif k.startswith('backbone'):
                backbone_state[k] = v
        
        if backbone_state:
            self.backbone.load_state_dict(backbone_state, strict=True)
            print(f"  Loaded backbone weights from {checkpoint_path}")
        else:
            print(f"  Warning: No backbone weights found in {checkpoint_path}")

    def forward(self, x):
        features = self.backbone(x)
        output = self.regressor(features)
        return output * self.image_size  # Scale to pixel space


class LocalizationLoss(nn.Module):
    """
    Combined loss for localization: MSE + IoU loss
    """
    def __init__(self, mse_weight=0.5, iou_weight=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.iou_weight = iou_weight
        self.mse_loss = nn.MSELoss()
        # Import IoULoss here to avoid circular import
        from losses.iou_loss import IoULoss
        self.iou_loss = IoULoss('mean')
    
    def forward(self, pred, target, image_size=224):
        mse = self.mse_loss(pred, target) / (image_size ** 2)
        iou = self.iou_loss(pred, target)
        return self.mse_weight * mse + self.iou_weight * iou