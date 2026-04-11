"""Classification components
"""

import torch
import torch.nn as nn
from models.layers import CustomDropout

class VGG11Classifier(nn.Module):
    """
    VGG11 for classification with custom dropout and batch norm.
    Architecture: VGG11 backbone → Flatten → FC4096 → BN → ReLU → Dropout → FC4096 → BN → ReLU → Dropout → FC37
    """
    def __init__(self, num_classes=37, in_channels=3, dropout_p=0.5, use_batch_norm=True):
        super().__init__()
        self.backbone = VGG11(in_channels, use_batch_norm)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )
        init_weights(self.classifier)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x):
        """Extract features before classifier (for visualization)"""
        return self.backbone(x)