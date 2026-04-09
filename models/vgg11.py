"""VGG11 encoder
"""

import torch
import torch.nn as nn


CFG = [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M']
def init_weights(module: nn.Module):
    """Apply Kaiming init to all Conv2d and Linear sub-modules."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # fan_out = good for deep conv stacks (backward variance)
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # fan_in = good for FC layers (forward variance), uniform distribution
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class VGG11(nn.Module):
    """Pure VGG11 backbone (features + adaptive avgpool)."""
    def __init__(self, in_channels=3, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.features = self._make(in_channels)
        self.avgpool  = nn.AdaptiveAvgPool2d((7,7))
        init_weights(self)

    def _make(self, in_ch):
        layers = []
        for v in CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(2,2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1)]
                if self.use_batch_norm: layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_ch = v
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.avgpool(self.features(x))   # (B,512,7,7)

class VGG11Encoder(nn.Module):
    """Pure VGG11 backbone (features + adaptive avgpool)."""
    def __init__(self, in_channels=3, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.features = self._make(in_channels)
        self.avgpool  = nn.AdaptiveAvgPool2d((7,7))
        init_weights(self)

    def _make(self, in_ch):
        layers = []
        for v in CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(2,2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1)]
                if self.use_batch_norm: layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_ch = v
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.avgpool(self.features(x))   # (B,512,7,7)
