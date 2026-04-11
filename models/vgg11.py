"""VGG11 encoder backbone."""

import torch
import torch.nn as nn
from models.layers import CustomDropout

# Standard VGG11 config: int = conv out_channels, 'M' = MaxPool
CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


def init_weights(module: nn.Module):
    """Kaiming init for Conv2d and Linear; ones/zeros for BN."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class VGG11(nn.Module):
    """
    VGG11 convolutional backbone (feature extractor only, no classifier head).
    Output: (B, 512, 7, 7) for 224x224 input.
    """

    def __init__(self, in_channels: int = 3, use_batch_norm: bool = True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.features = self._make_layers(in_channels)
        init_weights(self)

    def _make_layers(self, in_ch: int) -> nn.Sequential:
        layers = []
        for v in CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1, bias=not self.use_batch_norm))
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_ch = v
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class VGG11Encoder(nn.Module):
    """
    Full VGG11 encoder with adaptive pooling and FC layers.
    Used as a standalone feature extractor.
    """

    def __init__(self, in_channels: int = 3, use_batch_norm: bool = True):
        super().__init__()
        self.backbone = VGG11(in_channels, use_batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
        )
        init_weights(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)