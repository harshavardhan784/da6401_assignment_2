"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from models.layers import CustomDropout

CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

class VGG11Encoder(nn.Module):
    def __init__(self, in_channels=3, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm

        # Convolutional backbone — 5 blocks, 8 conv layers total
        self.features = self._make_layers(in_channels)

        # FC layers are part of VGG11 — must include per paper
        # AdaptiveAvgPool handles inputs that aren't exactly 224x224
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            # final Linear(4096, num_classes) goes in the VGG11 model, not encoder
        )

    def _make_layers(self, in_ch):
        layers = []
        for v in CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_ch = v
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        features = {}
        # Walk through manually to capture skip connections
        block_idx = 0
        enc_idx = 1
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                features[f"enc{enc_idx}"] = x
                enc_idx += 1
        x = self.avgpool(x)
        x_flat = torch.flatten(x, 1)
        x_flat = self.classifier(x_flat)
        if return_features:
            return x_flat, features
        return x_flat