"""VGG11-based object localization model."""

import torch
import torch.nn as nn
from models.layers import CustomDropout
from models.vgg11 import VGG11, init_weights

IMAGE_SIZE = 224  # Fixed VGG11 input size


class VGG11Localizer(nn.Module):
    """
    VGG11 backbone + regression head for bounding box prediction.
    Output: [cx, cy, w, h] in pixel space (0 to IMAGE_SIZE).
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5,
                 use_batch_norm: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.image_size = IMAGE_SIZE
        self.backbone = VGG11(in_channels, use_batch_norm)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # Output in [0, 1], then scaled to pixel space
        )
        init_weights(self.regressor)

    def load_backbone_weights(self, checkpoint_path: str):
        """Load pretrained backbone weights from a classifier checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        backbone_state = {
            k.replace('backbone.', ''): v
            for k, v in state_dict.items()
            if k.startswith('backbone.')
        }

        if backbone_state:
            self.backbone.load_state_dict(backbone_state, strict=True)
            print(f"  Loaded backbone weights from {checkpoint_path}")
        else:
            print(f"  Warning: No backbone weights found in {checkpoint_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Tensor of shape [B, 4] → [cx, cy, w, h] in pixel coordinates.
        """
        features = self.backbone(x)
        output = self.regressor(features)
        return output * self.image_size  # Scale [0,1] → pixel space