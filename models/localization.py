"""VGG11 localization head — outputs [cx, cy, w, h] in PIXEL space [0..image_size]."""
import torch
import torch.nn as nn
from models.layers import CustomDropout
from models.vgg11  import VGG11Encoder

IMAGE_SIZE = 224


class VGG11Localizer(nn.Module):
    """
    Predicts [cx, cy, w, h] in pixel space [0..IMAGE_SIZE].
    Sigmoid keeps raw output in [0,1], then we scale by IMAGE_SIZE.
    The dataset (pets_dataset.py) already returns GT in pixel space,
    so prediction and GT are now in the same coordinate system → IoU is correct.
    """
    def __init__(self, dropout_p=0.5, freeze_backbone=False):
        super().__init__()
        self.backbone = VGG11Encoder(return_features=False)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(inplace=True), CustomDropout(dropout_p),
            nn.Linear(1024, 256),          nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),   # → [0, 1]; multiply by IMAGE_SIZE below
        )
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_backbone_weights(self, path):
        ck = torch.load(path, map_location="cpu")
        sd = ck.get("model_state_dict", ck)
        enc = {k[len("features."):]: v for k, v in sd.items() if k.startswith("features.")}
        if enc:
            self.backbone.load_state_dict(enc, strict=True)
            print(f"  Backbone loaded from {path}")
        else:
            # try backbone. prefix (from this model's own checkpoint)
            enc2 = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
            if enc2:
                self.backbone.load_state_dict(enc2, strict=True)
                print(f"  Backbone loaded (backbone. prefix) from {path}")

    def forward(self, x):
        # Output in pixel space: [0..IMAGE_SIZE]
        return self.regressor(self.backbone(x)) * IMAGE_SIZE