# %%writefile /kaggle/working/models/localization.py
"""VGG11 localization — output in [0,1] (NO * image_size)."""
import torch
import torch.nn as nn
from models.layers import CustomDropout
from models.vgg11  import VGG11Encoder


class VGG11Localizer(nn.Module):
    """
    Predicts [cx, cy, w, h] all in [0, 1].
    FIX: removed * image_size. GT bbox from dataloader is also [0,1],
    so prediction and GT are in the same space -> IoU computed correctly.
    """
    def __init__(self, dropout_p=0.5, freeze_backbone=False):
        super().__init__()
        self.backbone = VGG11Encoder(return_features=False)
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 1024), nn.ReLU(inplace=True), CustomDropout(dropout_p),
            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 4), nn.Sigmoid(),   # [0,1] — DO NOT multiply by image_size
        )
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def load_backbone_weights(self, path):
        import torch
        ck = torch.load(path, map_location="cpu")
        sd = ck.get("model_state_dict", ck)
        enc = {k[len("features."):]:v for k,v in sd.items() if k.startswith("features.")}
        if enc:
            self.backbone.load_state_dict(enc, strict=True)
            print(f"  Backbone loaded from {path}")

    def forward(self, x):
        return self.regressor(self.backbone(x))  # [0, 1]
