# %%writefile /kaggle/working/models/classification.py
"""VGG11 classification model — matches friend's working ClassificationModel."""
import torch
import torch.nn as nn
from layers import CustomDropout
from vgg11  import VGG11Encoder


class VGG11Classifier(nn.Module):
    def __init__(self, num_classes=37, dropout_p=0.5):
        super().__init__()
        self.features = VGG11Encoder(return_features=False)
        self.avgpool  = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True), CustomDropout(dropout_p),
            nn.Linear(4096, 4096),    nn.ReLU(inplace=True), CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def set_encoder_weight(self, path):
        import torch
        sd = torch.load(path, map_location="cpu")
        # handle checkpoint dicts
        sd = sd.get("model_state_dict", sd)
        # strip "features." prefix if present
        enc_sd = {k[len("features."):]:v for k,v in sd.items() if k.startswith("features.")}
        if enc_sd:
            self.features.load_state_dict(enc_sd, strict=True)
        else:
            self.features.load_state_dict(sd, strict=False)
        print(f"  Encoder weights loaded from {path}")

    def forward(self, x):
        return self.classifier(self.avgpool(self.features(x)))
