"""Unified multi-task perception model."""

import torch
import torch.nn as nn

IMAGE_SIZE = 224

# VGG11 layer indices in the features Sequential that correspond to each encoder block.
# CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
# With BN the layers are: Conv,BN,ReLU, Pool, Conv,BN,ReLU, Pool, ...
# Without BN:              Conv,ReLU,   Pool, Conv,ReLU,   Pool, ...
# We slice the backbone.features Sequential manually in forward.


class MultiTaskPerceptionModel(nn.Module):
    """
    Shared VGG11 backbone with three task heads:
      • classification : [B, 37] logits
      • localization   : [B, 4]  [cx, cy, w, h] in pixel space
      • segmentation   : [B, 3, 224, 224] logits
    """

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3,
                 in_channels: int = 3,
                 classifier_path: str = "checkpoints/classifier.pth",
                 localizer_path:  str = "checkpoints/localizer.pth",
                 unet_path:       str = "checkpoints/unet.pth"):
        super().__init__()

        import gdown
        # https://drive.google.com/file/d/1bQatPpJxWBYuzZA949igWJh5OrADKrYM/view?usp=sharing
        gdown.download(id="1bQatPpJxWBYuzZA949igWJh5OrADKrYM", output=classifier_path, quiet=False)
        gdown.download(id="1gKu5L9hSIAFMJuOMxiUqHIVm5EbIScKD",  output=localizer_path,  quiet=False)
        gdown.download(id="1aWRiSNzmgdk3WbTOppXUfJ6Mkk6OUIA4",       output=unet_path,       quiet=False)

        from models.vgg11        import VGG11, init_weights
        from models.layers       import CustomDropout

        bn = True

        # ── Shared VGG11 backbone ────────────────────────────────────────
        # We keep enc1..enc5 + pool1..pool5 as a single Sequential and
        # slice it during forward to grab skip-connection feature maps.
        self.backbone = VGG11(in_channels, use_batch_norm=bn)
        # backbone.features layout (with BN):
        #  [0] Conv  [1] BN  [2] ReLU            ← enc1 output after [2]
        #  [3] MaxPool                             ← pool1
        #  [4] Conv  [5] BN  [6] ReLU            ← enc2 output after [6]
        #  [7] MaxPool                             ← pool2
        #  [8] Conv  [9] BN  [10] ReLU
        #  [11] Conv [12] BN [13] ReLU            ← enc3 output after [13]
        #  [14] MaxPool                            ← pool3
        #  [15] Conv [16] BN [17] ReLU
        #  [18] Conv [19] BN [20] ReLU            ← enc4 output after [20]
        #  [21] MaxPool                            ← pool4
        #  [22] Conv [23] BN [24] ReLU
        #  [25] Conv [26] BN [27] ReLU            ← enc5 output after [27]
        #  [28] MaxPool                            ← pool5 → (B,512,7,7)

        # ── Classification head ──────────────────────────────────────────
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_breeds),
        )

        # ── Localization head ────────────────────────────────────────────
        self.loc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        # ── Segmentation decoder ─────────────────────────────────────────
        from models.segmentation import _DoubleConv
        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _DoubleConv(512 + 512, 512, bn)

        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(256 + 512, 256, bn)

        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(128 + 256, 128, bn)

        self.up2  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(64  + 128, 64,  bn)

        self.up1  = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(32  + 64,  32,  bn)

        self.seg_head = nn.Conv2d(32, seg_classes, kernel_size=1)

        # ── Load pre-trained weights ─────────────────────────────────────
        self._load_classifier(classifier_path)
        self._load_localizer(localizer_path)
        self._load_unet(unet_path, seg_classes)

    # ── Weight loaders ────────────────────────────────────────────────────

    @staticmethod
    def _get_sd(path):
        ckpt = torch.load(path, map_location='cpu')
        return ckpt.get('model_state_dict', ckpt)

    def _load_classifier(self, path):
        sd = self._get_sd(path)
        bb  = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
        cls = {k[len('classifier.'):]: v for k, v in sd.items() if k.startswith('classifier.')}
        self.backbone.load_state_dict(bb, strict=True)
        self.cls_head.load_state_dict(cls, strict=True)
        print(f"  Loaded classifier from {path}")

    def _load_localizer(self, path):
        sd  = self._get_sd(path)
        loc = {k[len('regressor.'):]: v for k, v in sd.items() if k.startswith('regressor.')}
        self.loc_head.load_state_dict(loc, strict=True)
        print(f"  Loaded localizer from {path}")

    def _load_unet(self, path, seg_classes):
        sd = self._get_sd(path)
        # Map UNet attribute names → MultiTask attribute names
        remap = {
            'up5': 'up5', 'dec5': 'dec5',
            'up4': 'up4', 'dec4': 'dec4',
            'up3': 'up3', 'dec3': 'dec3',
            'up2': 'up2', 'dec2': 'dec2',
            'up1': 'up1', 'dec1': 'dec1',
            'head': 'seg_head',
        }
        new_sd = {}
        for k, v in sd.items():
            prefix = k.split('.')[0]
            if prefix in remap:
                new_k = remap[prefix] + k[len(prefix):]
                new_sd[new_k] = v
        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        print(f"  Loaded UNet decoder from {path}  "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            dict with keys 'classification', 'localization', 'segmentation'
        """
        feats = self.backbone.features

        # ── Run through encoder, saving skip maps ──
        # enc1: layers 0..2  (Conv-BN-ReLU)
        e1 = feats[:3](x)          # (B,  64, 224, 224)
        p1 = feats[3](e1)          # (B,  64, 112, 112)  pool1

        # enc2: layers 4..6
        e2 = feats[4:7](p1)        # (B, 128, 112, 112)
        p2 = feats[7](e2)          # (B, 128,  56,  56)  pool2

        # enc3: layers 8..13
        e3 = feats[8:14](p2)       # (B, 256,  56,  56)
        p3 = feats[14](e3)         # (B, 256,  28,  28)  pool3

        # enc4: layers 15..20
        e4 = feats[15:21](p3)      # (B, 512,  28,  28)
        p4 = feats[21](e4)         # (B, 512,  14,  14)  pool4

        # enc5: layers 22..27
        e5 = feats[22:28](p4)      # (B, 512,  14,  14)
        p5 = feats[28](e5)         # (B, 512,   7,   7)  pool5

        # ── Classification (cls_head starts with nn.Flatten) ─────────────
        cls_out = self.cls_head(p5)

        # ── Localization (loc_head starts with nn.Flatten) ────────────────
        loc_out = self.loc_head(p5) * IMAGE_SIZE

        # ── Segmentation decoder ──────────────────
        d5  = self.dec5(torch.cat([self.up5(p5), e5], dim=1))
        d4  = self.dec4(torch.cat([self.up4(d5), e4], dim=1))
        d3  = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2  = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1  = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg_out = self.seg_head(d1)

        return {
            'classification': cls_out,
            'localization':   loc_out,
            'segmentation':   seg_out,
        }