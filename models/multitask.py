"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
import gdown


# ─── VGG11 helpers (must match training definitions) ─────────────────────────

class CustomDropout(nn.Module):
    """Inverted dropout (no nn.Dropout / F.dropout internally)."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f'p must be in [0,1), got {p}')
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * mask / keep

    def extra_repr(self): return f'p={self.p}'


CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class VGG11(nn.Module):
    def __init__(self, in_channels=3, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.features = self._make(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def _make(self, in_ch):
        layers = []
        for v in CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1)]
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_ch = v
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.avgpool(self.features(x))  # (B, 512, 7, 7)


class _DConv(nn.Module):
    """Two Conv-BN-ReLU blocks used in the U-Net decoder."""
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ─── MultiTaskPerceptionModel ─────────────────────────────────────────────────

class MultiTaskPerceptionModel(nn.Module):
    """
    Shared VGG11 backbone → three task heads:
      • Classification  : (B, num_breeds) logits
      • Localization    : (B, 4)  [cx, cy, w, h] in pixel space
      • Segmentation    : (B, seg_classes, H, W) logits

    VGG11 feature indices with BatchNorm enabled:
      idx 0-2  : Conv(3→64)  + BN + ReLU        → (B,  64, 224, 224)   [skip e1]
      idx 3    : MaxPool                          → (B,  64, 112, 112)
      idx 4-6  : Conv(64→128)+ BN + ReLU         → (B, 128, 112, 112)   [skip e2]
      idx 7    : MaxPool                          → (B, 128,  56,  56)
      idx 8-13 : 2×(Conv+BN+ReLU) 128→256        → (B, 256,  56,  56)   [skip e3]
      idx 14   : MaxPool                          → (B, 256,  28,  28)
      idx 15-20: 2×(Conv+BN+ReLU) 256→512        → (B, 512,  28,  28)   [skip e4]
      idx 21   : MaxPool                          → (B, 512,  14,  14)
      idx 22-27: 2×(Conv+BN+ReLU) 512→512        → (B, 512,  14,  14)   [skip e5]
      idx 28   : MaxPool                          → (B, 512,   7,   7)
      avgpool  : AdaptiveAvgPool(7,7)             → (B, 512,   7,   7)   [global]
    """

    def __init__(self, num_breeds: int = 37, seg_classes: int = 2,
                 in_channels: int = 3, image_size: int = 224,
                 classifier_path: str = "checkpoints/classifier.pth",
                 localizer_path:  str = "checkpoints/localizer.pth",
                 unet_path:       str = "checkpoints/unet.pth"):
        super().__init__()

        os.makedirs("checkpoints", exist_ok=True)

        # ── Download weights from Google Drive ───────────────────────────────
        gdown.download(id="1ly8n8hye9XDcoOjp8Mqx4Wz5DAq8LAc2", output=classifier_path, quiet=False)
        gdown.download(id="1Z585cGenqPWQdOMTMgipvG2Xh7syC0Hq",  output=localizer_path,  quiet=False)
        gdown.download(id="1IwAfZohK42rjQ0O_psSJ0zfocqmPEmG1",  output=unet_path,       quiet=False)

        self.image_size  = image_size
        self.num_breeds  = num_breeds
        self.seg_classes = seg_classes

        # ── Shared backbone ──────────────────────────────────────────────────
        self.backbone = VGG11(in_channels=in_channels, use_batch_norm=True)

        # ── Classification head ──────────────────────────────────────────────
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),         nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_breeds),
        )

        # ── Localization head ────────────────────────────────────────────────
        self.localizer_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            CustomDropout(0.3),
            nn.Linear(1024, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        # ── Segmentation decoder ─────────────────────────────────────────────
        self.seg_bot  = _DConv(512, 1024)
        self.seg_up5  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.seg_dec5 = _DConv(512 + 512, 512)
        self.seg_up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.seg_dec4 = _DConv(256 + 512, 256)
        self.seg_up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.seg_dec3 = _DConv(128 + 256, 128)
        self.seg_up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.seg_dec2 = _DConv(64 + 128, 64)
        self.seg_up1  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.seg_dec1 = _DConv(32 + 64, 32)
        self.seg_head = nn.Conv2d(32, seg_classes, 1)

        # ── Load pretrained weights ──────────────────────────────────────────
        self._load_weights(classifier_path, localizer_path, unet_path)

        # Always start in eval mode so BN uses running stats, not batch stats
        self.eval()

    # ── weight loading ────────────────────────────────────────────────────────
    @staticmethod
    def _read_sd(path):
        if not os.path.exists(path):
            print(f"  ⚠️  {path} not found – skipping")
            return {}
        ck = torch.load(path, map_location='cpu')
        return ck.get('model_state_dict', ck)

    def _load_weights(self, cls_path, loc_path, unet_path):
        # Classifier → backbone + classifier_head
        sd = self._read_sd(cls_path)
        if sd:
            print(f"  ✅ Loading classifier from {cls_path}")
            bb  = {k[len('backbone.'):]:   v for k, v in sd.items() if k.startswith('backbone.')}
            clf = {k[len('classifier.'):]: v for k, v in sd.items() if k.startswith('classifier.')}
            if bb:  self.backbone.load_state_dict(bb, strict=False)
            if clf: self.classifier_head.load_state_dict(clf, strict=False)

        # Localizer → localizer_head only (backbone already loaded above)
        sd = self._read_sd(loc_path)
        if sd:
            print(f"  ✅ Loading localizer from {loc_path}")
            reg = {k[len('regressor.'):]: v for k, v in sd.items() if k.startswith('regressor.')}
            if reg: self.localizer_head.load_state_dict(reg, strict=False)

        # UNet → seg decoder
        sd = self._read_sd(unet_path)
        if sd:
            print(f"  ✅ Loading unet from {unet_path}")
            # UNet saves keys like 'bot.net.0.weight', 'up5.weight', etc.
            # We map them to 'seg_bot.net.0.weight', 'seg_up5.weight', etc.
            unet_prefixes = ['bot', 'up5', 'dec5', 'up4', 'dec4',
                             'up3', 'dec3', 'up2', 'dec2', 'up1', 'dec1', 'head']
            for prefix in unet_prefixes:
                layer_sd = {k[len(prefix) + 1:]: v
                            for k, v in sd.items()
                            if k.startswith(prefix + '.')}
                self_attr = f'seg_{prefix}'
                if layer_sd and hasattr(self, self_attr):
                    getattr(self, self_attr).load_state_dict(layer_sd, strict=False)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Single forward pass.

        Returns dict:
          'classification' : (B, num_breeds)              — raw logits
          'localization'   : (B, 4)                        — [cx, cy, w, h] pixels
          'segmentation'   : (B, seg_classes, H, W)        — raw logits
        """
        # Ensure eval mode so BN/Dropout behave correctly during inference
        self.eval()
        feats = self.backbone.features

        # Extract intermediate activations for U-Net skip connections.
        # Indices are fixed for BN-enabled VGG11 (see class docstring).
        e1 = feats[0:3](x)     # (B,  64, 224, 224)
        e2 = feats[3:7](e1)    # (B, 128, 112, 112)  — [3] is MaxPool
        e3 = feats[7:14](e2)   # (B, 256,  56,  56)  — [7] is MaxPool
        e4 = feats[14:21](e3)  # (B, 512,  28,  28)  — [14] is MaxPool
        e5 = feats[21:28](e4)  # (B, 512,  14,  14)  — [21] is MaxPool
        # [28] is last MaxPool → 7×7; avgpool is deterministic to (7,7)
        pooled = self.backbone.avgpool(feats[28](e5))  # (B, 512, 7, 7)

        # Classification
        cls_logits = self.classifier_head(pooled)

        # Localization  (Sigmoid output scaled to pixel space)
        loc_out = self.localizer_head(pooled) * self.image_size

        # Segmentation  (U-Net decoder)
        # e5 is 14×14; bottleneck expects 7×7 (same as UNet's self.bot(self.pool(e5)))
        _pool = torch.nn.functional.max_pool2d
        b  = self.seg_bot(_pool(e5, 2, 2))
        d5 = self.seg_dec5(torch.cat([self.seg_up5(b),  e5], 1))
        d4 = self.seg_dec4(torch.cat([self.seg_up4(d5), e4], 1))
        d3 = self.seg_dec3(torch.cat([self.seg_up3(d4), e3], 1))
        d2 = self.seg_dec2(torch.cat([self.seg_up2(d3), e2], 1))
        d1 = self.seg_dec1(torch.cat([self.seg_up1(d2), e1], 1))
        seg_logits = self.seg_head(d1)

        return {
            'classification': cls_logits,
            'localization':   loc_out,
            'segmentation':   seg_logits,
        }


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import torch

    # Build without downloading weights
    m = MultiTaskPerceptionModel.__new__(MultiTaskPerceptionModel)
    nn.Module.__init__(m)
    m.image_size  = 224
    m.num_breeds  = 37
    m.seg_classes = 2
    m.backbone         = VGG11(3, True)
    m.classifier_head  = nn.Sequential(
        nn.Flatten(), nn.Linear(512*7*7, 4096), nn.BatchNorm1d(4096), nn.ReLU(),
        CustomDropout(0.5), nn.Linear(4096, 4096), nn.BatchNorm1d(4096), nn.ReLU(),
        CustomDropout(0.5), nn.Linear(4096, 37))
    m.localizer_head = nn.Sequential(
        nn.Flatten(), nn.Linear(512*7*7, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
        CustomDropout(0.3), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 4), nn.Sigmoid())
    m.seg_bot  = _DConv(512, 1024)
    m.seg_up5  = nn.ConvTranspose2d(1024, 512, 2, stride=2); m.seg_dec5 = _DConv(512+512, 512)
    m.seg_up4  = nn.ConvTranspose2d(512,  256, 2, stride=2); m.seg_dec4 = _DConv(256+512, 256)
    m.seg_up3  = nn.ConvTranspose2d(256,  128, 2, stride=2); m.seg_dec3 = _DConv(128+256, 128)
    m.seg_up2  = nn.ConvTranspose2d(128,   64, 2, stride=2); m.seg_dec2 = _DConv( 64+128,  64)
    m.seg_up1  = nn.ConvTranspose2d( 64,   32, 2, stride=2); m.seg_dec1 = _DConv( 32+ 64,  32)
    m.seg_head = nn.Conv2d(32, 2, 1)
    m.eval()

    with torch.no_grad():
        out = MultiTaskPerceptionModel.forward(m, torch.randn(2, 3, 224, 224))

    assert out['classification'].shape == (2, 37),         f"cls: {out['classification'].shape}"
    assert out['localization'].shape   == (2, 4),           f"loc: {out['localization'].shape}"
    assert out['segmentation'].shape   == (2, 2, 224, 224), f"seg: {out['segmentation'].shape}"
    print("✅ All shapes correct")
    print(f"  classification : {out['classification'].shape}")
    print(f"  localization   : {out['localization'].shape}")
    print(f"  segmentation   : {out['segmentation'].shape}")