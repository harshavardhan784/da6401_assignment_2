"""VGG11 U-Net style segmentation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg11 import init_weights


def _conv_block(in_ch: int, out_ch: int, num_convs: int = 2,
                use_batch_norm: bool = True) -> nn.Sequential:
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        in_ch = out_ch
    return nn.Sequential(*layers)


class _DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks used in the decoder."""

    def __init__(self, in_ch: int, out_ch: int, use_batch_norm: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        init_weights(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VGG11UNet(nn.Module):
    """
    U-Net with VGG11 encoder and symmetric transposed-conv decoder.
    Input:  (B, in_channels, 224, 224)
    Output: (B, num_classes, 224, 224) logits
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3,
                 use_batch_norm: bool = True, freeze_backbone: bool = False):
        super().__init__()
        bn = use_batch_norm

        # ── Encoder (mirrors VGG11 blocks) ──────────────────────────────
        self.enc1 = _conv_block(in_channels, 64,  num_convs=1, use_batch_norm=bn)  # →224
        self.pool1 = nn.MaxPool2d(2)  # →112

        self.enc2 = _conv_block(64,  128, num_convs=1, use_batch_norm=bn)  # →112
        self.pool2 = nn.MaxPool2d(2)  # →56

        self.enc3 = _conv_block(128, 256, num_convs=2, use_batch_norm=bn)  # →56
        self.pool3 = nn.MaxPool2d(2)  # →28

        self.enc4 = _conv_block(256, 512, num_convs=2, use_batch_norm=bn)  # →28
        self.pool4 = nn.MaxPool2d(2)  # →14

        self.enc5 = _conv_block(512, 512, num_convs=2, use_batch_norm=bn)  # →14
        self.pool5 = nn.MaxPool2d(2)  # →7

        # ── Bottleneck ───────────────────────────────────────────────────
        self.bottleneck = _DoubleConv(512, 1024, use_batch_norm=bn)  # 7→7

        # ── Decoder (transposed convs + skip concat) ─────────────────────
        self.up5   = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5  = _DoubleConv(512 + 512, 512, use_batch_norm=bn)   # 7→14

        self.up4   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4  = _DoubleConv(256 + 512, 256, use_batch_norm=bn)   # 14→28

        self.up3   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3  = _DoubleConv(128 + 256, 128, use_batch_norm=bn)   # 28→56

        self.up2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2  = _DoubleConv(64  + 128, 64,  use_batch_norm=bn)   # 56→112

        self.up1   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1  = _DoubleConv(32  + 64,  32,  use_batch_norm=bn)   # 112→224

        # ── Classification head ──────────────────────────────────────────
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)
        init_weights(self.head)

        if freeze_backbone:
            for block in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
                for p in block.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)                    # (B,  64, 224, 224)
        e2 = self.enc2(self.pool1(e1))        # (B, 128, 112, 112)
        e3 = self.enc3(self.pool2(e2))        # (B, 256,  56,  56)
        e4 = self.enc4(self.pool3(e3))        # (B, 512,  28,  28)
        e5 = self.enc5(self.pool4(e4))        # (B, 512,  14,  14)

        # Bottleneck
        b  = self.bottleneck(self.pool5(e5))  # (B,1024,   7,   7)

        # Decoder
        d5 = self.dec5(torch.cat([self.up5(b),  e5], dim=1))  # (B, 512, 14, 14)
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))  # (B, 256, 28, 28)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 128, 56, 56)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B,  64,112,112)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B,  32,224,224)

        return self.head(d1)                  # (B, num_classes, 224, 224)


# ── Loss functions ────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss over all classes (macro average)."""

    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, H, W) logits  |  targets: (B, H, W) long
        num_classes = inputs.shape[1]
        probs = F.softmax(inputs, dim=1)                       # (B, C, H, W)
        targets_oh = F.one_hot(targets, num_classes)           # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()   # (B, C, H, W)

        intersection = (probs * targets_oh).sum(dim=(0, 2, 3))
        union        = probs.sum(dim=(0, 2, 3)) + targets_oh.sum(dim=(0, 2, 3))
        dice         = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """Cross-entropy + Dice loss."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss     = nn.CrossEntropyLoss()
        self.dice_loss   = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce_loss(pred, target) \
             + self.dice_weight * self.dice_loss(pred, target)