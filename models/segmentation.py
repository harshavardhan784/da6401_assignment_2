"""
Segmentation model (U-Net) for DA6401 Assignment 2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg11 import init_weights


def conv_block(in_channels, out_channels, num_convs=2, use_batch_norm=True):
    """Create a convolutional block with multiple conv layers"""
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    return nn.Sequential(*layers)


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks (standard U-Net decoder block)"""
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=not use_batch_norm),
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        init_weights(self.conv)
    
    def forward(self, x):
        return self.conv(x)


class VGG11UNet(nn.Module):
    """
    U-Net with VGG11 encoder (contracting path) and symmetric decoder.
    Uses transposed convolution for upsampling (no bilinear interpolation).
    """
    def __init__(self, num_classes=2, in_channels=3, use_batch_norm=True, freeze_backbone=False):
        super().__init__()
        bn = use_batch_norm
        
        # Encoder (contracting path) - exactly VGG11 structure
        # Each block: conv(s) → ReLU → (BN)
        self.enc1 = conv_block(in_channels, 64, num_convs=1, use_batch_norm=bn)   # 224 → 224
        self.pool1 = nn.MaxPool2d(2)  # 224 → 112
        
        self.enc2 = conv_block(64, 128, num_convs=1, use_batch_norm=bn)   # 112 → 112
        self.pool2 = nn.MaxPool2d(2)  # 112 → 56
        
        self.enc3 = conv_block(128, 256, num_convs=2, use_batch_norm=bn)  # 56 → 56
        self.pool3 = nn.MaxPool2d(2)  # 56 → 28
        
        self.enc4 = conv_block(256, 512, num_convs=2, use_batch_norm=bn)  # 28 → 28
        self.pool4 = nn.MaxPool2d(2)  # 28 → 14
        
        self.enc5 = conv_block(512, 512, num_convs=2, use_batch_norm=bn)  # 14 → 14
        self.pool5 = nn.MaxPool2d(2)  # 14 → 7
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, use_batch_norm=bn)  # 7 → 7
        
        # Decoder (expanding path) with transposed convolutions
        # up5: 1024 → 512, output 7 → 14
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(512 + 512, 512, use_batch_norm=bn)  # concat with enc5
        
        # up4: 512 → 256, output 14 → 28
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256 + 512, 256, use_batch_norm=bn)  # concat with enc4
        
        # up3: 256 → 128, output 28 → 56
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 256, 128, use_batch_norm=bn)  # concat with enc3
        
        # up2: 128 → 64, output 56 → 112
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 128, 64, use_batch_norm=bn)  # concat with enc2
        
        # up1: 64 → 32, output 112 → 224
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32 + 64, 32, use_batch_norm=bn)  # concat with enc1
        
        # Final classification head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)
        init_weights(self.head)
        
        if freeze_backbone:
            for param in self.enc1.parameters(): param.requires_grad = False
            for param in self.enc2.parameters(): param.requires_grad = False
            for param in self.enc3.parameters(): param.requires_grad = False
            for param in self.enc4.parameters(): param.requires_grad = False
            for param in self.enc5.parameters(): param.requires_grad = False

    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)          # (B, 64, 224, 224)
        e2 = self.enc2(self.pool1(e1))  # (B, 128, 112, 112)
        e3 = self.enc3(self.pool2(e2))  # (B, 256, 56, 56)
        e4 = self.enc4(self.pool3(e3))  # (B, 512, 28, 28)
        e5 = self.enc5(self.pool4(e4))  # (B, 512, 14, 14)
        
        # Bottleneck
        b = self.bottleneck(self.pool5(e5))  # (B, 1024, 7, 7)
        
        # Decoder with skip connections
        d5 = self.dec5(torch.cat([self.up5(b), e5], dim=1))  # (B, 512, 14, 14)
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))  # (B, 256, 28, 28)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 128, 56, 56)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 64, 112, 112)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 32, 224, 224)
        
        return self.head(d1)  # (B, num_classes, 224, 224)


class FocalLoss(nn.Module):
    """
    Focal Loss for segmentation to handle foreground-background imbalance.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities for the true class
        pt = torch.exp(-ce_loss)
        
        # Apply focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation - better for imbalanced data.
    Dice = 2|A∩B| / (|A| + |B|)
    """
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # inputs: logits (B, C, H, W), targets: (B, H, W)
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get foreground probability
        probs_fg = probs[:, 1, :, :]  # (B, H, W)
        targets_fg = targets.float()
        
        # Compute Dice
        intersection = (probs_fg * targets_fg).sum()
        union = probs_fg.sum() + targets_fg.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class CombinedSegmentationLoss(nn.Module):
    """
    Combined Cross Entropy + Dice Loss for segmentation
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.0, use_focal=False):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        if use_focal:
            self.ce_loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice