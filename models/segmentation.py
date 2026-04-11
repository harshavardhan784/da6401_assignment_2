"""VGG11 U-Net — uses VGG11Encoder(return_features=True) for real skip connections."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg11 import VGG11Encoder


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """
    FIX: encoder is VGG11Encoder(return_features=True) which returns
    (s1:64, s2:128, s3:256, s4:512, bottleneck:512) — real skip connections.
    Old code had a flat VGG11 + separate enc1-enc5 blocks that were
    completely disconnected, giving garbage decoder inputs and F1=0.
    """
    def __init__(self, num_classes=3, freeze_backbone=False):
        super().__init__()
        self.encoder = VGG11Encoder(return_features=True)
        if freeze_backbone:
            for p in self.encoder.parameters(): p.requires_grad = False

        self.up1  = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec1 = _double_conv(512+512, 512)   # cat with s4
        self.up2  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = _double_conv(256+256, 256)   # cat with s3
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _double_conv(128+128, 128)   # cat with s2
        self.up4  = nn.ConvTranspose2d(128,  64, 2, stride=2)
        self.dec4 = _double_conv( 64+ 64,  64)   # cat with s1
        self.up5  = nn.ConvTranspose2d( 64,  64, 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def set_encoder_weight(self, path):
        import torch
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        enc = {k[len("features."):]:v for k,v in sd.items() if k.startswith("features.")}
        if enc:
            self.encoder.load_state_dict(enc, strict=False)
            print(f"  Encoder loaded from {path}")

    def forward(self, x):
        s1, s2, s3, s4, bn = self.encoder(x)
        x = self.dec1(torch.cat([self.up1(bn), s4], dim=1))
        x = self.dec2(torch.cat([self.up2(x),  s3], dim=1))
        x = self.dec3(torch.cat([self.up3(x),  s2], dim=1))
        x = self.dec4(torch.cat([self.up4(x),  s1], dim=1))
        x = self.up5(x)
        return self.final_conv(x)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): super().__init__(); self.smooth=smooth
    def forward(self, pred, target):
        nc=pred.shape[1]; probs=F.softmax(pred,dim=1)
        oh=F.one_hot(target,nc).permute(0,3,1,2).float()
        inter=(probs*oh).sum(dim=(0,2,3))
        union=probs.sum(dim=(0,2,3))+oh.sum(dim=(0,2,3))
        return 1.0-((2.*inter+self.smooth)/(union+self.smooth)).mean()


class CombinedSegmentationLoss(nn.Module):
    def __init__(self, ce_w=1.0, dice_w=1.0):
        super().__init__()
        self.ce=nn.CrossEntropyLoss(); self.dice=DiceLoss()
        self.cw=ce_w; self.dw=dice_w
    def forward(self, pred, target):
        return self.cw*self.ce(pred,target)+self.dw*self.dice(pred,target)
