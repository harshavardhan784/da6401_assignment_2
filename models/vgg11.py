"""VGG11 encoder with optional skip-connection return."""
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """
    VGG11 conv backbone.
    forward(x, return_features=False) -> bottleneck tensor
    forward(x, return_features=True)  -> (s1, s2, s3, s4, bottleneck)
    This matches the friend's working code exactly.
    """
    def __init__(self, return_features=False):
        super().__init__()
        self.return_features = return_features
        self.block1 = nn.Sequential(conv_block(3,   64),  nn.MaxPool2d(2,2))
        self.block2 = nn.Sequential(conv_block(64,  128), nn.MaxPool2d(2,2))
        self.block3 = nn.Sequential(conv_block(128, 256), conv_block(256,256), nn.MaxPool2d(2,2))
        self.block4 = nn.Sequential(conv_block(256, 512), conv_block(512,512), nn.MaxPool2d(2,2))
        self.block5 = nn.Sequential(conv_block(512, 512), conv_block(512,512), nn.MaxPool2d(2,2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1 = self.block1(x)    # (B,  64, 112, 112)
        s2 = self.block2(s1)   # (B, 128,  56,  56)
        s3 = self.block3(s2)   # (B, 256,  28,  28)
        s4 = self.block4(s3)   # (B, 512,  14,  14)
        bn = self.block5(s4)   # (B, 512,   7,   7)
        if self.return_features:
            return s1, s2, s3, s4, bn
        return bn
