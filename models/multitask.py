import os
import torch
import torch.nn as nn

IMAGE_SIZE = 224


class MultiTaskPerceptionModel(nn.Module):
    """
    Shared VGG11 backbone + three heads:
      classification : [B, 37] logits
      localization   : [B, 4]  [cx, cy, w, h] pixel space
      segmentation   : [B, 3, 224, 224] logits
    """

    def __init__(self, num_breeds=37, seg_classes=3, in_channels=3,
                 classifier_path="checkpoints/classifier.pth",
                 localizer_path="checkpoints/localizer.pth",
                 unet_path="checkpoints/unet.pth"):
        super().__init__()

        from models.vgg11        import VGG11, init_weights
        from models.layers       import CustomDropout
        from models.segmentation import _DoubleConv

        bn = True
        # Shared backbone
        self.backbone = VGG11(in_channels, use_batch_norm=bn)

        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096), nn.BatchNorm1d(4096), nn.ReLU(True), CustomDropout(0.5),
            nn.Linear(4096, 4096),    nn.BatchNorm1d(4096), nn.ReLU(True), CustomDropout(0.5),
            nn.Linear(4096, num_breeds),
        )

        self.loc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 1024), nn.BatchNorm1d(1024), nn.ReLU(True), CustomDropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(True), nn.Linear(256, 4), nn.Sigmoid(),
        )

        self.up5=nn.ConvTranspose2d(512,512,2,stride=2); self.dec5=_DoubleConv(1024,512,bn)
        self.up4=nn.ConvTranspose2d(512,256,2,stride=2); self.dec4=_DoubleConv( 768,256,bn)
        self.up3=nn.ConvTranspose2d(256,128,2,stride=2); self.dec3=_DoubleConv( 384,128,bn)
        self.up2=nn.ConvTranspose2d(128, 64,2,stride=2); self.dec2=_DoubleConv( 192, 64,bn)
        self.up1=nn.ConvTranspose2d( 64, 32,2,stride=2); self.dec1=_DoubleConv(  96, 32,bn)
        self.seg_head=nn.Conv2d(32,seg_classes,1)

        # Load weights if paths exist
        if os.path.exists(classifier_path):
            self._load_cls(classifier_path)
        if os.path.exists(localizer_path):
            self._load_loc(localizer_path)
        if os.path.exists(unet_path):
            self._load_seg(unet_path)

    @staticmethod
    def _sd(path):
        ck=torch.load(path, map_location="cpu")
        return ck.get("model_state_dict", ck)

    def _load_cls(self, path):
        sd=self._sd(path)
        self.backbone.load_state_dict({k[len("backbone."):]:v for k,v in sd.items() if k.startswith("backbone.")}, strict=True)
        self.cls_head.load_state_dict({k[len("classifier."):]:v for k,v in sd.items() if k.startswith("classifier.")}, strict=True)
        print(f"  Loaded classifier from {path}")

    def _load_loc(self, path):
        sd=self._sd(path)
        self.loc_head.load_state_dict({k[len("regressor."):]:v for k,v in sd.items() if k.startswith("regressor.")}, strict=True)
        print(f"  Loaded localizer from {path}")

    def _load_seg(self, path):
        sd=self._sd(path)
        remap={"up5":"up5","dec5":"dec5","up4":"up4","dec4":"dec4",
               "up3":"up3","dec3":"dec3","up2":"up2","dec2":"dec2",
               "up1":"up1","dec1":"dec1","head":"seg_head"}
        new={remap[k.split(".")[0]]+k[len(k.split(".")[0]):]:v
              for k,v in sd.items() if k.split(".")[0] in remap}
        m,u=self.load_state_dict(new, strict=False)
        print(f"  Loaded UNet from {path} (missing={len(m)}, unexpected={len(u)})")

    def forward(self, x):
        f=self.backbone.features
        e1=f[:3](x);     p1=f[3](e1)
        e2=f[4:7](p1);   p2=f[7](e2)
        e3=f[8:14](p2);  p3=f[14](e3)
        e4=f[15:21](p3); p4=f[21](e4)
        e5=f[22:28](p4); p5=f[28](e5)

        cls_out = self.cls_head(p5)
        loc_out = self.loc_head(p5) * IMAGE_SIZE

        d5=self.dec5(torch.cat([self.up5(p5),e5],1))
        d4=self.dec4(torch.cat([self.up4(d5),e4],1))
        d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        seg_out=self.seg_head(d1)

        return {"classification":cls_out, "localization":loc_out, "segmentation":seg_out}
