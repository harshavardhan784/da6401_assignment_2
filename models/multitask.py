"""
Multi-task perception model for DA6401 Assignment 2
"""
import os
import torch
import torch.nn as nn
from models.vgg11 import VGG11, init_weights
from models.layers import CustomDropout


class MultiTaskPerceptionModel(nn.Module):
    """
    Unified multi-task model:
      - Classification  -> (B, 37)
      - Localization    -> (B, 4)   [cx, cy, w, h] pixel space [0..224]
      - Segmentation    -> (B, 3, H, W)  trimap classes: 0=pet, 1=background, 2=border
    """
    def __init__(self, num_classes=37, num_seg_classes=3,
                 use_batch_norm=True, dropout_p=0.5):
        super().__init__()

        os.makedirs('checkpoints', exist_ok=True)
        import gdown

        classifier_path = "checkpoints/classifier.pth"
        localizer_path  = "checkpoints/localizer.pth"
        unet_path       = "checkpoints/unet.pth"

        # ── Download checkpoints ──────────────────────────────────────────────
        if not os.path.exists(classifier_path):
            gdown.download(id="1Y4hCiPdUe1WLA9XsQD0nq9ocsH55__da",
                           output=classifier_path, quiet=False)
        if not os.path.exists(localizer_path):
            gdown.download(id="1gKu5L9hSIAFMJuOMxiUqHIVm5EbIScKD",
                           output=localizer_path,  quiet=False)
        if not os.path.exists(unet_path):
            gdown.download(id="1aWRiSNzmgdk3WbTOppXUfJ6Mkk6OUIA4",
                           output=unet_path,        quiet=False)

        # ── Shared backbone ───────────────────────────────────────────────────
        self.backbone = VGG11(in_channels=3, use_batch_norm=use_batch_norm)

        # ── Classification head (same structure as VGG11Classifier) ───────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

        # ── Localization head (same as VGG11Localizer) ────────────────────────
        self.localizer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            CustomDropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        # ── Segmentation head: full U-Net ─────────────────────────────────────
        from models.segmentation import VGG11UNet
        self.seg_model = VGG11UNet(num_classes=num_seg_classes,
                                   use_batch_norm=use_batch_norm)

        # Initialise with Kaiming (will be overwritten by loaded weights)
        init_weights(self.classifier)
        init_weights(self.localizer)

        # Load pretrained weights using full model copies (avoids key mismatches)
        self._load_pretrained(classifier_path, localizer_path, unet_path)

        self.eval()

    # -------------------------------------------------------------------------
    def forward(self, x):
        """
        Single forward pass → all three task outputs.
        Returns a dict (required by autograder):
            {
                'classification': Tensor (B, 37)
                'localization'  : Tensor (B, 4)          [cx,cy,w,h] in pixels
                'segmentation'  : Tensor (B, 3, H, W)
            }
        """
        features   = self.backbone(x)                  # (B, 512, 7, 7)
        cls_logits = self.classifier(features)         # (B, 37)
        bbox       = self.localizer(features) * 224.0  # (B, 4) Sigmoid*224
        seg_logits = self.seg_model(x)                 # (B, 3, H, W)

        return {
            'classification': cls_logits,
            'localization'  : bbox,
            'segmentation'  : seg_logits,
        }

    # -------------------------------------------------------------------------
    def _load_pretrained(self, cls_path, loc_path, seg_path):
        """Load weights from individual task checkpoints using full model copies."""

        # ── 1. Classification + backbone ──────────────────────────────────────
        if cls_path and os.path.exists(cls_path):
            from models.classification import VGG11Classifier
            # Load the full classifier model
            cls_model = VGG11Classifier(num_classes=37, use_batch_norm=True)
            ckpt = torch.load(cls_path, map_location='cpu')
            cls_model.load_state_dict(ckpt['model_state_dict'])
            # Copy backbone and classifier weights directly
            self.backbone.load_state_dict(cls_model.backbone.state_dict())
            self.classifier.load_state_dict(cls_model.classifier.state_dict())
            print(f"✓ Loaded backbone + classifier from {cls_path}")

            # Sanity check: verify a weight changed
            with torch.no_grad():
                print(f"  Sample classifier weight[0,0] = {self.classifier[1].weight[0,0].item():.6f}")

        # ── 2. Localization ───────────────────────────────────────────────────
        if loc_path and os.path.exists(loc_path):
            from models.localization import VGG11Localizer
            loc_model = VGG11Localizer(use_batch_norm=True)
            ckpt = torch.load(loc_path, map_location='cpu')
            loc_model.load_state_dict(ckpt['model_state_dict'])
            # Copy the regressor (localizer) weights
            self.localizer.load_state_dict(loc_model.regressor.state_dict())
            print(f"✓ Loaded localizer from {loc_path}")

        # ── 3. Segmentation U-Net ─────────────────────────────────────────────
        if seg_path and os.path.exists(seg_path):
            from models.segmentation import VGG11UNet
            # Check number of classes in saved model
            ckpt = torch.load(seg_path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt)
            saved_nc = sd['head.weight'].shape[0] if 'head.weight' in sd else 2
            if saved_nc != self.seg_model.head.weight.shape[0]:
                self.seg_model = VGG11UNet(num_classes=saved_nc, use_batch_norm=True)
                print(f"  Re-created seg_model with num_classes={saved_nc}")
            self.seg_model.load_state_dict(sd)
            print(f"✓ Loaded seg model from {seg_path}")