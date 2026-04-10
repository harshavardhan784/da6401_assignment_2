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

        # ── Classification head ───────────────────────────────────────────────
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

        # ── Localization head ─────────────────────────────────────────────────
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

        # Initialise with Kaiming before loading pretrained weights
        init_weights(self.classifier)
        init_weights(self.localizer)

        # Load pretrained weights from individual task checkpoints
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
        """Load weights from the three individual task checkpoints."""
        # Helper to remove 'module.' prefix if present (DataParallel)
        def strip_module(key):
            return key.replace('module.', '')

        # ── 1. Classifier + backbone ──────────────────────────────────────────
        if cls_path and os.path.exists(cls_path):
            ckpt = torch.load(cls_path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt)

            print(f"Loading from {cls_path}...")
            print("  Sample keys from checkpoint:", list(sd.keys())[:5])

            # Backbone weights (strip 'backbone.' prefix)
            backbone_sd = {}
            classifier_sd = {}

            for k, v in sd.items():
                k_clean = strip_module(k)
                if k_clean.startswith('backbone.'):
                    backbone_sd[k_clean[len('backbone.'):]] = v
                elif k_clean.startswith('classifier.'):
                    classifier_sd[k_clean[len('classifier.'):]] = v

            # Load backbone
            if backbone_sd:
                missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
                print(f"  ✓ Loaded backbone: {len(backbone_sd)} keys")
                if missing:
                    print(f"    Missing (ignored): {missing[:3]}...")
                if unexpected:
                    print(f"    Unexpected: {unexpected[:3]}...")
            else:
                print("  ⚠ No backbone keys found!")

            # Load classifier
            if classifier_sd:
                # Store a reference weight before loading
                with torch.no_grad():
                    before_weight = self.classifier[1].weight[0, 0].clone()
                missing, unexpected = self.classifier.load_state_dict(classifier_sd, strict=False)
                with torch.no_grad():
                    after_weight = self.classifier[1].weight[0, 0]
                print(f"  ✓ Loaded classifier: {len(classifier_sd)} keys")
                if missing:
                    print(f"    Missing: {missing[:3]}...")
                if unexpected:
                    print(f"    Unexpected: {unexpected[:3]}...")
                # Verify change
                if torch.allclose(before_weight, after_weight):
                    print("  ⚠ WARNING: weights did NOT change → loading failed!")
                else:
                    print("  ✓ Weights changed – loading successful")
            else:
                print("  ⚠ No classifier keys found!")

        # ── 2. Localizer ──────────────────────────────────────────────────────
        if loc_path and os.path.exists(loc_path):
            ckpt = torch.load(loc_path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt)

            localizer_sd = {}
            for k, v in sd.items():
                k_clean = strip_module(k)
                if k_clean.startswith('regressor.'):
                    localizer_sd[k_clean[len('regressor.'):]] = v
                elif k_clean.startswith('localizer.'):
                    localizer_sd[k_clean[len('localizer.'):]] = v

            if localizer_sd:
                missing, unexpected = self.localizer.load_state_dict(localizer_sd, strict=False)
                print(f"  ✓ Loaded localizer from {loc_path} ({len(localizer_sd)} keys)")
                if missing:
                    print(f"    Missing: {missing[:3]}...")
            else:
                print(f"  ⚠ No localizer keys found in {loc_path}.")

        # ── 3. Segmentation U-Net ─────────────────────────────────────────────
        if seg_path and os.path.exists(seg_path):
            ckpt = torch.load(seg_path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt)

            # If the saved checkpoint used a different num_classes, rebuild.
            if 'head.weight' in sd:
                saved_nc   = sd['head.weight'].shape[0]
                current_nc = self.seg_model.head.weight.shape[0]
                if saved_nc != current_nc:
                    from models.segmentation import VGG11UNet
                    self.seg_model = VGG11UNet(num_classes=saved_nc,
                                               use_batch_norm=True)
                    print(f"  ✓ Re-created seg_model with num_classes={saved_nc}")

            # Load segmentation weights
            missing, unexpected = self.seg_model.load_state_dict(sd, strict=False)
            print(f"  ✓ Loaded seg model from {seg_path} ({len(sd)} keys)")
            if missing:
                print(f"    Missing (expected): {missing[:3]}...")
            if unexpected:
                print(f"    Unexpected: {unexpected[:3]}...")