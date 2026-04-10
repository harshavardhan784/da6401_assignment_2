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
    def __init__(self, num_classes=37, num_seg_classes=3,   # ← 3, not 2
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
        # Sequential index → layer:
        #   0: Flatten
        #   1: Linear(25088, 4096)
        #   2: BatchNorm1d(4096)
        #   3: ReLU
        #   4: CustomDropout
        #   5: Linear(4096, 4096)
        #   6: BatchNorm1d(4096)
        #   7: ReLU
        #   8: CustomDropout
        #   9: Linear(4096, 37)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

        # ── Localization head ─────────────────────────────────────────────────
        # Matches VGG11Localizer's self.regressor exactly.
        # Output: Sigmoid × 224 → [0..224] pixel-space [cx, cy, w, h]
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

    # ─────────────────────────────────────────────────────────────────────────
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
        cls_logits = self.classifier(features)          # (B, 37)
        bbox       = self.localizer(features) * 224.0  # (B, 4) Sigmoid*224
        seg_logits = self.seg_model(x)                  # (B, 3, H, W)

        return {
            'classification': cls_logits,
            'localization'  : bbox,
            'segmentation'  : seg_logits,
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _load_pretrained(self, cls_path, loc_path, seg_path):
        """Load weights from the three individual task checkpoints."""

        # ── 1. Classifier + backbone ──────────────────────────────────────────
        if cls_path and os.path.exists(cls_path):
            ckpt = torch.load(cls_path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt)

            print(f"Loading from {cls_path}...")

            # Backbone weights  (strip 'backbone.' prefix)
            backbone_sd = {k[len('backbone.'):]: v
                           for k, v in sd.items() if k.startswith('backbone.')}
            if backbone_sd:
                self.backbone.load_state_dict(backbone_sd, strict=False)
                print(f"  ✓ Loaded backbone: {len(backbone_sd)} keys")

            # ── BUG FIX ────────────────────────────────────────────────────
            # WRONG (old): classifier_sd[k] = v  → key is 'classifier.1.weight'
            #              self.classifier.load_state_dict expects '1.weight'
            #              → ALL keys are "unexpected", NOTHING loads, F1=0
            #
            # FIXED: strip the 'classifier.' prefix so keys are '1.weight' etc.
            classifier_sd = {k[len('classifier.'):]: v
                             for k, v in sd.items() if k.startswith('classifier.')}

            if classifier_sd:
                missing, unexpected = self.classifier.load_state_dict(
                    classifier_sd, strict=False)
                print(f"  ✓ Loaded classifier: {len(classifier_sd)} keys")
                if missing:
                    print(f"    Missing (buffers expected): {missing[:3]}...")
                if unexpected:
                    print(f"    Unexpected: {unexpected[:3]}...")

                # Sanity-check: weight should be non-trivially non-zero
                with torch.no_grad():
                    w = self.classifier[1].weight
                    print(f"  ✓ Classifier[1] weight  mean={w.mean().item():.6f}  "
                          f"std={w.std().item():.6f}")
                    if w.std().item() < 1e-4:
                        print("  ⚠ WARNING: classifier weights look degenerate — "
                              "check the checkpoint file.")

        # ── 2. Localizer ──────────────────────────────────────────────────────
        if loc_path and os.path.exists(loc_path):
            ckpt = torch.load(loc_path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt)

            # VGG11Localizer stores the regression head as self.regressor
            # Strip 'regressor.' prefix → bare indices '1.weight', '2.weight'…
            # which match self.localizer (nn.Sequential with same structure).
            localizer_sd = {}
            for k, v in sd.items():
                if k.startswith('regressor.'):
                    localizer_sd[k[len('regressor.'):]] = v
                elif k.startswith('localizer.'):          # fallback key name
                    localizer_sd[k[len('localizer.'):]] = v

            if localizer_sd:
                missing, _ = self.localizer.load_state_dict(
                    localizer_sd, strict=False)
                print(f"  ✓ Loaded localizer from {loc_path} "
                      f"({len(localizer_sd)} keys)")
                if missing:
                    print(f"    Missing: {missing[:3]}...")
            else:
                print(f"  ⚠ No localizer keys found in {loc_path}. "
                      f"Available keys: {list(sd.keys())[:5]}")

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

            missing, unexpected = self.seg_model.load_state_dict(sd, strict=False)
            print(f"  ✓ Loaded seg model from {seg_path} ({len(sd)} keys)")
            if missing:
                print(f"    Missing: {missing[:3]}...")