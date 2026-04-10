"""
Multi-task perception model for DA6401 Assignment 2
FIXED VERSION - Matches notebook architecture
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
      - Segmentation    -> (B, 2, H, W)  binary classes: 0=background, 1=foreground
    
    All tasks share the VGG11 backbone (parameter efficient).
    """
    def __init__(self, num_classes=37, use_batch_norm=True, dropout_p=0.5):
        super().__init__()

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

        # ── Segmentation head (lightweight decoder) ───────────────────────────
        # Shares the backbone, just decodes from features (B, 512, 7, 7)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # 7 → 14
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 14 → 28
            nn.BatchNorm2d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 28 → 56
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),     # 56 → 112
            nn.BatchNorm2d(32) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),     # 112 → 224
            nn.BatchNorm2d(16) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),  # 2 classes: background, foreground
        )

        # Initialize weights
        init_weights(self.classifier)
        init_weights(self.localizer)
        init_weights(self.seg_head)

    # -------------------------------------------------------------------------
    def forward(self, x):
        """
        Single forward pass → all three task outputs.
        
        Returns a TUPLE (for compatibility with notebook and inference.py):
            (cls_logits, bbox, seg_logits)
            - cls_logits: Tensor (B, 37) - classification logits
            - bbox: Tensor (B, 4) - [cx, cy, w, h] in pixels [0-224]
            - seg_logits: Tensor (B, 2, H, W) - segmentation logits
        """
        features   = self.backbone(x)                  # (B, 512, 7, 7)
        cls_logits = self.classifier(features)         # (B, 37)
        bbox       = self.localizer(features) * 224.0  # (B, 4) Sigmoid*224
        seg_logits = self.seg_head(features)           # (B, 2, 224, 224)

        return cls_logits, bbox, seg_logits

    # -------------------------------------------------------------------------
    def load_pretrained_components(self, cls_path=None, loc_path=None, seg_path=None):
        """
        Load pretrained weights from individual task checkpoints.
        
        Args:
            cls_path: Path to classifier checkpoint
            loc_path: Path to localizer checkpoint  
            seg_path: Path to segmentation checkpoint (if trained separately)
        """
        # ── 1. Classification + backbone ──────────────────────────────────────
        if cls_path and os.path.exists(cls_path):
            from models.classification import VGG11Classifier
            print(f"Loading classifier from {cls_path}...")
            cls_model = VGG11Classifier(num_classes=37, use_batch_norm=True)
            ckpt = torch.load(cls_path, map_location='cpu')
            cls_model.load_state_dict(ckpt['model_state_dict'])
            # Copy backbone and classifier weights
            self.backbone.load_state_dict(cls_model.backbone.state_dict())
            self.classifier.load_state_dict(cls_model.classifier.state_dict())
            print(f"✓ Loaded backbone + classifier")

        # ── 2. Localization ───────────────────────────────────────────────────
        if loc_path and os.path.exists(loc_path):
            from models.localization import VGG11Localizer
            print(f"Loading localizer from {loc_path}...")
            loc_model = VGG11Localizer(use_batch_norm=True)
            ckpt = torch.load(loc_path, map_location='cpu')
            loc_model.load_state_dict(ckpt['model_state_dict'])
            # Copy the regressor (localizer) weights
            self.localizer.load_state_dict(loc_model.regressor.state_dict())
            print(f"✓ Loaded localizer")

        # ── 3. Segmentation (if available) ────────────────────────────────────
        if seg_path and os.path.exists(seg_path):
            print(f"Loading segmentation head from {seg_path}...")
            ckpt = torch.load(seg_path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt)
            
            # Try to extract just the seg_head weights if this is a full multitask checkpoint
            seg_head_state = {}
            for k, v in sd.items():
                if k.startswith('seg_head.'):
                    seg_head_state[k.replace('seg_head.', '')] = v
            
            if seg_head_state:
                self.seg_head.load_state_dict(seg_head_state)
                print(f"✓ Loaded segmentation head")
            else:
                print(f"⚠ No seg_head weights found in checkpoint")