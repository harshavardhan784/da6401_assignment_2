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
      - Localization    -> (B, 4)   [cx, cy, w, h] pixel space
      - Segmentation    -> (B, C, H, W)
    """
    def __init__(self, num_classes=37, num_seg_classes=2,
                 use_batch_norm=True, dropout_p=0.5):
        super().__init__()

        os.makedirs('checkpoints', exist_ok=True)
        import gdown
        classifier_path = "checkpoints/classifier.pth"
        localizer_path  = "checkpoints/localizer.pth"
        unet_path       = "checkpoints/unet.pth"

        gdown.download(id="1bQatPpJxWBYuzZA949igWJh5OrADKrYM",
                       output=classifier_path, quiet=False)
        gdown.download(id="1gKu5L9hSIAFMJuOMxiUqHIVm5EbIScKD",
                       output=localizer_path,  quiet=False)
        gdown.download(id="1aWRiSNzmgdk3WbTOppXUfJ6Mkk6OUIA4",
                       output=unet_path,        quiet=False)

        # Shared backbone
        self.backbone = VGG11(in_channels=3, use_batch_norm=use_batch_norm)

        # Classification head
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

        # Localization head
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

        # Segmentation: full U-Net, num_classes auto-detected from checkpoint
        from models.segmentation import VGG11UNet
        self.seg_model = VGG11UNet(num_classes=num_seg_classes,
                                   use_batch_norm=use_batch_norm)

        init_weights(self.classifier)
        init_weights(self.localizer)

        self._load_pretrained(classifier_path, localizer_path, unet_path)

        # CRITICAL: ensure eval mode after all weight loading and any
        # submodule re-creation inside _load_pretrained
        self.eval()

    def forward(self, x):
        """
        Returns dict (required by autograder):
            {
                'classification': Tensor (B, num_classes)
                'localization'  : Tensor (B, 4)
                'segmentation'  : Tensor (B, num_seg_classes, H, W)
            }
        """
        features   = self.backbone(x)
        cls_logits = self.classifier(features)
        bbox       = self.localizer(features) * 224
        seg_logits = self.seg_model(x)

        return {
            'classification': cls_logits,
            'localization'  : bbox,
            'segmentation'  : seg_logits,
        }

    def _load_pretrained(self, cls_path, loc_path, seg_path):
        """Load weights from the three individual task checkpoints."""

        # classifier.pth -> backbone + classifier head
        if cls_path and os.path.exists(cls_path):
            ckpt = torch.load(cls_path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt)

            backbone_sd   = {k[len('backbone.'):]: v
                             for k, v in sd.items() if k.startswith('backbone.')}
            classifier_sd = {k[len('classifier.'):]: v
                             for k, v in sd.items() if k.startswith('classifier.')}

            if backbone_sd:
                self.backbone.load_state_dict(backbone_sd, strict=False)
                print(f"  Loaded backbone from {cls_path}")
            if classifier_sd:
                self.classifier.load_state_dict(classifier_sd, strict=False)
                print(f"  Loaded classifier from {cls_path}")

        # localizer.pth -> localization head
        # VGG11Localizer saves its head under "regressor.*" keys
        if loc_path and os.path.exists(loc_path):
            ckpt = torch.load(loc_path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt)

            localizer_sd = {k[len('regressor.'):]: v
                            for k, v in sd.items() if k.startswith('regressor.')}
            if localizer_sd:
                self.localizer.load_state_dict(localizer_sd, strict=False)
                print(f"  Loaded localizer from {loc_path}")

        # unet.pth -> full segmentation U-Net
        if seg_path and os.path.exists(seg_path):
            ckpt = torch.load(seg_path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt)

            # Auto-detect num_classes from saved head weight
            if 'head.weight' in sd:
                saved_nc = sd['head.weight'].shape[0]
                current_nc = self.seg_model.head.weight.shape[0]
                if saved_nc != current_nc:
                    from models.segmentation import VGG11UNet
                    self.seg_model = VGG11UNet(num_classes=saved_nc,
                                               use_batch_norm=True)
                    print(f"  Re-created seg_model with num_classes={saved_nc}")

            self.seg_model.load_state_dict(sd, strict=False)
            print(f"  Loaded seg model from {seg_path}")