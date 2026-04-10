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
    Unified multi-task model that performs:
    - Classification (breed label)
    - Localization (bounding box)
    - Segmentation (pixel mask)
    
    All tasks share the VGG11 backbone.
    """
    def __init__(self, num_classes=37, use_batch_norm=True, dropout_p=0.5):
        super().__init__()
        
        # Download checkpoints from Google Drive
        import gdown
        classifier_path = "checkpoints/classifier.pth"
        localizer_path = "checkpoints/localizer.pth"
        unet_path = "checkpoints/unet.pth"
        
        # TODO: Replace these with your actual Google Drive file IDs
        # gdown.download(id="<classifier.pth drive id>", output=classifier_path, quiet=False)
        # gdown.download(id="<localizer.pth drive id>", output=localizer_path, quiet=False)
        # gdown.download(id="<unet.pth drive id>", output=unet_path, quiet=False)
        
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
        
        # Segmentation head (lightweight decoder)
        # We'll use a simple decoder for multi-task, not the full U-Net
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
        
        init_weights(self.classifier)
        init_weights(self.localizer)
        init_weights(self.seg_head)
        
        # Load pretrained weights if available
        self.load_pretrained_components(classifier_path, localizer_path, unet_path)
    
    def forward(self, x):
        # Shared backbone
        features = self.backbone(x)  # (B, 512, 7, 7)
        
        # Classification
        cls_logits = self.classifier(features)
        
        # Localization
        bbox = self.localizer(features) * 224  # IMAGE_SIZE = 224
        
        # Segmentation
        seg_logits = self.seg_head(features)  # (B, 2, 224, 224)
        
        return cls_logits, bbox, seg_logits
    
    def load_pretrained_components(self, cls_path=None, loc_path=None, seg_path=None):
        """Load pretrained weights for individual components"""
        if cls_path and os.path.exists(cls_path):
            checkpoint = torch.load(cls_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # Filter backbone and classifier weights
            backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() 
                            if k.startswith('backbone.')}
            classifier_state = {k.replace('classifier.', ''): v for k, v in state_dict.items()
                              if k.startswith('classifier.')}
            if backbone_state:
                self.backbone.load_state_dict(backbone_state, strict=False)
                print(f"  Loaded backbone from {cls_path}")
            if classifier_state:
                self.classifier.load_state_dict(classifier_state, strict=False)
                print(f"  Loaded classifier from {cls_path}")
        
        if loc_path and os.path.exists(loc_path):
            checkpoint = torch.load(loc_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            localizer_state = {k.replace('regressor.', ''): v for k, v in state_dict.items()
                             if 'regressor' in k}
            if localizer_state:
                self.localizer.load_state_dict(localizer_state, strict=False)
                print(f"  Loaded localizer from {loc_path}")
        
        # Note: For segmentation, the full U-Net architecture is different from our lightweight decoder
        # So we don't load those weights here