"""Unified multi-task model
"""

import torch
import torch.nn as nn
import gdown
import os
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
from models.segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, 
                 classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", 
                 unet_path: str = "unet.pth", use_batch_norm: bool = True, dropout_p: float = 0.5):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head (3: background, pet, border).
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
            use_batch_norm: Whether to use BatchNorm.
            dropout_p: Dropout probability.
        """
        super(MultiTaskPerceptionModel, self).__init__()
        
        # Download weights from Google Drive
        # IMPORTANT: Replace these IDs with your actual Google Drive file IDs
        classifier_id = "1ly8n8hye9XDcoOjp8Mqx4Wz5DAq8LAc2"  
        localizer_id = "1Z585cGenqPWQdOMTMgipvG2Xh7syC0Hq"   
        unet_id = "1IwAfZohK42rjQ0O_psSJ0zfocqmPEmG1"   
        
        # For now, check if files exist locally, otherwise download
        if not os.path.exists(classifier_path):
            print(f"Downloading classifier weights from Google Drive...")
            gdown.download(id=classifier_id, output=classifier_path, quiet=False)
            print(f"Please place your trained {classifier_path} in the current directory")
        
        if not os.path.exists(localizer_path):
            print(f"Downloading localizer weights from Google Drive...")
            gdown.download(id=localizer_id, output=localizer_path, quiet=False)
            print(f"Please place your trained {localizer_path} in the current directory")
        
        if not os.path.exists(unet_path):
            print(f"Downloading UNet weights from Google Drive...")
            gdown.download(id=unet_id, output=unet_path, quiet=False)
            print(f"Please place your trained {unet_path} in the current directory")
        
        # Shared encoder
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batch_norm=use_batch_norm)
        
        # Load pretrained weights for encoder from classifier (if available)
        if os.path.exists(classifier_path):
            checkpoint = torch.load(classifier_path, map_location='cpu')
            # Extract encoder weights
            encoder_state = {}
            for k, v in checkpoint['model_state_dict'].items():
                if k.startswith('encoder.'):
                    encoder_state[k.replace('encoder.', '')] = v
            if encoder_state:
                self.encoder.load_state_dict(encoder_state, strict=False)
                print(f"Loaded encoder weights from {classifier_path}")
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_breeds)
        )
        
        # Localization head
        self.localization_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4)  # x_center, y_center, width, height
        )
        
        # Segmentation decoder (U-Net style)
        self.segmentation_decoder = VGG11UNet(
            num_classes=seg_classes, 
            in_channels=in_channels, 
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm
        )
        
        # Load weights for heads if available
        if os.path.exists(classifier_path):
            classifier_checkpoint = torch.load(classifier_path, map_location='cpu')
            # Load classification head weights
            cls_state = {}
            for k, v in classifier_checkpoint['model_state_dict'].items():
                if k.startswith('classifier.'):
                    cls_state[k.replace('classifier.', '')] = v
            if cls_state:
                self.classification_head.load_state_dict(cls_state, strict=False)
                print(f"Loaded classification head from {classifier_path}")
        
        if os.path.exists(localizer_path):
            localizer_checkpoint = torch.load(localizer_path, map_location='cpu')
            # Load localization head weights
            loc_state = {}
            for k, v in localizer_checkpoint['model_state_dict'].items():
                if k.startswith('localization_head.'):
                    loc_state[k.replace('localization_head.', '')] = v
                elif k.startswith('encoder.'):
                    pass  # Already loaded from classifier
            if loc_state:
                self.localization_head.load_state_dict(loc_state, strict=False)
                print(f"Loaded localization head from {localizer_path}")
        
        if os.path.exists(unet_path):
            unet_checkpoint = torch.load(unet_path, map_location='cpu')
            # Load segmentation decoder weights
            seg_state = {}
            for k, v in unet_checkpoint['model_state_dict'].items():
                if k.startswith('encoder.'):
                    pass  # Already loaded from classifier
                elif k.startswith('decoder.') or k.startswith('upconvs') or k.startswith('decoder_convs') or k.startswith('final_conv'):
                    seg_state[k] = v
            if seg_state:
                self.segmentation_decoder.load_state_dict(seg_state, strict=False)
                print(f"Loaded segmentation decoder from {unet_path}")

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W] (224x224, normalized).
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor in pixel space.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # Get encoder features
        bottleneck, encoder_features = self.encoder(x, return_features=True)
        
        # Classification branch
        cls_out = self.classification_head(bottleneck)
        
        # Localization branch (output in pixel space)
        loc_out = self.localization_head(bottleneck)
        # Scale to pixel space (224x224)
        loc_out = loc_out * 224
        
        # Segmentation branch
        seg_out = self.segmentation_decoder(x)  # Use the full UNet forward
        
        return {
            'classification': cls_out,
            'localization': loc_out,
            'segmentation': seg_out
        }
    
    def freeze_encoder(self):
        """Freeze the encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze the encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True