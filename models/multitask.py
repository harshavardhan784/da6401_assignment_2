"""Unified multi-task model
"""

import torch
import torch.nn as nn
import gdown
import os

# ============================================
# Model Architecture (same as training)
# ============================================
class CustomDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * mask / keep


CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


class VGG11(nn.Module):
    def __init__(self, in_channels=3, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.features = self._make(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def _make(self, in_ch):
        layers = []
        for v in CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1)]
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_ch = v
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.avgpool(self.features(x))


def _vgg_block(ic, oc, num_convs, bn):
    """Build one VGG encoder stage: num_convs × (Conv-[BN]-ReLU). No pooling."""
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(ic, oc, 3, padding=1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm2d(oc))
        layers.append(nn.ReLU(inplace=True))
        ic = oc
    return nn.Sequential(*layers)


class _DConv(nn.Module):
    """Two Conv-BN-ReLU blocks used in the decoder."""
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.net(x)


class VGG11UNet(nn.Module):
    """U-Net with VGG11 encoder structure."""
    def __init__(self, num_classes=2, in_channels=3, use_batch_norm=True, freeze_backbone=False):
        super().__init__()
        bn = use_batch_norm
        
        # Encoder: exact VGG11 conv counts
        self.enc1 = _vgg_block(in_channels, 64, 1, bn)
        self.enc2 = _vgg_block(64, 128, 1, bn)
        self.enc3 = _vgg_block(128, 256, 2, bn)
        self.enc4 = _vgg_block(256, 512, 2, bn)
        self.enc5 = _vgg_block(512, 512, 2, bn)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bot = _DConv(512, 1024)
        
        # Decoder
        self.up5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec5 = _DConv(512 + 512, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _DConv(256 + 512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _DConv(128 + 256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _DConv(64 + 128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _DConv(32 + 64, 32)
        self.head = nn.Conv2d(32, num_classes, 1)
        
        if freeze_backbone:
            for e in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
                for p in e.parameters():
                    p.requires_grad = False
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        b = self.bot(self.pool(e5))
        d5 = self.dec5(torch.cat([self.up5(b), e5], 1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ============================================
# Unified Multi-Task Model
# ============================================
class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""
    
    # Google Drive IDs for pretrained weights
    FILE_IDS = {  
        'classifier': '1ly8n8hye9XDcoOjp8Mqx4Wz5DAq8LAc2',  # Replace with actual ID
        'localizer': '1Z585cGenqPWQdOMTMgipvG2Xh7syC0Hq',    # Replace with actual ID
        'unet': '1IwAfZohK42rjQ0O_psSJ0zfocqmPEmG1'               # Replace with actual ID
    }

    def __init__(self, num_breeds: int = 37, seg_classes: int = 2, 
                 in_channels: int = 3, image_size: int = 224,
                 classifier_path: str = "classifier.pth", 
                 localizer_path: str = "localizer.pth", 
                 unet_path: str = "unet.pth",
                 download_weights: bool = False):
        """
        Initialize the shared backbone/heads using these trained weights.
        
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head (2 for binary).
            in_channels: Number of input channels.
            image_size: Input image size (assumed square).
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
            download_weights: Whether to download weights from Google Drive.
        """
        super().__init__()
        
        self.image_size = image_size
        self.num_breeds = num_breeds
        self.seg_classes = seg_classes
        
        # Download weights if requested
        if download_weights:
            self._download_weights(classifier_path, localizer_path, unet_path)
        
        # ========== Shared Backbone (VGG11 features) ==========
        self.backbone = VGG11(in_channels=in_channels, use_batch_norm=True)
        
        # ========== Classification Head ==========
        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_breeds),
        )
        
        # ========== Localization Head ==========
        self.localizer_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )
        
        # ========== Segmentation Head (U-Net decoder) ==========
        # Using separate decoder for segmentation
        self.seg_bottleneck = _DConv(512, 1024)
        self.seg_up5 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.seg_dec5 = _DConv(512 + 512, 512)
        self.seg_up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.seg_dec4 = _DConv(256 + 512, 256)
        self.seg_up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.seg_dec3 = _DConv(128 + 256, 128)
        self.seg_up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.seg_dec2 = _DConv(64 + 128, 64)
        self.seg_up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.seg_dec1 = _DConv(32 + 64, 32)
        self.seg_head = nn.Conv2d(32, seg_classes, 1)
        
        # Load pretrained weights if available
        self._load_pretrained_weights(classifier_path, localizer_path, unet_path)
    
    def _download_weights(self, classifier_path, localizer_path, unet_path):
        """Download pretrained weights from Google Drive."""
        print("Downloading pretrained weights...")
        
        if not os.path.exists(classifier_path):
            gdown.download(id=self.FILE_IDS['classifier'], output=classifier_path, quiet=False)
        if not os.path.exists(localizer_path):
            gdown.download(id=self.FILE_IDS['localizer'], output=localizer_path, quiet=False)
        if not os.path.exists(unet_path):
            gdown.download(id=self.FILE_IDS['unet'], output=unet_path, quiet=False)
    
    def _load_pretrained_weights(self, classifier_path, localizer_path, unet_path):
        """Load pretrained weights into respective heads."""
        
        # Load classifier weights
        if os.path.exists(classifier_path):
            print(f"Loading classifier weights from {classifier_path}")
            checkpoint = torch.load(classifier_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Filter and load backbone + classifier head
            backbone_state = {k.replace('backbone.', ''): v 
                            for k, v in state_dict.items() 
                            if k.startswith('backbone.')}
            classifier_state = {k.replace('classifier.', ''): v 
                              for k, v in state_dict.items() 
                              if k.startswith('classifier.')}
            
            if backbone_state:
                self.backbone.load_state_dict(backbone_state, strict=False)
            if classifier_state:
                self.classifier_head.load_state_dict(classifier_state, strict=False)
        
        # Load localizer weights
        if os.path.exists(localizer_path):
            print(f"Loading localizer weights from {localizer_path}")
            checkpoint = torch.load(localizer_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load backbone (if not already loaded) and localizer head
            backbone_state = {k.replace('backbone.', ''): v 
                            for k, v in state_dict.items() 
                            if k.startswith('backbone.')}
            localizer_state = {k.replace('regressor.', ''): v 
                             for k, v in state_dict.items() 
                             if k.startswith('regressor.')}
            
            if backbone_state and not any(p.requires_grad for p in self.backbone.parameters()):
                self.backbone.load_state_dict(backbone_state, strict=False)
            if localizer_state:
                self.localizer_head.load_state_dict(localizer_state, strict=False)
        
        # Load U-Net weights (for segmentation decoder)
        if os.path.exists(unet_path):
            print(f"Loading segmentation weights from {unet_path}")
            checkpoint = torch.load(unet_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load segmentation-specific layers
            seg_layers = ['bot', 'up5', 'dec5', 'up4', 'dec4', 'up3', 'dec3', 
                         'up2', 'dec2', 'up1', 'dec1', 'head']
            for layer in seg_layers:
                layer_state = {k.replace(f'seg_{layer}', layer): v 
                             for k, v in state_dict.items() 
                             if k.startswith(f'seg_{layer}')}
                if layer_state and hasattr(self, f'seg_{layer}'):
                    getattr(self, f'seg_{layer}').load_state_dict(layer_state, strict=False)
    
    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
            
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor (cx, cy, w, h in pixel space).
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor.
        """
        # Get intermediate features from backbone
        # For segmentation, we need skip connections
        e1 = self.backbone.features[:5](x)  # After first conv block
        e2 = self.backbone.features[5:10](self.backbone.features[4](e1))  # After pool + conv
        e3 = self.backbone.features[10:16](self.backbone.features[9](e2))
        e4 = self.backbone.features[16:23](self.backbone.features[15](e3))
        e5 = self.backbone.features[23:30](self.backbone.features[22](e4))
        
        # Global features for classification & localization
        global_features = self.backbone.avgpool(e5)  # [B, 512, 7, 7]
        
        # Classification
        cls_logits = self.classifier_head(global_features)
        
        # Localization (predicts cx, cy, w, h in pixel space)
        loc_output = self.localizer_head(global_features) * self.image_size
        
        # Segmentation (using decoder)
        b = self.seg_bottleneck(e5)
        d5 = self.seg_dec5(torch.cat([self.seg_up5(b), e5], 1))
        d4 = self.seg_dec4(torch.cat([self.seg_up4(d5), e4], 1))
        d3 = self.seg_dec3(torch.cat([self.seg_up3(d4), e3], 1))
        d2 = self.seg_dec2(torch.cat([self.seg_up2(d3), e2], 1))
        d1 = self.seg_dec1(torch.cat([self.seg_up1(d2), e1], 1))
        seg_logits = self.seg_head(d1)
        
        return {
            'classification': cls_logits,
            'localization': loc_output,
            'segmentation': seg_logits
        }
    
    def predict(self, x: torch.Tensor, return_probs: bool = False):
        """
        Convenience method for inference with post-processing.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_probs: If True, return softmax probabilities for classification
        
        Returns:
            Dictionary with predictions:
            - class_ids: [B] predicted class indices
            - class_names: List of predicted breed names
            - bboxes: [B, 4] bounding boxes in [x1, y1, x2, y2] format
            - seg_masks: [B, H, W] predicted segmentation masks
        """
        outputs = self.forward(x)
        
        # Classification
        class_logits = outputs['classification']
        class_probs = torch.softmax(class_logits, dim=1)
        class_ids = torch.argmax(class_probs, dim=1)
        
        # Localization (convert cx,cy,w,h to x1,y1,x2,y2)
        bbox_params = outputs['localization']  # [B, 4] -> cx, cy, w, h
        cx, cy, w, h = bbox_params[:, 0], bbox_params[:, 1], bbox_params[:, 2], bbox_params[:, 3]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        bboxes = torch.stack([x1, y1, x2, y2], dim=1)
        
        # Segmentation
        seg_masks = torch.argmax(outputs['segmentation'], dim=1)
        
        results = {
            'class_ids': class_ids,
            'bboxes': bboxes,
            'seg_masks': seg_masks
        }
        
        if return_probs:
            results['class_probs'] = class_probs
        
        return results
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze the shared backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def freeze_head(self, head_name: str, freeze: bool = True):
        """Freeze/unfreeze a specific head."""
        if head_name == 'classification':
            for param in self.classifier_head.parameters():
                param.requires_grad = not freeze
        elif head_name == 'localization':
            for param in self.localizer_head.parameters():
                param.requires_grad = not freeze
        elif head_name == 'segmentation':
            seg_layers = ['seg_bottleneck', 'seg_up5', 'seg_dec5', 'seg_up4', 'seg_dec4',
                         'seg_up3', 'seg_dec3', 'seg_up2', 'seg_dec2', 'seg_up1', 'seg_dec1', 'seg_head']
            for layer_name in seg_layers:
                layer = getattr(self, layer_name)
                for param in layer.parameters():
                    param.requires_grad = not freeze


# ============================================
# Usage Example
# ============================================
if __name__ == "__main__":
    # Initialize multi-task model
    model = MultiTaskPerceptionModel(
        num_breeds=37,
        seg_classes=2,
        classifier_path='classifier.pth',
        localizer_path='localizer.pth',
        unet_path='unet.pth'
    )
    
    # Set to evaluation mode
    model.eval()
    
    # Test with random input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
        predictions = model.predict(dummy_input)
    
    print(f"Classification output shape: {outputs['classification'].shape}")
    print(f"Localization output shape: {outputs['localization'].shape}")
    print(f"Segmentation output shape: {outputs['segmentation'].shape}")
    print(f"Predicted class IDs: {predictions['class_ids']}")
    print(f"Predicted bboxes shape: {predictions['bboxes'].shape}")
    print(f"Predicted masks shape: {predictions['seg_masks'].shape}")