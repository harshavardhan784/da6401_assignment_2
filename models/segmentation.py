"""Segmentation model - U-Net style with VGG11 encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg11 import VGG11Encoder


class VGG11UNet(nn.Module):
    """U-Net style segmentation network with VGG11 encoder."""
    
    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5, use_batch_norm: bool = True):
        """
        Initialize the VGG11UNet model.
        
        Args:
            num_classes: Number of output classes (3 for trimap: background, pet, border).
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
            use_batch_norm: Whether to use BatchNorm in the encoder.
        """
        super(VGG11UNet, self).__init__()
        
        # Encoder (contracting path)
        self.encoder = VGG11Encoder(in_channels=in_channels, use_batch_norm=use_batch_norm)
        
        # Encoder channels at each block (after pooling)
        # Block 1: 64, Block 2: 128, Block 3: 256, Block 4: 512, Block 5: 512
        encoder_channels = [64, 128, 256, 512, 512]
        
        # Decoder (expansive path) with transposed convolutions
        self.upconvs = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        
        # Reverse the encoder channels for decoder
        decoder_channels = encoder_channels[::-1]
        
        for i in range(len(decoder_channels) - 1):
            # Transposed convolution for upsampling (learnable)
            self.upconvs.append(
                nn.ConvTranspose2d(
                    decoder_channels[i], 
                    decoder_channels[i + 1], 
                    kernel_size=2, 
                    stride=2
                )
            )
            
            # Double convolution after concatenation with skip connection
            # Input: upsampled features + skip connection from encoder
            in_ch = decoder_channels[i + 1] + encoder_channels[-(i + 2)]
            out_ch = decoder_channels[i + 1]
            
            conv_block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
                nn.ReLU(inplace=True)
            )
            self.decoder_convs.append(conv_block)
        
        # Final convolution to produce segmentation mask
        self.final_conv = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W] (224x224).
        
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Get encoder features with skip connections
        bottleneck, encoder_features = self.encoder(x, return_features=True)
        
        # Decoder path
        x = bottleneck
        
        # Process from deepest to shallowest
        for i, (upconv, decoder_conv) in enumerate(zip(self.upconvs, self.decoder_convs)):
            # Upsample
            x = upconv(x)
            
            # Get corresponding encoder feature (skip connection)
            # i=0: block_4, i=1: block_3, i=2: block_2, i=3: block_1
            skip_key = f'block_{len(self.upconvs) - i}'
            skip = encoder_features[skip_key]
            
            # Handle size mismatch (due to padding/flooring)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)
            
            # Double convolution
            x = decoder_conv(x)
        
        # Final convolution to produce segmentation logits
        x = self.final_conv(x)
        
        return x