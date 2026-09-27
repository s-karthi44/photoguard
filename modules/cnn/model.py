"""
=========================================================
PhotoGuard AI
Dual-Stream Universal AI Image Detector Model

Author : Vinay Kirithic

Description:
PyTorch Neural Network architecture featuring a Dual-Stream design:
1. Spatial Stream: EfficientNet-B0 backbone for visual macro-features.
2. Noise/Frequency Stream: SRM High-Pass Filters + FFT Spectrum + Conv Layers
   for camera sensor noise (PRNU) and AI latent grid detection.
=========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from modules.filters import SRMFilterLayer, FFTSpectrumExtractor


class PhotoGuardDualStreamNet(nn.Module):
    """
    Dual-Stream Neural Network for Universal AI Detection.
    - Stream A (Spatial): Extracts RGB visual features.
    - Stream B (Noise & Frequency): Extracts SRM High-Pass Residuals & FFT Spectrum.
    - Attention Fusion Layer: Merges spatial and spectral representations.
    """
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.3):
        super(PhotoGuardDualStreamNet, self).__init__()
        
        # --- STREAM A: SPATIAL RGB BRANCH (EfficientNet-B0) ---
        spatial_backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.spatial_features = spatial_backbone.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        spatial_dim = spatial_backbone.classifier[1].in_features  # 1280
        
        # --- STREAM B: NOISE & FREQUENCY BRANCH ---
        self.srm_layer = SRMFilterLayer()
        self.fft_layer = FFTSpectrumExtractor()
        
        # Input to Noise Stream: 3 SRM channels + 1 FFT Channel = 4 channels
        self.noise_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        noise_dim = 128
        
        # --- FUSION & CLASSIFICATION HEAD ---
        combined_dim = spatial_dim + noise_dim  # 1280 + 128 = 1408
        
        # Attention Gate for Channel-wise Feature Weighting
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(combined_dim // 2, combined_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, 3, H, W) RGB Image
        Output: (B, 2) Logits [AI, Real]
        """
        # 1. Spatial Stream Processing
        spatial_feat = self.spatial_features(x)
        spatial_vec = torch.flatten(self.spatial_pool(spatial_feat), 1)
        
        # 2. Noise & Frequency Stream Processing
        srm_feat = self.srm_layer(x)          # (B, 3, H, W)
        fft_feat = self.fft_layer(x)          # (B, 1, H, W)
        noise_input = torch.cat([srm_feat, fft_feat], dim=1) # (B, 4, H, W)
        
        noise_vec = torch.flatten(self.noise_conv(noise_input), 1) # (B, 128)
        
        # 3. Feature Fusion & Attention
        fused_vec = torch.cat([spatial_vec, noise_vec], dim=1) # (B, 1408)
        attn_weights = self.attention(fused_vec)
        weighted_fused_vec = fused_vec * attn_weights
        
        # 4. Classification
        logits = self.classifier(weighted_fused_vec)
        return logits


# Wrapper function for backward compatibility
def get_model(num_classes: int = 2, dropout_rate: float = 0.3) -> nn.Module:
    return PhotoGuardDualStreamNet(num_classes=num_classes, dropout_rate=dropout_rate)


def build_model(num_classes: int = 2, dropout_rate: float = 0.3) -> nn.Module:
    return PhotoGuardDualStreamNet(num_classes=num_classes, dropout_rate=dropout_rate)