"""
=========================================================
PhotoGuard AI
High-Frequency Noise & Frequency Feature Extractor Layer

Author : Vinay Kirithic

Description:
Implements Spatial Rich Model (SRM) High-Pass Convolutional Filters
and 2D Fast Fourier Transform (FFT) Magnitude Spectrum analysis to
extract microscopic camera sensor noise (PRNU) and AI latent diffusion
grid artifacts.
=========================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SRMFilterLayer(nn.Module):
    """
    Applies fixed Spatial Rich Model (SRM) high-pass filter kernels
    to suppress visual scene content (colors/shapes) and isolate
    high-frequency residual noise patterns.
    """
    def __init__(self):
        super(SRMFilterLayer, self).__init__()
        
        # Define standard SRM kernels (3x3 and 5x5 filters)
        # Filter 1: KV (1st order difference)
        kv_kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32) / 4.0

        # Filter 2: 2nd order edge residual
        s2_kernel = np.array([
            [-1, 2, -1],
            [2, -4, 2],
            [-1, 2, -1]
        ], dtype=np.float32) / 4.0

        # Filter 3: 3rd order High-Pass Filter
        s3_kernel = np.array([
            [-1, 2, -1],
            [2, -4, 2],
            [0, 0, 0]
        ], dtype=np.float32) / 2.0

        kernels = [kv_kernel, s2_kernel, s3_kernel]
        
        # Build 3-channel (RGB) weights tensor (3 filters * 3 RGB channels = 3 output channels)
        weight_tensor = torch.zeros((3, 3, 3, 3), dtype=torch.float32)
        for i, k in enumerate(kernels):
            for c in range(3):
                weight_tensor[i, c] = torch.from_numpy(k)

        self.srm_weights = nn.Parameter(weight_tensor, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, 3, H, W) RGB Image Tensor
        Returns: (B, 3, H, W) SRM High-Pass Residual Noise Tensor
        """
        # Apply 2D Conv with fixed SRM weights
        residuals = F.conv2d(x, self.srm_weights, padding=1)
        # Truncate values to accentuate residual spikes
        residuals = torch.clamp(residuals, -3.0, 3.0)
        return residuals


class FFTSpectrumExtractor(nn.Module):
    """
    Computes 2D Discrete Fast Fourier Transform magnitude spectrum
    to detect periodic lattice grid spikes left by AI diffusion upsampling.
    """
    def __init__(self):
        super(FFTSpectrumExtractor, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, 3, H, W) RGB Image Tensor
        Returns: (B, 1, H, W) Normalized Log-Magnitude Frequency Spectrum
        """
        # Convert RGB to grayscale for frequency domain analysis
        gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        
        # Apply 2D Real FFT
        fft2d = torch.fft.fft2(gray)
        # Shift zero frequency component to center
        fft_shifted = torch.fft.fftshift(fft2d)
        
        # Calculate Log Magnitude Spectrum
        magnitude = torch.abs(fft_shifted)
        log_magnitude = torch.log(1.0 + magnitude)
        
        # Min-Max Normalization per batch
        min_val = log_magnitude.view(log_magnitude.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        max_val = log_magnitude.view(log_magnitude.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-6
        norm_spectrum = (log_magnitude - min_val) / (max_val - min_val)
        
        return norm_spectrum
