"""
FusionSR-v3 discriminator for optional GAN training.

VGG-style architecture with spectral normalization (SRGAN / ESRGAN).
Only instantiated when CONFIG['use_gan'] = True. Disabled by default.
"""

import torch.nn as nn


class VGGStyleDiscriminator(nn.Module):
    """
    VGG-style discriminator with spectral normalization.

    Input:  HR-resolution image [B, 3, H, W]
    Output: real/fake scalar [B, 1]

    Architecture:
        8 conv blocks (64→64→128→128→256→256→512→512)
        stride-2 every other block for downsampling
        spectral norm on all conv layers for training stability
        global average pool → linear classifier

    ~2.8M parameters. Lightweight enough for T4 alongside the generator.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        def sn_conv(in_ch, out_ch, stride=1):
            return nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True)
            )

        # progressive downsampling: H→H/2→H/4→H/8→H/16
        self.features = nn.Sequential(
            # block 1: base_channels, no downsample
            sn_conv(in_channels, base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(base_channels, base_channels, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            # block 2: 2× channels
            sn_conv(base_channels, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(base_channels * 2, base_channels * 2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            # block 3: 4× channels
            sn_conv(base_channels * 2, base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(base_channels * 4, base_channels * 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            # block 4: 8× channels
            sn_conv(base_channels * 4, base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            sn_conv(base_channels * 8, base_channels * 8, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
