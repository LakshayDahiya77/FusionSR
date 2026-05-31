"""
FusionSR-v3 loss functions.

Components:
    CharbonnierLoss   — smooth L1 variant (EDSR / RCAN)
    VGGPerceptualLoss — L1 in VGG19 feature space (SRGAN / ESRGAN)
    GANLoss           — relativistic average GAN (ESRGAN / Real-ESRGAN)
    CombinedSRLoss    — pixel + perceptual wrapper (GAN handled by trainer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss — smooth L1 variant.
    L = mean( sqrt( (pred - target)² + ε² ) )

    Better than L1: differentiable everywhere, smoother gradients near zero.
    Better than L2: less sensitive to outliers.
    ε=1e-3 is the standard value from EDSR/RCAN papers.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class VGGPerceptualLoss(nn.Module):
    """
    VGG19 perceptual loss (SRGAN / ESRGAN).

    Extracts features from pretrained VGG19 at conv layers BEFORE activation
    (ESRGAN improvement over SRGAN), and computes L1 distance between
    predicted and target feature maps.

    Extraction points:
        conv1_2 (idx 2), conv2_2 (7), conv3_4 (16), conv4_4 (25), conv5_4 (34)

    All VGG weights are frozen. Input is normalized with ImageNet statistics.
    """

    def __init__(self, layer_weights=None):
        super().__init__()
        import torchvision

        vgg = torchvision.models.vgg19(weights="IMAGENET1K_V1").features

        # build sequential slices — each ends at a conv output (before ReLU)
        # slice 0: layers [0..2]  → conv1_1, relu, conv1_2
        # slice 1: layers [3..7]  → relu, pool, conv2_1, relu, conv2_2
        # slice 2: layers [8..16] → ... conv3_4
        # slice 3: layers [17..25]→ ... conv4_4
        # slice 4: layers [26..34]→ ... conv5_4
        boundaries = [3, 8, 17, 26, 35]
        self.slices = nn.ModuleList()
        prev = 0
        for b in boundaries:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:b]))
            prev = b

        # freeze all VGG weights
        for p in self.parameters():
            p.requires_grad = False

        self.weights = layer_weights or [0.1, 0.1, 1.0, 1.0, 1.0]

        # ImageNet normalization constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # normalize to ImageNet distribution
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = torch.tensor(0.0, device=pred.device)
        x_pred, x_target = pred, target
        for i, slice_net in enumerate(self.slices):
            x_pred = slice_net(x_pred)
            x_target = slice_net(x_target)
            loss = loss + self.weights[i] * F.l1_loss(x_pred, x_target.detach())

        return loss

    def train(self, mode=True):
        """VGG always stays in eval mode — prevents BN stat updates."""
        return super().train(False)


class GANLoss(nn.Module):
    """
    Relativistic average GAN loss (ESRGAN / Real-ESRGAN).

    Generator:     D(fake) should score higher than average D(real)
    Discriminator: D(real) should score higher than average D(fake)

    More stable than standard GAN loss. Uses BCEWithLogits for numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def generator_loss(
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        """Generator wants fake to look more real than average real."""
        return self.bce(
            fake_logits - real_logits.detach().mean(),
            torch.ones_like(fake_logits),
        )

    def discriminator_loss(
        self, fake_logits: torch.Tensor, real_logits: torch.Tensor
    ) -> torch.Tensor:
        """Discriminator wants real > avg(fake) and fake < avg(real)."""
        d_real = self.bce(
            real_logits - fake_logits.detach().mean(),
            torch.ones_like(real_logits),
        )
        d_fake = self.bce(
            fake_logits - real_logits.detach().mean(),
            torch.zeros_like(fake_logits),
        )
        return (d_real + d_fake) / 2


class CombinedSRLoss(nn.Module):
    """
    Combined loss for FusionSR-v3 training.

    L_total = pixel_weight × L_charbonnier  [+ perceptual_weight × L_vgg]

    GAN loss is handled separately by the Trainer (needs discriminator access).
    Returns (total_loss, loss_dict) for W&B logging of individual components.
    """

    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        use_perceptual: bool = False,
    ):
        super().__init__()
        self.pixel_loss = CharbonnierLoss(eps=1e-3)
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual

        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        loss_dict = {}

        p_loss = self.pixel_loss(pred, target)
        total = self.pixel_weight * p_loss
        loss_dict["pixel"] = p_loss.item()

        if self.use_perceptual:
            vgg_loss = self.perceptual_loss(pred, target)
            total = total + self.perceptual_weight * vgg_loss
            loss_dict["perceptual"] = vgg_loss.item()

        loss_dict["total"] = total.item()
        return total, loss_dict
