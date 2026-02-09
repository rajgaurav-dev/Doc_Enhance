import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -------------------------------------------------
# L1 Loss (Pixel-wise)
# -------------------------------------------------
def l1_loss(pred, gt):
    """
    Standard L1 (MAE) loss
    """
    return F.l1_loss(pred, gt)


# -------------------------------------------------
# SSIM Loss (Lightweight Approximation)
# -------------------------------------------------
def ssim_loss(pred, gt, eps=1e-2):
    """
    Simple SSIM-like loss.
    NOTE: This is a lightweight approximation, not full SSIM.
    Good enough for training stability.
    """
    num = 2 * pred * gt + eps
    den = pred.pow(2) + gt.pow(2) + eps
    return 1.0 - torch.mean(num / den)


# -------------------------------------------------
# Perceptual Loss (VGG-based)
# -------------------------------------------------
class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    Compares high-level visual similarity instead of pixels.
    """

    def __init__(self, layer=16):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:layer]

        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg.eval()

    def forward(self, pred, gt):
        """
        pred, gt: [B, 3, H, W] in range [0,1]
        """
        # VGG expects normalized input
        pred_vgg = self.vgg(pred)
        gt_vgg = self.vgg(gt)

        return F.l1_loss(pred_vgg, gt_vgg)


# -------------------------------------------------
# Illumination Loss (Low-frequency consistency)
# -------------------------------------------------
def illumination_loss(pred, gt, kernel_size=31):
    """
    Encourages similar illumination (low-frequency content)
    Very important for deshadowing & document enhancement
    """
    padding = kernel_size // 2
    blur = nn.AvgPool2d(kernel_size, stride=1, padding=padding)

    return F.l1_loss(blur(pred), blur(gt))


# -------------------------------------------------
# Combined Loss Helper (Optional)
# -------------------------------------------------
class CombinedLoss(nn.Module):
    """
    Convenience wrapper to combine multiple losses
    """

    def __init__(
        self,
        w_l1=1.0,
        w_ssim=0.2,
        w_perceptual=0.1,
        w_illumination=0.1,
    ):
        super().__init__()

        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_perc = w_perceptual
        self.w_illu = w_illumination

        self.perceptual = PerceptualLoss()

    def forward(self, pred, gt):
        loss = 0.0

        if self.w_l1 > 0:
            loss += self.w_l1 * l1_loss(pred, gt)

        if self.w_ssim > 0:
            loss += self.w_ssim * ssim_loss(pred, gt)

        if self.w_perc > 0:
            loss += self.w_perc * self.perceptual(pred, gt)

        if self.w_illu > 0:
            loss += self.w_illu * illumination_loss(pred, gt)

        return loss
