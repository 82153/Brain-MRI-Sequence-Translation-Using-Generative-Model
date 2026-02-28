import torch
import torch.nn as nn
import segmentation_models_pytorch_3d as smp

class GradientDifferenceLoss3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Z, Y):
        dz_d = torch.abs(Z[:, :, 1:, :, :] - Z[:, :, :-1, :, :])
        dy_d = torch.abs(Y[:, :, 1:, :, :] - Y[:, :, :-1, :, :])

        dz_h = torch.abs(Z[:, :, :, 1:, :] - Z[:, :, :, :-1, :])
        dy_h = torch.abs(Y[:, :, :, 1:, :] - Y[:, :, :, :-1, :])

        dz_w = torch.abs(Z[:, :, :, :, 1:] - Z[:, :, :, :, :-1])
        dy_w = torch.abs(Y[:, :, :, :, 1:] - Y[:, :, :, :, :-1])

        loss = (
            ((dz_d - dy_d) ** 2).mean() +
            ((dz_h - dy_h) ** 2).mean() +
            ((dz_w - dy_w) ** 2).mean()
        )

        return loss

class GANLoss(nn.Module):
    """
    pix2pix / PatchGAN용 adversarial loss
    """
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target_is_real):
        if target_is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)
        return self.loss(pred, target)

class GeneratorLoss(nn.Module):
    """
    pix2pix Generator loss
    = adversarial + lambda * L1
    """
    def __init__(self, gan_loss, lambda_l1=100.0
                 , lambda_diff = 20.0
                 ):
        super().__init__()
        self.gan_loss = gan_loss
        self.lambda_l1 = lambda_l1
        self.l1 = nn.L1Loss()
        self.lambda_diff = lambda_diff
        self.diff_loss = GradientDifferenceLoss3D()

    def forward(self, pred_fake, fake, target):
        loss_adv = self.gan_loss(pred_fake, target_is_real=True)
        loss_l1  = self.l1(fake, target)
        loss_diff = self.diff_loss(fake, target)
        loss = loss_adv + self.lambda_l1 * loss_l1 + self.lambda_diff * loss_diff
        return loss, loss_adv, loss_l1, loss_diff

class DiceBCELoss(torch.nn.Module):
    def __init__(self, device, dice_weight, bce_weight, dice, bce):
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice = smp.losses.DiceLoss(**dice)

        pos_weight = bce.get("pos_weight", None)
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight], device=device)

        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target):
        target = target.unsqueeze(1).float()

        loss = (
            self.dice_weight * self.dice(logits, target)
            + self.bce_weight * self.bce(logits, target)
        )
        return loss