import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch

def compute_psnr(pred, gt, data_range=2.0, eps=1e-8):
    mse = np.mean((pred - gt) ** 2)
    return 20 * np.log10(data_range) - 10 * np.log10(mse + eps)

def compute_ssim(pred, gt, data_range=2.0):
    # slice-wise 평균
    D = pred.shape[0]
    vals = []
    for z in range(D):
        vals.append(ssim(pred[z], gt[z], data_range=data_range))
    return np.mean(vals)

def compute_acc(logits, target):
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()

def dice_metric(logits, target, eps=1e-8):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    target = target.unsqueeze(1).float()

    intersection = (preds * target).sum(dim=(2,3,4))
    union = preds.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))

    dice = (2 * intersection + eps) / (union + eps)

    return dice.mean().item()