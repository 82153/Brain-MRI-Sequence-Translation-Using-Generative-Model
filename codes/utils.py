import numpy as np
from os.path import join as osp
import nibabel as nib
import torch
import random

def load_nii_pre(row, base_dir):
    img_p1  = row["img_p1"]
    img_p99 = row["img_p99"]
    gt_p1 = row["gt_p1"]
    gt_p99 = row["gt_p99"]
    img = nib.load(osp(base_dir, row['source_nii'])).get_fdata(dtype=np.float32)
    gt = nib.load(osp(base_dir, row['target_nii'])).get_fdata(dtype=np.float32)

    img = np.clip(img, img_p1, img_p99)
    img = (img - img_p1) / (img_p99 - img_p1 + 1e-8) * 2 - 1
    img = np.transpose(img, (2, 0, 1))

    gt = np.clip(gt, gt_p1, gt_p99)
    gt = (gt - gt_p1) / (gt_p99 - gt_p1 + 1e-8) * 2 - 1
    gt = np.transpose(gt, (2, 0, 1))
    return img, gt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def load_nii_pre_cls(row, base_dir):
    img = nib.load(osp(base_dir, row['source_nii'])).get_fdata(dtype=np.float32)

    if "img_p1" in row.index:
        img_p1  = row["img_p1"]
        img_p99 = row["img_p99"]
        img = np.clip(img, img_p1, img_p99)
        img = (img - img_p1) / (img_p99 - img_p1 + 1e-8) * 2 - 1

    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)  

    cls = row['source_seq']
    return img, cls

def pad_to_multiple(img, mask, multiple=32):
    D, H, W = img.shape

    pd = (multiple - D % multiple) % multiple
    ph = (multiple - H % multiple) % multiple
    pw = (multiple - W % multiple) % multiple

    img = np.pad(img, ((0, pd), (0, ph), (0, pw)), mode='constant')
    mask = np.pad(mask, ((0, pd), (0, ph), (0, pw)), mode='constant')

    return img, mask

def load_nii_pre_seg(row, base_dir):
    img_p1  = row["img_p1"]
    img_p99 = row["img_p99"]

    img = nib.load(osp(base_dir, row['img_path'])).get_fdata(dtype=np.float32)
    mask = nib.load(osp(base_dir, row['seg_path'])).get_fdata()
    img, mask = pad_to_multiple(img, mask, multiple=32)

    img = np.clip(img, img_p1, img_p99)
    img = (img - img_p1) / (img_p99 - img_p1 + 1e-8) * 2 - 1

    mask = (mask > 0).astype(np.int64)

    img = np.transpose(img, (2, 0, 1))
    mask = np.transpose(mask, (2, 0, 1))

    img = torch.from_numpy(img).float().unsqueeze(0)
    mask = torch.from_numpy(mask).long()
    return img, mask