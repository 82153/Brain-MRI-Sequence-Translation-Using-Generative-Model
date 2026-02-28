from torch.utils.data.dataset import Dataset
import nibabel as nib
import torch.nn.functional as F
from os.path import join as osp
import torch
import numpy as np
from codes.utils import load_nii_pre, load_nii_pre_cls, load_nii_pre_seg

class MRIDataset(Dataset):
    def __init__(self, df, base_dir, patch_size =(64, 64, 64), brain_prob = 0.7):
        self.base_dir = base_dir
        self.df = df
        self.patch_size = patch_size
        self.brain_prob = brain_prob
        self.seq_map = {
            "t1":0,
            "t2":1,
            "flair":2
        }

        self.z_min_frac = 0.014701
        self.z_max_frac = 0.897270
        self.y_min_frac = 0.208986
        self.y_max_frac = 0.777663
        self.x_min_frac = 0.186373
        self.x_max_frac = 0.889814

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_dir = osp(self.base_dir, self.df['source_nii'].iloc[idx])
        gt_dir = osp(self.base_dir, self.df['target_nii'].iloc[idx])

        img = nib.load(img_dir).get_fdata(dtype = np.float32)
        gt = nib.load(gt_dir).get_fdata(dtype = np.float32)

        img_p1 = self.df.loc[idx, "img_p1"]
        img_p99 = self.df.loc[idx, 'img_p99']
        gt_p1 = self.df.loc[idx, "gt_p1"]
        gt_p99 = self.df.loc[idx, "gt_p99"]

        img = np.clip(img, img_p1, img_p99)
        gt = np.clip(gt, gt_p1, gt_p99)

        img = (img - img_p1) / (img_p99 - img_p1 + 1e-8) * 2 - 1
        gt = (gt- gt_p1) / (gt_p99 - gt_p1 + 1e-8) * 2 - 1

        img = np.transpose(img, (2, 0, 1))
        gt = np.transpose(gt, (2, 0, 1))

        D, H, W = img.shape
        pd, ph, pw = self.patch_size

        z_min_b = int(self.z_min_frac * D)
        z_max_b = int(self.z_max_frac * D)

        y_min_b = int(self.y_min_frac * H)
        y_max_b = int(self.y_max_frac * H)

        x_min_b = int(self.x_min_frac * W)
        x_max_b = int(self.x_max_frac * W)

        if np.random.rand() < self.brain_prob:
            z0 = np.random.randint(z_min_b, z_max_b)
            y0 = np.random.randint(y_min_b, y_max_b)
            x0 = np.random.randint(x_min_b, x_max_b)
        else:
            while 1:
                z0 = np.random.randint(0, D)
                y0 = np.random.randint(0, H)
                x0 = np.random.randint(0, W)
                if (z_min_b <= 0 < z_max_b and y_min_b <= 0 < y_max_b and x_min_b <= 0 < x_max_b):
                    continue
                else:
                    break

        z = np.clip(z0-pd//2, 0, D-pd)
        y = np.clip(y0-ph//2, 0, H-ph)
        x = np.clip(x0-pw//2, 0, W-pw)

        img = img[z:z+pd, y:y+ph, x:x+pw]
        gt = gt[z:z+pd, y:y+ph, x:x+pw]

        img = torch.from_numpy(img).float().unsqueeze(0)
        gt = torch.from_numpy(gt).float().unsqueeze(0)

        cond = torch.tensor(self.seq_map[self.df['target_seq'].iloc[idx]], dtype = torch.long)
        cond = F.one_hot(cond, num_classes = 3).float()
        pos = torch.tensor([(z0 / D) * 2 - 1, (y0/H) * 2 - 1, (x0/W) * 2 - 1], dtype = torch.float32)
        return img, gt, cond, pos

class TestMRIDataset(Dataset):
    def __init__(self, df, base_dir):
        self.base_dir = base_dir
        self.df = df
        self.seq_map = {'t1':0, 't2': 1, 'flair':2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src, tgt = load_nii_pre(row, self.base_dir)
        src = torch.from_numpy(src).float()
        tgt = torch.from_numpy(tgt).float()
        src = src.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
        cond = torch.tensor(self.seq_map[row['target_seq']], dtype = torch.long)
        return src, tgt, cond
    
class CLSMRIDataset(Dataset):
    def __init__(self, df, base_dir):
        self.base_dir = base_dir
        self.df = df
        self.seq_map = {"t1":0, "t2":1, "flair":2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img, cls = load_nii_pre_cls(row, self.base_dir)

        cls = torch.tensor(self.seq_map[cls], dtype=torch.long)

        return img, cls
    
class SegMRIDataset(Dataset):
    def __init__(self, df, base_dir):
        self.base_dir = base_dir
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img, mask = load_nii_pre_seg(row, self.base_dir)
        return img, mask