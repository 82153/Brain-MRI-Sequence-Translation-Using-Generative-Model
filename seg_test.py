import torch
from tqdm import tqdm
import segmentation_models_pytorch_3d as smp
import pandas as pd
from os.path import join as osp
from codes.dataset import SegMRIDataset
from torch.utils.data import DataLoader
from codes.metric import dice_metric
from codes.loss import DiceBCELoss
import argparse
from omegaconf import OmegaConf
import gc

@torch.no_grad()
def test(model, loader, device, criterion, metric):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    count = 0

    pbar = tqdm(loader, desc="[Test]")
    for img, mask in pbar:
        img = img.to(device)
        mask = mask.to(device)

        logits = model(img)
        loss = criterion(logits, mask)

        dice = metric(logits, mask)   

        dice_val = float(dice)

        total_loss += loss.item()
        total_dice += dice_val
        count += 1

    return {
        "loss": total_loss / max(1, count),
        "dice": total_dice / max(1, count)
    }
    
def main(cfg):
    device = cfg.device
    model = getattr(smp, cfg.model.name)(**cfg.model.parameters).to(device)
    
    if cfg.test.pt_path != "":
        checkpoint = torch.load(cfg.test.pt_path, map_location="cuda")
        model.load_state_dict(checkpoint)
    
    test_df = pd.read_csv(osp(cfg.base_dir, cfg.test.df_path))
    test_dataset = SegMRIDataset(test_df, cfg.base_dir)
    test_loader = DataLoader(test_dataset, cfg.batch_size, shuffle = False, num_workers = cfg.num_workers)
    criterion = DiceBCELoss(device=device,**cfg.loss.parameters)
    metric = dice_metric
    
    his = test(model, test_loader, device, criterion, metric)
    print(f"loss: {his['loss']:.4f} | dice: {his['dice']:.4f}")
    
if __name__=="__main__":
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/seg_config.yaml")
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    main(cfg)