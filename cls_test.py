import torch
from tqdm import tqdm
import timm_3d
from codes.dataset import CLSMRIDataset
from os.path import join as osp
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
from omegaconf import OmegaConf
import gc
import argparse

@torch.no_grad()
def test(model, loader, device, criterion, num_classes=3):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    count = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"[Test]")

    for img, cls in pbar:
        img = img.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)

        logits = model(img)
        loss = criterion(logits, cls)

        pred = logits.argmax(dim=1)

        acc = (pred == cls).float().mean().item()

        total_loss += loss.item()
        total_acc += acc
        count += 1

        all_preds.append(pred.detach().cpu())
        all_targets.append(cls.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for t, p in zip(all_targets, all_preds):
        conf_mat[t.long(), p.long()] += 1

    return {
        "loss": total_loss / max(1, count),
        "acc": total_acc / max(1, count),
        "confusion_matrix": conf_mat
    }
    
def main(cfg):
    device = cfg.device
    model = timm_3d.create_model(cfg.model.name, **cfg.model.parameters).to(device)
    if cfg.test.pt_path != "":
        checkpoint = torch.load(cfg.test.pt_path, map_location = device)
        model.load_state_dict(checkpoint)
    
    test_df = pd.read_csv(osp(cfg.base_dir, cfg.test.df_path))
    test_dataset = CLSMRIDataset(test_df, cfg.base_dir)
    test_loader = DataLoader(test_dataset, cfg.batch_size, shuffle = False, num_workers = cfg.num_workers)
    
    criterion = getattr(nn, cfg.loss.name)(**cfg.loss.parameters)
    his = test(model, test_loader, device, criterion, cfg.model.parameters.num_classes)
    print(f"loss: {his['loss']:.4f} | acc: {his['acc']:.4f}")
    
if __name__=="__main__":
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cls_config.yaml")
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    main(cfg)