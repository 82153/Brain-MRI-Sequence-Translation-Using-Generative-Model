import os
import gc
import json
from omegaconf import OmegaConf
import argparse
import torch
from codes.utils import set_seed
from codes.dataset import SegMRIDataset
import pandas as pd
from os.path import join as osp
from torch.utils.data import DataLoader
import segmentation_models_pytorch_3d as smp
from codes.loss import DiceBCELoss
from codes.metric import dice_metric
from codes.seg_trainer import train
import torch.optim as optim
import torch.optim.lr_scheduler as sch

def main(cfg):
    set_seed(cfg.seed)
    device = cfg.device
    
    train_df = pd.read_csv(osp(cfg.base_dir, cfg.train.df_path))
    valid_df = pd.read_csv(osp(cfg.base_dir, cfg.valid.df_path))
    
    train_dataset = SegMRIDataset(train_df, cfg.base_dir)
    valid_dataset = SegMRIDataset(valid_df, cfg.base_dir)
    
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle = True, num_workers = cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, cfg.batch_size, shuffle = False, num_workers = cfg.num_workers)
    
    model = getattr(smp, cfg.model.name)(**cfg.model.parameters).to(device)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = getattr(optim, cfg.optimizer.name)(model.parameters(), **cfg.optimizer.parameters)
    
    criterion = DiceBCELoss(device=device,**cfg.loss.parameters)
    metric = dice_metric
    
    scheduler = getattr(sch, cfg.scheduler.name)(optimizer, **cfg.scheduler.parameters)
    
    history = train(model, train_loader, valid_loader, cfg.epochs, optimizer, criterion, metric, scaler, device, cfg.save_dir, scheduler)
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4) 
        
if __name__=='__main__':
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/seg_config.yaml")
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
        
    main(cfg)