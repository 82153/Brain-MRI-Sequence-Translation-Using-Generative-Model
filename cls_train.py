from omegaconf import OmegaConf
import torch
from codes.utils import set_seed
from codes.dataset import CLSMRIDataset
import pandas as pd
from os.path import join as osp
from torch.utils.data import DataLoader
import timm_3d
import torch.optim as optim
import torch.nn as nn
from codes.cls_trainer import train
import os
import argparse
import gc
import json

def main(cfg):
    set_seed(cfg.seed)
    
    device = cfg.device
    train_df = pd.read_csv(osp(cfg.base_dir, cfg.train.df_path))
    valid_df = pd.read_csv(osp(cfg.base_dir, cfg.valid.df_path))
    
    train_dataset = CLSMRIDataset(train_df, cfg.base_dir)
    valid_dataset = CLSMRIDataset(valid_df, cfg.base_dir)
    
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle = True, num_workers = cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, cfg.batch_size, shuffle = False, num_workers = cfg.num_workers)
    
    model = timm_3d.create_model(cfg.model.name, **cfg.model.parameters).to(device)
    
    optimizer = getattr(optim, cfg.optimizer.name)(model.parameters(), **cfg.optimizer.parameters)
    
    criterion = getattr(nn, cfg.loss.name)(**cfg.loss.parameters)
    scaler = torch.cuda.amp.GradScaler()
    
    history = train(model, train_loader, valid_loader, cfg.epochs, optimizer, criterion, scaler, device, cfg.save_dir)
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4) 

if __name__=='__main__':
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cls_config.yaml")
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
        
    main(cfg)