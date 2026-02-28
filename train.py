from omegaconf import OmegaConf
import torch
from codes.utils import set_seed
from codes.dataset import MRIDataset, TestMRIDataset
import pandas as pd
from os.path import join as osp
from torch.utils.data import DataLoader
from codes.model import Generator, Discriminator
import torch.optim as optim
from codes.loss import GeneratorLoss, GANLoss
import torch.optim.lr_scheduler as sch
import argparse
from codes.trainer import train, residual_train
import os
import gc
import json

def main(cfg):
    set_seed(cfg.seed)
    device = cfg.device
    train_df = pd.read_csv(osp(cfg.base_dir, cfg.train.df_path))
    valid_df = pd.read_csv(osp(cfg.base_dir, cfg.valid.df_path))
    
    train_dataset = MRIDataset(train_df, cfg.base_dir, cfg.train.patch_size, cfg.train.brain_prob)
    valid_dataset = TestMRIDataset(valid_df, cfg.base_dir)
    
    train_loader = DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle = True, num_workers = cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size = cfg.batch_size, shuffle = False, num_workers = cfg.num_workers)
    
    D = Discriminator(in_channels = cfg.model.in_channels, num_domains = cfg.model.num_domain).to(device)
    G = Generator(in_channels = cfg.model.in_channels, out_channels = cfg.model.out_channels, 
                  num_domains = cfg.model.num_domain, num_pos = cfg.model.num_pos, init_features = cfg.model.init_features).to(device)
    
    optimizer = getattr(optim, cfg.optimizer)
    optimizer_G = optimizer(G.parameters(), lr = cfg.lr)
    optimizer_D = optimizer(D.parameters(), lr = cfg.lr)
    
    d_loss_fn = GANLoss()
    g_loss_fn = GeneratorLoss(d_loss_fn, cfg.loss.lambda_l1, cfg.loss.lambda_diff)
    
    scheduler = getattr(sch, cfg.scheduler.name)
    scheduler_d = scheduler(optimizer_D, **cfg.scheduler.parameters)
    scheduler_g = scheduler(optimizer_G, **cfg.scheduler.parameters)
    
    scaler = torch.cuda.amp.GradScaler()
    
    if cfg.train.is_residual:
        history = residual_train(D, G, cfg.epochs, train_loader, valid_loader, optimizer_D, optimizer_G, scheduler_d, scheduler_g, d_loss_fn, g_loss_fn, 
                          device, scaler, cfg.save_dir, cfg.max_patience)
    else:
        history = train(D, G, cfg.epochs, train_loader, valid_loader, optimizer_D, optimizer_G, scheduler_d, scheduler_g, d_loss_fn, g_loss_fn, 
                          device, scaler, cfg.save_dir, cfg.max_patience)
    
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4) 
    
if __name__=='__main__':
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
        
    main(cfg)