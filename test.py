from tqdm import tqdm
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from codes.model import Generator
from codes.dataset import TestMRIDataset
import pandas as pd
from codes.metric import compute_psnr, compute_ssim
from os.path import join as osp
from torch.utils.data import DataLoader
import gc

@torch.no_grad()
def infer_full_volume_sliding_window(
    G,
    img,
    cond_id: int,
    patch_size=(64,64,64),
    stride=(32,32,32),
    device="cuda",
):
    G.eval()

    D, H, W = img.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # 1) 시작 좌표 리스트 만들기 (끝도 포함)
    z_starts = list(range(0, max(D - pd + 1, 1), sd))
    y_starts = list(range(0, max(H - ph + 1, 1), sh))
    x_starts = list(range(0, max(W - pw + 1, 1), sw))

    if len(z_starts) == 0: z_starts = [0]
    if len(y_starts) == 0: y_starts = [0]
    if len(x_starts) == 0: x_starts = [0]

    if z_starts[-1] != D - pd: z_starts.append(D - pd)
    if y_starts[-1] != H - ph: y_starts.append(H - ph)
    if x_starts[-1] != W - pw: x_starts.append(W - pw)

    # 2) 누적 버퍼 (count 방식)
    out_sum = torch.zeros((1,1,D,H,W), device=device, dtype=torch.float32)
    w_sum   = torch.zeros((1,1,D,H,W), device=device, dtype=torch.float32)

    # 3) cond one-hot
    cond_id_t = torch.tensor(cond_id, dtype=torch.long, device=device)
    cond = F.one_hot(cond_id_t, num_classes=3).float().unsqueeze(0)  # [1,3]

    # 4) 슬라이딩 윈도우 inference
    img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = img_t[:, :, z:z+pd, y:y+ph, x:x+pw]

                # 중심 좌표 기반 pos
                z0 = z + pd // 2
                y0 = y + ph // 2
                x0 = x + pw // 2

                pos = torch.tensor(
                    [[
                        (z0 / D) * 2 - 1,
                        (y0 / H) * 2 - 1,
                        (x0 / W) * 2 - 1
                    ]],
                    device=device,
                    dtype=torch.float32
                )

                pred = G(patch, cond, pos)
                # residual = G(patch, cond, pos)
                # pred = patch + residual
                # pred = torch.clamp(pred, -1, 1)

                # count 방식
                out_sum[:, :, z:z+pd, y:y+ph, x:x+pw] += pred
                w_sum[:,   :, z:z+pd, y:y+ph, x:x+pw] += 1

    out = out_sum / (w_sum + 1e-8)
    return out

@torch.no_grad()
def test_model(G, test_loader, stride, device="cuda"):

    G.eval()

    psnr_list = []
    ssim_list = []

    for src, tgt, cond in tqdm(test_loader, desc="Testing"):
        src = src.to(device)  # [B,1,D,H,W]
        tgt = tgt.to(device)
        cond = cond.to(device)

        B, _, D, H, W = src.shape

        for i in range(B):

            # cond 처리
            ci = cond[i]
            if ci.ndim > 0 and ci.numel() > 1:
                cond_id = int(torch.argmax(ci).item())
            else:
                cond_id = int(ci.item())

            # ---- 전체 볼륨 inference ----
            src_np = src[i,0].detach().cpu().numpy()  # [D,H,W]
            pred = infer_full_volume_sliding_window(
                G,
                src_np,
                cond_id,
                stride = stride,
                device=device
            )
            pred = pred.squeeze(0).squeeze(0).cpu().numpy()

            gt_np = tgt[i,0].detach().cpu().numpy()

            # ---- PSNR ----
            psnr = compute_psnr(pred, gt_np, data_range=2.0)
            psnr_list.append(psnr)

            # ---- SSIM ----
            ssim_val = compute_ssim(pred, gt_np, data_range=2.0)
            ssim_list.append(ssim_val)

    results = {
        "PSNR_mean": np.mean(psnr_list),
        "SSIM_mean": np.mean(ssim_list),
    }
    return results

def main(cfg):
    device = cfg.device
    checkpoint = torch.load(cfg.test.pt_path, map_location = device)
    G = Generator(in_channels = cfg.model.in_channels, out_channels = cfg.model.out_channels, 
                  num_domains = cfg.model.num_domain, num_pos = cfg.model.num_pos, init_features = cfg.model.init_features).to(device)
    G.load_state_dict(checkpoint['generator'])
    
    test_df = pd.read_csv(osp(cfg.base_dir, cfg.test.df_path))
    test_dataset = TestMRIDataset(test_df, cfg.base_dir)
    test_loader = DataLoader(test_dataset, batch_size = cfg.batch_size, shuffle = False, num_workers = cfg.num_workers)
    result = test_model(G, test_loader, cfg.test.stride, device)
    print(result)
    
if __name__=="__main__":
    gc.collect()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    main(cfg)