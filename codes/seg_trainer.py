import torch
from tqdm import tqdm
from datetime import datetime
from os.path import join as osp

@torch.no_grad()
def validate(model, epoch, loader, device, criterion, metric):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"[Epoch {epoch}] Valid")


    for img, mask in pbar:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits = model(img)
        loss = criterion(logits, mask)

        dice = metric(logits, mask)

        total_loss += loss.item()
        total_dice += dice
        count += 1

    return {
        "loss": total_loss / max(1, count),
        "dice": total_dice / max(1, count)
    }


def train(model, train_loader, val_loader, epochs, optimizer, criterion, metric, scaler, device, save_dir, scheduler):
    model = model.to(device)

    best_val_dice = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": []
    }

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_dice = 0.0
        count = 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Train")

        for img, mask in pbar:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                logits = model(img)
                loss = criterion(logits, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            dice = metric(logits, mask)

            total_loss += loss.item()
            total_dice += dice
            count += 1

        train_loss = total_loss / max(1, count)
        train_dice = total_dice / max(1, count)
        scheduler.step()
        # validation
        val_stats = validate(model, epoch, val_loader, device, criterion, metric)

        print(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f} | "
            f"train_dice={train_dice:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"val_dice={val_stats['dice']:.4f}"
        )

        # history 저장
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_stats["loss"])
        history["val_dice"].append(val_stats["dice"])

        # best save
        if val_stats["dice"] > best_val_dice:
            best_val_dice = val_stats["dice"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = osp(save_dir, f"binary_t2_seg_best_model_{timestamp}_epoch{epoch}_loss{best_val_dice:.4f}")
            torch.save(model.state_dict(), save_path)
            print(f"===== Best model saved {save_path}")

    return history