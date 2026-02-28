from codes.metric import compute_acc
import torch
from tqdm import tqdm
from os.path import join as osp
from datetime import datetime

@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    count = 0

    for img, cls in loader:
        img = img.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)

        logits = model(img)
        loss = criterion(logits, cls)

        acc = compute_acc(logits, cls)

        total_loss += loss.item()
        total_acc += acc
        count += 1

    return {
        "loss": total_loss / max(1, count),
        "acc": total_acc / max(1, count)
    }


def train(model, train_loader, val_loader, epochs, optimizer, criterion, scaler, device, save_dir):
    model = model.to(device)

    best_val_loss = float("inf")

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_acc = 0.0
        count = 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Train")

        for img, cls in pbar:
            img = img.to(device, non_blocking=True)
            cls = cls.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                logits = model(img)
                loss = criterion(logits, cls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = compute_acc(logits.detach(), cls)

            total_loss += loss.item()
            total_acc += acc
            count += 1

        train_loss = total_loss / max(1, count)
        train_acc  = total_acc / max(1, count)

        # validation
        val_stats = validate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_stats['loss']:.4f}, val_acc={val_stats['acc']:.4f}"
        )

        # history 저장
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])

        # best save
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = osp(save_dir, f"cls_best_model_{timestamp}_epoch{epoch}_loss{best_val_loss:.4f}")
            torch.save(model.state_dict(), save_path)
            print(f"===== Best model saved {save_path}")

    return history