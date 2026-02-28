from tqdm import tqdm
import torch 
from os.path import join as osp
from datetime import datetime

@torch.no_grad()
def validate(
    discriminator,
    generator,
    valid_loader,
    d_loss_fn,
    g_loss_fn,
    device,
):
    discriminator.eval()
    generator.eval()

    d_epoch = 0.0
    g_epoch = 0.0
    g_adv_epoch = 0.0
    g_l1_epoch = 0.0
    g_diff_epoch = 0.0
    total_epoch = 0.0
    count = 0

    pbar = tqdm(valid_loader, desc="Valid", leave=False)

    for source, target, cond, pos in pbar:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        cond   = cond.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking = True)

        with torch.cuda.amp.autocast():
            fake = generator(source, cond, pos)

            pred_real = discriminator(source, target, cond)
            loss_d_real = d_loss_fn(pred_real, True)

            pred_fake_d = discriminator(source, fake, cond)
            loss_d_fake = d_loss_fn(pred_fake_d, False)

            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            pred_fake = discriminator(source, fake, cond)
            loss_g, loss_g_adv, loss_g_l1, loss_g_diff = g_loss_fn(
                pred_fake=pred_fake,
                fake=fake,
                target=target,
            )

        total_loss = loss_d + loss_g

        d_epoch += loss_d.item()
        g_epoch += loss_g.item()
        g_adv_epoch += loss_g_adv.item()
        g_l1_epoch += loss_g_l1.item()
        g_diff_epoch += loss_g_diff.item()
        total_epoch += total_loss.item()
        count += 1

    return {
        "d_loss": d_epoch / count,
        "g_loss": g_epoch / count,
        "g_adv": g_adv_epoch / count,
        "g_l1": g_l1_epoch / count,
        "g_diff": g_diff_epoch / count,
        "total_loss": total_epoch / count,
    }
    
def train(
    discriminator,
    generator,
    epochs,
    train_loader,
    valid_loader,
    optimizer_d,
    optimizer_g,
    scheduler_d,
    scheduler_g,
    d_loss_fn,
    g_loss_fn,
    device,
    scaler,
    save_dir,
    max_patience = 10
):
    history = {
        "train": [],
        "val": [],
    }

    best_val_total = float("inf")
    patience = 0

    for epoch in range(epochs):
        # TRAIN
        discriminator.train()
        generator.train()

        d_epoch = 0.0
        g_epoch = 0.0
        g_adv_epoch = 0.0
        g_l1_epoch = 0.0
        g_diff_epoch = 0.0
        total_epoch = 0.0
        count = 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Train", leave=True)

        for source, target, cond, pos in pbar:
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            cond   = cond.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking = True)

            # Discriminator
            optimizer_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    fake = generator(source, cond, pos)

                pred_real = discriminator(source, target, cond)
                loss_d_real = d_loss_fn(pred_real, True)

                pred_fake = discriminator(source, fake.detach(), cond)
                loss_d_fake = d_loss_fn(pred_fake, False)

                loss_d = 0.5 * (loss_d_real + loss_d_fake)

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)

            # Generator
            optimizer_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                fake = generator(source, cond, pos)
                pred_fake = discriminator(source, fake, cond)

                loss_g, loss_g_adv, loss_g_l1, loss_g_diff = g_loss_fn(
                    pred_fake=pred_fake,
                    fake=fake,
                    target=target,
                )

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            total_loss = loss_d + loss_g

            d_epoch += loss_d.item()
            g_epoch += loss_g.item()
            g_adv_epoch += loss_g_adv.item()
            g_l1_epoch += loss_g_l1.item()
            g_diff_epoch += loss_g_diff.item()
            total_epoch += total_loss.item()
            count += 1

        train_metrics = {
            "d_loss": d_epoch / count,
            "g_loss": g_epoch / count,
            "g_adv": g_adv_epoch / count,
            "g_l1": g_l1_epoch / count,
            "g_diff": g_diff_epoch / count,
            "total_loss": total_epoch / count,
        }

        print(
            f"[Epoch {epoch}] "
            f"Train | "
            f"D: {d_epoch / count:.4f} "
            f"G: {g_epoch / count:.4f} "
            f"G_adv: {g_adv_epoch / count:.4f} "
            f"G_L1: {g_l1_epoch / count:.4f} "
            f"G_diff: {g_diff_epoch / count:.4f} "
            f"Total: {total_epoch / count:.4f}"
        )

        # VALIDATION
        val_metrics = validate(
            discriminator=discriminator,
            generator=generator,
            valid_loader=valid_loader,
            d_loss_fn=d_loss_fn,
            g_loss_fn=g_loss_fn,
            device=device,
        )

        print(
            f"[Epoch {epoch}] "
            f"Valid | "
            f"D: {val_metrics['d_loss']:.4f} "
            f"G: {val_metrics['g_loss']:.4f} "
            f"G_adv: {val_metrics['g_adv']:.4f} "
            f"G_L1: {val_metrics['g_l1']:.4f} "
            f"G_diff: {val_metrics['g_diff']:.4f} "
            f"Total: {val_metrics['total_loss']:.4f}"
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics['total_loss'] < best_val_total:
            best_val_total = val_metrics['total_loss']
            patience = 0

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            loss_str = f"{best_val_total:.4f}"

            filename = (
                f"residual_best_{timestamp}_epoch_{epoch}_valloss_{loss_str}.pt"
            )
            save_path = osp(save_dir, filename)

            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "val_total_loss": best_val_total,
                },
                save_path,
            )

            print(f"===== Best model saved → {filename}")

        # else:
        #     patience += 1
        #     if patience == max_patience:
        #         print(f"===== Early stopping at epoch {epoch}")
        #         break
        scheduler_d.step()
        scheduler_g.step()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_str = f"{val_metrics['g_l1']:.4f}"

    filename = (
        f"residual_final_{timestamp}_valloss_{loss_str}.pt"
    )
    save_path = osp(save_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "val_total_loss": best_val_total,
        },
        save_path,
    )
    return history

@torch.no_grad()
def residual_validate(
    discriminator,
    generator,
    valid_loader,
    d_loss_fn,
    g_loss_fn,
    device,
):
    discriminator.eval()
    generator.eval()

    d_epoch = 0.0
    g_epoch = 0.0
    g_adv_epoch = 0.0
    g_l1_epoch = 0.0
    g_diff_epoch = 0.0
    total_epoch = 0.0
    count = 0

    pbar = tqdm(valid_loader, desc="Valid", leave=False)

    for source, target, cond, pos in pbar:
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        cond   = cond.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking = True)

        with torch.cuda.amp.autocast():
            residual = generator(source, cond, pos)
            fake = source + residual # [-2, 2]의 범위
            fake = torch.clamp(fake, -1, 1)

            pred_real = discriminator(source, target, cond)
            loss_d_real = d_loss_fn(pred_real, True)

            pred_fake_d = discriminator(source, fake, cond)
            loss_d_fake = d_loss_fn(pred_fake_d, False)

            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            pred_fake = discriminator(source, fake, cond)
            loss_g, loss_g_adv, loss_g_l1, loss_g_diff = g_loss_fn(
                pred_fake=pred_fake,
                fake=fake,
                target=target,
            )
            # loss_g, loss_g_adv, loss_g_l1 = g_loss_fn(
            #     pred_fake=pred_fake,
            #     fake=fake,
            #     target=target,
            # )

        total_loss = loss_d + loss_g

        d_epoch += loss_d.item()
        g_epoch += loss_g.item()
        g_adv_epoch += loss_g_adv.item()
        g_l1_epoch += loss_g_l1.item()
        g_diff_epoch += loss_g_diff.item()
        total_epoch += total_loss.item()
        count += 1

    return {
        "d_loss": d_epoch / count,
        "g_loss": g_epoch / count,
        "g_adv": g_adv_epoch / count,
        "g_l1": g_l1_epoch / count,
        "g_diff": g_diff_epoch / count,
        "total_loss": total_epoch / count,
    }
    
def residual_train(
    discriminator,
    generator,
    epochs,
    train_loader,
    valid_loader,
    optimizer_d,
    optimizer_g,
    scheduler_d,
    scheduler_g,
    d_loss_fn,
    g_loss_fn,
    device,
    scaler,
    save_dir,
    max_patience = 10
):
    history = {
        "train": [],
        "val": [],
    }

    best_val_total = float("inf")
    patience = 0

    for epoch in range(epochs):
        # TRAIN
        discriminator.train()
        generator.train()

        d_epoch = 0.0
        g_epoch = 0.0
        g_adv_epoch = 0.0
        g_l1_epoch = 0.0
        g_diff_epoch = 0.0
        total_epoch = 0.0
        count = 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Train", leave=True)

        for source, target, cond, pos in pbar:
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            cond   = cond.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking = True)

            # Discriminator
            optimizer_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    residual = generator(source, cond, pos)
                    fake = source + residual # [-2, 2]의 범위
                    fake = torch.clamp(fake, -1, 1)

                pred_real = discriminator(source, target, cond)
                loss_d_real = d_loss_fn(pred_real, True)

                pred_fake = discriminator(source, fake.detach(), cond)
                loss_d_fake = d_loss_fn(pred_fake, False)

                loss_d = 0.5 * (loss_d_real + loss_d_fake)

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)

            # Generator
            optimizer_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                residual = generator(source, cond, pos)
                fake = source + residual # [-2, 2]의 범위
                fake = torch.clamp(fake, -1, 1)
                pred_fake = discriminator(source, fake, cond)

                loss_g, loss_g_adv, loss_g_l1, loss_g_diff = g_loss_fn(
                    pred_fake=pred_fake,
                    fake=fake,
                    target=target,
                )
                # loss_g, loss_g_adv, loss_g_l1 = g_loss_fn(
                #     pred_fake=pred_fake,
                #     fake=fake,
                #     target=target,
                # )

            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            total_loss = loss_d + loss_g

            d_epoch += loss_d.item()
            g_epoch += loss_g.item()
            g_adv_epoch += loss_g_adv.item()
            g_l1_epoch += loss_g_l1.item()
            g_diff_epoch += loss_g_diff.item()
            total_epoch += total_loss.item()
            count += 1

        train_metrics = {
            "d_loss": d_epoch / count,
            "g_loss": g_epoch / count,
            "g_adv": g_adv_epoch / count,
            "g_l1": g_l1_epoch / count,
            "g_diff": g_diff_epoch / count,
            "total_loss": total_epoch / count,
        }

        print(
            f"[Epoch {epoch}] "
            f"Train | "
            f"D: {d_epoch / count:.4f} "
            f"G: {g_epoch / count:.4f} "
            f"G_adv: {g_adv_epoch / count:.4f} "
            f"G_L1: {g_l1_epoch / count:.4f} "
            f"G_diff: {g_diff_epoch / count:.4f} "
            f"Total: {total_epoch / count:.4f}"
        )

        # VALIDATION
        val_metrics = residual_validate(
            discriminator=discriminator,
            generator=generator,
            valid_loader=valid_loader,
            d_loss_fn=d_loss_fn,
            g_loss_fn=g_loss_fn,
            device=device,
        )

        print(
            f"[Epoch {epoch}] "
            f"Valid | "
            f"D: {val_metrics['d_loss']:.4f} "
            f"G: {val_metrics['g_loss']:.4f} "
            f"G_adv: {val_metrics['g_adv']:.4f} "
            f"G_L1: {val_metrics['g_l1']:.4f} "
            f"G_diff: {val_metrics['g_diff']:.4f} "
            f"Total: {val_metrics['total_loss']:.4f}"
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics['total_loss'] < best_val_total:
            best_val_total = val_metrics['total_loss']
            patience = 0

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            loss_str = f"{best_val_total:.4f}"

            filename = (
                f"residual_best_{timestamp}_epoch_{epoch}_valloss_{loss_str}.pt"
            )
            save_path = osp(save_dir, filename)

            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "val_total_loss": best_val_total,
                },
                save_path,
            )

            print(f"===== Best model saved → {filename}")

        # else:
        #     patience += 1
        #     if patience == max_patience:
        #         print(f"===== Early stopping at epoch {epoch}")
        #         break
        scheduler_d.step()
        scheduler_g.step()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_str = f"{val_metrics['g_l1']:.4f}"

    filename = (
        f"residual_final_{timestamp}_valloss_{loss_str}.pt"
    )
    save_path = osp(save_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "val_total_loss": best_val_total,
        },
        save_path,
    )
    return history