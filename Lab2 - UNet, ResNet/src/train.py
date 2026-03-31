import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from oxford_pet import OxfordPetSegDataset, load_name_list
from utils import compute_loss, dice_score_from_logits, ensure_dir
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet


def build_model(name, base_channels=64):
    if name == "unet":
        return UNet(in_channels=3, out_channels=1, base_channels=base_channels)
    if name == "resnet34_unet":
        return ResNet34UNet(out_channels=1)
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "resnet34_unet"], required=True)
    parser.add_argument("--data-root", default="dataset/oxford-iiit-pet")
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--save-dir", default="saved_models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = ensure_dir(args.save_dir)
    run_name = f"{args.model}_img{args.img_size}_lr{args.lr}_wd{args.weight_decay}".replace(".", "p")
    checkpoint_path = save_dir / f"{run_name}.pth"

    train_names = load_name_list(Path(args.split_root) / "train.txt")
    val_names = load_name_list(Path(args.split_root) / "val.txt")
    train_ds = OxfordPetSegDataset(args.data_root, train_names, img_size=args.img_size, augment=True, return_mask=True)
    val_ds = OxfordPetSegDataset(args.data_root, val_names, img_size=args.img_size, augment=False, return_mask=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_dice = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_dice_sum = 0.0
        train_count = 0

        for images, masks in train_dl:
            images = images.to(device)
            masks = masks.to(device)

            opt.zero_grad()
            outputs = model(images)
            loss = compute_loss(outputs, masks, bce_loss)
            loss.backward()
            opt.step()

            batch_size = images.size(0)
            train_loss_sum += loss.item() * batch_size
            train_dice_sum += dice_score_from_logits(outputs, masks).item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / train_count
        train_dice = train_dice_sum / train_count

        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for images, masks in val_dl:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = compute_loss(outputs, masks, bce_loss)
                batch_size = images.size(0)
                val_loss_sum += loss.item() * batch_size
                val_dice_sum += dice_score_from_logits(outputs, masks).item() * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / val_count
        val_dice = val_dice_sum / val_count
        scheduler.step(val_loss)

        print(
            f"epoch {epoch:02d} "
            f"train_loss {train_loss:.4f} train_dice {train_dice:.4f} "
            f"val_loss {val_loss:.4f} val_dice {val_dice:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    print("Training finished")
    print(f"Best val dice: {best_val_dice:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()

