import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from oxford_pet import OxfordPetSegDataset, load_name_list
from utils import compute_loss, dice_score_from_logits
from train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "resnet34_unet"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="dataset/oxford-iiit-pet")
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_names = load_name_list(Path(args.split_root) / "val.txt")
    val_ds = OxfordPetSegDataset(args.data_root, val_names, img_size=args.img_size, augment=False, return_mask=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model, args.base_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    bce_loss = nn.BCEWithLogitsLoss()

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

    print(f"val_loss={val_loss_sum / val_count:.4f}")
    print(f"val_dice={val_dice_sum / val_count:.4f}")


if __name__ == "__main__":
    main()

