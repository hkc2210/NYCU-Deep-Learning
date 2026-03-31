import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from oxford_pet import OxfordPetSegDataset, load_name_list
from train import build_model
from utils import dice_score_from_logits, rle_encode, split_outputs


def predict_with_tta(model, images):
    main_logits, _ = split_outputs(model(images))
    probs = torch.sigmoid(main_logits)

    h_images = torch.flip(images, dims=[3])
    h_logits, _ = split_outputs(model(h_images))
    h_probs = torch.sigmoid(h_logits)
    h_probs = torch.flip(h_probs, dims=[3])

    v_images = torch.flip(images, dims=[2])
    v_logits, _ = split_outputs(model(v_images))
    v_probs = torch.sigmoid(v_logits)
    v_probs = torch.flip(v_probs, dims=[2])

    hv_images = torch.flip(images, dims=[2, 3])
    hv_logits, _ = split_outputs(model(hv_images))
    hv_probs = torch.sigmoid(hv_logits)
    hv_probs = torch.flip(hv_probs, dims=[2, 3])

    return (probs + h_probs + v_probs + hv_probs) / 4


def search_threshold(model, val_dl, device, thresholds):
    best_threshold = None
    best_dice = 0.0
    for threshold in thresholds:
        dice_sum = 0.0
        count = 0
        with torch.no_grad():
            for images, masks in val_dl:
                images = images.to(device)
                masks = masks.to(device)
                probs = predict_with_tta(model, images)
                preds = (probs > threshold).float()
                intersection = (preds * masks).sum(dim=(1, 2, 3))
                union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                dice = (2 * intersection + 1e-7) / (union + 1e-7)
                dice_sum += dice.sum().item()
                count += images.size(0)
        avg_dice = dice_sum / count
        print(f"threshold={threshold:.2f}, val_dice={avg_dice:.4f}")
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_threshold = threshold
    return best_threshold, best_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "resnet34_unet"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="dataset/oxford-iiit-pet")
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58])
    parser.add_argument("--submission-path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.base_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    val_names = load_name_list(Path(args.split_root) / "val.txt")
    test_names = load_name_list(Path(args.split_root) / args.test_file)
    val_ds = OxfordPetSegDataset(args.data_root, val_names, img_size=args.img_size, augment=False, return_mask=True)
    test_ds = OxfordPetSegDataset(args.data_root, test_names, img_size=args.img_size, augment=False, return_mask=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    threshold = args.threshold
    if threshold is None:
        threshold, best_dice = search_threshold(model, val_dl, device, args.thresholds)
        print(f"best threshold = {threshold:.2f}, best val dice = {best_dice:.4f}")

    rows = []
    with torch.no_grad():
        for images, image_ids, original_sizes in test_dl:
            images = images.to(device)
            probs = predict_with_tta(model, images).cpu()
            for i, image_id in enumerate(image_ids):
                width = int(original_sizes[0][i].item())
                height = int(original_sizes[1][i].item())
                prob_map = probs[i : i + 1]
                prob_map = F.interpolate(prob_map, size=(height, width), mode="bilinear", align_corners=False)
                pred_mask = (prob_map[0, 0].numpy() > threshold).astype(np.uint8)
                rows.append((image_id, rle_encode(pred_mask)))

    submission_path = Path(args.submission_path)
    with submission_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(rows)

    print(f"submission threshold: {threshold}")
    print(f"submission path: {submission_path}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    import numpy as np

    main()

