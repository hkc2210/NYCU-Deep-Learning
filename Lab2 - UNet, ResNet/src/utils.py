from pathlib import Path

import numpy as np
import torch


def split_outputs(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0], list(outputs[1:])
    return outputs, []


def soft_dice_loss(logits, masks, eps=1e-7):
    probs = torch.sigmoid(logits)
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def single_loss(logits, masks, bce_loss):
    return 0.1 * bce_loss(logits, masks) + 0.9 * soft_dice_loss(logits, masks)


def compute_loss(outputs, masks, bce_loss):
    main_logits, aux_logits = split_outputs(outputs)
    loss = single_loss(main_logits, masks, bce_loss)
    if aux_logits:
        aux_weights = [0.4, 0.25, 0.15]
        for weight, aux in zip(aux_weights, aux_logits):
            loss = loss + weight * single_loss(aux, masks, bce_loss)
    return loss


def dice_score_from_logits(outputs, masks, threshold=0.5, eps=1e-7):
    logits, _ = split_outputs(outputs)
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()


def rle_encode(mask):
    mask = mask.astype(np.uint8)
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

