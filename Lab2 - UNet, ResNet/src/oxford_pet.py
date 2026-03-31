from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms import ColorJitter


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def load_name_list(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class OxfordPetSegDataset(Dataset):
    def __init__(
        self,
        root,
        names,
        img_size=320,
        augment=False,
        return_mask=True,
        color_jitter=None,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "annotations" / "trimaps"
        self.names = names
        self.img_size = img_size
        self.augment = augment
        self.return_mask = return_mask
        self.color_jitter = color_jitter or ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        image = Image.open(self.image_dir / f"{name}.jpg").convert("RGB")
        original_size = image.size

        if self.return_mask:
            mask = Image.open(self.mask_dir / f"{name}.png").convert("L")
        else:
            mask = None

        if self.augment:
            image = self.color_jitter(image)
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            angle = random.uniform(-10.0, 10.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=2)

        image = TF.resize(image, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        image = TF.to_tensor(image)
        image = TF.normalize(image, MEAN, STD)

        if self.return_mask:
            mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
            mask = (mask == 1).float().unsqueeze(0)
            return image, mask

        return image, name, original_size

