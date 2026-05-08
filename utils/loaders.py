import glob
import numpy as np
from PIL import Image
import cv2
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
# Make sure you have swin_unet.py from https://github.com/HuCaoFighting/Swin-Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os, shutil
import natsort



def apply_clahe(img):
    """Apply CLAHE on an RGB image (per channel in LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)



class FundusBmpDataset(Dataset):
    def __init__(self, root_dir, size=512, augment=False, margin=2, test=False):
        """
        root_dir: path containing case folders
        size: int (e.g., 512) or tuple/list like (H, W) or [H, W]
        augment: whether to apply augmentation (only for training/val)
        margin: crop margin factor
        test: if True, load all 7 annotators' masks and do resize/normalize only
        """
        self.root_dir = root_dir
        # normalize size -> (H, W) ints
        if isinstance(size, (list, tuple)):
            if len(size) != 2:
                raise ValueError(f"`size` as list/tuple must be length 2, got {size}")
            self.size_h, self.size_w = int(size[0]), int(size[1])
        else:
            self.size_h = self.size_w = int(size)

        self.margin = margin
        self.test = test
        self.samples = []

        # collect samples
        for case in sorted(os.listdir(root_dir)):
            case_path = os.path.join(root_dir, case)
            if not os.path.isdir(case_path):
                continue

            img_path = os.path.join(case_path, f"{case}.jpg")

            disc_paths = [os.path.join(case_path, f"{case}_seg_disc_{i}.png") for i in range(1, 8)]
            cup_paths  = [os.path.join(case_path, f"{case}_seg_cup_{i}.png")  for i in range(1, 8)]

            if self.test:
                if os.path.exists(img_path) and all(os.path.exists(p) for p in disc_paths + cup_paths):
                    self.samples.append((img_path, disc_paths, cup_paths, case))
            else:
                # for training/val we also keep all annotators
                if os.path.exists(img_path) and all(os.path.exists(p) for p in disc_paths + cup_paths):
                    self.samples.append((img_path, disc_paths, cup_paths, case))

        # transformations (only for train/val, not for test)
        if not self.test:
            if augment:
                self.transform = A.Compose([
                    A.Resize(self.size_h, self.size_w),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(self.size_h, self.size_w),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2()
                ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.test:
            # ---------- TEST MODE (7 raters): return all masks ----------
            img_path, disc_paths, cup_paths, case = self.samples[idx]

            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Could not read image at {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = apply_clahe(img)

            discs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in disc_paths]
            cups  = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in cup_paths]
            if any(m is None for m in discs + cups):
                raise RuntimeError(f"Could not read one of the masks for case {case}")

            discs = [(m > 130).astype(np.uint8) for m in discs]
            cups  = [(m > 130).astype(np.uint8) for m in cups]

            # crop window = union of all (disc ∪ cup)
            combined = np.zeros_like(discs[0], dtype=np.uint8)
            for m in discs + cups:
                combined = np.logical_or(combined, m).astype(np.uint8)

            ys, xs = np.where(combined > 0)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                w, h = x_max - x_min, y_max - y_min
                cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
                r = int(max(w, h) * self.margin / 2)
                x1, x2 = max(0, cx-r), min(img.shape[1], cx+r)
                y1, y2 = max(0, cy-r), min(img.shape[0], cy+r)

                img   = img[y1:y2, x1:x2]
                discs = [m[y1:y2, x1:x2] for m in discs]
                cups  = [m[y1:y2, x1:x2] for m in cups]

            # build final 7 cup masks
            final_masks = []
            for i in range(7):
                mask = (cups[i] == 1).astype(np.uint8)
                mask_resized = cv2.resize(mask, (self.size_w, self.size_h), interpolation=cv2.INTER_NEAREST)
                final_masks.append(torch.from_numpy(mask_resized).long())

            # resize + normalize image
            img_resized = cv2.resize(img, (self.size_w, self.size_h), interpolation=cv2.INTER_CUBIC)
            img_tensor  = TF.to_tensor(img_resized)                   # [0,1]
            img_tensor  = TF.normalize(img_tensor, [0.5]*3, [0.5]*3)  # [-1,1]

            return img_tensor, torch.stack(final_masks), case

        else:
            # ---------- TRAIN/VAL MODE (majority vote mask) ----------
            img_path, disc_paths, cup_paths, case = self.samples[idx]

            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Could not read image at {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = apply_clahe(img)

            discs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in disc_paths]
            cups  = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in cup_paths]
            if any(m is None for m in discs + cups):
                raise RuntimeError(f"Could not read mask(s) for case {case}")

            discs = [(m > 130).astype(np.uint8) for m in discs]
            cups  = [(m > 130).astype(np.uint8) for m in cups]

            # majority vote cup mask
            cup_stack = np.stack(cups, axis=0)    # [A,H,W]
            cup_mean  = cup_stack.mean(axis=0)    # [H,W] float [0,1]
            mask = (cup_mean >= 0.5).astype(np.uint8)  # hard binary {0,1}

            # crop around disc union
            disc_union = np.logical_or.reduce(discs).astype(np.uint8)
            ys, xs = np.where(disc_union > 0)
            if len(xs) > 0 and len(ys) > 0:
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                w, h = x_max - x_min, y_max - y_min
                cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
                r = int(max(w, h) * self.margin / 2)
                x1, x2 = max(0, cx-r), min(img.shape[1], cx+r)
                y1, y2 = max(0, cy-r), min(img.shape[0], cy+r)
                img  = img[y1:y2, x1:x2]
                mask = mask[y1:y2, x1:x2]

            augmented = self.transform(image=img, mask=mask)
            img_t  = augmented["image"]
            mask_t = augmented["mask"]

            return img_t, mask_t.long(), case