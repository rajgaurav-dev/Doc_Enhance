import os
import re
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Match input_X.jpg ↔ gt_X.jpg safely
# --------------------------------------------------
def get_matched_pairs(input_dir, gt_dir):
    input_files = os.listdir(input_dir)
    gt_files = os.listdir(gt_dir)

    input_map = {}
    for f in input_files:
        m = re.search(r'input_(\d+)', f)
        if m:
            input_map[m.group(1)] = f

    gt_map = {}
    for f in gt_files:
        m = re.search(r'gt_(\d+)', f)
        if m:
            gt_map[m.group(1)] = f

    common_ids = sorted(set(input_map.keys()) & set(gt_map.keys()))
    pairs = [(input_map[i], gt_map[i]) for i in common_ids]

    print(f"Matched pairs : {len(pairs)}")
    print(f"Missing GT    : {len(input_map) - len(common_ids)}")
    print(f"Missing Input : {len(gt_map) - len(common_ids)}")

    return pairs


# --------------------------------------------------
# Train / Val / Test split
# --------------------------------------------------
def split_pairs(pairs, seed=42):
    train_pairs, temp_pairs = train_test_split(
        pairs, test_size=0.25, random_state=seed
    )

    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=0.4, random_state=seed
    )

    print(f"Train : {len(train_pairs)}")
    print(f"Val   : {len(val_pairs)}")
    print(f"Test  : {len(test_pairs)}")

    return train_pairs, val_pairs, test_pairs


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class DocumentDataset(Dataset):
    def __init__(
        self,
        input_dir,
        gt_dir,
        pairs,
        patch_size=256,
        training=True
    ):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.pairs = pairs
        self.patch_size = patch_size
        self.training = training

    def __len__(self):
        return len(self.pairs)

    def random_crop(self, img, gt):
        h, w, _ = img.shape
        ps = self.patch_size

        if h < ps or w < ps:
            img = cv2.resize(img, (ps, ps))
            gt = cv2.resize(gt, (ps, ps))
            return img, gt

        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        return img[y:y+ps, x:x+ps], gt[y:y+ps, x:x+ps]

    def center_crop(self, img, gt):
        h, w, _ = img.shape
        ps = self.patch_size
        x = (w - ps) // 2
        y = (h - ps) // 2
        return img[y:y+ps, x:x+ps], gt[y:y+ps, x:x+ps]

    def __getitem__(self, idx):
        inp_name, gt_name = self.pairs[idx]

        inp = cv2.imread(os.path.join(self.input_dir, inp_name))
        gt  = cv2.imread(os.path.join(self.gt_dir, gt_name))

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        gt  = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        if self.training:
            inp, gt = self.random_crop(inp, gt)
        else:
            inp, gt = self.center_crop(inp, gt)

        inp = inp.astype(np.float32) / 255.0
        gt  = gt.astype(np.float32) / 255.0

        inp = torch.from_numpy(inp).permute(2, 0, 1).contiguous()
        gt  = torch.from_numpy(gt).permute(2, 0, 1).contiguous()

        return inp, gt
