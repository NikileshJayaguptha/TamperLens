from __future__ import annotations

import csv
import random
from pathlib import Path

import common  # noqa: F401
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.pipeline.classifier import INPUT_SIZE, make_region_input


def read_rows(csv_paths: list[str], split: str | None = None) -> list[dict]:
    rows = []
    for p in csv_paths:
        base = Path(p).resolve().parent
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if split and r["split"] != split:
                    continue
                r["image"] = str(base / r["image"]) if not Path(r["image"]).is_absolute() else r["image"]
                r["mask"] = str(base / r["mask"]) if not Path(r["mask"]).is_absolute() else r["mask"]
                r["label"] = int(r["label"])
                rows.append(r)
    return rows


class RegionDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool = False):
        self.rows = rows
        self.train = train

    def __len__(self):
        return len(self.rows)

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        if random.random() < 0.5:
            img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
        if random.random() < 0.3:
            q = random.randint(70, 95)
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return img, mask

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = cv2.imread(r["image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE)
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask > 127
        if self.train:
            img, mask = self._augment(img, mask)
        x = make_region_input(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask, INPUT_SIZE)
        return torch.from_numpy(x), torch.tensor(float(r["label"]))
