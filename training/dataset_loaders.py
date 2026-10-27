from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MASK_SUFFIXES = ["_mask", "-mask", "_gt", "-gt", "_GT", "_m", "mask"]

PRESETS: dict[str, tuple[str, str, str | None]] = {
    "cocoglide": ("fake", "mask", "real"),
    "generic": ("images", "masks", "real"),
}


@dataclass
class Sample:
    image: Path
    mask: Path | None
    dataset: str


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def _key(path: Path, strip: bool) -> str:
    stem = path.stem
    if strip:
        for suf in MASK_SUFFIXES:
            if stem.endswith(suf) and len(stem) > len(suf):
                return stem[: -len(suf)]
    return stem


def find_pairs(images_dir: Path, masks_dir: Path, dataset: str) -> list[Sample]:
    masks = {_key(p, strip=True): p for p in _files(masks_dir)}
    samples, missing = [], 0
    for img in _files(images_dir):
        mask = masks.get(_key(img, strip=False)) or masks.get(_key(img, strip=True))
        if mask is None:
            missing += 1
            continue
        samples.append(Sample(img, mask, dataset))
    if missing:
        print(f"[{dataset}] {missing} edited images had no matching mask and were skipped")
    return samples


def load_pairs_csv(path: str, dataset: str) -> list[Sample]:
    import csv

    base = Path(path).resolve().parent
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            img = Path(row["image"])
            mask = Path(row["mask"]) if row.get("mask") else None
            out.append(Sample(img if img.is_absolute() else base / img,
                              None if mask is None else (mask if mask.is_absolute() else base / mask),
                              dataset))
    return out


def load_dataset(name: str, root: str | None = None, preset: str | None = None,
                 images: str | None = None, masks: str | None = None,
                 real: str | None = None, pairs_csv: str | None = None) -> list[Sample]:
    if pairs_csv:
        samples = load_pairs_csv(pairs_csv, name)
        print(f"[{name}] {len(samples)} images from {pairs_csv}")
        return samples
    if preset:
        if not root:
            raise ValueError("--root is required with --preset")
        img_rel, mask_rel, real_rel = PRESETS[preset]
        images = images or str(Path(root) / img_rel)
        masks = masks or str(Path(root) / mask_rel)
        if real is None and real_rel and (Path(root) / real_rel).exists():
            real = str(Path(root) / real_rel)
    if not images or not masks:
        raise ValueError("Give --preset with --root, or both --images and --masks")

    samples = find_pairs(Path(images), Path(masks), name)
    if real:
        samples += [Sample(p, None, name) for p in _files(Path(real))]
    n_real = sum(s.mask is None for s in samples)
    print(f"[{name}] {len(samples) - n_real} edited images with masks, {n_real} real images")
    if not samples:
        raise SystemExit("No images found. Check the folder paths.")
    return samples


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Could not read mask {path}")
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127
