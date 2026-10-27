from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

import common  # noqa: F401
import cv2
import numpy as np
import torch
from tqdm import tqdm

from app import config
from app.pipeline.analyze import resize_max_side
from app.pipeline.segment import GridSegmenter, SamSegmenter, postprocess_masks
from dataset_loaders import PRESETS, load_dataset, read_mask

FIELDS = ["image", "mask", "label", "source", "split", "dataset"]


def split_for(source: str, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.md5(source.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def gt_components(gt: np.ndarray, min_frac: float) -> list[np.ndarray]:
    n, labels = cv2.connectedComponents(gt.astype(np.uint8))
    comps = [labels == i for i in range(1, n)]
    return [c for c in comps if c.sum() >= min_frac * gt.size] or ([gt] if gt.any() else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="dataset name stored in the CSV")
    ap.add_argument("--root")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--images")
    ap.add_argument("--masks")
    ap.add_argument("--real")
    ap.add_argument("--pairs-csv", help="CSV of image,mask pairs (overrides the folder options)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", choices=["sam", "grid"], default="sam")
    ap.add_argument("--sam-checkpoint", default=config.SAM_CHECKPOINT)
    ap.add_argument("--pos-thresh", type=float, default=0.5)
    ap.add_argument("--neg-thresh", type=float, default=0.05)
    ap.add_argument("--max-neg", type=int, default=6, help="max negative regions kept per image")
    ap.add_argument("--no-gt-regions", action="store_true",
                    help="don't add the GT mask as a positive (only SAM regions)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="only process N images (for a quick test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    samples = load_dataset(args.name, args.root, args.preset, args.images, args.masks, args.real,
                           args.pairs_csv)
    rng.shuffle(samples)
    if args.limit:
        samples = samples[: args.limit]

    if args.regions == "sam":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        segmenter = SamSegmenter(args.sam_checkpoint, config.SAM_MODEL_TYPE, device,
                                 config.SAM_POINTS_PER_SIDE, config.MAX_REGIONS, config.MIN_REGION_FRAC)
    else:
        segmenter = GridSegmenter()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    csv_path = out / "regions.csv"
    counts = {0: 0, 1: 0}

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, s in enumerate(tqdm(samples, desc="images")):
            img = cv2.imread(str(s.image), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = resize_max_side(img, config.MAX_SIDE)
            gt = read_mask(s.mask, img.shape[:2]) if s.mask else None

            regions = segmenter.segment(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pos, neg = [], []
            for m in regions:
                if gt is None:
                    neg.append(m)
                    continue
                overlap = np.logical_and(m, gt).sum() / max(m.sum(), 1)
                if overlap >= args.pos_thresh:
                    pos.append(m)
                elif overlap <= args.neg_thresh:
                    neg.append(m)
            if gt is not None and not args.no_gt_regions:
                pos = postprocess_masks(pos + gt_components(gt, config.MIN_REGION_FRAC),
                                        max_regions=10**6, min_frac=0.0)
            rng.shuffle(neg)
            neg = neg[: args.max_neg]
            if not pos and not neg:
                continue

            img_name = f"{i:06d}.png"
            cv2.imwrite(str(out / "images" / img_name), img)
            split = split_for(str(s.mask or s.image), args.val_frac, args.test_frac)
            for k, (m, label) in enumerate([(m, 1) for m in pos] + [(m, 0) for m in neg]):
                mask_name = f"{i:06d}_{k}.png"
                cv2.imwrite(str(out / "masks" / mask_name), m.astype(np.uint8) * 255)
                writer.writerow({
                    "image": f"images/{img_name}", "mask": f"masks/{mask_name}", "label": label,
                    "source": str(s.image), "split": split, "dataset": s.dataset,
                })
                counts[label] += 1

    print(f"Wrote {csv_path}: {counts[1]} AI-edited regions, {counts[0]} real regions")


if __name__ == "__main__":
    main()
