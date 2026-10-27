from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import time
import urllib.request
from pathlib import Path

import common  # noqa: F401
from sqlalchemy import select

from app import config
from app.db import SessionLocal
from app.models import Region
from app.pipeline.classifier import RegionClassifier, build_model
from region_dataset import read_rows
from train import fit


def labelled_rows() -> list[dict]:
    rows = []
    with SessionLocal() as db:
        for r in db.scalars(select(Region).where(Region.label.is_not(None))):
            folder = config.RESULT_DIR / str(r.analysis_id)
            image, mask = folder / "image.png", folder / "masks" / f"{r.index}.png"
            if not (image.exists() and mask.exists()):
                continue
            h = int(hashlib.md5(str(r.analysis_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            rows.append({
                "image": str(image), "mask": str(mask), "label": 1 if r.label == "ai" else 0,
                "source": str(r.analysis_id), "split": "val" if h < 0.15 else "train",
                "dataset": "user-labels",
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=config.CLASSIFIER_WEIGHTS, help="weights to start from")
    ap.add_argument("--out", default=config.CLASSIFIER_WEIGHTS)
    ap.add_argument("--replay-csv", action="append", default=[],
                    help="original prepared dataset(s) to mix in")
    ap.add_argument("--replay-ratio", type=float, default=2.0,
                    help="dataset regions per user-labelled region")
    ap.add_argument("--min-labels", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reload-url", default=None)
    args = ap.parse_args()

    user = labelled_rows()
    print(f"{len(user)} labelled regions "
          f"({sum(r['label'] for r in user)} AI, {sum(1 - r['label'] for r in user)} real)")
    if len(user) < args.min_labels:
        raise SystemExit(f"Need at least {args.min_labels} labels to fine-tune.")

    train_rows = [r for r in user if r["split"] == "train"]
    val_rows = [r for r in user if r["split"] == "val"]
    if args.replay_csv:
        replay = read_rows(args.replay_csv, "train")
        random.shuffle(replay)
        train_rows += replay[: int(len(train_rows) * args.replay_ratio)]
        val_rows += read_rows(args.replay_csv, "val")[: max(len(val_rows) * 2, 50)]

    if Path(args.weights).exists():
        model = RegionClassifier(args.weights, "cpu").model
        print(f"Starting from {args.weights}")
    else:
        model = build_model(pretrained=True)
        print("No existing weights; starting from ImageNet")

    out = Path(args.out)
    if out.exists():
        backup = out.with_suffix(f".{time.strftime('%Y%m%d-%H%M%S')}.bak.pt")
        shutil.copy(out, backup)
        print(f"Backed up current weights to {backup}")

    version = f"finetuned-{time.strftime('%Y%m%d-%H%M')}"
    tmp = out.with_suffix(".tmp.pt")
    metrics = fit(model, train_rows, val_rows, tmp, version, args.epochs,
                  args.batch_size, args.lr, args.workers)
    tmp.replace(out)
    print("best:", metrics)

    if args.reload_url:
        req = urllib.request.Request(args.reload_url, method="POST")
        with urllib.request.urlopen(req) as resp:
            print("API reloaded:", resp.read().decode())


if __name__ == "__main__":
    main()
