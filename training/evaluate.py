import argparse
import json

import common  # noqa: F401
import torch
from torch.utils.data import DataLoader

from app.pipeline.classifier import RegionClassifier
from region_dataset import RegionDataset, read_rows
from train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--split", default="test", help="train | val | test | all")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = RegionClassifier(args.weights, device)
    rows = read_rows(args.csv, None if args.split == "all" else args.split)

    results = {}
    for name in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == name]
        loader = DataLoader(RegionDataset(subset), batch_size=args.batch_size, num_workers=4)
        results[name] = evaluate(clf.model, loader, device)
    print(json.dumps({"model": clf.version, "split": args.split, "results": results}, indent=2))


if __name__ == "__main__":
    main()
