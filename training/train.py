from __future__ import annotations

import argparse
import time

import common  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.pipeline.classifier import build_model, save_checkpoint
from region_dataset import RegionDataset, read_rows


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device)).squeeze(1))
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    metrics["auc"] = float(roc_auc_score(labels, probs)) if len(set(labels.tolist())) > 1 else float("nan")
    return metrics


def fit(model, train_rows, val_rows, out, version, epochs=10, batch_size=32, lr=3e-4,
        workers=4, device=None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_pos = sum(r["label"] for r in train_rows)
    n_neg = len(train_rows) - n_pos
    print(f"train: {len(train_rows)} regions ({n_pos} AI, {n_neg} real) | val: {len(val_rows)} | device: {device}")
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(RegionDataset(train_rows, train=True), batch_size=batch_size,
                              shuffle=True, num_workers=workers, drop_last=len(train_rows) > batch_size)
    val_loader = DataLoader(RegionDataset(val_rows), batch_size=batch_size, num_workers=workers)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best, best_metrics = -1.0, {}

    for epoch in range(1, epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()

        m = evaluate(model, val_loader, device) if val_rows else {"auc": float("nan"), "f1": 0.0}
        key = m["auc"] if not np.isnan(m["auc"]) else m["f1"]
        print(f"epoch {epoch}: loss {total / len(train_rows):.4f}  val AUC {m['auc']:.3f}  "
              f"F1 {m['f1']:.3f}  ({time.time() - t0:.0f}s)")
        if key > best:
            best, best_metrics = key, {**m, "epoch": epoch}
            save_checkpoint(model, out, version, best_metrics)
            print(f"  saved best -> {out}")
    return best_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True)
    ap.add_argument("--out", default="../models/region_classifier.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--version", default=None, help="name stored with the weights")
    args = ap.parse_args()

    train_rows = read_rows(args.csv, "train")
    val_rows = read_rows(args.csv, "val")
    version = args.version or f"resnet18-{time.strftime('%Y%m%d-%H%M')}"
    metrics = fit(build_model(pretrained=True), train_rows, val_rows, args.out, version,
                  args.epochs, args.batch_size, args.lr, args.workers)
    print("best:", metrics)


if __name__ == "__main__":
    main()
