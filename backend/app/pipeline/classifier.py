from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    except Exception as exc:
        print(f"Could not load ImageNet weights ({exc}); starting from random init")
        model = resnet18(weights=None)
    old = model.conv1.weight.data
    conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv.weight.data[:, :3] = old
    conv.weight.data[:, 3:] = old.mean(dim=1, keepdim=True)
    model.conv1 = conv
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def make_region_input(img_rgb: np.ndarray, mask: np.ndarray, size: int = INPUT_SIZE,
                      context: float = 0.15) -> np.ndarray:
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        y0, y1, x0, x1 = 0, h, 0, w
    else:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad_y = int((y1 - y0) * context) + 4
    pad_x = int((x1 - x0) * context) + 4
    y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)

    crop = img_rgb[y0:y1, x0:x1]
    mcrop = mask[y0:y1, x0:x1].astype(np.uint8)

    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    sq = np.zeros((side, side, 3), np.uint8)
    msq = np.zeros((side, side), np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    sq[oy:oy + ch, ox:ox + cw] = crop
    msq[oy:oy + ch, ox:ox + cw] = mcrop

    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    sq = cv2.resize(sq, (size, size), interpolation=interp)
    msq = cv2.resize(msq, (size, size), interpolation=cv2.INTER_NEAREST)

    rgb = (sq.astype(np.float32) / 255.0 - MEAN) / STD
    m = (msq.astype(np.float32) - 0.5) / 0.5
    return np.concatenate([rgb.transpose(2, 0, 1), m[None]], axis=0).astype(np.float32)


def save_checkpoint(model: nn.Module, path: str | Path, version: str, metrics: dict | None = None):
    torch.save(
        {"state_dict": model.state_dict(), "input_size": INPUT_SIZE,
         "version": version, "metrics": metrics or {}},
        str(path),
    )


class RegionClassifier:
    def __init__(self, weights_path: str | Path, device: str = "cpu"):
        ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.device = device
        self.input_size = int(ckpt.get("input_size", INPUT_SIZE))
        self.version = str(ckpt.get("version", Path(weights_path).stem))
        self.metrics = ckpt.get("metrics", {})

    @torch.no_grad()
    def predict(self, img_rgb: np.ndarray, masks: list[np.ndarray], batch_size: int = 16) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(masks), batch_size):
            batch = np.stack([make_region_input(img_rgb, m, self.input_size) for m in masks[i:i + batch_size]])
            logits = self.model(torch.from_numpy(batch).to(self.device)).squeeze(1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores
