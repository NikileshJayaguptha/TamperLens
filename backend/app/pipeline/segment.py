from __future__ import annotations

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def postprocess_masks(
    masks: list[np.ndarray], max_regions: int, min_frac: float, dup_iou: float = 0.9
) -> list[np.ndarray]:
    if not masks:
        return []
    total = masks[0].size
    masks = sorted((m for m in masks if m.sum() >= min_frac * total), key=lambda m: -int(m.sum()))
    kept: list[np.ndarray] = []
    for m in masks:
        if all(_iou(m, k) < dup_iou for k in kept):
            kept.append(m)
        if len(kept) >= max_regions:
            break
    return kept


class SamSegmenter:
    def __init__(self, checkpoint: str, model_type: str = "vit_b", device: str = "cpu",
                 points_per_side: int = 16, max_regions: int = 64, min_frac: float = 0.002):
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self.generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
            min_mask_region_area=100,
        )
        self.max_regions = max_regions
        self.min_frac = min_frac

    def segment(self, img_rgb: np.ndarray) -> list[np.ndarray]:
        outputs = self.generator.generate(img_rgb)
        masks = [o["segmentation"].astype(bool) for o in outputs]
        return postprocess_masks(masks, self.max_regions, self.min_frac)


class GridSegmenter:

    def __init__(self, cells: int = 6):
        self.cells = cells

    def segment(self, img_rgb: np.ndarray) -> list[np.ndarray]:
        h, w = img_rgb.shape[:2]
        ys = np.linspace(0, h, self.cells + 1, dtype=int)
        xs = np.linspace(0, w, self.cells + 1, dtype=int)
        masks = []
        for i in range(self.cells):
            for j in range(self.cells):
                m = np.zeros((h, w), bool)
                m[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = True
                masks.append(m)
        return masks
