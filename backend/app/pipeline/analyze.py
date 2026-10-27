from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .noise import NoiseAnalyzer


@dataclass
class RegionResult:
    mask: np.ndarray
    bbox: list[int]
    polygons: list[list[list[int]]]
    area_frac: float
    noise_score: float
    classifier_score: float | None
    score: float
    extra: dict = field(default_factory=dict)


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def mask_to_polygons(mask: np.ndarray, max_polys: int = 4) -> list[list[list[int]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_polys]
    polys = []
    for c in contours:
        eps = 0.002 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append(approx.astype(int).tolist())
    return polys


def analyze_image(img_bgr: np.ndarray, segmenter, classifier=None,
                  classifier_weight: float = 0.7) -> list[RegionResult]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = segmenter.segment(img_rgb)
    if not masks:
        return []

    noise = NoiseAnalyzer(img_bgr)
    noise_scores = [noise.score(m) for m in masks]
    cls_scores = classifier.predict(img_rgb, masks) if classifier is not None else [None] * len(masks)

    total = float(masks[0].size)
    results = []
    for m, ns, cs in zip(masks, noise_scores, cls_scores):
        x, y, w, h = cv2.boundingRect(m.astype(np.uint8))
        if cs is None:
            final = ns["noise_score"]
        else:
            final = classifier_weight * cs + (1 - classifier_weight) * ns["noise_score"]
        results.append(RegionResult(
            mask=m,
            bbox=[int(x), int(y), int(w), int(h)],
            polygons=mask_to_polygons(m),
            area_frac=float(m.sum()) / total,
            noise_score=float(ns["noise_score"]),
            classifier_score=None if cs is None else float(cs),
            score=float(final),
            extra={"noise_z": ns["noise_z"], "ela_z": ns["ela_z"]},
        ))
    results.sort(key=lambda r: -r.score)
    return results


def heat_map(shape: tuple[int, int], regions: list[RegionResult]) -> np.ndarray:
    heat = np.zeros(shape, np.float32)
    for r in regions:
        heat[r.mask] = np.maximum(heat[r.mask], r.score)
    return heat


def render_heatmap_png(heat: np.ndarray) -> np.ndarray:
    colored = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (np.clip((heat - 0.2) / 0.8, 0, 1) * 220).astype(np.uint8)
    return np.dstack([colored, alpha])


def overall_score(regions: list[RegionResult], min_area: float = 0.005) -> float:
    eligible = [r.score for r in regions if r.area_frac >= min_area]
    return float(max(eligible)) if eligible else 0.0
