from __future__ import annotations

import cv2
import numpy as np


def _robust_stats(values: np.ndarray) -> tuple[float, float]:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) * 1.4826
    return med, max(mad, 1e-6)


def local_noise_map(img_bgr: np.ndarray, block: int = 16) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    mean = cv2.blur(residual, (block, block))
    sq_mean = cv2.blur(residual * residual, (block, block))
    var = np.maximum(sq_mean - mean * mean, 0.0)
    return np.log(var + 1e-3)


def ela_map(img_bgr: np.ndarray, quality: int = 90, block: int = 8) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return np.zeros(img_bgr.shape[:2], np.float32)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = np.abs(img_bgr.astype(np.float32) - decoded.astype(np.float32)).mean(axis=2)
    return cv2.blur(diff, (block, block))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class NoiseAnalyzer:

    def __init__(self, img_bgr: np.ndarray, sample_step: int = 2):
        self.noise = local_noise_map(img_bgr)
        self.ela = ela_map(img_bgr)
        self.step = sample_step

    def _z(self, feature: np.ndarray, mask: np.ndarray) -> float:
        f = feature[:: self.step, :: self.step]
        m = mask[:: self.step, :: self.step]
        inside, outside = f[m], f[~m]
        if inside.size < 20 or outside.size < 20:
            return 0.0
        in_med = float(np.median(inside))
        out_med, out_spread = _robust_stats(outside)
        return abs(in_med - out_med) / out_spread

    def score(self, mask: np.ndarray) -> dict[str, float]:
        z_noise = self._z(self.noise, mask)
        z_ela = self._z(self.ela, mask)
        s_noise = _sigmoid(1.5 * (z_noise - 2.0))
        s_ela = _sigmoid(1.5 * (z_ela - 2.0))
        return {
            "noise_z": z_noise,
            "ela_z": z_ela,
            "noise_score": 0.6 * s_noise + 0.4 * s_ela,
        }
