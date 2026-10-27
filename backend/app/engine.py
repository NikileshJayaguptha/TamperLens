from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch

from . import config
from .pipeline.classifier import RegionClassifier
from .pipeline.segment import SamSegmenter

log = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.segmenter: SamSegmenter | None = None
        self.classifier: RegionClassifier | None = None

    def get_segmenter(self) -> SamSegmenter:
        if self.segmenter is None:
            if not Path(config.SAM_CHECKPOINT).exists():
                raise RuntimeError(
                    f"SAM checkpoint not found at {config.SAM_CHECKPOINT}. "
                    "Run scripts/download_models.sh first."
                )
            log.info("Loading SAM (%s) on %s", config.SAM_MODEL_TYPE, DEVICE)
            self.segmenter = SamSegmenter(
                config.SAM_CHECKPOINT, config.SAM_MODEL_TYPE, DEVICE,
                config.SAM_POINTS_PER_SIDE, config.MAX_REGIONS, config.MIN_REGION_FRAC,
            )
        return self.segmenter

    def load_classifier(self) -> RegionClassifier | None:
        path = Path(config.CLASSIFIER_WEIGHTS)
        if not path.exists():
            log.warning("No classifier weights at %s; scoring with noise analysis only", path)
            self.classifier = None
        else:
            self.classifier = RegionClassifier(path, DEVICE)
            log.info("Loaded classifier %s", self.classifier.version)
        return self.classifier

    @property
    def model_version(self) -> str | None:
        return self.classifier.version if self.classifier else None


engine = Engine()
