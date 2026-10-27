import os
from pathlib import Path

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://detector:detector@localhost:5432/detector"
)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"

SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "./models/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")
SAM_POINTS_PER_SIDE = int(os.getenv("SAM_POINTS_PER_SIDE", "16"))
MAX_REGIONS = int(os.getenv("MAX_REGIONS", "64"))
MIN_REGION_FRAC = float(os.getenv("MIN_REGION_FRAC", "0.002"))

CLASSIFIER_WEIGHTS = os.getenv("CLASSIFIER_WEIGHTS", "./models/region_classifier.pt")
CLASSIFIER_WEIGHT = float(os.getenv("CLASSIFIER_WEIGHT", "0.7"))

MAX_SIDE = int(os.getenv("MAX_SIDE", "1024"))

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")]
