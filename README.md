# AI Edit Detector

AI Edit Detector finds AI-edited areas in photos. It splits the image into regions with Segment Anything, scores each region with noise analysis and a PyTorch classifier, and shows the result as a heatmap you can review and label.

## Features

- Region-by-region detection using Segment Anything (SAM ViT-B)
- Noise and JPEG error level (ELA) analysis for every region
- ResNet-18 region classifier trained on public AI-inpainting datasets
- Heatmap and region outlines over the image
- Label regions as real or AI, then fine-tune the classifier on those labels
- Keyboard review: `j`/`k` to move, `r` for real, `a` for AI

## Quick Start

1. Download the SAM checkpoint into `models/`
   ```bash
   ./scripts/download_models.sh
   ```
2. Build and run everything
   ```bash
   docker compose up --build
   ```
3. Open:
   - Frontend: http://localhost:3000
   - Backend: http://localhost:8000

Without trained classifier weights, the app scores regions with noise analysis only.

## Tech Stack

- Frontend: Next.js, React, TypeScript
- Backend: FastAPI, SQLAlchemy
- Detection: Segment Anything, OpenCV, PyTorch (ResNet-18)
- Database: PostgreSQL

## API Endpoints

- `POST /api/analyses`: upload an image
- `GET /api/analyses`: list analyses
- `GET /api/analyses/{id}`: scores and regions for one analysis
- `GET /api/analyses/{id}/heatmap`: heatmap PNG
- `DELETE /api/analyses/{id}`: delete an analysis
- `PUT /api/regions/{id}/label`: label a region as `real` or `ai`
- `GET /api/labels/stats`: label counts
- `POST /api/model/reload`: reload classifier weights
- `GET /api/health`: health check

## Training

Use a machine with a GPU.

```bash
cd training
pip install -r requirements.txt
python prepare_regions.py --name tgif --pairs-csv /path/to/pairs.csv --out prepared/tgif
python train.py --csv prepared/tgif/regions.csv --out ../models/region_classifier.pt
python evaluate.py --weights ../models/region_classifier.pt --csv prepared/tgif/regions.csv
```

To fine-tune on labels from the dashboard:

```bash
DATA_DIR=../data python finetune_from_labels.py --weights ../models/region_classifier.pt --out ../models/region_classifier.pt
```

Then restart the backend or call `POST /api/model/reload`.

## Requirements

- Docker
- Python 3.11+ for training
- A GPU is optional. On CPU, each image takes roughly 10–30 s.
