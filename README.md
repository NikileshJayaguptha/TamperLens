# AI Edit Detector

Finds AI-edited areas in photos.

1. **Segment Anything (SAM)** splits the image into regions.
2. **OpenCV noise analysis** scores each region. It compares the region's noise residual and JPEG error level (ELA) against the rest of the photo.
3. **A PyTorch classifier** (ResNet-18 with RGB + region mask input) scores each region. It is trained on a **public AI-inpainting dataset with ground-truth masks**.
4. A **FastAPI** backend stores results in **PostgreSQL**. A **Next.js** dashboard shows a heatmap and lets users label regions as real or AI. Those labels are used to fine-tune the classifier.

Everything runs with **Docker Compose**.

```
backend/      FastAPI app + detection pipeline (SAM, noise analysis, classifier)
training/     dataset loading, region extraction, training, evaluation, fine-tuning from labels
frontend/     Next.js dashboard
scripts/      model download
```

## Datasets

The classifier needs photos where only part of the image was AI-edited, with a mask showing which pixels were edited.

| Dataset | What it is | Where to get it |
|---|---|---|
| **TGIF** | ~75k images inpainted with Stable Diffusion 2, SDXL and Adobe Firefly, with masks | [github.com/IDLabMedia/tgif-dataset](https://github.com/IDLabMedia/tgif-dataset) |
| **CocoGlide** | COCO photos with objects replaced by GLIDE inpainting, with masks | linked from the TruFor repo ([github.com/grip-unina/TruFor](https://github.com/grip-unina/TruFor)) |

**Recommended setup:** train on TGIF and test on CocoGlide. That shows whether the model works on a generator it never saw during training.

For TGIF, use the **spliced** subsets (`SD2-sp`, `PS-sp`), where only the masked pixels are synthetic. The "fully regenerated" subsets pass the whole image through the generator, so pixels outside the mask aren't truly real. That makes them wrong as negatives for a region classifier.

Check the folder layout of the copy you download. The loader supports three ways to point at the data:

* `--preset cocoglide --root <dir>` for `fake/`, `mask/` and `real/` folders
* `--images <dir> --masks <dir> [--real <dir>]` for folders where masks match images by file name. Suffixes such as `_mask` or `_gt` are ignored when matching.
* `--pairs-csv pairs.csv` with columns `image,mask` for any other layout. Leave `mask` empty for a real image. This is the easiest option when several edited variants share one mask, as in TGIF.

## Quick start (Docker)

```bash
./scripts/download_models.sh          # SAM ViT-B checkpoint -> models/
docker compose up --build
```

Open http://localhost:3000. Without trained weights the app still runs and scores regions with noise analysis only. The dashboard then shows `model noise-only`.

## Training the classifier

Use a machine with a GPU. Install PyTorch for your CUDA version from pytorch.org first.

```bash
cd training
pip install -r requirements.txt

# 1. Turn each dataset into labelled regions, using SAM the same way the API does
python prepare_regions.py --name tgif --pairs-csv /datasets/tgif/pairs_sp.csv --out prepared/tgif
python prepare_regions.py --name cocoglide --preset cocoglide --root /datasets/CocoGlide --out prepared/cocoglide

# 2. Train
python train.py --csv prepared/tgif/regions.csv --out ../models/region_classifier.pt --epochs 10

# 3. Evaluate: held-out TGIF test split, then cross-dataset on CocoGlide
python evaluate.py --weights ../models/region_classifier.pt --csv prepared/tgif/regions.csv
python evaluate.py --weights ../models/region_classifier.pt --csv prepared/cocoglide/regions.csv --split all
```

Then restart the backend, or call `POST /api/model/reload`.

**How regions are labelled:** each SAM region in an edited image is labelled **AI** if at least 50% of it lies inside the ground-truth mask. It is labelled **real** if at most 5% does. Anything in between is skipped. The ground-truth mask is also added as an AI example. Every region from an unedited photo is labelled real.

Splits are made per source photo, so regions from one photo never appear in both train and test. Use `--regions grid` for a fast first pass without SAM, and `--limit 50` to test the pipeline end to end.

## Fine-tuning from dashboard labels

Labels clicked in the dashboard are saved on the `regions` table. Each region's mask is saved under `data/results/<analysis>/masks/`.

```bash
cd training
DATABASE_URL=postgresql+psycopg://detector:detector@localhost:5432/detector DATA_DIR=../data \
python finetune_from_labels.py \
    --weights ../models/region_classifier.pt --out ../models/region_classifier.pt \
    --replay-csv prepared/tgif/regions.csv \
    --reload-url http://localhost:8000/api/model/reload
```

This mixes in some original training regions so the model doesn't forget the dataset. It backs up the previous weights, then hot-reloads the API.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/api/analyses` | upload an image (multipart `file`); analysis runs in the background |
| `GET` | `/api/analyses` | recent analyses |
| `GET` | `/api/analyses/{id}` | status, overall score, regions with polygons and scores |
| `GET` | `/api/analyses/{id}/image` | working image (resized to `MAX_SIDE`) |
| `GET` | `/api/analyses/{id}/heatmap` | transparent heatmap PNG |
| `DELETE` | `/api/analyses/{id}` | delete |
| `PUT` | `/api/regions/{id}/label` | `{"label": "real" \| "ai" \| null}` |
| `GET` | `/api/labels/stats` | label counts |
| `POST` | `/api/model/reload` | reload classifier weights |
| `GET` | `/api/health` | which models are loaded |

## Scoring

```
region score = 0.7 × classifier + 0.3 × noise score          (CLASSIFIER_WEIGHT)
noise score  = 0.6 × σ(noise-level z-score) + 0.4 × σ(ELA z-score)
image score  = highest region score (regions ≥ 0.5% of the image)
heatmap      = per pixel, the highest score of any region covering it
```

The z-scores compare the region's median against the median and MAD of everything outside it.

## Configuration

Set these as environment variables on the backend: `DATABASE_URL`, `DATA_DIR`, `SAM_CHECKPOINT`, `SAM_MODEL_TYPE` (`vit_b`/`vit_l`/`vit_h`), `SAM_POINTS_PER_SIDE` (default 16, which is faster on CPU), `MAX_REGIONS`, `MIN_REGION_FRAC`, `CLASSIFIER_WEIGHTS`, `CLASSIFIER_WEIGHT`, `MAX_SIDE`, `CORS_ORIGINS`.

On CPU, SAM ViT-B takes roughly 10–30 s per image with these settings. A GPU is much faster.
