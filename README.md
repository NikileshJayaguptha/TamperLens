# AI Edit Detector

Upload a photo and it shows which parts look AI-edited. The image is split into regions with Segment Anything, and each region gets a score from noise analysis and a small PyTorch classifier. Results show up as a heatmap in a web dashboard, where you can mark regions as real or AI to help retrain the model.

## Run it

You need Docker.

```bash
./scripts/download_models.sh
docker compose up --build
```

Then open http://localhost:3000.

The API runs at http://localhost:8000.

## Train the classifier (optional)

Without trained weights the app only uses noise analysis. To train the classifier on a dataset of inpainted images with masks (for example TGIF):

```bash
cd training
pip install -r requirements.txt
python prepare_regions.py --name tgif --pairs-csv /path/to/pairs.csv --out prepared/tgif
python train.py --csv prepared/tgif/regions.csv --out ../models/region_classifier.pt
```

Then restart the backend.
