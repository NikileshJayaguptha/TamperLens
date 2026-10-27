#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
if [ -f models/sam_vit_b_01ec64.pth ]; then
  echo "SAM checkpoint already present."
else
  curl -L --fail -o models/sam_vit_b_01ec64.pth "$URL"
fi
echo "Done. Train the classifier with training/train.py to create models/region_classifier.pt"
