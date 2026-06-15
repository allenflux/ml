#!/usr/bin/env bash

set -euo pipefail

.venv/bin/python harvest_people_images.py \
  --target-cache 1000 \
  --workers 1 \
  --shuffle-buffer 0 \
  --request-timeout 4 \
  --min-width 256 \
  --min-height 256 \
  --min-person-height-ratio 0.25 \
  --max-detection-side 384 \
  --detection-mode face-only \
  --max-image-bytes 3000000
