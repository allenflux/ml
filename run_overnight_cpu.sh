#!/usr/bin/env bash

set -euo pipefail

.venv/bin/python train.py \
  --device cpu \
  --dataset imagefolder \
  --data-root my_images/train \
  --profile overnight-cpu
