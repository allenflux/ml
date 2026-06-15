#!/bin/zsh

set -euo pipefail

python train.py \
  --device cpu \
  --dataset imagefolder \
  --data-root my_images/train \
  --profile overnight-cpu
