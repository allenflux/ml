#!/bin/zsh

set -euo pipefail

.venv/bin/python laion_people_api.py \
  --mode serve \
  --host 127.0.0.1 \
  --port 8000 \
  --output-dir outputs/laion_people_api
