#!/usr/bin/env bash

set -euo pipefail

.venv/bin/python laion_people_api.py \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 8 \
  --max-cache 1000 \
  --shuffle-buffer 0 \
  --request-timeout 4 \
  --min-width 256 \
  --min-height 256 \
  --min-person-height-ratio 0.25
