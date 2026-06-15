#!/usr/bin/env bash

set -euo pipefail

.venv/bin/python laion_people_api.py \
  --mode serve \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --output-dir outputs/laion_people_api
