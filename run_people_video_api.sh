#!/usr/bin/env bash

set -euo pipefail

.venv/bin/python people_video_api.py \
  --host "${HOST:-0.0.0.0}" \
  --port "${VIDEO_PORT:-8001}" \
  --max-cache "${VIDEO_MAX_CACHE:-5000}" \
  --refresh-seconds "${VIDEO_REFRESH_SECONDS:-5}" \
  --output-dir outputs/people_video_api
