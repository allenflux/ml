#!/usr/bin/env bash

set -euo pipefail

.venv/bin/python people_media_api.py \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --image-max-cache "${IMAGE_MAX_CACHE:-5000}" \
  --video-max-cache "${VIDEO_MAX_CACHE:-5000}" \
  --refresh-seconds "${MEDIA_REFRESH_SECONDS:-5}" \
  --image-output-dir outputs/laion_people_api \
  --video-output-dir outputs/people_video_api
