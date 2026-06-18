#!/usr/bin/env bash

set -euo pipefail

args=(
  --source-path "${VIDEO_SOURCE_PATH:-data/videos/urls.csv}" \
  --split "${VIDEO_SPLIT:-train}" \
  --shuffle-buffer "${VIDEO_SHUFFLE_BUFFER:-0}" \
  --target-cache "${VIDEO_TARGET_CACHE:-500}" \
  --max-video-bytes "${MAX_VIDEO_BYTES:-100000000}" \
  --min-duration "${MIN_VIDEO_DURATION:-4}" \
  --max-duration "${MAX_VIDEO_DURATION:-7}" \
  --sample-frames "${VIDEO_SAMPLE_FRAMES:-5}" \
  --max-detection-side "${MAX_DETECTION_SIDE:-384}"
)

if [[ -n "${VIDEO_DATASET:-}" ]]; then
  args+=(--dataset "$VIDEO_DATASET")
fi

.venv/bin/python harvest_people_videos.py "${args[@]}"
