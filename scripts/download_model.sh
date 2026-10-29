#!/usr/bin/env bash
# Fetch the MediaPipe pose landmarker model (not committed, ~9 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
curl -fL -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
