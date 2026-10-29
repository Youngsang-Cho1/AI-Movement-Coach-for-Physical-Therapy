# AI Movement Coach for Physical Therapy

Webcam app giving real-time feedback on PT exercises using MediaPipe pose estimation.
Pilot/feasibility project — **not a clinical tool**.

Exercises: seated knee extension, sit-to-stand, shoulder raise.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
scripts/download_model.sh
pytest
```
