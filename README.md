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

## Run
```bash
coach knee_extension --source 0                       # webcam, side view
coach sit_to_stand --source clip.mp4 --no-display --out runs/clip
coach shoulder_raise --source 0                       # webcam, front view
```
Therapist targets live in `configs/<exercise>.json` (e.g. `target_angle`, `hold_s`).
When tracking is unreliable (low landmark confidence, no person, jumps) the app withholds
movement advice and asks you to adjust camera/lighting.

## Layout
- `src/coach/exercises/` — per-exercise state machines (reps, hold, tempo, lean)
- `src/coach/tracker.py` — MediaPipe pose wrapper; `run.py` — webcam/video runner + latency logging
- `src/coach/evaluation/` — metrics + `evaluate.py`; `tools/annotate_angle.py` — manual reference angles
- `docs/evaluation_protocol.md`, `docs/consent_template.md`

## Status
Logic is unit-tested on synthetic skeletons. **Not yet validated on real volunteer video** —
results and limitations go in `docs/evaluation_report.md` after data collection.
