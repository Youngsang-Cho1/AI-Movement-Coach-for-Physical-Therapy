# Evaluation protocol

1. **Record** 2-3 consenting volunteers, several sets per exercise (side view for knee/sit-to-stand,
   front view for shoulder raise). Save under `data/videos/` (git-ignored).
2. **Split**: sets for development (tuning thresholds) vs `test` (never used for tuning).
   Reserve one volunteer for test only if possible.
3. **Labels** per video (`data/labels/<id>.json`, see `*.example.json`):
   - `manual_reps`: attempts counted by hand (an attempt = a full cycle, even if poor).
   - `pt_feedback`: per attempt, the corrective codes a physical therapist would give
     (`extend_more`, `hold_longer`, `ascent_too_fast`, `descent_too_fast`, `stand_fully`,
     `raise_higher`, `trunk_lean`, ...). Attempts with no problem are omitted.
   - `ref_angles`: independent angles via `python -m coach.tools.annotate_angle` (or goniometer).
   - `pt_times` (optional): time the feedback should ideally fire, to measure feedback delay.
4. **Conditions**: vary `lighting` (normal/dim/backlit) and `camera` (side/oblique/high...).
5. **Run**: `python -m coach.evaluation.evaluate data/manifest.json` -> `results/tables.md`.
6. Report errors, wrong feedback (`fp_pairs`/`fn_pairs` in results.json) and tracking failures.

Targets (project goals, not clinical standards): MAE <= 5 deg, exact reps in >= 95% of sets,
feedback precision >= 90%, recall >= 85%. Limits: 2D reference angle vs 3D model angle, small n.
