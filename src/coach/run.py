"""Run an exercise coach on a webcam (--source 0) or a video file."""
import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .exercises import REGISTRY, make_exercise
from .overlay import draw
from .tracker import PoseTracker

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_overrides(exercise, path=None):
    path = Path(path) if path else CONFIG_DIR / f"{exercise}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def process(source, exercise, overrides=None, display=False, save_video=None, tracker=None,
            max_seconds=None):
    """Returns (rows, events, summary). Latency = frame grabbed -> feedback ready (+ overlay if shown)."""
    ex = make_exercise(exercise, overrides)
    tracker = tracker or PoseTracker()
    live = isinstance(source, int)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    rows, events, idx, t0 = [], [], 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        grabbed = time.perf_counter()
        t = grabbed - t0 if live else idx / fps
        if max_seconds and t > max_seconds:
            break
        pose = tracker.process(frame, int(t * 1000))
        upd = ex.update(t, pose)
        if display or save_video:
            draw(frame, pose, upd, ex)
        latency_ms = (time.perf_counter() - grabbed) * 1000
        for e in upd.events:
            events.append({**asdict(e), "latency_ms": latency_ms})
        rows.append({"frame": idx, "t": round(t, 4), "latency_ms": round(latency_ms, 2),
                     "pose": pose is not None, "reliable": upd.reliable, "reason": upd.reason,
                     "phase": upd.phase, "reps": upd.reps, "good_reps": upd.good_reps,
                     **{k: round(float(v), 3) for k, v in upd.metrics.items()}})
        if save_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(save_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if display:
            cv2.imshow("AI Movement Coach (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        idx += 1
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    lat = np.array([r["latency_ms"] for r in rows]) if rows else np.zeros(1)
    summary = {
        "exercise": exercise, "frames": len(rows), "fps": fps,
        "reps": ex.reps, "good_reps": ex.good_reps,
        "no_pose_frames": sum(not r["pose"] for r in rows),
        "unreliable_frames": sum(not r["reliable"] for r in rows),
        "latency_ms_mean": float(lat.mean()), "latency_ms_p95": float(np.percentile(lat, 95)),
        "config": asdict(ex.cfg),
    }
    return rows, events, summary


def write_outputs(out: Path, rows, events, summary):
    out.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    (out / "events.json").write_text(json.dumps(events, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exercise", choices=sorted(REGISTRY))
    p.add_argument("--source", default="0", help="webcam index or video path")
    p.add_argument("--config", help="JSON of therapist targets (default: configs/<exercise>.json)")
    p.add_argument("--out", type=Path, help="write frames.csv/events.json/summary.json here")
    p.add_argument("--save-video", type=Path, help="write annotated video")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-seconds", type=float)
    a = p.parse_args(argv)
    src = int(a.source) if a.source.isdigit() else a.source
    rows, events, summary = process(src, a.exercise, load_overrides(a.exercise, a.config),
                                    display=not a.no_display, save_video=a.save_video,
                                    max_seconds=a.max_seconds)
    if a.out:
        write_outputs(a.out, rows, events, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))
    for e in events:
        print(f"  t={e['t']:.2f}s attempt {e['attempt']}: {e['code']}")


if __name__ == "__main__":
    main()
