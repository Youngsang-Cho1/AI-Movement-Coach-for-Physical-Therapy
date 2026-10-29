"""Record a labelled test clip from the webcam after confirming volunteer consent.

Saves data/videos/<id>.mp4 and data/consent/<volunteer>.json (both git-ignored).
Keys: space start/stop recording, q quit.
"""
import argparse
import json
import time
from datetime import date
from pathlib import Path

import cv2

DATA = Path(__file__).resolve().parents[3] / "data"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volunteer", required=True, help="anonymous ID, e.g. v1")
    p.add_argument("--exercise", required=True)
    p.add_argument("--set", type=int, required=True, help="set number")
    p.add_argument("--lighting", default="normal")
    p.add_argument("--camera", default="side")
    p.add_argument("--source", type=int, default=0)
    a = p.parse_args(argv)

    consent = DATA / "consent" / f"{a.volunteer}.json"
    if not consent.exists():
        if input(f"Has {a.volunteer} signed docs/consent_template.md? [y/N] ").lower() != "y":
            raise SystemExit("No consent, not recording.")
        consent.parent.mkdir(parents=True, exist_ok=True)
        consent.write_text(json.dumps({"volunteer": a.volunteer, "date": str(date.today())}))

    vid_id = f"{a.volunteer}_{a.exercise}_{a.lighting}_{a.camera}_set{a.set}"
    out = DATA / "videos" / f"{vid_id}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer, t0 = None, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer:
            writer.write(frame)
        shown = frame.copy()
        label = f"REC {time.time() - t0:.0f}s" if writer else "space = start"
        cv2.putText(shown, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if writer else (255, 255, 255), 2)
        cv2.imshow("record", shown)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            if writer:
                break
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            t0 = time.time()
        elif k == ord("q"):
            break
    cap.release()
    if writer:
        writer.release()
        print(f"saved {out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
