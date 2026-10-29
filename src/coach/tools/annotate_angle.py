"""Independent reference angles: click 3 points (A, vertex B, C) on video frames.

Knee: hip, knee, ankle. Shoulder elevation: hip, shoulder, elbow.
Keys: s skip frame, r redo clicks, q quit. Output CSV columns: t,angle (2D image angle).
"""
import argparse
import csv
from pathlib import Path

import cv2

from ..geometry import angle_deg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("--times", help="comma-separated seconds, e.g. 3.0,5.5")
    p.add_argument("--every", type=float, default=1.0, help="step in seconds if --times omitted")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    times = ([float(x) for x in a.times.split(",")] if a.times
             else [i * a.every for i in range(int(dur / a.every) + 1)])
    pts, rows = [], []

    def on_click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
            pts.append((x, y))

    cv2.namedWindow("annotate")
    cv2.setMouseCallback("annotate", on_click)
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, img = cap.read()
        if not ok:
            break
        pts.clear()
        while True:
            vis = img.copy()
            for q in pts:
                cv2.circle(vis, q, 5, (0, 255, 255), -1)
            cv2.putText(vis, f"t={t:.2f}s clicks {len(pts)}/3", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("annotate", vis)
            k = cv2.waitKey(30) & 0xFF
            if len(pts) == 3:
                rows.append((t, angle_deg(*pts)))
                break
            if k == ord("s"):
                break
            if k == ord("r"):
                pts.clear()
            if k == ord("q"):
                times = []
                break
        if not times:
            break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "angle"])
        w.writerows(rows)
    print(f"saved {len(rows)} reference angles -> {a.out}")


if __name__ == "__main__":
    main()
