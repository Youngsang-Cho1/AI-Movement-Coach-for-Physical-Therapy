import cv2

from .frame import SKELETON

GREEN, RED, WHITE, GREY = (80, 200, 80), (60, 60, 230), (255, 255, 255), (60, 60, 60)


def _text(img, s, org, color=WHITE, scale=0.7):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw(img, pose, upd, ex):
    h, w = img.shape[:2]
    if pose is not None:
        col = GREEN if upd.reliable else RED
        for a, b in SKELETON:
            if pose.vis[a] > 0.3 and pose.vis[b] > 0.3:
                cv2.line(img, tuple((pose.img[a] * (w, h)).astype(int)),
                         tuple((pose.img[b] * (w, h)).astype(int)), col, 3)
    cv2.rectangle(img, (0, 0), (w, 110), GREY, -1)
    _text(img, f"{ex.name}  reps {upd.reps}  good {upd.good_reps}  [{upd.phase}]", (10, 28))
    m = upd.metrics
    key = ex.primary
    if key in m:
        extra = f"  hold {m['hold_s']:.1f}s" if "hold_s" in m else ""
        extra += f"  lean {m['lean']:+.0f}" if "lean" in m else ""
        _text(img, f"{key} {m[key]:.0f} deg{extra}", (10, 58))
    if upd.message:
        _text(img, upd.message, (10, 92), RED if not upd.reliable else GREEN, 0.75)
    return img
