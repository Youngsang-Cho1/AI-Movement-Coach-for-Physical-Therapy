import math

import numpy as np


def angle_deg(a, b, c) -> float:
    """Interior angle at vertex b formed by points a-b-c, in degrees."""
    v1 = np.asarray(a, float) - np.asarray(b, float)
    v2 = np.asarray(c, float) - np.asarray(b, float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def vector_angle_deg(u, v) -> float:
    return angle_deg(np.asarray(u, float), np.zeros(len(u)), np.asarray(v, float))


def lean_deg(top_xy, bottom_xy) -> float:
    """Signed tilt of the bottom->top segment from vertical (image y points down)."""
    dx = top_xy[0] - bottom_xy[0]
    dy = bottom_xy[1] - top_xy[1]
    return math.degrees(math.atan2(dx, dy))
