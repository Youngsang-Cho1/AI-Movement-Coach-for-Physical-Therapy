"""Synthetic skeleton streams for testing exercise logic without a camera."""
import numpy as np

from coach.frame import PoseFrame

FPS = 30


def _blank():
    return np.zeros((33, 3)), np.ones(33)


def knee_frame(theta_deg, vis=1.0):
    w, v = _blank()
    th = np.radians(theta_deg)
    hip, knee = np.array([0.0, 0.0, 0]), np.array([0.4, 0.0, 0])
    ankle = knee + 0.4 * np.array([-np.cos(th), np.sin(th), 0])
    for i in (23, 24):
        w[i] = hip
    for i in (25, 26):
        w[i] = knee
    for i in (27, 28):
        w[i] = ankle
    return PoseFrame(w[:, :2].copy(), w, v * vis)


def shoulder_frame(elev_deg, lean_deg=0.0):
    w, v = _blank()
    lean = 0.5 * np.tan(np.radians(lean_deg))
    w[11], w[12] = [0.15 + lean, 0, 0], [-0.15 + lean, 0, 0]
    w[23], w[24] = [0.1, 0.5, 0], [-0.1, 0.5, 0]
    e = np.radians(elev_deg)
    w[13] = w[11] + 0.3 * np.array([np.sin(e), np.cos(e), 0])
    w[14] = w[12] + 0.3 * np.array([-np.sin(e), np.cos(e), 0])
    return PoseFrame(w[:, :2].copy(), w, v)


def ramp(points):
    """Piecewise-linear signal from [(t_s, value), ...] sampled at FPS."""
    ts, vs = zip(*points)
    t = np.arange(0, ts[-1], 1 / FPS)
    return t, np.interp(t, ts, vs)


def run(ex, t, values, make):
    events = []
    for ti, vi in zip(t, values):
        events += ex.update(ti, make(vi)).events
    return events
