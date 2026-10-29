from dataclasses import dataclass, field, fields, replace

import numpy as np

from ..filters import OneEuroFilter
from ..frame import JOINT_NAMES, JOINTS, PoseFrame


@dataclass
class CommonConfig:
    min_vis: float = 0.5          # landmark visibility needed to trust a joint
    min_good_frames: int = 3      # consecutive reliable frames before acting
    abort_gap_s: float = 1.0      # unreliable this long -> abandon the current rep
    max_jump_deg: float = 45.0    # per-frame jump treated as a tracking glitch
    side: str = "auto"            # "L", "R" or "auto"
    msg_hold_s: float = 2.0       # how long feedback text stays on screen


@dataclass
class Feedback:
    code: str
    text: str
    t: float
    attempt: int


@dataclass
class Update:
    t: float
    reliable: bool
    reason: str
    metrics: dict
    phase: str
    reps: int
    good_reps: int
    events: list = field(default_factory=list)
    message: str = ""


def load_config(cls, overrides: dict | None = None):
    cfg = cls()
    valid = {f.name for f in fields(cls)}
    unknown = set(overrides or {}) - valid
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return replace(cfg, **(overrides or {}))


class Exercise:
    """Reliability gating + feedback bookkeeping; subclasses supply measure() and step()."""

    name = ""
    primary = ""        # metric key compared against reference angles
    rest_phase = "rest"

    def __init__(self, cfg):
        self.cfg = cfg
        self._side = None if cfg.side == "auto" else cfg.side
        self.reps = self.good_reps = self.attempt = 0
        self._last_ok_t = None
        self._good = 0
        self._last_primary = None
        self._msg, self._msg_until = "", 0.0
        self._reset_motion()

    def _reset_motion(self):
        self.phase = self.rest_phase
        self._filters = {}

    # --- subclass hooks -------------------------------------------------
    def required(self, f: PoseFrame) -> list[int]:
        raise NotImplementedError

    def measure(self, f: PoseFrame, t: float) -> dict:
        raise NotImplementedError

    def step(self, t: float, m: dict) -> list[Feedback]:
        raise NotImplementedError

    # --- helpers ---------------------------------------------------------
    def smooth(self, key, value, t, **kw):
        flt = self._filters.setdefault(key, OneEuroFilter(**kw))
        return flt(value, t)

    def pick_side(self, f: PoseFrame, parts=("hip", "knee", "ankle")) -> str:
        if self._side is None:
            score = {s: np.mean([f.vis[JOINTS[s][p]] for p in parts]) for s in "LR"}
            self._side = max(score, key=score.get)
        return self._side

    def fb(self, code, text, t) -> Feedback:
        self._msg, self._msg_until = text, t + self.cfg.msg_hold_s
        return Feedback(code, text, t, self.attempt)

    def _check(self, f):
        if f is None:
            return False, "no person detected"
        weak = [JOINT_NAMES[i] for i in self.required(f) if f.vis[i] < self.cfg.min_vis]
        if weak:
            return False, "low confidence: " + ", ".join(weak)
        return True, ""

    # --- main entry ------------------------------------------------------
    def update(self, t: float, f: PoseFrame | None) -> Update:
        if self._last_ok_t is None:
            self._last_ok_t = t
        ok, reason = self._check(f)
        metrics = {}
        if ok:
            metrics = self.measure(f, t)
            prim = metrics.get(self.primary)
            if prim is None or np.isnan(prim):
                ok, reason = False, "degenerate pose"
            elif (self._last_primary is not None
                  and abs(prim - self._last_primary) > self.cfg.max_jump_deg):
                ok, reason = False, "tracking jump"
            self._last_primary = prim
        if ok:
            self._good += 1
            self._last_ok_t = t
        else:
            self._good = 0
            self._last_primary = None if reason != "tracking jump" else self._last_primary
            if t - self._last_ok_t > self.cfg.abort_gap_s and self.phase != self.rest_phase:
                self._reset_motion()
        if not ok or self._good < self.cfg.min_good_frames:
            # tracking unreliable: withhold movement advice
            msg = f"Tracking unreliable ({reason or 'stabilising'}) - adjust camera/lighting"
            return Update(t, False, reason or "stabilising", {}, self.phase,
                          self.reps, self.good_reps, [], msg)
        events = self.step(t, metrics)
        msg = self._msg if t < self._msg_until else ""
        return Update(t, True, "", metrics, self.phase, self.reps, self.good_reps, events, msg)
