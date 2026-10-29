from dataclasses import dataclass

from ..frame import JOINTS, PoseFrame
from ..geometry import angle_deg
from .base import CommonConfig, Exercise


@dataclass
class KneeExtensionConfig(CommonConfig):
    target_angle: float = 160.0   # therapist target: extend at least this far (180 = straight)
    hold_s: float = 2.0           # therapist target: hold at target this long
    start_angle: float = 125.0    # leaving flexed rest position
    return_angle: float = 110.0   # back at rest -> rep ends
    hold_tol: float = 5.0         # may sag this far below target and still count as holding
    turn_drop: float = 8.0        # drop from peak that signals a turn-around
    min_rom: float = 10.0         # peak must exceed start_angle by this to be a real attempt


class KneeExtension(Exercise):
    """Side view of a seated person. Knee angle = hip-knee-ankle interior angle."""

    name, primary = "knee_extension", "angle"

    def __init__(self, cfg=None):
        super().__init__(cfg or KneeExtensionConfig())

    def _reset_motion(self):
        super()._reset_motion()
        self._peak = 0.0
        self._hold_t0 = None

    def required(self, f):
        j = JOINTS[self.pick_side(f)]
        return [j["hip"], j["knee"], j["ankle"]]

    def measure(self, f: PoseFrame, t):
        j = JOINTS[self.pick_side(f)]
        raw = angle_deg(f.world[j["hip"]], f.world[j["knee"]], f.world[j["ankle"]])
        m = {"angle_raw": raw}
        if raw == raw:
            m["angle"] = self.smooth("angle", raw, t)
        else:
            m["angle"] = raw
        return m

    def step(self, t, m):
        c, a, ev = self.cfg, m["angle"], []
        if self.phase == "rest":
            if a > c.start_angle:
                self.phase, self._peak = "extending", a
                self.attempt += 1
        elif self.phase == "extending":
            self._peak = max(self._peak, a)
            if a >= c.target_angle:
                self.phase, self._hold_t0 = "holding", t
            elif a < self._peak - c.turn_drop:
                if self._peak < c.start_angle + c.min_rom:   # jitter, not an attempt
                    self.attempt -= 1
                    self.phase = "rest"
                else:
                    ev.append(self.fb("extend_more", "Straighten your knee more", t))
                    self.phase = "returning"
        elif self.phase == "holding":
            self._peak = max(self._peak, a)
            m["hold_s"] = t - self._hold_t0
            if a < c.target_angle - c.hold_tol:
                if m["hold_s"] < c.hold_s:
                    ev.append(self.fb("hold_longer", "Hold the position longer", t))
                else:
                    self.good_reps += 1
                    ev.append(self.fb("good_rep", "Good rep", t))
                self.phase = "returning"
        elif self.phase == "returning":
            if a < c.return_angle:
                self.phase = "rest"
                self.reps += 1
            elif a >= c.target_angle:      # went up again without resting
                self.phase, self._hold_t0 = "holding", t
        m["peak"] = self._peak
        return ev
