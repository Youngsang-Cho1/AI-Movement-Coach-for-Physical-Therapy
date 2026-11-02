from dataclasses import dataclass

from ..frame import JOINTS, PoseFrame
from ..geometry import angle_deg
from .base import CommonConfig, Exercise


@dataclass
class SitToStandConfig(CommonConfig):
    sit_angle: float = 120.0      # knee angle below this = seated
    stand_angle: float = 160.0    # knee angle above this = standing
    stand_hyst: float = 8.0
    turn_drop: float = 10.0
    min_rom: float = 10.0
    # tempo targets (s), timed between the sit and stand angle thresholds
    min_ascent_s: float = 0.6
    max_ascent_s: float = 3.0
    min_descent_s: float = 0.6
    max_descent_s: float = 4.0


class SitToStand(Exercise):
    """Side view. A rep is counted when standing is reached; tempo judged per phase."""

    name, primary = "sit_to_stand", "angle"
    rest_phase = "seated"

    def __init__(self, cfg=None):
        super().__init__(cfg or SitToStandConfig())

    def _reset_motion(self):
        super()._reset_motion()
        self._peak = 0.0
        self._t0 = 0.0
        self._ascent_ok = True
        self.last_ascent_s = self.last_descent_s = None

    def required(self, f):
        j = JOINTS[self.pick_side(f)]
        return [j["hip"], j["knee"], j["ankle"]]

    def measure(self, f: PoseFrame, t):
        j = JOINTS[self.pick_side(f)]
        raw = angle_deg(f.world[j["hip"]], f.world[j["knee"]], f.world[j["ankle"]])
        return {"angle_raw": raw, "angle": self.smooth("angle", raw, t) if raw == raw else raw}

    def step(self, t, m):
        c, a, ev = self.cfg, m["angle"], []
        if self.phase == "seated":
            self.armed = self.armed or a < c.sit_angle
            if self.armed and a > c.sit_angle:
                self.phase, self._peak, self._t0, self._ascent_ok = "rising", a, t, True
                self.attempt += 1
        elif self.phase == "rising":
            self._peak = max(self._peak, a)
            if a >= c.stand_angle:
                self.phase, self.reps = "standing", self.reps + 1
                self.last_ascent_s = t - self._t0
                if self.last_ascent_s < c.min_ascent_s:
                    ev.append(self.fb("ascent_too_fast", "Rise more slowly", t))
                    self._ascent_ok = False
                elif self.last_ascent_s > c.max_ascent_s:
                    ev.append(self.fb("ascent_too_slow", "Stand up a bit faster", t))
                    self._ascent_ok = False
            elif a < self._peak - c.turn_drop:
                if self._peak < c.sit_angle + c.min_rom:
                    self.attempt -= 1
                else:
                    ev.append(self.fb("stand_fully", "Stand up all the way", t))
                self.phase = "lowering_abort"
        elif self.phase == "standing":
            if a < c.stand_angle - c.stand_hyst:
                self.phase, self._t0 = "lowering", t
        elif self.phase == "lowering":
            if a >= c.stand_angle:
                self.phase = "standing"
            elif a < c.sit_angle:
                self.phase = "seated"
                self.last_descent_s = t - self._t0
                if self.last_descent_s < c.min_descent_s:
                    ev.append(self.fb("descent_too_fast", "Lower yourself more slowly", t))
                elif self.last_descent_s > c.max_descent_s:
                    ev.append(self.fb("descent_too_slow", "Sit down a bit faster", t))
                elif self._ascent_ok:
                    self.good_reps += 1
                    ev.append(self.fb("good_rep", "Good tempo", t))
        elif self.phase == "lowering_abort":
            if a < c.sit_angle:
                self.phase = "seated"
            elif a >= c.stand_angle:
                self.phase = "standing"
                self.reps += 1
        return ev
