from dataclasses import dataclass

from ..frame import JOINTS, PoseFrame
from ..geometry import lean_deg, vector_angle_deg
from .base import CommonConfig, Exercise


@dataclass
class ShoulderRaiseConfig(CommonConfig):
    target_angle: float = 80.0    # therapist target: raise arm to at least this elevation
    start_angle: float = 25.0
    return_angle: float = 15.0
    hold_tol: float = 5.0
    turn_drop: float = 8.0
    min_rom: float = 15.0
    max_lean_deg: float = 10.0    # sideways trunk lean tolerated


class ShoulderRaise(Exercise):
    """Front view. Elevation = angle between trunk (shoulder->hip) and upper arm (shoulder->elbow)."""

    name, primary = "shoulder_raise", "elevation"
    rest_phase = "down"

    def __init__(self, cfg=None):
        super().__init__(cfg or ShoulderRaiseConfig())

    def _reset_motion(self):
        super()._reset_motion()
        self._peak = 0.0
        self._lean_flagged = False

    def _sides(self):
        return "LR" if self.cfg.side == "auto" else self.cfg.side

    def required(self, f):
        idx = [JOINTS[s][p] for s in "LR" for p in ("shoulder", "hip")]  # trunk lean needs both
        return idx + [JOINTS[s]["elbow"] for s in self._sides()]

    def measure(self, f: PoseFrame, t):
        elev = {}
        for s in self._sides():
            j = JOINTS[s]
            sh = f.world[j["shoulder"]]
            elev[s] = vector_angle_deg(f.world[j["hip"]] - sh, f.world[j["elbow"]] - sh)
        raw = max(elev.values())
        shoulders = (f.world[11, :2] + f.world[12, :2]) / 2
        hips = (f.world[23, :2] + f.world[24, :2]) / 2
        lean = lean_deg(shoulders, hips)
        return {"elevation_raw": raw,
                "elevation": self.smooth("elev", raw, t) if raw == raw else raw,
                "lean": self.smooth("lean", lean, t)}

    def step(self, t, m):
        c, e, ev = self.cfg, m["elevation"], []
        if self.phase == "down":
            self.armed = self.armed or e < c.start_angle
            if self.armed and e > c.start_angle:
                self.phase, self._peak, self._lean_flagged = "raising", e, False
                self.attempt += 1
        elif self.phase == "raising":
            self._peak = max(self._peak, e)
            if e >= c.target_angle:
                self.phase = "top"
            elif e < self._peak - c.turn_drop:
                if self._peak < c.start_angle + c.min_rom:
                    self.attempt -= 1
                    self.phase = "down"
                else:
                    ev.append(self.fb("raise_higher", "Raise your arm higher", t))
                    self.phase = "lowering"
        elif self.phase == "top":
            self._peak = max(self._peak, e)
            if e < c.target_angle - c.hold_tol:
                self.phase = "lowering"
        elif self.phase == "lowering":
            if e < c.return_angle:
                self.phase = "down"
                self.reps += 1
                if self._peak >= c.target_angle and not self._lean_flagged:
                    self.good_reps += 1
                    ev.append(self.fb("good_rep", "Good rep", t))
        if self.phase != "down" and abs(m["lean"]) > c.max_lean_deg and not self._lean_flagged:
            self._lean_flagged = True
            ev.append(self.fb("trunk_lean", "Keep your trunk upright", t))
        return ev
