from dataclasses import dataclass

import numpy as np

# MediaPipe Pose landmark indices (33-point BlazePose topology)
JOINTS = {
    "L": dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27),
    "R": dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28),
}
JOINT_NAMES = {idx: f"{side} {name}" for side, d in JOINTS.items() for name, idx in d.items()}
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


@dataclass
class PoseFrame:
    img: np.ndarray    # (33, 2) normalized image coords, y down
    world: np.ndarray  # (33, 3) metric coords centered on hips, y down
    vis: np.ndarray    # (33,) landmark visibility in [0, 1]
