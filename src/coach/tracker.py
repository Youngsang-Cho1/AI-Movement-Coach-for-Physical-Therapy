from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .frame import PoseFrame

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_full.task"


class PoseTracker:
    """MediaPipe PoseLandmarker (single person) in synchronous VIDEO mode."""

    def __init__(self, model_path=DEFAULT_MODEL, min_conf=0.5):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} missing; run scripts/download_model.sh")
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, bgr: np.ndarray, ts_ms: int) -> PoseFrame | None:
        ts_ms = max(int(ts_ms), self._last_ts + 1)  # MediaPipe needs strictly increasing
        self._last_ts = ts_ms
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        res = self._lm.detect_for_video(mp.Image(mp.ImageFormat.SRGB, rgb), ts_ms)
        if not res.pose_landmarks:
            return None
        img, world = res.pose_landmarks[0], res.pose_world_landmarks[0]
        return PoseFrame(
            img=np.array([[p.x, p.y] for p in img]),
            world=np.array([[p.x, p.y, p.z] for p in world]),
            vis=np.array([p.visibility for p in img]),
        )

    def close(self):
        self._lm.close()
