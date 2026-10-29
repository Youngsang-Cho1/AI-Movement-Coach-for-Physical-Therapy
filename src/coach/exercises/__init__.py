from .base import load_config
from .knee_extension import KneeExtension, KneeExtensionConfig
from .shoulder_raise import ShoulderRaise, ShoulderRaiseConfig
from .sit_to_stand import SitToStand, SitToStandConfig

REGISTRY = {
    "knee_extension": (KneeExtension, KneeExtensionConfig),
    "sit_to_stand": (SitToStand, SitToStandConfig),
    "shoulder_raise": (ShoulderRaise, ShoulderRaiseConfig),
}


def make_exercise(name: str, overrides: dict | None = None):
    cls, cfg_cls = REGISTRY[name]
    return cls(load_config(cfg_cls, overrides))
