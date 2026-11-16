from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

CLASSES = ("person", "helmet", "safety_vest", "forklift")


@dataclass(frozen=True)
class Frame:
    camera_id: str
    session_id: str
    frame_id: int
    source_width: int
    source_height: int
    source_pts: float | None
    received_at_utc: str
    received_monotonic: float
    image: np.ndarray = field(repr=False, compare=False)
    preprocess_transform: dict = field(default_factory=lambda: {"coordinate_space":"original_pixels", "library_restores":True})

    @property
    def key(self):
        return self.camera_id, self.session_id, self.frame_id

    def metadata(self):
        return {k:v for k,v in vars(self).items() if k != "image"}


@dataclass
class Detection:
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None
    observed: bool = True
    last_observed_frame: int | None = None
    last_observed_monotonic: float | None = None
    key: str | None = None


@dataclass
class RuleObservation:
    key: str
    kind: str
    condition: bool | None
    tracks: list[str] = field(default_factory=list)
    zone_id: str | None = None
    basis: dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"
    confirm_s: float = 1.0
    clear_s: float = 1.0
