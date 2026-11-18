from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class BBox:
    """Ограничивающая рамка лица."""

    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class DetectionOutput:
    """Унифицированный результат работы любого детектора."""

    detector: str
    faces: List[BBox] = field(default_factory=list)
    time_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.faces)

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "faces": [f.to_dict() for f in self.faces],
            "count": self.count,
            "time_ms": round(self.time_ms, 2),
        }


class BaseDetector:
    """Абстрактный интерфейс детектора."""

    name: str = "base"

    def detect(self, image_bgr: np.ndarray) -> DetectionOutput:
        raise NotImplementedError