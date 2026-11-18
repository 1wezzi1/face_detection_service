"""Ленивая инициализация детекторов — модели грузятся один раз за процесс."""

from .base import BaseDetector
from .haar_detector import HaarDetector
from .mtcnn_detector import MtcnnDetector

_instances: dict[str, BaseDetector] = {}


def get_detector(name: str) -> BaseDetector:
    name = name.lower()
    if name not in ("haar", "mtcnn"):
        raise ValueError(f"Неизвестный детектор: {name}")
    if name not in _instances:
        _instances[name] = (
            HaarDetector() if name == "haar" else MtcnnDetector()
        )
    return _instances[name]