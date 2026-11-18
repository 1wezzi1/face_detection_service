import time

import numpy as np
from facenet_pytorch import MTCNN

from .base import BaseDetector, BBox, DetectionOutput


class MtcnnDetector(BaseDetector):
    name = "mtcnn"

    def __init__(self, confidence_threshold: float = 0.9):
        self._threshold = confidence_threshold
        # keep_all=True — возвращает все лица, а не только самое крупное
        self._detector = MTCNN(keep_all=True, device="cpu")

    def detect(self, image_bgr: np.ndarray) -> DetectionOutput:
        # MTCNN ожидает RGB
        image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])

        t0 = time.perf_counter()
        boxes, probs = self._detector.detect(image_rgb)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        faces = []
        if boxes is not None:
            height, width = image_bgr.shape[:2]
            for (x1, y1, x2, y2), p in zip(boxes, probs):
                if p is None or p < self._threshold:
                    continue
                x = max(0, min(int(x1), width - 1))
                y = max(0, min(int(y1), height - 1))
                w = min(int(x2 - x1), width - x)
                h = min(int(y2 - y1), height - y)
                if w <= 0 or h <= 0:
                    continue
                faces.append(
                    BBox(x=x, y=y, w=w, h=h, confidence=float(p))
                )

        return DetectionOutput(
            detector=self.name, faces=faces, time_ms=elapsed_ms
        )