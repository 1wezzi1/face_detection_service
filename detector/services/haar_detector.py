import time

import cv2
import numpy as np

from .base import BaseDetector, BBox, DetectionOutput


class HaarDetector(BaseDetector):
    name = "haar"

    def __init__(self):
        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("Не удалось загрузить каскад Хаара")

    def detect(self, image_bgr: np.ndarray) -> DetectionOutput:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        t0 = time.perf_counter()
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        faces = [
            BBox(x=int(x), y=int(y), w=int(w), h=int(h))
            for (x, y, w, h) in rects
        ]
        return DetectionOutput(
            detector=self.name, faces=faces, time_ms=elapsed_ms
        )