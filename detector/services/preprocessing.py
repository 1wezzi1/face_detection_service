import cv2
import numpy as np


def _gamma_correct(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]
    ).astype("uint8")
    return cv2.LUT(image_bgr, table)


def apply_low_light(image_bgr: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Имитация низкой освещённости (затемнение)."""
    return _gamma_correct(image_bgr, gamma)


def apply_bright_light(
    image_bgr: np.ndarray, gamma: float = 0.5
) -> np.ndarray:
    """Имитация засветки (осветление)."""
    return _gamma_correct(image_bgr, gamma)


def preprocess(image_bgr: np.ndarray, lighting: str) -> np.ndarray:
    """Применяет преобразование освещённости по строковому режиму."""
    if lighting == "low":
        return apply_low_light(image_bgr)
    if lighting == "bright":
        return apply_bright_light(image_bgr)
    return image_bgr