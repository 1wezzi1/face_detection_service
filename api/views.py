import hashlib
import os
import uuid

import cv2
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from detector.models import DetectionLog
from detector.services import get_detector, preprocess


def _error(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"success": False, "error": message}, status=status)


@csrf_exempt
@require_POST
def detect_view(request):
    # --- 1. Валидация входа ---
    uploaded = request.FILES.get("image")
    if uploaded is None:
        return _error("Поле 'image' обязательно")

    ext = os.path.splitext(uploaded.name)[1].lower()
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_IMAGE_EXTENSIONS))
        return _error(f"Недопустимый формат: {ext}. Разрешены: {allowed}")

    if uploaded.size > settings.MAX_UPLOAD_SIZE:
        return _error("Файл превышает 10 МБ", status=413)

    detector_type = request.POST.get("detector_type", "haar").lower()
    if detector_type not in ("haar", "mtcnn"):
        return _error("detector_type должен быть 'haar' или 'mtcnn'")

    lighting = request.POST.get("lighting", "normal").lower()
    if lighting not in ("normal", "low", "bright"):
        return _error(
            "lighting должен быть 'normal', 'low' или 'bright'"
        )

    # --- 2. Чтение и сохранение файла ---
    raw_bytes = uploaded.read()
    image_hash = hashlib.sha256(raw_bytes).hexdigest()

    filename = f"{uuid.uuid4().hex}{ext}"
    saved_path = default_storage.save(
        os.path.join("uploads", filename), ContentFile(raw_bytes)
    )

    # --- 3. Декодирование и предобработка ---
    image = cv2.imdecode(
        np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        return _error("Не удалось прочитать изображение")
    image = preprocess(image, lighting)

    # --- 4. Инференс ---
    try:
        detector = get_detector(detector_type)
        result = detector.detect(image)
    except Exception as exc:
        return _error(f"Ошибка детекции: {exc}", status=500)

    # --- 5. Логирование в БД ---
    DetectionLog.objects.create(
        image_hash=image_hash,
        detector_type=detector_type,
        lighting_mode=lighting,
        face_count=result.count,
        time_ms=result.time_ms,
    )

    # --- 6. Ответ ---
    return JsonResponse(
        {
            "success": True,
            "result": result.to_dict(),
            "lighting": lighting,
            "image_url": settings.MEDIA_URL + saved_path,
        }
    )