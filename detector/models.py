from django.db import models


class DetectionLog(models.Model):
    DETECTOR_CHOICES = [
        ("haar", "Haar Cascade"),
        ("mtcnn", "MTCNN"),
    ]
    LIGHTING_CHOICES = [
        ("normal", "Normal"),
        ("low", "Low"),
        ("bright", "Bright"),
    ]

    image_hash = models.CharField(max_length=64, db_index=True)
    detector_type = models.CharField(max_length=10, choices=DETECTOR_CHOICES)
    lighting_mode = models.CharField(
        max_length=10, choices=LIGHTING_CHOICES, default="normal"
    )
    face_count = models.IntegerField(default=0)
    time_ms = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Результат детекции"
        verbose_name_plural = "Результаты детекции"

    def __str__(self):
        return (
            f"{self.detector_type}/{self.lighting_mode} → "
            f"{self.face_count} лиц за {self.time_ms:.1f} мс"
        )