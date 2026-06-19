from django.conf import settings
from django.db import models


class ChartSheet(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chart_sheets",
    )
    dataset = models.ForeignKey(
        "datasets.Dataset",
        on_delete=models.CASCADE,
        related_name="chart_sheets",
    )
    name = models.CharField(max_length=255)
    chart_type = models.CharField(max_length=80, default="bar")
    chart_config_json = models.JSONField(default=dict, blank=True)
    chart_data_json = models.JSONField(default=dict, blank=True)
    settings_json = models.JSONField(default=dict, blank=True)
    ai_interpretation_json = models.JSONField(default=dict, blank=True)
    user_notes_json = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "-created_at"]

    def __str__(self):
        return f"{self.name} ({self.dataset})"


class ChartInterpretation(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="chart_interpretations",
    )
    sheet = models.ForeignKey(
        ChartSheet,
        on_delete=models.CASCADE,
        related_name="interpretations",
    )
    mode = models.CharField(max_length=80, default="note")
    title = models.CharField(max_length=255, blank=True)
    content = models.TextField(blank=True)
    is_visible = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title or f"Interpretation {self.id}"
