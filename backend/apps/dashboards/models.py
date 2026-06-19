from django.conf import settings
from django.db import models


class Dashboard(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="visualization_dashboards",
    )
    dataset = models.ForeignKey(
        "datasets.Dataset",
        on_delete=models.CASCADE,
        related_name="visualization_dashboards",
    )
    name = models.CharField(max_length=255)
    dashboard_type = models.CharField(max_length=80, default="visualization")
    layout_json = models.JSONField(default=dict, blank=True)
    filters_json = models.JSONField(default=dict, blank=True)
    theme_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at", "-created_at"]

    def __str__(self):
        return f"{self.name} ({self.dataset})"


class DashboardItem(models.Model):
    dashboard = models.ForeignKey(
        Dashboard,
        on_delete=models.CASCADE,
        related_name="items",
    )
    sheet = models.ForeignKey(
        "charts.ChartSheet",
        on_delete=models.CASCADE,
        related_name="dashboard_items",
    )
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    w = models.IntegerField(default=6)
    h = models.IntegerField(default=4)
    config_json = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["y", "x", "id"]

    def __str__(self):
        return f"{self.dashboard} -> {self.sheet}"
