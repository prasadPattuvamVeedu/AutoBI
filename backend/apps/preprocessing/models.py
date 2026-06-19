from django.conf import settings
from django.db import models


class PreprocessingPlan(models.Model):
    STATUS_DRAFT = "draft"
    STATUS_ACTIVE = "active"
    STATUS_ARCHIVED = "archived"

    STATUS_CHOICES = [
        (STATUS_DRAFT, "Draft"),
        (STATUS_ACTIVE, "Active"),
        (STATUS_ARCHIVED, "Archived"),
    ]

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="preprocessing_plans",
    )
    dataset = models.ForeignKey(
        "datasets.Dataset",
        on_delete=models.CASCADE,
        related_name="preprocessing_plans",
    )
    dataset_version = models.ForeignKey(
        "datasets.DatasetVersion",
        on_delete=models.CASCADE,
        related_name="preprocessing_plans",
    )
    name = models.CharField(max_length=255)
    target_column = models.CharField(max_length=255, blank=True, null=True)
    plan_json = models.JSONField(default=dict, blank=True)
    required_columns_json = models.JSONField(default=list, blank=True)
    feature_mapping_json = models.JSONField(default=dict, blank=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_ACTIVE,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.owner})"


class PredictionDataset(models.Model):
    VALIDATION_PENDING = "pending"
    VALIDATION_PASSED = "passed"
    VALIDATION_FAILED = "failed"

    VALIDATION_STATUS_CHOICES = [
        (VALIDATION_PENDING, "Pending"),
        (VALIDATION_PASSED, "Passed"),
        (VALIDATION_FAILED, "Failed"),
    ]

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="prediction_datasets",
    )
    source_dataset = models.ForeignKey(
        "datasets.Dataset",
        on_delete=models.CASCADE,
        related_name="prediction_datasets",
    )
    uploaded_file = models.FileField(upload_to="prediction_datasets/", null=True, blank=True)
    file_type = models.CharField(max_length=50, blank=True)
    file_size = models.PositiveBigIntegerField(null=True, blank=True)
    row_count = models.PositiveIntegerField(null=True, blank=True)
    column_count = models.PositiveIntegerField(null=True, blank=True)
    columns_json = models.JSONField(default=list, blank=True)
    validation_status = models.CharField(
        max_length=20,
        choices=VALIDATION_STATUS_CHOICES,
        default=VALIDATION_PENDING,
    )
    validation_errors_json = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"PredictionDataset {self.id} ({self.owner})"


class PredictionPreparationJob(models.Model):
    STATUS_PENDING = "pending"
    STATUS_VALIDATED = "validated"
    STATUS_PREPARED = "prepared"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_VALIDATED, "Validated"),
        (STATUS_PREPARED, "Prepared"),
    ]

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="prediction_preparation_jobs",
    )
    preprocessing_plan = models.ForeignKey(
        PreprocessingPlan,
        on_delete=models.CASCADE,
        related_name="preparation_jobs",
    )
    prediction_dataset = models.ForeignKey(
        PredictionDataset,
        on_delete=models.CASCADE,
        related_name="preparation_jobs",
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
    )
    validation_result_json = models.JSONField(default=dict, blank=True)
    prepared_preview_json = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"PreparationJob {self.id} ({self.owner})"
