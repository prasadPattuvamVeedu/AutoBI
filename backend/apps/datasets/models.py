from django.conf import settings
from django.db import models


class Dataset(models.Model):
    UPLOAD_MODE_LOCAL_UPLOAD = "local_upload"
    UPLOAD_MODE_CUSTOMER_CLOUD = "customer_cloud"

    UPLOAD_MODE_CHOICES = [
        (UPLOAD_MODE_LOCAL_UPLOAD, "Local upload"),
        (UPLOAD_MODE_CUSTOMER_CLOUD, "Customer cloud"),
    ]

    STORAGE_TYPE_LOCAL_TEMP = "local_temp"
    STORAGE_TYPE_LOCAL_DISK = "local_disk"
    STORAGE_TYPE_CUSTOMER_CLOUD = "customer_cloud"

    STORAGE_TYPE_CHOICES = [
        (STORAGE_TYPE_LOCAL_TEMP, "Local temporary"),
        (STORAGE_TYPE_LOCAL_DISK, "Local disk"),
        (STORAGE_TYPE_CUSTOMER_CLOUD, "Customer cloud"),
    ]

    STATUS_UPLOADED = "uploaded"
    STATUS_PROCESSING = "processing"
    STATUS_READY = "ready"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_UPLOADED, "Uploaded"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_READY, "Ready"),
        (STATUS_FAILED, "Failed"),
    ]

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="datasets",
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    file_name = models.CharField(max_length=255, blank=True)
    original_filename = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to="datasets/", null=True, blank=True)
    file_type = models.CharField(max_length=50, blank=True)
    file_size = models.PositiveBigIntegerField(null=True, blank=True)
    row_count = models.PositiveIntegerField(null=True, blank=True)
    column_count = models.PositiveIntegerField(null=True, blank=True)
    columns_json = models.JSONField(default=list, blank=True)
    upload_mode = models.CharField(
        max_length=30,
        choices=UPLOAD_MODE_CHOICES,
        default=UPLOAD_MODE_LOCAL_UPLOAD,
    )
    storage_type = models.CharField(
        max_length=30,
        choices=STORAGE_TYPE_CHOICES,
        default=STORAGE_TYPE_LOCAL_DISK,
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_UPLOADED,
    )
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.owner})"

    def delete(self, *args, **kwargs):
        storage = self.file.storage if self.file else None
        file_name = self.file.name if self.file else None
        super().delete(*args, **kwargs)
        if storage and file_name:
            storage.delete(file_name)


class DatasetVersion(models.Model):
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="versions",
    )
    version_number = models.PositiveIntegerField()
    file = models.FileField(upload_to="dataset_versions/", null=True, blank=True)
    is_cleaned = models.BooleanField(default=False)
    transformation_log = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-version_number"]
        unique_together = ("dataset", "version_number")

    def __str__(self):
        return f"{self.dataset.name} v{self.version_number}"

    def delete(self, *args, **kwargs):
        storage = self.file.storage if self.file else None
        file_name = self.file.name if self.file else None
        super().delete(*args, **kwargs)
        if storage and file_name:
            storage.delete(file_name)


class ColumnSchema(models.Model):
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="columns",
    )
    column_name = models.CharField(max_length=255)
    detected_type = models.CharField(max_length=100)
    missing_count = models.PositiveIntegerField(default=0)
    unique_count = models.PositiveIntegerField(default=0)
    role = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return f"{self.dataset.name}: {self.column_name}"


class DatasetProfile(models.Model):
    dataset = models.OneToOneField(
        Dataset,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    profile_json = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile for {self.dataset.name}"
