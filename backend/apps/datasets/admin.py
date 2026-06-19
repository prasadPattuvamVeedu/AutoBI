from django.contrib import admin

from .models import ColumnSchema, Dataset, DatasetProfile, DatasetVersion


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "owner",
        "file_name",
        "original_filename",
        "file_type",
        "file_size",
        "row_count",
        "column_count",
        "upload_mode",
        "storage_type",
        "status",
        "created_at",
    )
    search_fields = (
        "name",
        "description",
        "original_filename",
        "owner__username",
        "owner__email",
    )
    list_filter = ("upload_mode", "storage_type", "status", "file_type", "created_at")


@admin.register(DatasetVersion)
class DatasetVersionAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "version_number", "version_type", "is_cleaned", "is_active", "created_at")
    search_fields = ("dataset__name",)
    list_filter = ("version_type", "is_cleaned", "is_active", "created_at")


@admin.register(ColumnSchema)
class ColumnSchemaAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "dataset",
        "column_name",
        "detected_type",
        "missing_count",
        "unique_count",
        "role",
        "created_at",
    )
    search_fields = ("dataset__name", "column_name", "detected_type", "role")
    list_filter = ("detected_type", "role", "created_at")


@admin.register(DatasetProfile)
class DatasetProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "created_at", "updated_at")
