from django.contrib import admin

from .models import Dataset, DatasetProfile


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "owner",
        "original_filename",
        "file_type",
        "file_size",
        "row_count",
        "column_count",
        "status",
        "created_at",
    )
    search_fields = (
        "name",
        "original_filename",
        "owner__username",
        "owner__email",
    )
    list_filter = ("status", "file_type", "created_at")


@admin.register(DatasetProfile)
class DatasetProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "created_at", "updated_at")
