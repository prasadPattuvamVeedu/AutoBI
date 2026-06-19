from django.contrib import admin

from .models import ChartInterpretation, ChartSheet


@admin.register(ChartSheet)
class ChartSheetAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "dataset", "owner", "chart_type", "updated_at")
    list_filter = ("chart_type", "created_at", "updated_at")
    search_fields = ("name", "dataset__name", "owner__username", "owner__email")


@admin.register(ChartInterpretation)
class ChartInterpretationAdmin(admin.ModelAdmin):
    list_display = ("id", "sheet", "owner", "mode", "title", "is_visible", "created_at")
    list_filter = ("mode", "is_visible", "created_at")
    search_fields = ("title", "content", "sheet__name", "owner__username", "owner__email")
