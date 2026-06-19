from django.contrib import admin

from .models import Dashboard, DashboardItem


class DashboardItemInline(admin.TabularInline):
    model = DashboardItem
    extra = 0


@admin.register(Dashboard)
class DashboardAdmin(admin.ModelAdmin):
    inlines = [DashboardItemInline]
    list_display = ("id", "name", "dataset", "owner", "dashboard_type", "updated_at")
    list_filter = ("dashboard_type", "created_at", "updated_at")
    search_fields = ("name", "dataset__name", "owner__username", "owner__email")


@admin.register(DashboardItem)
class DashboardItemAdmin(admin.ModelAdmin):
    list_display = ("id", "dashboard", "sheet", "x", "y", "w", "h")
    search_fields = ("dashboard__name", "sheet__name")
