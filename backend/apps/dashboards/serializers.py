from rest_framework import serializers

from apps.charts.models import ChartSheet
from apps.charts.serializers import ChartSheetSerializer
from apps.datasets.models import Dataset

from .models import Dashboard, DashboardItem


class DashboardItemSerializer(serializers.ModelSerializer):
    sheet_detail = ChartSheetSerializer(source="sheet", read_only=True)

    class Meta:
        model = DashboardItem
        fields = ("id", "dashboard", "sheet", "sheet_detail", "x", "y", "w", "h", "config_json")
        read_only_fields = ("id", "dashboard", "sheet_detail")

    def validate_sheet(self, sheet):
        request = self.context.get("request")
        dashboard = self.context.get("dashboard")
        if request and not ChartSheet.objects.filter(id=sheet.id, owner=request.user).exists():
            raise serializers.ValidationError("Sheet not found.")
        if dashboard and sheet.dataset_id != dashboard.dataset_id:
            raise serializers.ValidationError("Sheet must belong to the dashboard dataset.")
        return sheet


class DashboardSerializer(serializers.ModelSerializer):
    items = DashboardItemSerializer(many=True, read_only=True)

    class Meta:
        model = Dashboard
        fields = (
            "id",
            "dataset",
            "name",
            "dashboard_type",
            "layout_json",
            "filters_json",
            "theme_json",
            "items",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("id", "items", "created_at", "updated_at")

    def validate_dataset(self, dataset):
        request = self.context.get("request")
        if request and not Dataset.objects.filter(id=dataset.id, owner=request.user).exists():
            raise serializers.ValidationError("Dataset not found.")
        return dataset


class DashboardLayoutSerializer(serializers.Serializer):
    items = serializers.ListField(child=serializers.DictField(), required=False, default=list)
    layout_json = serializers.JSONField(required=False, default=dict)
