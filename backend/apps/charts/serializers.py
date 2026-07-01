from rest_framework import serializers

from apps.datasets.models import Dataset

from .models import ChartInterpretation, ChartSheet


class ChartInterpretationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChartInterpretation
        fields = (
            "id",
            "sheet",
            "mode",
            "title",
            "content",
            "is_visible",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("id", "sheet", "created_at", "updated_at")


class ChartSheetSerializer(serializers.ModelSerializer):
    interpretations = ChartInterpretationSerializer(many=True, read_only=True)

    class Meta:
        model = ChartSheet
        fields = (
            "id",
            "dataset",
            "name",
            "chart_type",
            "chart_config_json",
            "chart_data_json",
            "settings_json",
            "ai_interpretation_json",
            "user_notes_json",
            "interpretations",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("id", "ai_interpretation_json", "user_notes_json", "created_at", "updated_at")

    def validate_dataset(self, dataset):
        request = self.context.get("request")
        if request and not Dataset.objects.filter(id=dataset.id, owner=request.user).exists():
            raise serializers.ValidationError("Dataset not found.")
        return dataset


class ChartGenerateSerializer(serializers.Serializer):
    chart_type = serializers.CharField(required=False, allow_blank=True)
    x_column = serializers.CharField(required=False, allow_blank=True)
    y_column = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    dimension = serializers.CharField(required=False, allow_blank=True)
    measure = serializers.CharField(required=False, allow_blank=True)
    value_column = serializers.CharField(required=False, allow_blank=True)
    size_column = serializers.CharField(required=False, allow_blank=True)
    color_by_column = serializers.CharField(required=False, allow_blank=True)
    group_by_column = serializers.CharField(required=False, allow_blank=True)
    secondary_y_column = serializers.CharField(required=False, allow_blank=True)
    latitude_column = serializers.CharField(required=False, allow_blank=True)
    longitude_column = serializers.CharField(required=False, allow_blank=True)
    location_column = serializers.CharField(required=False, allow_blank=True)
    aggregation = serializers.CharField(required=False, allow_blank=True)
    top_n = serializers.IntegerField(required=False, allow_null=True, min_value=1, max_value=1000)
    bottom_n = serializers.IntegerField(required=False, allow_null=True, min_value=1, max_value=1000)
    bins = serializers.IntegerField(required=False, allow_null=True, min_value=1, max_value=100)
    max_columns = serializers.IntegerField(required=False, allow_null=True, min_value=2, max_value=50)
    sort_order = serializers.CharField(required=False, allow_blank=True)
    sort_by = serializers.CharField(required=False, allow_blank=True)
    date_granularity = serializers.CharField(required=False, allow_blank=True)
    primary_series_type = serializers.CharField(required=False, allow_blank=True)
    secondary_series_type = serializers.CharField(required=False, allow_blank=True)
    secondary_axis_color = serializers.CharField(required=False, allow_blank=True)
    title = serializers.CharField(required=False, allow_blank=True)
    description = serializers.CharField(required=False, allow_blank=True)
    columns = serializers.ListField(child=serializers.CharField(), required=False)
    selected_columns = serializers.ListField(child=serializers.CharField(), required=False)
    multi_series = serializers.ListField(child=serializers.DictField(), required=False, default=list)
    filters = serializers.ListField(child=serializers.DictField(), required=False, default=list)
    calculated_fields = serializers.ListField(child=serializers.DictField(), required=False, default=list)
    settings_json = serializers.JSONField(required=False, default=dict)

    def validate_settings_json(self, value):
        if value in (None, ""):
            return {}
        if not isinstance(value, dict):
            raise serializers.ValidationError("settings_json must be an object.")
        return value


class DashboardCommandConfigSerializer(serializers.Serializer):
    theme = serializers.CharField(required=False, allow_blank=True)
    filters = serializers.ListField(child=serializers.DictField(), required=False, default=list)
    tooltip_fields = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    drill_path = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    conditional_rules = serializers.ListField(child=serializers.DictField(), required=False, default=list)
    forecast = serializers.DictField(required=False, default=dict)
    what_if = serializers.DictField(required=False, default=dict)


class DashboardForecastSerializer(serializers.Serializer):
    date_column = serializers.CharField(required=False, allow_blank=True)
    value_column = serializers.CharField(required=False, allow_blank=True)
    periods = serializers.IntegerField(required=False, min_value=1, max_value=120, default=6)
    method = serializers.CharField(required=False, allow_blank=True, default="moving_average")


class DashboardWhatIfSerializer(serializers.Serializer):
    variable = serializers.CharField(required=False, allow_blank=True)
    change_type = serializers.CharField(required=False, allow_blank=True, default="percentage")
    change_value = serializers.FloatField(required=False, default=0)
    affected_measure = serializers.CharField(required=False, allow_blank=True)


class DashboardDrillSerializer(serializers.Serializer):
    drill_path = serializers.ListField(child=serializers.CharField(), required=False, default=list)
    filters = serializers.ListField(child=serializers.DictField(), required=False, default=list)

class ChartNoteSerializer(serializers.Serializer):
    title = serializers.CharField(required=False, allow_blank=True)
    content = serializers.CharField(required=False, allow_blank=True)
    mode = serializers.CharField(required=False, allow_blank=True, default="note")
    is_visible = serializers.BooleanField(required=False, default=True)
