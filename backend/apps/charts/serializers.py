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
    y_column = serializers.CharField(required=False, allow_blank=True)
    dimension = serializers.CharField(required=False, allow_blank=True)
    measure = serializers.CharField(required=False, allow_blank=True)
    size_column = serializers.CharField(required=False, allow_blank=True)
    color_by_column = serializers.CharField(required=False, allow_blank=True)
    aggregation = serializers.CharField(required=False, allow_blank=True)
    top_n = serializers.IntegerField(required=False, min_value=1, max_value=1000)
    sort_order = serializers.CharField(required=False, allow_blank=True)
    title = serializers.CharField(required=False, allow_blank=True)
    columns = serializers.ListField(child=serializers.CharField(), required=False)
    selected_columns = serializers.ListField(child=serializers.CharField(), required=False)
    settings_json = serializers.JSONField(required=False, default=dict)


class ChartNoteSerializer(serializers.Serializer):
    title = serializers.CharField(required=False, allow_blank=True)
    content = serializers.CharField(required=False, allow_blank=True)
    mode = serializers.CharField(required=False, allow_blank=True, default="note")
    is_visible = serializers.BooleanField(required=False, default=True)
