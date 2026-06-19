from rest_framework import serializers

from .models import (
    PreprocessingPlan,
    PredictionDataset,
    PredictionPreparationJob,
)


class PreprocessingPlanSerializer(serializers.ModelSerializer):
    class Meta:
        model = PreprocessingPlan
        fields = (
            "id",
            "dataset",
            "dataset_version",
            "name",
            "target_column",
            "plan_json",
            "required_columns_json",
            "feature_mapping_json",
            "status",
            "created_at",
            "updated_at",
        )
        read_only_fields = (
            "id",
            "owner",
            "created_at",
            "updated_at",
        )


class PredictionDatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField(write_only=True)
    source_dataset_id = serializers.IntegerField()


class PredictionDatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionDataset
        fields = (
            "id",
            "source_dataset",
            "uploaded_file",
            "file_type",
            "file_size",
            "row_count",
            "column_count",
            "columns_json",
            "validation_status",
            "validation_errors_json",
            "created_at",
        )
        read_only_fields = fields


class PredictionPreparationJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionPreparationJob
        fields = (
            "id",
            "preprocessing_plan",
            "prediction_dataset",
            "status",
            "validation_result_json",
            "prepared_preview_json",
            "created_at",
        )
        read_only_fields = fields


class ValidatePredictionDatasetSerializer(serializers.Serializer):
    prediction_dataset_id = serializers.IntegerField()


class PreparePredictionDatasetSerializer(serializers.Serializer):
    prediction_dataset_id = serializers.IntegerField()


class ManualFeatureInferRulesSerializer(serializers.Serializer):
    source_column = serializers.CharField()
    new_feature_name = serializers.CharField(required=False, allow_blank=True)
    output_type = serializers.CharField(required=False, allow_blank=True)
    source_values = serializers.ListField(required=False, default=list)
    expected_outputs = serializers.ListField(required=False, default=list)


class ManualFeaturePreviewSerializer(ManualFeatureInferRulesSerializer):
    rule_json = serializers.JSONField()


class ManualFeatureSavePlaceholderSerializer(ManualFeaturePreviewSerializer):
    preview_rows = serializers.ListField(required=False, default=list)
