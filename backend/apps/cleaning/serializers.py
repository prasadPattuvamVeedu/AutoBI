from rest_framework import serializers


class DatasetVersionListSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    version_number = serializers.IntegerField()
    file = serializers.CharField()
    is_cleaned = serializers.BooleanField()
    transformation_log = serializers.JSONField()
    created_at = serializers.DateTimeField()


class CleaningApplySerializer(serializers.Serializer):
    # Accept nested JSON objects for actions so frontend can pass selected_method and params
    actions = serializers.ListField(
        child=serializers.JSONField(),
        required=False,
        default=list,
    )
    apply_safe = serializers.BooleanField(required=False, default=False)


class DatasetRollbackSerializer(serializers.Serializer):
    version_id = serializers.IntegerField()


class OutlierPreviewRequestSerializer(serializers.Serializer):
    column_name = serializers.CharField()
    method = serializers.CharField()
    params = serializers.JSONField(required=False, default=dict)
