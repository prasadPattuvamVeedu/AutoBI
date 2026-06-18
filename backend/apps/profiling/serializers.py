from rest_framework import serializers


class ProfileSummarySerializer(serializers.Serializer):
    row_count = serializers.IntegerField()
    column_count = serializers.IntegerField()
    duplicate_row_count = serializers.IntegerField()
    total_missing_cells = serializers.IntegerField()
    total_missing_percentage = serializers.FloatField()
    quality_score = serializers.FloatField()


class ProfileColumnSerializer(serializers.Serializer):
    column_name = serializers.CharField()
    detected_type = serializers.CharField()
    missing_count = serializers.IntegerField()
    missing_percentage = serializers.FloatField()
    unique_count = serializers.IntegerField(allow_null=True)
    unique_ratio = serializers.FloatField(required=False)
    mean = serializers.JSONField(required=False, allow_null=True)
    median = serializers.JSONField(required=False, allow_null=True)
    mode = serializers.JSONField(required=False, allow_null=True)
    min = serializers.JSONField(required=False, allow_null=True)
    max = serializers.JSONField(required=False, allow_null=True)
    range = serializers.JSONField(required=False, allow_null=True)
    std = serializers.JSONField(required=False, allow_null=True)
    percentile_25 = serializers.JSONField(required=False, allow_null=True)
    percentile_50 = serializers.JSONField(required=False, allow_null=True)
    percentile_75 = serializers.JSONField(required=False, allow_null=True)
    iqr = serializers.JSONField(required=False, allow_null=True)
    top_value = serializers.JSONField(required=False, allow_null=True)
    top_frequency = serializers.JSONField(required=False, allow_null=True)
    role = serializers.CharField()
    recommendation = serializers.CharField()


class DatasetProfileSummarySerializer(serializers.Serializer):
    dataset_id = serializers.IntegerField()
    summary = ProfileSummarySerializer()
    numeric_columns = serializers.ListField(child=serializers.CharField())
    categorical_columns = serializers.ListField(child=serializers.CharField())
    datetime_columns = serializers.ListField(child=serializers.CharField())
    boolean_columns = serializers.ListField(child=serializers.CharField())
    text_columns = serializers.ListField(child=serializers.CharField())
    id_like_columns = serializers.ListField(child=serializers.CharField())
    constant_columns = serializers.ListField(child=serializers.CharField())
    high_cardinality_columns = serializers.ListField(child=serializers.CharField())
    review_columns = serializers.ListField(child=serializers.CharField(), required=False)
    columns = ProfileColumnSerializer(many=True)
