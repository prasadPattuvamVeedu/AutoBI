from rest_framework import serializers


class EdaGraphRequestSerializer(serializers.Serializer):
    eda_mode = serializers.ChoiceField(choices=["advanced", "validation"], required=False, default="advanced")
    dataset_version = serializers.CharField(required=False, allow_blank=True, default="raw")
    chart_type = serializers.CharField()
    column = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    x_column = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    y_column = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    columns = serializers.ListField(child=serializers.CharField(), required=False, allow_empty=True)
    target_column = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    bins = serializers.CharField(required=False, allow_blank=True, default="auto")
    sample_size = serializers.CharField(required=False, allow_blank=True, allow_null=True)
