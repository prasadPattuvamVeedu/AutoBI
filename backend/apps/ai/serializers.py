from rest_framework import serializers


class AiTaskRequestSerializer(serializers.Serializer):
    task = serializers.CharField(max_length=120)
    dataset_id = serializers.IntegerField(required=False, allow_null=True)
    context = serializers.JSONField(required=False, default=dict)
    question = serializers.CharField(required=False, allow_blank=True, allow_null=True)
