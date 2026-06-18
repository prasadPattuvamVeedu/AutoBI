from rest_framework import serializers

from .models import ColumnSchema, Dataset, DatasetProfile, DatasetVersion
from .services import (
    build_dataset_profile,
    build_preview,
    detect_column_schema,
    get_dataset_file_extension,
    get_dataset_shape,
    read_dataset_file,
    validate_dataset_file,
)


class DatasetVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetVersion
        fields = (
            "id",
            "version_number",
            "file",
            "is_cleaned",
            "transformation_log",
            "created_at",
        )
        read_only_fields = fields


class ColumnSchemaSerializer(serializers.ModelSerializer):
    class Meta:
        model = ColumnSchema
        fields = (
            "id",
            "column_name",
            "detected_type",
            "missing_count",
            "unique_count",
            "role",
            "created_at",
        )
        read_only_fields = fields


class DatasetSerializer(serializers.ModelSerializer):
    uploaded_at = serializers.DateTimeField(source="created_at", read_only=True)

    class Meta:
        model = Dataset
        fields = (
            "id",
            "name",
            "description",
            "file_name",
            "file_type",
            "file_size",
            "row_count",
            "column_count",
            "columns_json",
            "upload_mode",
            "storage_type",
            "status",
            "created_at",
            "uploaded_at",
            "updated_at",
        )
        read_only_fields = fields


class DatasetDetailSerializer(DatasetSerializer):
    versions = DatasetVersionSerializer(many=True, read_only=True)
    columns = ColumnSchemaSerializer(many=True, read_only=True)
    preview_json = serializers.SerializerMethodField()
    profile_json = serializers.SerializerMethodField()

    class Meta(DatasetSerializer.Meta):
        fields = DatasetSerializer.Meta.fields + (
            "preview_json",
            "profile_json",
            "versions",
            "columns",
        )

    def get_preview_json(self, obj):
        if not obj.file:
            return {"columns": [], "rows": []}

        try:
            obj.file.open("rb")
            df = read_dataset_file(obj.file)
        except ValueError:
            return {"columns": [], "rows": []}
        finally:
            obj.file.close()

        return build_preview(df, limit=20)

    def get_profile_json(self, obj):
        try:
            return obj.profile.profile_json
        except DatasetProfile.DoesNotExist:
            return {}


class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField(write_only=True)
    name = serializers.CharField(required=False, allow_blank=True, max_length=255)
    description = serializers.CharField(required=False, allow_blank=True)
    confirm_large_file = serializers.BooleanField(required=False, default=False)

    def validate_file(self, file):
        error = validate_dataset_file(file)
        if error:
            raise serializers.ValidationError(error)
        return file

    def create(self, validated_data):
        file = validated_data["file"]
        try:
            df = read_dataset_file(file)
        except ValueError as exc:
            raise serializers.ValidationError({"file": str(exc)}) from exc

        row_count, column_count = get_dataset_shape(df)
        columns_json = detect_column_schema(df)
        profile_json = build_dataset_profile(df)
        file.seek(0)

        file_name = file.name
        file_type = get_dataset_file_extension(file_name).lstrip(".")
        name = validated_data.get("name") or PathlessFileName(file_name)

        dataset = Dataset.objects.create(
            owner=self.context["request"].user,
            name=name,
            description=validated_data.get("description", ""),
            file_name=file_name,
            original_filename=file_name,
            file=file,
            file_type=file_type,
            file_size=file.size,
            row_count=row_count,
            column_count=column_count,
            columns_json=columns_json,
            upload_mode=Dataset.UPLOAD_MODE_LOCAL_UPLOAD,
            storage_type=Dataset.STORAGE_TYPE_LOCAL_DISK,
            status=Dataset.STATUS_READY,
        )

        ColumnSchema.objects.bulk_create(
            [
                ColumnSchema(
                    dataset=dataset,
                    column_name=column["column_name"],
                    detected_type=column["detected_type"],
                    missing_count=column["missing_count"],
                    unique_count=column["unique_count"],
                    role=column["recommended_role"],
                )
                for column in columns_json
            ]
        )
        DatasetProfile.objects.update_or_create(
            dataset=dataset,
            defaults={"profile_json": profile_json},
        )
        return dataset


def PathlessFileName(file_name):
    return file_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
