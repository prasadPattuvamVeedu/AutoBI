import os

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
            "version_type",
            "parent_version",
            "preview_rows",
            "columns",
            "transformation_log",
            "transformation_plan_json",
            "is_active",
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
    schema_json = serializers.SerializerMethodField()
    preview_json = serializers.SerializerMethodField()
    profile_json = serializers.SerializerMethodField()
    latest_cleaned_version = serializers.SerializerMethodField()
    latest_feature_engineered_version = serializers.SerializerMethodField()
    active_version_type = serializers.SerializerMethodField()
    active_version_id = serializers.SerializerMethodField()
    active_preview_rows = serializers.SerializerMethodField()
    active_columns = serializers.SerializerMethodField()
    active_processing_stage = serializers.SerializerMethodField()

    class Meta(DatasetSerializer.Meta):
        fields = DatasetSerializer.Meta.fields + (
            "schema_json",
            "preview_json",
            "profile_json",
            "versions",
            "columns",
            "latest_cleaned_version",
            "latest_feature_engineered_version",
            "active_version_type",
            "active_version_id",
            "active_preview_rows",
            "active_columns",
            "active_processing_stage",
        )

    def _get_latest_cleaned_version(self, obj):
        return (
            obj.versions.filter(
                is_cleaned=True,
                version_type=DatasetVersion.VERSION_TYPE_CLEANED,
                is_active=True,
                file__isnull=False,
            )
            .exclude(file="")
            .order_by("-version_number")
            .first()
        )

    def _get_latest_feature_engineered_version(self, obj):
        return (
            obj.versions.filter(
                version_type=DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED,
                is_active=True,
                file__isnull=False,
            )
            .exclude(file="")
            .order_by("-version_number")
            .first()
        )

    def _get_active_version(self, obj):
        return self._get_latest_feature_engineered_version(obj) or self._get_latest_cleaned_version(obj)

    def _build_file_preview(self, file_field, limit=20):
        if not file_field:
            return {"columns": [], "rows": []}

        try:
            file_field.open("rb")
            df = read_dataset_file(file_field)
        except ValueError:
            return {"columns": [], "rows": []}
        finally:
            file_field.close()

        return build_preview(df, limit=limit)

    def _normalize_schema_rows(self, value):
        if not value:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, dict):
            normalized_rows = []
            for key, row in value.items():
                if isinstance(row, dict):
                    row_data = dict(row)
                    row_data.setdefault("column_name", key)
                    normalized_rows.append(row_data)
                else:
                    normalized_rows.append({"column_name": key, "detected_type": row})
            return normalized_rows

        return []

    def _first_schema_rows(self, *values):
        for value in values:
            rows = self._normalize_schema_rows(value)
            if rows:
                return rows
        return []

    def get_schema_json(self, obj):
        profile_json = self.get_profile_json(obj)
        preview_json = self.get_preview_json(obj)
        preview_columns = preview_json.get("columns", []) if isinstance(preview_json, dict) else []

        return self._first_schema_rows(
            getattr(obj, "schema_json", None),
            obj.columns_json,
            profile_json.get("columns") if isinstance(profile_json, dict) else None,
            profile_json.get("column_profiles") if isinstance(profile_json, dict) else None,
            [{"column_name": column} for column in preview_columns],
        )

    def get_preview_json(self, obj):
        return self._build_file_preview(obj.file, limit=20)

    def get_profile_json(self, obj):
        try:
            return obj.profile.profile_json
        except DatasetProfile.DoesNotExist:
            return {}

    def get_latest_cleaned_version(self, obj):
        version = self._get_latest_cleaned_version(obj)
        return self._serialize_version_payload(version, "cleaned")

    def get_latest_feature_engineered_version(self, obj):
        version = self._get_latest_feature_engineered_version(obj)
        return self._serialize_version_payload(version, "feature_engineered")

    def _serialize_version_payload(self, version, fallback_type):
        if version is None:
            return None

        preview = {
            "columns": version.columns or [],
            "rows": version.preview_rows or [],
        }
        if not preview["columns"] and not preview["rows"]:
            preview = self._build_file_preview(version.file, limit=20)

        return {
            "id": version.id,
            "version_number": version.version_number,
            "version_type": version.version_type or fallback_type,
            "parent_version_id": version.parent_version_id,
            "is_cleaned": version.is_cleaned,
            "is_active": version.is_active,
            "created_at": version.created_at,
            "preview_rows": preview.get("rows", []),
            "columns": preview.get("columns", []),
            "transformation_log": version.transformation_log,
            "transformation_plan_json": version.transformation_plan_json,
        }

    def get_active_version_type(self, obj):
        version = self._get_active_version(obj)
        return version.version_type if version is not None else "original"

    def get_active_version_id(self, obj):
        version = self._get_active_version(obj)
        return version.id if version is not None else None

    def get_active_preview_rows(self, obj):
        version = self._get_active_version(obj)
        if version is not None:
            if version.preview_rows:
                return version.preview_rows
            return self._build_file_preview(version.file, limit=20).get("rows", [])
        return self.get_preview_json(obj).get("rows", [])

    def get_active_columns(self, obj):
        version = self._get_active_version(obj)
        if version is not None:
            if version.columns:
                return version.columns
            return self._build_file_preview(version.file, limit=20).get("columns", [])
        return self.get_preview_json(obj).get("columns", [])

    def get_active_processing_stage(self, obj):
        return self.get_active_version_type(obj)


def _version_stage_label(version_type):
    labels = {
        DatasetVersion.VERSION_TYPE_CLEANED: "Cleaning",
        DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED: "Feature Engineering",
        DatasetVersion.VERSION_TYPE_ML_READY: "ML Readiness",
        DatasetVersion.VERSION_TYPE_ORIGINAL: "Original",
    }
    return labels.get(version_type, "Manual Transform")


class TransformedDatasetSerializer(serializers.ModelSerializer):
    name = serializers.SerializerMethodField()
    source_dataset_id = serializers.IntegerField(source="dataset_id", read_only=True)
    source_dataset_name = serializers.CharField(source="dataset.name", read_only=True)
    rows = serializers.SerializerMethodField()
    columns_count = serializers.SerializerMethodField()
    column_count = serializers.SerializerMethodField()
    file_size = serializers.SerializerMethodField()
    pipeline_stage = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()
    preview_json = serializers.SerializerMethodField()
    schema_json = serializers.SerializerMethodField()

    class Meta:
        model = DatasetVersion
        fields = (
            "id",
            "name",
            "source_dataset_id",
            "source_dataset_name",
            "version_number",
            "version_type",
            "pipeline_stage",
            "status",
            "rows",
            "columns_count",
            "column_count",
            "file_size",
            "is_cleaned",
            "is_active",
            "preview_json",
            "schema_json",
            "transformation_log",
            "transformation_plan_json",
            "created_at",
        )
        read_only_fields = fields

    def get_name(self, obj):
        if obj.file:
            return os.path.basename(obj.file.name)
        return f"{obj.dataset.name}_v{obj.version_number}"

    def get_rows(self, obj):
        return len(obj.preview_rows or []) if not obj.file else obj.dataset.row_count

    def get_columns_count(self, obj):
        return len(obj.columns or []) or obj.dataset.column_count

    def get_column_count(self, obj):
        return self.get_columns_count(obj)

    def get_file_size(self, obj):
        if not obj.file:
            return None
        try:
            return obj.file.size
        except (OSError, ValueError):
            return None

    def get_pipeline_stage(self, obj):
        return _version_stage_label(obj.version_type)

    def get_status(self, obj):
        return "ML Ready" if obj.version_type == DatasetVersion.VERSION_TYPE_ML_READY else "Ready"

    def get_preview_json(self, obj):
        return {"columns": obj.columns or [], "rows": obj.preview_rows or []}

    def get_schema_json(self, obj):
        return obj.columns or []


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
