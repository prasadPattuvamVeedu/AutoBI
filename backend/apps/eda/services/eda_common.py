"""
AutoBI EDA common helpers.

Shared by:
- Advanced EDA: before cleaning / before transformation.
- Validation EDA: after cleaning / after feature engineering comparison.

Important:
Column classification is not reimplemented here. EDA reuses the existing
DatasetProfile / ColumnSchema / dataset-services profile logic so numeric,
categorical, datetime, id-like, text, and boolean buckets stay identical to
Profile.
"""

from apps.datasets.models import Dataset, DatasetProfile, DatasetVersion
from apps.datasets.services import make_json_safe, read_dataset_file


VERSION_ALIASES = {
    "raw": DatasetVersion.VERSION_TYPE_ORIGINAL,
    "original": DatasetVersion.VERSION_TYPE_ORIGINAL,
    "cleaned": DatasetVersion.VERSION_TYPE_CLEANED,
    "feature_engineered": DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED,
    "engineered": DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED,
    "ml_ready": DatasetVersion.VERSION_TYPE_ML_READY,
    "transformed": None,
    "active": None,
    "latest": None,
}


def _read_file_field_to_dataframe(file_field):
    if not file_field:
        raise ValueError("Dataset file is not available.")

    try:
        file_field.open("rb")
        return read_dataset_file(file_field)
    finally:
        try:
            file_field.close()
        except Exception:
            pass


def _get_dataset(dataset_id):
    return Dataset.objects.get(pk=dataset_id)


def _get_latest_version(dataset, version_type=None):
    queryset = DatasetVersion.objects.filter(dataset=dataset, is_active=True)
    if version_type:
        queryset = queryset.filter(version_type=version_type)
    return queryset.order_by("-version_number", "-created_at").first()


def load_eda_dataframe(dataset_id, dataset_version="raw"):
    """
    Load a dataset version as a pandas DataFrame for EDA.

    This is safe integration code only. It reuses the existing dataset file
    reader instead of adding new pandas parsing logic here.
    """
    dataset = _get_dataset(dataset_id)
    requested_version = str(dataset_version or "raw").lower()
    mapped_version = VERSION_ALIASES.get(requested_version, requested_version)

    if mapped_version == DatasetVersion.VERSION_TYPE_ORIGINAL:
        return _read_file_field_to_dataframe(dataset.file)

    version = None
    if mapped_version in {
        DatasetVersion.VERSION_TYPE_CLEANED,
        DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED,
        DatasetVersion.VERSION_TYPE_ML_READY,
    }:
        version = _get_latest_version(dataset, mapped_version)
    else:
        version = _get_latest_version(dataset)

    if version and version.file:
        return _read_file_field_to_dataframe(version.file)

    return _read_file_field_to_dataframe(dataset.file)


def load_eda_comparison_dataframes(dataset_id, before_version="raw", after_version="cleaned"):
    df_before = load_eda_dataframe(dataset_id, dataset_version=before_version)
    df_after = load_eda_dataframe(dataset_id, dataset_version=after_version)
    return df_before, df_after


def _profile_from_storage(dataset):
    try:
        profile_json = dataset.profile.profile_json
        if isinstance(profile_json, dict) and profile_json:
            return profile_json
    except DatasetProfile.DoesNotExist:
        pass

    return {}


def _normalise_profile_groups(profile_json):
    """
    Support both profile shapes currently present in the project:
    1. apps.datasets.services.build_dataset_profile:
       row_count, numeric_columns, categorical_columns, columns, ...
    2. apps.profiling.services.build_dataset_profile_response:
       summary, numeric_columns, categorical_columns, columns, ...
    """
    if not isinstance(profile_json, dict):
        profile_json = {}

    def as_list(key):
        return [str(col) for col in profile_json.get(key, []) if col is not None]

    groups = {
        "numeric_columns": as_list("numeric_columns"),
        "categorical_columns": as_list("categorical_columns"),
        "datetime_columns": as_list("datetime_columns"),
        "boolean_columns": as_list("boolean_columns"),
        "text_columns": as_list("text_columns"),
        "id_like_columns": as_list("id_like_columns"),
        "constant_columns": as_list("constant_columns"),
        "high_cardinality_columns": as_list("high_cardinality_columns"),
        "possible_target_columns": as_list("possible_target_columns"),
        "review_columns": as_list("review_columns"),
    }

    # Fallback from column-level metadata when bucket arrays are missing.
    columns = profile_json.get("columns") or profile_json.get("column_profiles") or []
    if columns and not any(groups[key] for key in ["numeric_columns", "categorical_columns", "datetime_columns", "boolean_columns", "text_columns", "id_like_columns"]):
        for column in columns:
            if not isinstance(column, dict):
                continue
            name = str(column.get("column_name") or column.get("name") or "")
            detected_type = str(column.get("detected_type") or column.get("type") or "").lower()
            if not name:
                continue
            if detected_type == "numeric":
                groups["numeric_columns"].append(name)
            elif detected_type in {"categorical", "categorical_numeric", "numeric_categorical"}:
                groups["categorical_columns"].append(name)
            elif detected_type == "datetime":
                groups["datetime_columns"].append(name)
            elif detected_type == "boolean":
                groups["boolean_columns"].append(name)
            elif detected_type in {"id", "id_like"}:
                groups["id_like_columns"].append(name)
            elif detected_type == "high_cardinality_categorical":
                groups["high_cardinality_columns"].append(name)
            else:
                groups["text_columns"].append(name)

    return groups


def load_profile_column_groups(dataset_id, df=None):
    dataset = _get_dataset(dataset_id)
    profile_json = _profile_from_storage(dataset)
    return _normalise_profile_groups(profile_json)


def get_eda_column_groups(df, dataset_id=None):
    """
    Return column groups for EDA without creating a separate detection system.

    If dataset_id is supplied, this reuses the stored profile. Without a
    dataset_id, EDA intentionally returns empty groups rather than detecting
    column types again.
    """
    if dataset_id is not None:
        return load_profile_column_groups(dataset_id, df=df)

    return _normalise_profile_groups({})


def validate_eda_columns(df, columns):
    requested_columns = [str(column) for column in (columns or []) if column is not None]
    available = {str(column): column for column in df.columns}
    valid_columns = [column for column in requested_columns if column in available]
    invalid_columns = [column for column in requested_columns if column not in available]

    return {
        "valid_columns": valid_columns,
        "invalid_columns": invalid_columns,
    }


def sample_eda_dataframe(df, sample_size=None, random_state=42):
    if sample_size in [None, "", "auto", "Auto"]:
        return df

    if isinstance(sample_size, str):
        cleaned = sample_size.lower().replace("rows", "").replace(",", "").strip()
        if cleaned in {"full dataset", "full"}:
            return df
        try:
            sample_size = int(cleaned)
        except (TypeError, ValueError):
            return df

    try:
        sample_size = int(sample_size)
    except (TypeError, ValueError):
        return df

    if sample_size <= 0 or sample_size >= len(df):
        return df

    return df.sample(n=sample_size, random_state=random_state)


def build_eda_chart_payload(title, chart_type, rows, columns=None, metadata=None):
    return make_json_safe({
        "ok": True,
        "title": title,
        "chart_type": chart_type,
        "columns": columns or [],
        "rows": rows or [],
        "metadata": metadata or {},
    })


def build_eda_image_payload(title, image_path=None, image_base64=None, metadata=None):
    return make_json_safe({
        "ok": True,
        "title": title,
        "chart_type": "image",
        "image_path": image_path,
        "image_base64": image_base64,
        "metadata": metadata or {},
    })


def build_eda_error_payload(message, details=None):
    return {
        "ok": False,
        "message": message,
        "details": details or {},
    }
