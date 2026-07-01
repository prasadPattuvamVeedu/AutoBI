"""
AutoBI Day 7 Cleaning Services

Manual pandas/data-intelligence implementation file.
Rules:
- Profiling is responsible for column meaning/type detection.
- Cleaning reuses profiling output where possible.
- AI Assistant explanations should be generated later by AI module, not hardcoded here.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.stats.mstats import winsorize

from django.core.files.base import ContentFile

from apps.datasets.models import DatasetProfile, DatasetVersion
from apps.datasets.services import build_preview, make_json_safe, read_dataset_file
from apps.profiling.services import build_dataset_profile_response, detect_column_type


def get_latest_cleaned_dataset_version(dataset):
    return (
        DatasetVersion.objects.filter(
            dataset=dataset,
            is_cleaned=True,
            version_type=DatasetVersion.VERSION_TYPE_CLEANED,
            is_active=True,
            file__isnull=False,
        )
        .exclude(file="")
        .order_by("-version_number")
        .first()
    )


def load_dataset_dataframe(dataset):
    """
    Load the active cleaned working file for cleaning reports and transformations.
    If one or more cleaned DatasetVersion objects exist for this dataset, use the
    latest cleaned file so each new approved action builds on the current working
    dataset state.
    Otherwise fall back to the original uploaded dataset file.
    """
    latest_version = get_latest_cleaned_dataset_version(dataset)

    if latest_version and latest_version.file:
        with latest_version.file.open("rb") as f:
            return read_dataset_file(f)

    if not dataset.file:
        raise ValueError("Dataset does not have an associated file.")

    with dataset.file.open("rb") as f:
        return read_dataset_file(f)


def analyze_missing_values(df):
    """
    Analyze missing values in the dataset.
    """
    missing_report = []
    total_rows = len(df)

    for column in df.columns:
        missing_count = int(df[column].isnull().sum())
        missing_percentage = (
            (missing_count / total_rows) * 100 if total_rows > 0 else 0
        )

        if missing_count > 0:
            missing_report.append(
                {
                    "column_name": str(column),
                    "missing_count": missing_count,
                    "missing_percentage": round(float(missing_percentage), 2),
                }
            )

    return missing_report


def analyze_duplicates(df):
    """
    Analyze duplicate row count and duplicate percentage.
    """
    duplicate_count = int(df.duplicated().sum())
    total_rows = df.shape[0]

    duplicate_percentage = (
        round(float((duplicate_count / total_rows) * 100), 2)
        if total_rows > 0
        else 0
    )

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
    }


def _safe_float(value, digits=2):
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _normalize_schema_type(value):
    if value is None:
        return ""
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


OUTLIER_ANALYSIS_ALLOWED_SCHEMA_TYPES = {
    "continuous numeric",
    "numeric",
    "numerical",
    "float",
    "decimal",
    "currency / amount",
    "currency",
    "amount",
    "percentage",
    "ratio",
}

OUTLIER_ANALYSIS_BLOCKED_SCHEMA_TYPES = {
    "identifier",
    "id",
    "id like",
    "year",
    "month",
    "day",
    "date",
    "datetime",
    "time",
    "binary",
    "boolean",
    "categorical",
    "nominal",
    "ordinal",
    "ordinal / numeric category",
    "numeric category",
    "postal code",
    "zip code",
    "zipcode",
    "latitude",
    "longitude",
    "text",
    "review text",
    "ignore",
}


def _get_dataset_schema_metadata(dataset):
    """
    Read the user-approved Schema metadata saved from the Schema page.

    This is used by Cleaning so columns approved as Year, Identifier, Binary,
    Ordinal/Numeric Category, Postal Code, etc. are not treated as continuous
    numeric outlier columns just because their raw pandas dtype is numeric.
    """
    if dataset is None:
        return {}

    profile = DatasetProfile.objects.filter(dataset=dataset).first()
    if not profile or not isinstance(profile.profile_json, dict):
        return {}

    profile_json = profile.profile_json or {}
    semantic_schema = profile_json.get("semantic_schema") or profile_json.get("active_schema_metadata") or {}
    metadata = {}

    if isinstance(semantic_schema, dict):
        for column_name, item in semantic_schema.items():
            if isinstance(item, dict):
                metadata[str(column_name)] = item
            else:
                metadata[str(column_name)] = {"semantic_type": item}

    for row in profile_json.get("columns") or profile_json.get("column_profiles") or []:
        if not isinstance(row, dict):
            continue
        column_name = row.get("column_name")
        if not column_name:
            continue
        metadata.setdefault(str(column_name), {})
        metadata[str(column_name)].update(
            {
                key: row.get(key)
                for key in [
                    "approved_semantic_type",
                    "ai_semantic_type",
                    "ai_data_category",
                    "detected_type",
                    "semantic_status",
                    "approved_raw_dtype",
                    "ai_raw_dtype",
                    "raw_dtype",
                    "raw_type",
                    "raw_dtype_status",
                ]
                if row.get(key) is not None
            }
        )

    return metadata


def _resolve_schema_type_for_outlier(column_name, detected_type, schema_metadata):
    item = (schema_metadata or {}).get(str(column_name)) or {}
    if not isinstance(item, dict):
        item = {"semantic_type": item}

    schema_type = (
        item.get("semantic_type")
        or item.get("approved_semantic_type")
        or item.get("ai_semantic_type")
        or item.get("ai_data_category")
        or item.get("approved_raw_dtype")
        or item.get("ai_raw_dtype")
        or item.get("raw_dtype")
        or item.get("raw_type")
        or item.get("detected_type")
        or detected_type
    )

    normalized = _normalize_schema_type(schema_type)
    return str(schema_type or detected_type or ""), normalized




def _resolve_schema_type_for_cleaning(column_name, detected_type, schema_metadata):
    """Resolve the effective type used by Cleaning recommendations.

    User-approved Schema metadata wins over pandas/raw dtype. This prevents
    columns such as YrSold, MSSubClass, OverallQual, and Id from being treated
    as normal continuous numeric columns during imputation and assistant cards.
    """
    schema_type, normalized = _resolve_schema_type_for_outlier(column_name, detected_type, schema_metadata)

    if normalized in {"identifier", "id", "id like", "ignore"}:
        return "id_like", schema_type
    if normalized in {"categorical", "category", "high cardinality categorical", "nominal", "postal code", "zip code", "zipcode", "city", "state", "country", "string"}:
        return "categorical", schema_type
    if normalized in {"binary", "boolean", "bool"}:
        return "categorical", schema_type
    if normalized in {"ordinal", "ordinal / numeric category", "numeric category", "discrete numeric"}:
        return "numeric_categorical", schema_type
    if normalized in {"year", "month", "day", "date", "datetime", "datetime64[ns]", "time", "duration"}:
        return "datetime", schema_type
    if normalized in {"text", "review text", "email", "phone", "url", "product code", "invoice number", "transaction id", "json", "mixed"}:
        return "text", schema_type
    if normalized in {"continuous numeric", "numeric", "numerical", "float", "float64", "integer", "int64", "decimal", "currency / amount", "currency", "amount", "percentage", "ratio", "latitude", "longitude"}:
        return "numeric", schema_type

    return detected_type or "review", schema_type

def _should_run_outlier_analysis(column_name, detected_type, schema_metadata):
    """
    Use approved Schema type first. This prevents Year/ID/code columns from
    being flagged as numeric outlier columns.
    """
    schema_type, normalized = _resolve_schema_type_for_outlier(
        column_name,
        detected_type,
        schema_metadata,
    )

    if normalized in OUTLIER_ANALYSIS_BLOCKED_SCHEMA_TYPES:
        return False, schema_type

    if normalized in OUTLIER_ANALYSIS_ALLOWED_SCHEMA_TYPES:
        return True, schema_type

    # If the user has not approved schema metadata yet, keep the old behavior.
    return detected_type == "numeric", schema_type


def build_outlier_column_statistics(series):
    """
    Build method-independent numeric statistics for the Outlier Analysis UI.

    Manual implementation note for the user:
    - Shapiro-Wilk / normality is intentionally left as a manual hook.
    - Implement it here later and return shapiro_p_value + normality_label.
    - Recommended manual logic:
        sample = numeric_series.dropna()
        if len(sample) > 5000: sample = sample.sample(5000, random_state=42)
        statistic, p_value = manual_shapiro_wilk(sample)
        normality_label = "Normal" if p_value >= 0.05 else "Not Normal"
    """
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    count = int(numeric_series.count())

    if count == 0:
        return {
            "min_value": None,
            "max_value": None,
            "mean": None,
            "median": None,
            "std": None,
            "variance": None,
            "skewness": None,
            "kurtosis": None,
            "normality_label": "Not tested",
            "shapiro_p_value": None,
            "shapiro_manual_required": True,
            "lowest_values": [],
            "highest_values": [],
            "distribution_label": "No numeric data",
        }

    skewness = numeric_series.skew() if count >= 3 else None
    kurtosis = numeric_series.kurtosis() if count >= 4 else None

    if skewness is None or pd.isna(skewness):
        distribution_label = "Unknown"
    elif skewness >= 1:
        distribution_label = "Right Skewed"
    elif skewness >= 0.5:
        distribution_label = "Slight Right Skew"
    elif skewness <= -1:
        distribution_label = "Left Skewed"
    elif skewness <= -0.5:
        distribution_label = "Slight Left Skew"
    else:
        distribution_label = "Nearly Symmetric"

    lowest_values = numeric_series.sort_values().head(5).tolist()
    highest_values = numeric_series.sort_values(ascending=False).head(5).tolist()

    return {
        "min_value": _safe_float(numeric_series.min()),
        "max_value": _safe_float(numeric_series.max()),
        "mean": _safe_float(numeric_series.mean()),
        "median": _safe_float(numeric_series.median()),
        "std": _safe_float(numeric_series.std()),
        "variance": _safe_float(numeric_series.var()),
        "skewness": _safe_float(skewness),
        "kurtosis": _safe_float(kurtosis),
        "normality_label": "Manual Shapiro pending",
        "shapiro_p_value": None,
        "shapiro_manual_required": True,
        "lowest_values": [_safe_float(v) for v in lowest_values],
        "highest_values": [_safe_float(v) for v in highest_values],
        "distribution_label": distribution_label,
    }


def recommend_outlier_action(stats, outlier_percentage):
    """
    Recommend the next action without applying transformations automatically.
    Transformation actions are intended to be forwarded to Feature Engineering.
    Row deletion/capping actions remain Cleaning actions.
    """
    skewness = stats.get("skewness")
    kurtosis = stats.get("kurtosis")
    min_value = stats.get("min_value")

    abs_skew = abs(float(skewness)) if skewness is not None else 0
    kurt = float(kurtosis) if kurtosis is not None else 0
    outlier_pct = float(outlier_percentage or 0)

    if outlier_pct < 1 and abs_skew < 0.5 and kurt < 3:
        return {
            "recommended_method": "keep_review",
            "recommended_action_label": "Keep / Review",
            "recommended_destination": "Cleaning",
            "recommendation_reason": "Low outlier percentage and almost symmetric distribution. Review only; no automatic deletion needed.",
            "alternative_methods": ["zscore_detection", "iqr_detection"],
            "avoid_methods": ["delete_outlier_rows"],
        }

    if skewness is not None and float(skewness) > 1:
        if min_value is not None and float(min_value) > 0:
            method = "log_transform"
            label = "Forward Log Transform"
        elif min_value is not None and float(min_value) >= 0:
            method = "sqrt_transform"
            label = "Forward Square Root Transform"
        else:
            method = "yeojohnson_transform"
            label = "Forward Yeo-Johnson Transform"
        return {
            "recommended_method": method,
            "recommended_action_label": label,
            "recommended_destination": "Feature Engineering",
            "recommendation_reason": "Right-skewed distribution with outliers. Prefer transformation so information is preserved for modelling.",
            "alternative_methods": ["winsorization", "percentile_capping", "boxcox_transform"],
            "avoid_methods": ["delete_outlier_rows"],
        }

    if skewness is not None and float(skewness) < -1:
        return {
            "recommended_method": "yeojohnson_transform",
            "recommended_action_label": "Forward Yeo-Johnson Transform",
            "recommended_destination": "Feature Engineering",
            "recommendation_reason": "Left-skewed distribution. Use a transformation that can handle zero or negative values.",
            "alternative_methods": ["winsorization", "percentile_capping"],
            "avoid_methods": ["delete_outlier_rows"],
        }

    if outlier_pct > 10 and kurt >= 3:
        return {
            "recommended_method": "winsorization",
            "recommended_action_label": "Winsorize / Cap",
            "recommended_destination": "Cleaning",
            "recommendation_reason": "High outlier share and heavy tails. Capping is safer than deleting many rows.",
            "alternative_methods": ["percentile_capping", "modified_zscore_detection"],
            "avoid_methods": ["delete_outlier_rows"],
        }

    return {
        "recommended_method": "review_extreme_values",
        "recommended_action_label": "Review Extremes",
        "recommended_destination": "Cleaning",
        "recommendation_reason": "Outliers are present, but distribution statistics do not strongly justify automatic transformation or deletion. Confirm using min/max and extreme values.",
        "alternative_methods": ["iqr_detection", "zscore_detection", "winsorization"],
        "avoid_methods": ["delete_outlier_rows_without_review"],
    }


def analyze_outliers(df, schema_metadata=None):
    """
    Detect outliers only for columns detected as true numeric by profiling logic.
    Uses IQR method for Day 7 cleaning report.
    This only detects outliers; it does not remove them.
    """
    outlier_report = []
    row_count = df.shape[0]

    schema_metadata = schema_metadata or {}

    for column in df.columns:
        series = df[column]
        detected_type = detect_column_type(column, series, row_count)
        schema_type, normalized_schema_type = _resolve_schema_type_for_outlier(
            column,
            detected_type,
            schema_metadata,
        )
        should_analyze, schema_type = _should_run_outlier_analysis(
            column,
            detected_type,
            schema_metadata,
        )

        if not should_analyze:
            continue

        numeric_series = pd.to_numeric(series, errors="coerce").dropna()

        if numeric_series.empty:
            continue

        q1 = numeric_series.quantile(0.25)
        q3 = numeric_series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = numeric_series[
            (numeric_series < lower_bound) | (numeric_series > upper_bound)
        ]
        outlier_count = int(outliers.count())
        outlier_percentage = (
            (outlier_count / row_count) * 100 if row_count > 0 else 0
        )

        if outlier_count > 0:
            stats = build_outlier_column_statistics(series)
            recommendation = recommend_outlier_action(stats, outlier_percentage)
            outlier_report.append(
                {
                    "column_name": str(column),
                    "detected_type": detected_type,
                    "schema_type": schema_type,
                    "approved_schema_type": schema_type,
                    "method": "IQR",
                    "lower_bound": round(float(lower_bound), 2),
                    "upper_bound": round(float(upper_bound), 2),
                    "outlier_count": outlier_count,
                    "outlier_percentage": round(float(outlier_percentage), 2),
                    "issue_code": "OUTLIERS_DETECTED",
                    "recommended_action_code": "REVIEW_OUTLIER_HANDLING",
                    "safety": "Review",
                    "requires_user_confirmation": True,
                    **stats,
                    **recommendation,
                }
            )

    return outlier_report


def analyze_data_types(profile):
    """
    Use profiling output to find columns needing datatype confirmation.
    Cleaning should not redetect column meaning again.
    """
    datatype_issues = []

    for column in profile.get("columns", []):
        detected_type = column.get("detected_type")

        if detected_type in ["id_like", "numeric_categorical", "review"]:
            datatype_issues.append(
                {
                    "column_name": column.get("column_name"),
                    "raw_dtype": column.get("raw_dtype"),
                    "detected_type": detected_type,
                    "issue_code": "DATATYPE_REVIEW_REQUIRED",
                    "recommended_action_code": "CONFIRM_COLUMN_TYPE",
                    "safety": "Review",
                    "requires_user_confirmation": True,
                    "options": [
                        "numeric",
                        "categorical",
                        "identifier",
                        "datetime",
                        "boolean",
                        "text",
                    ],
                }
            )

    return datatype_issues


def analyze_cardinality(profile):
    """
    Analyze high-cardinality columns using profiling output.
    """
    cardinality_report = []

    for column in profile.get("columns", []):
        column_name = column.get("column_name")
        detected_type = column.get("detected_type")
        unique_count = column.get("unique_count", 0) or 0
        unique_ratio = column.get("unique_ratio", 0) or 0

        if detected_type in ["categorical", "text", "numeric_categorical", "id_like"]:
            if unique_count > 50 or unique_ratio >= 0.5:
                cardinality_report.append(
                    {
                        "column_name": column_name,
                        "detected_type": detected_type,
                        "unique_count": int(unique_count),
                        "unique_ratio": round(float(unique_ratio), 4),
                        "issue_code": "HIGH_CARDINALITY",
                        "recommended_action_code": "REVIEW_ENCODING_OR_COLUMN_TYPE",
                        "safety": "Review",
                        "requires_user_confirmation": True,
                    }
                )

    return cardinality_report


def analyze_constant_features(profile):
    """
    Analyze constant columns using profiling output.
    """
    constant_report = []

    for column in profile.get("columns", []):
        unique_count = column.get("unique_count", 0) or 0

        if unique_count == 1:
            constant_report.append(
                {
                    "column_name": column.get("column_name"),
                    "detected_type": column.get("detected_type"),
                    "unique_count": 1,
                    "issue_code": "CONSTANT_COLUMN",
                    "recommended_action_code": "DROP_COLUMN_CANDIDATE",
                    "safety": "Safe",
                    "requires_user_confirmation": True,
                }
            )

    return constant_report


def analyze_datetime_columns(profile):
    """
    Analyze datetime columns using profiling output.
    """
    datetime_report = []

    for column in profile.get("columns", []):
        if column.get("detected_type") == "datetime":
            datetime_report.append(
                {
                    "column_name": column.get("column_name"),
                    "detected_type": "datetime",
                    "missing_percentage": column.get("missing_percentage", 0),
                    "unique_count": column.get("unique_count", 0),
                    "issue_code": "DATETIME_COLUMN_DETECTED",
                    "recommended_action_code": "REVIEW_DATETIME_FEATURE_EXTRACTION",
                    "safety": "Review",
                    "requires_user_confirmation": True,
                }
            )

    return datetime_report


IMPUTATION_METHODS = {
    "numeric": [
        "mean_imputation",
        "median_imputation",
        "mode_imputation",
        "fill_zero",
        "custom_value",
        "drop_rows",
        "drop_column",
        "missing_indicator",
        "forward_fill",
        "backward_fill",
        "interpolation_linear",
        "group_median_imputation",
        "group_mean_imputation",
        "rolling_mean_imputation",
        "knn_imputation",
        "model_based_imputation",
        "mice_imputation",
    ],
    "categorical": [
        "mode_imputation",
        "missing_category",
        "custom_value",
        "drop_rows",
        "drop_column",
        "missing_indicator",
        "rare_category_grouping",
        "group_mode_imputation",
        "predictive_imputation",
    ],
    "numeric_categorical": [
        "mode_imputation",
        "missing_category",
        "custom_value",
        "drop_rows",
        "drop_column",
        "missing_indicator",
    ],
    "text": [
        "empty_string",
        "missing_token",
        "custom_value",
        "drop_rows",
        "drop_column",
        "missing_indicator",
    ],
    "datetime": [
        "forward_fill",
        "backward_fill",
        "custom_value",
        "drop_rows",
        "drop_column",
        "missing_indicator",
    ],
    "id_like": [
        "keep_missing",
        "missing_indicator",
        "drop_rows",
        "drop_column",
    ],
    "review": [
        "manual_review",
        "custom_value",
        "drop_rows",
        "drop_column",
        "missing_indicator",
    ],
}


def suggest_imputation_method(profile_column, missing_item):
    """
    Decide the best default imputation method for one column.
    Return only method names/codes for UI selection.
    Do not apply the imputation here.
    """
    column_name = profile_column.get("column_name") or missing_item.get("column_name")
    detected_type = profile_column.get("detected_type", "review")
    missing_percentage = missing_item.get("missing_percentage", 0) or 0

    iqr = profile_column.get("iqr")
    std = profile_column.get("std")
    mean = profile_column.get("mean")
    median = profile_column.get("median")

    available_methods = IMPUTATION_METHODS.get(
        detected_type,
        IMPUTATION_METHODS["review"],
    )

    safety = "Review"
    suggested_method = "manual_review"
    confidence = "Medium"

    if missing_percentage == 0:
        suggested_method = "no_imputation_required"
        available_methods = ["no_imputation_required"]
        safety = "Safe"
        confidence = "High"

    elif missing_percentage >= 85:
        suggested_method = "missing_indicator"
        safety = "Review"
        confidence = "Medium"

    elif detected_type == "numeric":
        if mean is not None and median is not None and std is not None:
            difference = abs(float(mean) - float(median))
            reference = abs(float(median)) if float(median) != 0 else 1

            if difference / reference < 0.25:
                suggested_method = "mean_imputation"
                confidence = "Medium"
            else:
                suggested_method = "median_imputation"
                confidence = "Medium"

        elif iqr is not None and iqr > 0:
            suggested_method = "median_imputation"
            confidence = "Medium"

        else:
            suggested_method = "median_imputation"
            confidence = "Low"

    elif detected_type in ["categorical", "numeric_categorical"]:
        suggested_method = "mode_imputation"

    elif detected_type == "text":
        suggested_method = "missing_token"

    elif detected_type == "datetime":
        # Without a confirmed time order, missing_indicator is the safest
        # selectable option. The user can still choose forward/backward fill.
        suggested_method = "missing_indicator"

    elif detected_type == "id_like":
        suggested_method = "missing_indicator"

    else:
        suggested_method = "manual_review"

    if suggested_method not in available_methods:
        available_methods = [suggested_method] + available_methods

    return {
        "column_name": column_name,
        "detected_type": detected_type,
        "issue_code": "MISSING_VALUES",
        "ai_suggested_method": suggested_method,
        "available_methods": available_methods,
        "selected_method": suggested_method,
        "safety": safety,
        "confidence": confidence,
        "requires_user_confirmation": True,
        "missing_percentage": round(float(missing_percentage), 2),
    }


def analyze_imputation_options(profile, missing_report, schema_metadata=None):
    """
    Build selectable imputation options for all columns with missing values.

    Uses approved Schema stat type/AI stat type before raw pandas dtype so a
    numeric-looking column that was approved as Year, Binary, Identifier,
    Ordinal/Numeric Category or Categorical receives the correct imputation
    method family.
    """
    profile_map = {
        column.get("column_name"): column
        for column in profile.get("columns", [])
    }
    schema_metadata = schema_metadata or {}

    options = []

    for missing_item in missing_report:
        column_name = missing_item.get("column_name")
        profile_column = dict(profile_map.get(column_name, {"column_name": column_name}))
        effective_type, schema_type = _resolve_schema_type_for_cleaning(
            column_name,
            profile_column.get("detected_type", "review"),
            schema_metadata,
        )
        profile_column["original_detected_type"] = profile_column.get("detected_type")
        profile_column["detected_type"] = effective_type
        profile_column["schema_type"] = schema_type
        options.append(suggest_imputation_method(profile_column, missing_item))

    return options


def get_group_column_options(profile, target_column):
    """
    Return valid categorical/group columns for group-based imputation.
    """
    group_options = []

    for column in profile.get("columns", []):
        column_name = column.get("column_name")
        detected_type = column.get("detected_type")
        unique_count = column.get("unique_count", 0) or 0
        unique_ratio = column.get("unique_ratio", 0) or 0

        if column_name == target_column:
            continue

        if detected_type not in ["categorical", "numeric_categorical", "boolean"]:
            continue

        if unique_count < 2:
            continue

        if unique_count > 50 or unique_ratio >= 0.5:
            continue

        group_options.append(
            {
                "column_name": column_name,
                "detected_type": detected_type,
                "unique_count": int(unique_count),
                "unique_ratio": round(float(unique_ratio), 4),
            }
        )

    return group_options


def suggest_group_column(profile, target_column):
    """
    Suggest best group column from valid options.
    """
    options = get_group_column_options(profile, target_column)

    if not options:
        return None

    options = sorted(
        options,
        key=lambda item: (item["unique_count"], item["unique_ratio"]),
    )

    return options[0]["column_name"]


def apply_imputation_method(
    df,
    column_name,
    method,
    custom_value=None,
    group_column=None,
    time_column=None,
    params=None,
):
    """
    Apply the user-selected imputation method to a copied DataFrame.
    Never modifies the original dataset file.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    output_df = df.copy()
    params = params or {}

    log = {
        "action": "imputation",
        "column_name": column_name,
        "method": method,
        "custom_value": custom_value,
        "params": params,
    }

    if method == "mean_imputation":
        mean_value = output_df[column_name].mean()
        output_df[column_name] = output_df[column_name].fillna(mean_value)
        log["imputed_value"] = float(mean_value) if pd.notna(mean_value) else None

    elif method == "median_imputation":
        median_value = output_df[column_name].median()
        output_df[column_name] = output_df[column_name].fillna(median_value)
        log["imputed_value"] = float(median_value) if pd.notna(median_value) else None

    elif method == "mode_imputation":
        mode_value = output_df[column_name].mode(dropna=True)
        if mode_value.empty:
            raise ValueError(f"Column '{column_name}' has no mode value to impute.")
        mode_value = mode_value.iloc[0]
        output_df[column_name] = output_df[column_name].fillna(mode_value)
        log["imputed_value"] = str(mode_value)

    elif method == "fill_zero":
        output_df[column_name] = output_df[column_name].fillna(0)
        log["imputed_value"] = 0

    elif method == "custom_value":
        if custom_value is None:
            raise ValueError("Custom value must be provided for custom_value imputation.")
        output_df[column_name] = output_df[column_name].fillna(custom_value)
        log["imputed_value"] = custom_value

    elif method == "missing_category":
        output_df[column_name] = output_df[column_name].fillna("Missing")
        log["imputed_value"] = "Missing"

    elif method == "empty_string":
        output_df[column_name] = output_df[column_name].fillna("")
        log["imputed_value"] = ""

    elif method == "missing_token":
        output_df[column_name] = output_df[column_name].fillna("MISSING")
        log["imputed_value"] = "MISSING"

    elif method == "forward_fill":
        output_df[column_name] = output_df[column_name].ffill()
        log["imputed_value"] = "forward_fill"

    elif method == "backward_fill":
        output_df[column_name] = output_df[column_name].bfill()
        log["imputed_value"] = "backward_fill"

    elif method == "interpolation_linear":
        output_df[column_name] = pd.to_numeric(output_df[column_name], errors="coerce")
        output_df[column_name] = output_df[column_name].interpolate(method="linear")
        log["imputed_value"] = "linear_interpolation"

    elif method == "rolling_mean_imputation":
        window = params.get("window", 3)
        if time_column:
            if time_column not in output_df.columns:
                raise ValueError(f"Time column '{time_column}' does not exist.")
            output_df = output_df.sort_values(time_column)

        rolling_values = (
            pd.to_numeric(output_df[column_name], errors="coerce")
            .rolling(window=window, min_periods=1)
            .mean()
        )
        output_df[column_name] = output_df[column_name].fillna(rolling_values)
        global_mean = pd.to_numeric(output_df[column_name], errors="coerce").mean()
        output_df[column_name] = output_df[column_name].fillna(global_mean)
        log["window"] = window
        log["fallback"] = "global_mean"

    elif method == "drop_rows":
        rows_before = len(output_df)
        output_df = output_df.dropna(subset=[column_name])
        log["action"] = "drop_rows"
        log["rows_removed"] = rows_before - len(output_df)

    elif method == "drop_column":
        output_df = output_df.drop(columns=[column_name])
        log["action"] = "drop_column"
        log["column_removed"] = column_name

    elif method == "missing_indicator":
        indicator_column = f"{column_name}_was_missing"
        output_df[indicator_column] = output_df[column_name].isnull().astype(int)
        log["indicator_column"] = indicator_column

    elif method == "group_mean_imputation":
        if group_column is None:
            raise ValueError("group_column is required for group_mean_imputation.")

        if group_column not in output_df.columns:
            raise ValueError(f"Group column '{group_column}' does not exist in the DataFrame.")

        group_values = output_df.groupby(group_column)[column_name].transform("mean")
        output_df[column_name] = output_df[column_name].fillna(group_values)

        global_mean = output_df[column_name].mean()
        output_df[column_name] = output_df[column_name].fillna(global_mean)

        log["group_column"] = group_column
        log["fallback"] = "global_mean"

    elif method == "group_median_imputation":
        if group_column is None:
            raise ValueError("group_column is required for group_median_imputation.")

        if group_column not in output_df.columns:
            raise ValueError(f"Group column '{group_column}' does not exist in the DataFrame.")

        group_values = output_df.groupby(group_column)[column_name].transform("median")
        output_df[column_name] = output_df[column_name].fillna(group_values)

        global_median = output_df[column_name].median()
        output_df[column_name] = output_df[column_name].fillna(global_median)

        log["group_column"] = group_column
        log["fallback"] = "global_median"

    elif method == "group_mode_imputation":
        if group_column is None:
            raise ValueError("group_column is required for group_mode_imputation.")

        if group_column not in output_df.columns:
            raise ValueError(f"Group column '{group_column}' does not exist in the DataFrame.")

        def group_mode(series):
            mode_values = series.mode(dropna=True)
            return mode_values.iloc[0] if not mode_values.empty else None

        group_values = output_df.groupby(group_column)[column_name].transform(group_mode)
        output_df[column_name] = output_df[column_name].fillna(group_values)

        global_mode = output_df[column_name].mode(dropna=True)
        if not global_mode.empty:
            output_df[column_name] = output_df[column_name].fillna(global_mode.iloc[0])

        log["group_column"] = group_column
        log["fallback"] = "global_mode"

    else:
        raise ValueError(f"Imputation method '{method}' is not implemented.")

    return output_df, log


ZIPCODE_COLUMN_NAME_HINTS = [
    "zip",
    "zipcode",
    "zip_code",
    "postal",
    "postal_code",
    "postcode",
    "post_code",
    "pincode",
    "pin_code",
]


def suggest_zipcode_column_handling(profile_column):
    """
    Decide how zipcode/postal/pincode columns should be treated.
    """
    column_name = str(profile_column.get("column_name", ""))
    column_lower = column_name.lower()

    detected_type = profile_column.get("detected_type", "review")
    unique_count = profile_column.get("unique_count", 0) or 0
    unique_ratio = profile_column.get("unique_ratio", 0) or 0

    is_zipcode_like = any(
        hint in column_lower
        for hint in ZIPCODE_COLUMN_NAME_HINTS
    )

    if not is_zipcode_like:
        return None

    if unique_count > 50 or unique_ratio >= 0.5:
        selected_type = "identifier"
        recommended_action_code = "CONFIRM_IDENTIFIER_OR_CATEGORICAL"
    else:
        selected_type = "categorical"
        recommended_action_code = "CONFIRM_LOCATION_CATEGORY"

    return {
        "column_name": column_name,
        "detected_type": detected_type,
        "issue_code": "ZIPCODE_OR_POSTAL_CODE_DETECTED",
        "recommended_action_code": recommended_action_code,
        "available_type_options": [
            "identifier",
            "categorical",
            "text",
            "ignore",
        ],
        "selected_type": selected_type,
        "safety": "Review",
        "requires_user_confirmation": True,
        "do_not_use_for": [
            "outlier_detection",
            "scaling",
            "continuous_numeric_statistics",
        ],
    }


def analyze_zipcode_issues(profile):
    zipcode_issues = []

    for column in profile.get("columns", []):
        result = suggest_zipcode_column_handling(column)
        if result:
            zipcode_issues.append(result)

    return zipcode_issues


def calculate_outlier_preview(series, method, params=None):
    """
    Preview outlier count/percentage for the selected detection method.
    This function does not clean, cap, remove, or transform data.

    Supported detection methods:
    - iqr_detection
    - zscore_detection
    - modified_zscore_detection

    Transformation methods such as iqr_capping, winsorization,
    percentile_capping, log_transform, and removals return the same
    preview rule they are based on or raise a clear validation error.
    """
    params = params or {}

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    total_count = int(numeric_series.count())

    if total_count == 0:
        return {
            "method": method,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "lower_bound": None,
            "upper_bound": None,
            "threshold": None,
            "message": "No numeric values available for outlier preview.",
        }

    # Map transformation methods to their preview/detection base.
    preview_method = method
    if method in ["iqr_capping"]:
        preview_method = "iqr_detection"
    elif method in ["zscore_removal"]:
        preview_method = "zscore_detection"
    elif method in ["winsorization", "percentile_capping"]:
        preview_method = "percentile_detection"
    elif method in ["log_transform", "ignore"]:
        return {
            "method": method,
            "preview_method": method,
            "outlier_count": 0,
            "outlier_percentage": 0.0,
            "lower_bound": None,
            "upper_bound": None,
            "threshold": None,
            "message": "This method is a transformation/ignore option and does not have a direct outlier preview count.",
        }

    if preview_method == "iqr_detection":
        multiplier = params.get("iqr_multiplier", 1.5)

        q1 = numeric_series.quantile(0.25)
        q3 = numeric_series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            return {
                "method": method,
                "preview_method": preview_method,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "lower_bound": None,
                "upper_bound": None,
                "threshold": multiplier,
                "message": "IQR is zero or unavailable, so IQR outliers cannot be detected.",
            }

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outliers = numeric_series[
            (numeric_series < lower_bound) | (numeric_series > upper_bound)
        ]

        outlier_count = int(outliers.count())

        return {
            "method": method,
            "preview_method": preview_method,
            "outlier_count": outlier_count,
            "outlier_percentage": round(float((outlier_count / total_count) * 100), 2),
            "lower_bound": round(float(lower_bound), 2),
            "upper_bound": round(float(upper_bound), 2),
            "threshold": multiplier,
            "message": "Preview calculated using IQR detection.",
        }

    if preview_method == "zscore_detection":
        threshold = params.get("zscore_threshold", 3)

        if numeric_series.std() == 0 or pd.isna(numeric_series.std()):
            return {
                "method": method,
                "preview_method": preview_method,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "lower_bound": None,
                "upper_bound": None,
                "threshold": threshold,
                "message": "Standard deviation is zero or unavailable, so Z-score outliers cannot be detected.",
            }

        z_scores = np.abs(zscore(numeric_series, nan_policy="omit"))
        outlier_mask = z_scores > threshold
        outlier_count = int(np.sum(outlier_mask))

        return {
            "method": method,
            "preview_method": preview_method,
            "outlier_count": outlier_count,
            "outlier_percentage": round(float((outlier_count / total_count) * 100), 2),
            "lower_bound": None,
            "upper_bound": None,
            "threshold": threshold,
            "message": "Preview calculated using Z-score detection.",
        }

    if preview_method == "modified_zscore_detection":
        threshold = params.get("modified_zscore_threshold", 3.5)

        median = numeric_series.median()
        mad = np.median(np.abs(numeric_series - median))

        if mad == 0 or pd.isna(mad):
            return {
                "method": method,
                "preview_method": preview_method,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "lower_bound": None,
                "upper_bound": None,
                "threshold": threshold,
                "message": "MAD is zero or unavailable, so modified Z-score outliers cannot be detected.",
            }

        modified_z_scores = 0.6745 * (numeric_series - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        outlier_count = int(np.sum(outlier_mask))

        return {
            "method": method,
            "preview_method": preview_method,
            "outlier_count": outlier_count,
            "outlier_percentage": round(float((outlier_count / total_count) * 100), 2),
            "lower_bound": None,
            "upper_bound": None,
            "threshold": threshold,
            "message": "Preview calculated using modified Z-score detection.",
        }

    if preview_method == "percentile_detection":
        lower_pct = params.get("lower_percentile", 0.01)
        upper_pct = params.get("upper_percentile", 0.99)

        lower_bound = numeric_series.quantile(lower_pct)
        upper_bound = numeric_series.quantile(upper_pct)

        outliers = numeric_series[
            (numeric_series < lower_bound) | (numeric_series > upper_bound)
        ]

        outlier_count = int(outliers.count())

        return {
            "method": method,
            "preview_method": preview_method,
            "outlier_count": outlier_count,
            "outlier_percentage": round(float((outlier_count / total_count) * 100), 2),
            "lower_bound": round(float(lower_bound), 2),
            "upper_bound": round(float(upper_bound), 2),
            "threshold": {
                "lower_percentile": lower_pct,
                "upper_percentile": upper_pct,
            },
            "message": "Preview calculated using percentile bounds.",
        }

    raise ValueError(f"Unsupported outlier preview method: {method}")


def build_outlier_preview(dataset, column_name, method, params=None):
    """
    Build an outlier preview for one dataset column and selected method.
    This is intended for an API view used when the frontend dropdown changes.
    """
    df = load_dataset_dataframe(dataset)

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataset.")

    row_count = df.shape[0]
    detected_type = detect_column_type(column_name, df[column_name], row_count)

    schema_metadata = _get_dataset_schema_metadata(dataset)
    should_analyze, schema_type = _should_run_outlier_analysis(
        column_name,
        detected_type,
        schema_metadata,
    )

    if not should_analyze:
        return make_json_safe(
            {
                "column_name": column_name,
                "detected_type": detected_type,
                "schema_type": schema_type,
                "approved_schema_type": schema_type,
                "method": method,
                "outlier_count": 0,
                "outlier_percentage": 0.0,
                "message": "Outlier preview is only available for approved continuous numeric columns. This column is treated as schema type: " + str(schema_type),
            }
        )

    preview = calculate_outlier_preview(
        df[column_name],
        method,
        params=params,
    )

    stats = build_outlier_column_statistics(df[column_name])
    recommendation = recommend_outlier_action(stats, preview.get("outlier_percentage", 0))

    preview.update(
        {
            "column_name": column_name,
            "detected_type": detected_type,
            "schema_type": schema_type,
            "approved_schema_type": schema_type,
            **stats,
            **recommendation,
        }
    )

    return make_json_safe(preview)


COLUMN_OUTLIER_METHOD_OPTIONS = [
    "iqr_detection",
    "iqr_capping",
    "zscore_detection",
    "zscore_removal",
    "modified_zscore_detection",
    "percentile_capping",
    "winsorization",
    "log_transform",
    "sqrt_transform",
    "cube_root_transform",
    "boxcox_transform",
    "yeojohnson_transform",
    "review_extreme_values",
    "keep_review",
    "ignore",
]


def suggest_column_outlier_method(profile_column, outlier_item):
    """
    Suggest best column-wise outlier method for one numeric continuous feature.
    Does not apply/remove/cap outliers.
    """
    column_name = profile_column.get("column_name") or outlier_item.get("column_name")
    detected_type = profile_column.get("detected_type")

    skewness = profile_column.get("skewness")
    kurtosis = profile_column.get("kurtosis")
    outlier_percentage = outlier_item.get("outlier_percentage", 0) or 0

    available_methods = COLUMN_OUTLIER_METHOD_OPTIONS.copy()
    suggested_method = "iqr_detection"
    safety = "Review"
    confidence = "Medium"

    if detected_type != "numeric":
        available_methods = ["ignore"]
        suggested_method = "ignore"
        safety = "Safe"
        confidence = "High"

    elif outlier_percentage >= 30:
        # Very high outlier share means the column may be naturally skewed or
        # wrongly typed. Keep this as detection/review, not an unsupported
        # transformation method.
        suggested_method = "iqr_detection"
        safety = "Review"
        confidence = "Medium"

    elif skewness is not None and pd.notna(skewness):
        skewness_value = abs(float(skewness))

        if skewness_value < 0.5:
            suggested_method = "zscore_detection"
            confidence = "High"
        elif skewness_value < 1:
            suggested_method = "iqr_detection"
            confidence = "Medium"
        else:
            suggested_method = "iqr_detection"
            confidence = "High"

    elif kurtosis is not None and pd.notna(kurtosis) and float(kurtosis) > 3:
        suggested_method = "modified_zscore_detection"
        confidence = "Medium"

    else:
        suggested_method = "iqr_detection"
        confidence = "Medium"

    if suggested_method not in available_methods:
        available_methods = [suggested_method] + available_methods

    return {
        "column_name": column_name,
        "analysis_type": "column_outlier",
        "detected_type": detected_type,
        "issue_code": "OUTLIERS_DETECTED",
        "ai_suggested_method": suggested_method,
        "available_methods": available_methods,
        "selected_method": suggested_method,
        "safety": safety,
        "confidence": confidence,
        "requires_user_confirmation": True,
        "outlier_percentage": round(float(outlier_percentage), 2),
        "min_value": outlier_item.get("min_value"),
        "max_value": outlier_item.get("max_value"),
        "mean": outlier_item.get("mean"),
        "median": outlier_item.get("median"),
        "std": outlier_item.get("std"),
        "variance": outlier_item.get("variance"),
        "skewness": outlier_item.get("skewness"),
        "kurtosis": outlier_item.get("kurtosis"),
        "normality_label": outlier_item.get("normality_label"),
        "shapiro_p_value": outlier_item.get("shapiro_p_value"),
        "shapiro_manual_required": outlier_item.get("shapiro_manual_required", True),
        "distribution_label": outlier_item.get("distribution_label"),
        "lowest_values": outlier_item.get("lowest_values", []),
        "highest_values": outlier_item.get("highest_values", []),
        "recommended_method": outlier_item.get("recommended_method"),
        "recommended_action_label": outlier_item.get("recommended_action_label"),
        "recommended_destination": outlier_item.get("recommended_destination"),
        "recommendation_reason": outlier_item.get("recommendation_reason"),
        "alternative_methods": outlier_item.get("alternative_methods", []),
        "avoid_methods": outlier_item.get("avoid_methods", []),
        "preview_available": True,
        "preview_methods": [
            "iqr_detection",
            "zscore_detection",
            "modified_zscore_detection",
            "percentile_capping",
            "iqr_capping",
            "zscore_removal",
        ],
    }


def analyze_column_outlier_options(profile, outlier_report):
    profile_map = {
        column.get("column_name"): column
        for column in profile.get("columns", [])
    }

    options = []

    for outlier_item in outlier_report:
        column_name = outlier_item.get("column_name")
        profile_column = profile_map.get(column_name, {"column_name": column_name})
        options.append(suggest_column_outlier_method(profile_column, outlier_item))

    return options


def apply_column_outlier_method(df, column_name, method, params=None):
    """
    Apply selected column-wise outlier handling.
    Detection-only methods return unchanged output_df with a log.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found.")

    output_df = df.copy()
    params = params or {}

    log = {
        "action": "outlier_handling",
        "column_name": column_name,
        "method": method,
        "params": params,
    }

    series = pd.to_numeric(
        output_df[column_name],
        errors="coerce",
    )

    if series.dropna().empty:
        return output_df, log

    if method == "iqr_detection":
        return output_df, log

    elif method == "iqr_capping":
        multiplier = params.get("iqr_multiplier", 1.5)

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        output_df[column_name] = series.clip(lower, upper)

        log["lower_bound"] = float(lower)
        log["upper_bound"] = float(upper)

    elif method == "zscore_detection":
        return output_df, log

    elif method == "zscore_removal":
        threshold = params.get("zscore_threshold", 3)

        z_scores = np.abs(zscore(series, nan_policy="omit"))
        mask = z_scores <= threshold

        rows_before = len(output_df)
        output_df = output_df[mask | series.isna()]

        log["rows_removed"] = rows_before - len(output_df)
        log["threshold"] = threshold

    elif method == "modified_zscore_detection":
        return output_df, log

    elif method == "percentile_capping":
        lower_pct = params.get("lower_percentile", 0.01)
        upper_pct = params.get("upper_percentile", 0.99)

        lower = series.quantile(lower_pct)
        upper = series.quantile(upper_pct)

        output_df[column_name] = series.clip(lower, upper)

        log["lower_percentile"] = lower_pct
        log["upper_percentile"] = upper_pct
        log["lower_bound"] = float(lower)
        log["upper_bound"] = float(upper)

    elif method == "winsorization":
        lower_limit = params.get("lower_limit", 0.01)
        upper_limit = params.get("upper_limit", 0.01)

        output_df[column_name] = winsorize(
            series,
            limits=[lower_limit, upper_limit],
        )

        log["lower_limit"] = lower_limit
        log["upper_limit"] = upper_limit

    elif method in ["log_transform", "sqrt_transform", "cube_root_transform", "boxcox_transform", "yeojohnson_transform"]:
        # Do not transform columns inside Cleaning. These actions are forwarded
        # to Feature Engineering so the user can implement and approve the
        # transformation pipeline manually.
        log["action"] = "forward_to_feature_engineering"
        log["forward_to"] = "feature_engineering"
        log["manual_implementation_required"] = True
        log["message"] = (
            f"{method} is recommended for column '{column_name}', but was not applied in Cleaning. "
            "Create this transformation in Feature Engineering manually."
        )
        return output_df, log

    elif method in ["keep_review", "review_extreme_values"]:
        log["action"] = "review_only"
        log["message"] = "Review-only outlier action. Dataset was not changed."
        return output_df, log

    elif method == "ignore":
        return output_df, log

    else:
        raise ValueError(f"Unsupported outlier method: {method}")

    return output_df, log


DATASET_ANOMALY_METHOD_OPTIONS = [
    "isolation_forest",
    "mahalanobis_distance",
    "local_outlier_factor",
    "dbscan",
    "cooks_distance",
    "ignore",
]


def suggest_dataset_anomaly_method(profile):
    """
    Suggest dataset-level anomaly detection method using multiple numeric columns together.
    """
    numeric_columns = profile.get("numeric_columns", [])
    row_count = (
        profile.get("summary", {}).get("row_count")
        or profile.get("summary", {}).get("rows")
        or 0
    )
    feature_count = len(numeric_columns)

    if feature_count < 2:
        return None

    suggested_method = "isolation_forest"
    confidence = "Medium"

    if row_count and row_count > feature_count * 10 and feature_count <= 10:
        suggested_method = "mahalanobis_distance"
        confidence = "Medium"

    if feature_count > 10:
        suggested_method = "isolation_forest"
        confidence = "High"

    return {
        "analysis_type": "dataset_anomaly",
        "columns": numeric_columns,
        "issue_code": "MULTIVARIATE_ANOMALY_CHECK_AVAILABLE",
        "ai_suggested_method": suggested_method,
        "available_methods": DATASET_ANOMALY_METHOD_OPTIONS,
        "selected_method": suggested_method,
        "safety": "Review",
        "confidence": confidence,
        "requires_user_confirmation": True,
    }


def apply_dataset_anomaly_method(df, columns, method, params=None):
    """
    Apply user-selected dataset-level anomaly method on selected numeric features.
    Adds anomaly flag/score columns; does not remove rows automatically.
    """
    params = params or {}
    output_df = df.copy()

    log = {
        "action": "dataset_anomaly_detection",
        "columns": columns,
        "method": method,
        "params": params,
    }

    if not columns or len(columns) < 2:
        raise ValueError("At least two numeric columns are required.")

    for column in columns:
        if column not in output_df.columns:
            raise ValueError(f"Column '{column}' not found.")

    X = output_df[columns].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    if method == "ignore":
        return output_df, log

    elif method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        contamination = params.get("contamination", 0.05)

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
        )

        labels = model.fit_predict(X)
        scores = model.decision_function(X)

        output_df["anomaly_flag"] = (labels == -1).astype(int)
        output_df["anomaly_score"] = scores

        log["contamination"] = contamination
        log["anomaly_count"] = int(output_df["anomaly_flag"].sum())

    elif method == "mahalanobis_distance":
        threshold = params.get("threshold", 3.0)

        X_values = X.values
        mean_vector = np.mean(X_values, axis=0)
        covariance_matrix = np.cov(X_values, rowvar=False)
        inv_covariance_matrix = np.linalg.pinv(covariance_matrix)

        diff = X_values - mean_vector
        mahalanobis_distances = np.sqrt(
            np.sum(diff.dot(inv_covariance_matrix) * diff, axis=1)
        )

        output_df["anomaly_score"] = mahalanobis_distances
        output_df["anomaly_flag"] = (mahalanobis_distances > threshold).astype(int)

        log["threshold"] = threshold
        log["anomaly_count"] = int(output_df["anomaly_flag"].sum())

    elif method == "local_outlier_factor":
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import StandardScaler

        n_neighbors = params.get("n_neighbors", 20)
        contamination = params.get("contamination", 0.05)

        X_scaled = StandardScaler().fit_transform(X)

        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
        )

        labels = model.fit_predict(X_scaled)

        output_df["anomaly_flag"] = (labels == -1).astype(int)
        output_df["anomaly_score"] = model.negative_outlier_factor_

        log["n_neighbors"] = n_neighbors
        log["contamination"] = contamination
        log["anomaly_count"] = int(output_df["anomaly_flag"].sum())

    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)

        X_scaled = StandardScaler().fit_transform(X)

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
        )

        labels = model.fit_predict(X_scaled)

        output_df["anomaly_flag"] = (labels == -1).astype(int)
        output_df["cluster_label"] = labels

        log["eps"] = eps
        log["min_samples"] = min_samples
        log["anomaly_count"] = int(output_df["anomaly_flag"].sum())

    elif method == "cooks_distance":
        import statsmodels.api as sm

        target_column = params.get("target_column")

        if not target_column:
            raise ValueError("target_column is required for cooks_distance.")

        if target_column not in output_df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        X_cooks = output_df[columns].apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(output_df[target_column], errors="coerce")

        valid_mask = X_cooks.notna().all(axis=1) & y.notna()
        X_valid = X_cooks[valid_mask]
        y_valid = y[valid_mask]

        if len(X_valid) <= len(columns) + 1:
            raise ValueError("Not enough valid rows for Cook's Distance.")

        X_valid_const = sm.add_constant(X_valid)
        model = sm.OLS(y_valid, X_valid_const).fit()

        influence = model.get_influence()
        cooks_distance_values = influence.cooks_distance[0]

        threshold = params.get("threshold", 4 / len(X_valid))

        output_df["anomaly_score"] = np.nan
        output_df.loc[valid_mask, "anomaly_score"] = cooks_distance_values

        output_df["anomaly_flag"] = 0
        output_df.loc[valid_mask, "anomaly_flag"] = (
            cooks_distance_values > threshold
        ).astype(int)

        log["target_column"] = target_column
        log["threshold"] = threshold
        log["anomaly_count"] = int(output_df["anomaly_flag"].sum())

    else:
        raise ValueError(f"Unsupported dataset anomaly method: {method}")

    return output_df, log


def _profile_map_from_report(report):
    """Return profile columns by name when a report includes profiling context."""
    profile = report.get("profile") or {}
    return {
        column.get("column_name"): column
        for column in profile.get("columns", [])
        if column.get("column_name")
    }


def _compact_profile_stats(profile_column):
    """Small per-column evidence payload for the Cleaning Assistant UI/LLM."""
    if not isinstance(profile_column, dict):
        return {}

    keys = [
        "raw_dtype",
        "detected_type",
        "missing_count",
        "missing_percentage",
        "unique_count",
        "unique_ratio",
        "mean",
        "median",
        "mode",
        "min",
        "max",
        "std",
        "variance",
        "iqr",
        "skewness",
        "kurtosis",
        "outlier_count",
        "outlier_percentage",
    ]
    return {key: profile_column.get(key) for key in keys if profile_column.get(key) is not None}


def _build_cleaning_explanation(issue_code, item, method, profile_column):
    """Create deterministic evidence text; LLM can later rewrite it."""
    detected_type = item.get("detected_type") or profile_column.get("detected_type") or "unknown"
    missing_pct = item.get("missing_percentage")
    outlier_pct = item.get("outlier_percentage")

    if issue_code == "MISSING_VALUES":
        detail = f"{missing_pct}% missing" if missing_pct is not None else "missing values detected"
        return (
            f"Suggested {method} because {item.get('column_name')} is detected as {detected_type} "
            f"with {detail}. Review the sample rows and statistics before applying."
        )
    if issue_code == "OUTLIERS_DETECTED":
        detail = f"{outlier_pct}% outliers" if outlier_pct is not None else "outliers detected"
        skew = profile_column.get("skewness")
        skew_text = f" Skewness is {skew}." if skew is not None else ""
        return (
            f"Suggested {method} because {item.get('column_name')} is numeric with {detail}."
            f"{skew_text} Detection methods do not delete rows; capping/removal must be chosen deliberately."
        )
    return "Review this recommendation using the column type, sample rows, and profile statistics before applying."


def generate_recommendations(report):
    """
    Generate structured recommendation cards for the frontend Cleaning Assistant.
    Each row is enriched with the best selectable method, available methods,
    missing/outlier counts, and per-column profile statistics.
    """
    recommendations = []
    profile_map = _profile_map_from_report(report)
    imputation_map = {
        item.get("column_name"): item
        for item in report.get("imputation_options", [])
        if item.get("column_name")
    }
    outlier_option_map = {
        item.get("column_name"): item
        for item in report.get("column_outlier_options", [])
        if item.get("column_name")
    }

    def add_recommendation(item, prefix, issue_code, action_code, confidence, safety):
        column_name = item.get("column_name") or "dataset"
        profile_column = profile_map.get(column_name, {})
        method_payload = {}
        if issue_code == "MISSING_VALUES":
            method_payload = imputation_map.get(column_name, {})
        elif issue_code == "OUTLIERS_DETECTED":
            method_payload = outlier_option_map.get(column_name, {})

        selected_method = (
            method_payload.get("selected_method")
            or method_payload.get("ai_suggested_method")
            or item.get("selected_method")
        )
        available_methods = method_payload.get("available_methods") or item.get("available_methods") or []

        recommendations.append(
            {
                "id": f"{prefix}-{column_name}",
                "column_name": column_name,
                "issue": issue_code,
                "suggested_action": selected_method or action_code,
                "issue_code": issue_code,
                "recommended_action_code": action_code,
                "ai_suggested_method": selected_method,
                "selected_method": selected_method,
                "available_methods": available_methods,
                "detected_type": method_payload.get("detected_type") or item.get("detected_type") or profile_column.get("detected_type"),
                "schema_type": method_payload.get("schema_type") or item.get("schema_type"),
                "raw_dtype": profile_column.get("raw_dtype"),
                "missing_count": item.get("missing_count") or method_payload.get("missing_count"),
                "missing_percentage": item.get("missing_percentage") or method_payload.get("missing_percentage"),
                "outlier_count": item.get("outlier_count") or method_payload.get("outlier_count"),
                "outlier_percentage": item.get("outlier_percentage") or method_payload.get("outlier_percentage"),
                "lower_bound": item.get("lower_bound"),
                "upper_bound": item.get("upper_bound"),
                "profile_statistics": _compact_profile_stats(profile_column),
                "ai_explanation": _build_cleaning_explanation(issue_code, item, selected_method or action_code, profile_column),
                "confidence": method_payload.get("confidence", confidence),
                "safety": method_payload.get("safety", safety),
                "requires_user_confirmation": True,
            }
        )

    for item in report.get("missing_values", []):
        add_recommendation(
            item,
            "missing",
            "MISSING_VALUES",
            "REVIEW_IMPUTATION",
            "Medium",
            "Review",
        )

    duplicates = report.get("duplicates", {})
    if duplicates.get("duplicate_count", 0) > 0:
        recommendations.append(
            {
                "id": "duplicates-dataset",
                "column_name": "Dataset",
                "issue": "DUPLICATE_ROWS",
                "suggested_action": "REVIEW_DUPLICATE_REMOVAL",
                "issue_code": "DUPLICATE_ROWS",
                "recommended_action_code": "REVIEW_DUPLICATE_REMOVAL",
                "ai_explanation": "",
                "confidence": "High",
                "safety": "Safe",
                "requires_user_confirmation": True,
            }
        )

    for item in report.get("outliers", []):
        add_recommendation(
            item,
            "outlier",
            "OUTLIERS_DETECTED",
            "REVIEW_OUTLIER_HANDLING",
            "Medium",
            "Review",
        )

    for item in report.get("datatype_issues", []):
        add_recommendation(
            item,
            "datatype",
            "DATATYPE_REVIEW_REQUIRED",
            "CONFIRM_COLUMN_TYPE",
            "Medium",
            "Review",
        )

    for item in report.get("cardinality_issues", []):
        add_recommendation(
            item,
            "cardinality",
            "HIGH_CARDINALITY",
            "REVIEW_ENCODING_OR_COLUMN_TYPE",
            "Medium",
            "Review",
        )

    for item in report.get("constant_features", []):
        add_recommendation(
            item,
            "constant",
            "CONSTANT_COLUMN",
            "DROP_COLUMN_CANDIDATE",
            "High",
            "Safe",
        )

    for item in report.get("datetime_columns", []):
        add_recommendation(
            item,
            "datetime",
            "DATETIME_COLUMN_DETECTED",
            "REVIEW_DATETIME_FEATURE_EXTRACTION",
            "Medium",
            "Review",
        )

    for item in report.get("zipcode_issues", []):
        add_recommendation(
            item,
            "zipcode",
            "ZIPCODE_OR_POSTAL_CODE_DETECTED",
            item.get("recommended_action_code", "CONFIRM_IDENTIFIER_OR_CATEGORICAL"),
            "Medium",
            "Review",
        )

    dataset_anomaly = report.get("dataset_anomaly")
    if dataset_anomaly:
        recommendations.append(
            {
                "id": "dataset-anomaly",
                "column_name": "Dataset",
                "issue": "MULTIVARIATE_ANOMALY_CHECK_AVAILABLE",
                "suggested_action": "REVIEW_DATASET_ANOMALIES",
                "issue_code": "MULTIVARIATE_ANOMALY_CHECK_AVAILABLE",
                "recommended_action_code": "REVIEW_DATASET_ANOMALIES",
                "ai_explanation": "",
                "confidence": dataset_anomaly.get("confidence", "Medium"),
                "safety": "Review",
                "requires_user_confirmation": True,
            }
        )

    return recommendations


def build_cleaning_report(dataset):
    """
    Build complete Day 7 cleaning report.
    """
    df = load_dataset_dataframe(dataset)
    profile = build_dataset_profile_response(dataset, df)

    missing_values = analyze_missing_values(df)
    schema_metadata = _get_dataset_schema_metadata(dataset)
    outliers = analyze_outliers(df, schema_metadata=schema_metadata)

    report = {
        "missing_values": missing_values,
        "imputation_options": analyze_imputation_options(profile, missing_values, schema_metadata=schema_metadata),
        "duplicates": analyze_duplicates(df),
        "outliers": outliers,
        "column_outlier_options": analyze_column_outlier_options(profile, outliers),
        "dataset_anomaly": suggest_dataset_anomaly_method(profile),
        "datatype_issues": analyze_data_types(profile),
        "cardinality_issues": analyze_cardinality(profile),
        "constant_features": analyze_constant_features(profile),
        "datetime_columns": analyze_datetime_columns(profile),
        "zipcode_issues": analyze_zipcode_issues(profile),
        # Internal context used to enrich Cleaning Assistant cards.
        # It is removed before returning the API response.
        "profile": profile,
    }

    report["recommendations"] = generate_recommendations(report)

    for recommendation in report.get("recommendations", []):
        column_name = recommendation.get("column_name")
        if column_name in df.columns:
            sample_values = (
                df[column_name]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(6)
                .tolist()
            )
            recommendation["sample_values_from_preview"] = sample_values

    report.pop("profile", None)

    report["summary"] = {
        "missing_values": sum(
            item.get("missing_count", 0) for item in report.get("missing_values", [])
        ),
        "duplicate_rows": report.get("duplicates", {}).get("duplicate_count", 0),
        "outliers": sum(
            item.get("outlier_count", 0) for item in report.get("outliers", [])
        ),
        "datatype_issues": len(report.get("datatype_issues", [])),
        "recommendations": len(report.get("recommendations", [])),
        "columns": len(df.columns),
    }

    return make_json_safe(report)


def apply_transformations(dataset, transformations=None, apply_safe=False):
    """
    Apply user-selected cleaning transformations.
    Never modifies original dataset.file.
    Always saves output as a new DatasetVersion.
    """
    df = load_dataset_dataframe(dataset)
    output_df = df.copy()
    transformation_logs = []

    for action in transformations or []:
        issue_code = action.get("issue_code")
        column_name = action.get("column_name")
        selected_method = action.get("selected_method")

        if issue_code == "MISSING_VALUES":
            output_df, log = apply_imputation_method(
                output_df,
                column_name,
                selected_method,
                custom_value=action.get("custom_value"),
                group_column=action.get("group_column"),
                time_column=action.get("time_column"),
                params=action.get("params"),
            )
            transformation_logs.append(log)

        elif issue_code == "OUTLIERS_DETECTED":
            output_df, log = apply_column_outlier_method(
                output_df,
                column_name,
                selected_method,
                params=action.get("params"),
            )
            transformation_logs.append(log)

        elif issue_code == "MULTIVARIATE_ANOMALY_CHECK_AVAILABLE":
            output_df, log = apply_dataset_anomaly_method(
                output_df,
                action.get("columns"),
                selected_method,
                params=action.get("params"),
            )
            transformation_logs.append(log)

        else:
            transformation_logs.append(
                {
                    "action": "ignored_or_not_implemented",
                    "issue_code": issue_code,
                    "column_name": column_name,
                    "selected_method": selected_method,
                }
            )

    return save_cleaned_dataset_version(dataset, output_df, transformation_logs)


def apply_cleaning_actions(dataset, actions=None, apply_safe=False):
    """
    Compatibility wrapper for existing views.
    """
    return apply_transformations(
        dataset,
        transformations=actions,
        apply_safe=apply_safe,
    )


def _get_accumulated_cleaning_actions(version):
    if version is None:
        return []

    existing_log = version.transformation_log or {}
    existing_actions = (
        existing_log.get("cleaning_plan_json")
        or existing_log.get("actions")
        or []
    )
    return existing_actions if isinstance(existing_actions, list) else []


def save_cleaned_dataset_version(dataset, output_df, transformation_log):
    source_version = get_latest_cleaned_dataset_version(dataset)
    accumulated_actions = _get_accumulated_cleaning_actions(source_version)
    latest_actions = transformation_log or []
    merged_actions = accumulated_actions + latest_actions

    last_version = (
        DatasetVersion.objects
        .filter(dataset=dataset)
        .order_by("-version_number")
        .first()
    )

    version_number = (last_version.version_number + 1) if last_version else 1

    original_name = dataset.file.name if dataset.file else dataset.name or "dataset"
    extension = Path(original_name).suffix.lower() or ".csv"

    buffer = BytesIO()

    if extension in [".xlsx", ".xls"]:
        output_df.to_excel(buffer, index=False)
        output_extension = ".xlsx"
    else:
        output_df.to_csv(buffer, index=False)
        output_extension = ".csv"

    buffer.seek(0)

    safe_name = "".join(
        c if c.isalnum() or c in {"_", "-", "."} else "_"
        for c in (dataset.name or "dataset")
    )

    cleaned_file = ContentFile(
        buffer.read(),
        name=f"{safe_name}_cleaned_v{version_number}{output_extension}",
    )

    preview = build_preview(output_df, limit=20)

    return DatasetVersion.objects.create(
        dataset=dataset,
        version_number=version_number,
        file=cleaned_file,
        is_cleaned=True,
        version_type=DatasetVersion.VERSION_TYPE_CLEANED,
        parent_version=source_version,
        preview_rows=make_json_safe(preview.get("rows", [])),
        columns=make_json_safe(preview.get("columns", [])),
        transformation_plan_json=make_json_safe({
            "cleaning_plan_json": merged_actions,
        }),
        is_active=True,
        transformation_log=make_json_safe({
            "actions": merged_actions,
            "cleaning_plan_json": merged_actions,
            "latest_actions": latest_actions,
            "source_version_id": source_version.id if source_version else None,
            "source_version_number": source_version.version_number if source_version else None,
            "active_version_type": "cleaned",
            "original_shape": getattr(dataset, "shape", None),
            "cleaned_shape": list(output_df.shape),
        }),
    )


def rollback_to_version(dataset, version_id):
    """
    Rollback creates a new DatasetVersion from an existing version file.
    This does not decide cleaning logic.
    """
    source_version = DatasetVersion.objects.filter(dataset=dataset, id=version_id).first()

    if source_version is None:
        raise ValueError("Rollback version not found for this dataset.")

    if not source_version.file:
        raise ValueError("Rollback version does not have a stored file.")

    source_version.file.open("rb")
    try:
        content = source_version.file.read()
    finally:
        source_version.file.close()

    last_version = (
        DatasetVersion.objects.filter(dataset=dataset)
        .order_by("-version_number")
        .first()
    )
    version_number = (last_version.version_number + 1) if last_version else 1

    safe_name = "".join(
        c if c.isalnum() or c in {"_", "-", "."} else "_"
        for c in (dataset.name or "dataset")
    )
    extension = Path(source_version.file.name).suffix or ".csv"
    restored_file = ContentFile(
        content,
        name=f"{safe_name}_rollback_v{version_number}{extension}",
    )

    return DatasetVersion.objects.create(
        dataset=dataset,
        version_number=version_number,
        file=restored_file,
        is_cleaned=source_version.is_cleaned,
        version_type=source_version.version_type,
        parent_version=source_version,
        preview_rows=source_version.preview_rows,
        columns=source_version.columns,
        transformation_plan_json=source_version.transformation_plan_json,
        is_active=True,
        transformation_log={
            "action": "rollback",
            "rolled_back_version_id": source_version.id,
            "rolled_back_version_number": source_version.version_number,
        },
    )
