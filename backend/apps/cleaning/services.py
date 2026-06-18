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

from apps.datasets.models import DatasetVersion
from apps.datasets.services import make_json_safe, read_dataset_file
from apps.profiling.services import build_dataset_profile_response, detect_column_type


def load_dataset_dataframe(dataset):
    """
    Load the most recent dataset file for cleaning reports and transformations.
    If one or more DatasetVersion objects exist for this dataset, use the latest
    version file so reports reflect the current working dataset state.
    Otherwise fall back to the original uploaded dataset file.
    """
    latest_version = (
        DatasetVersion.objects.filter(dataset=dataset)
        .order_by("-version_number")
        .first()
    )

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


def analyze_outliers(df):
    """
    Detect outliers only for columns detected as true numeric by profiling logic.
    Uses IQR method for Day 7 cleaning report.
    This only detects outliers; it does not remove them.
    """
    outlier_report = []
    row_count = df.shape[0]

    for column in df.columns:
        series = df[column]
        detected_type = detect_column_type(column, series, row_count)

        if detected_type != "numeric":
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
            outlier_report.append(
                {
                    "column_name": str(column),
                    "detected_type": detected_type,
                    "method": "IQR",
                    "lower_bound": round(float(lower_bound), 2),
                    "upper_bound": round(float(upper_bound), 2),
                    "outlier_count": outlier_count,
                    "outlier_percentage": round(float(outlier_percentage), 2),
                    "issue_code": "OUTLIERS_DETECTED",
                    "recommended_action_code": "REVIEW_OUTLIER_HANDLING",
                    "safety": "Review",
                    "requires_user_confirmation": True,
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
        suggested_method = "manual_review"

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


def analyze_imputation_options(profile, missing_report):
    """
    Build selectable imputation options for all columns with missing values.
    """
    profile_map = {
        column.get("column_name"): column
        for column in profile.get("columns", [])
    }

    options = []

    for missing_item in missing_report:
        column_name = missing_item.get("column_name")
        profile_column = profile_map.get(column_name, {"column_name": column_name})
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


COLUMN_OUTLIER_METHOD_OPTIONS = [
    "iqr_detection",
    "iqr_capping",
    "zscore_detection",
    "zscore_removal",
    "modified_zscore_detection",
    "percentile_capping",
    "winsorization",
    "log_transform",
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
        suggested_method = "review_only"
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

    elif method == "log_transform":
        if (series.dropna() <= 0).any():
            raise ValueError("log_transform requires positive values only.")

        output_df[column_name] = np.log1p(series)

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


def generate_recommendations(report):
    """
    Generate structured recommendation codes for frontend cards.
    Do not hardcode AI explanations here.
    Later AI Assistant module can convert issue_code/action_code into natural language.
    """
    recommendations = []

    def add_recommendation(item, prefix, issue_code, action_code, confidence, safety):
        column_name = item.get("column_name") or "dataset"
        recommendations.append(
            {
                "id": f"{prefix}-{column_name}",
                "column_name": column_name,
                "issue": issue_code,
                "suggested_action": action_code,
                "issue_code": issue_code,
                "recommended_action_code": action_code,
                "ai_explanation": "",
                "confidence": confidence,
                "safety": safety,
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
    outliers = analyze_outliers(df)

    report = {
        "missing_values": missing_values,
        "imputation_options": analyze_imputation_options(profile, missing_values),
        "duplicates": analyze_duplicates(df),
        "outliers": outliers,
        "column_outlier_options": analyze_column_outlier_options(profile, outliers),
        "dataset_anomaly": suggest_dataset_anomaly_method(profile),
        "datatype_issues": analyze_data_types(profile),
        "cardinality_issues": analyze_cardinality(profile),
        "constant_features": analyze_constant_features(profile),
        "datetime_columns": analyze_datetime_columns(profile),
        "zipcode_issues": analyze_zipcode_issues(profile),
    }

    report["recommendations"] = generate_recommendations(report)

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


def save_cleaned_dataset_version(dataset, output_df, transformation_log):
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

    return DatasetVersion.objects.create(
        dataset=dataset,
        version_number=version_number,
        file=cleaned_file,
        is_cleaned=True,
        transformation_log={
            "actions": transformation_log,
            "original_shape": getattr(dataset, "shape", None),
            "cleaned_shape": list(output_df.shape),
        },
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
        transformation_log={
            "action": "rollback",
            "rolled_back_version_id": source_version.id,
            "rolled_back_version_number": source_version.version_number,
        },
    )
