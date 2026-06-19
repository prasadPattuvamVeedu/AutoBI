from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


SUPPORTED_DATASET_EXTENSIONS = {".csv", ".xlsx"}


def get_dataset_file_extension(file_name):
    return Path(file_name).suffix.lower()


def validate_dataset_file(file):
    if not file:
        return "A dataset file is required."

    extension = get_dataset_file_extension(file.name)
    if extension not in SUPPORTED_DATASET_EXTENSIONS:
        return "Only CSV and XLSX files are supported."

    if file.size == 0:
        return "The uploaded file is empty."

    return None


def read_dataset_file(file):
    extension = get_dataset_file_extension(file.name)
    file.seek(0)

    try:
        if extension == ".csv":
            return pd.read_csv(file)
        if extension == ".xlsx":
            return pd.read_excel(file)
    except (EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("The uploaded file could not be read as a valid dataset.") from exc

    raise ValueError("Only CSV and XLSX files are supported.")


def get_dataset_shape(df):
    return int(df.shape[0]), int(df.shape[1])


def make_json_safe(value):
    if value is None:
        return None

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, dict):
        return {
            str(make_json_safe(key)): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return [make_json_safe(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, (pd.Timestamp, datetime, date)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, Decimal):
        if value.is_nan() or value.is_infinite():
            return None
        return float(value)

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(value, (str, int, bool)):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def build_preview(df, limit=20, selected_columns=None):
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 20

    limit = max(1, min(limit, 100))
    available_columns = [str(column) for column in df.columns]
    column_lookup = {str(column): column for column in df.columns}

    if selected_columns:
        selected_columns = [
            column_lookup[str(column)]
            for column in selected_columns
            if str(column) in column_lookup
        ]
    else:
        selected_columns = list(df.columns)

    if not selected_columns:
        selected_columns = list(df.columns)

    limited_df = df[selected_columns].head(limit)
    rows = [
        {str(column): make_json_safe(value) for column, value in row.items()}
        for row in limited_df.to_dict(orient="records")
    ]
    columns = [str(column) for column in limited_df.columns]

    return {
        "columns": columns,
        "rows": rows,
        "row_limit": limit,
        "available_columns": available_columns,
    }


def _safe_unique_count(series):
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        unique_values = set()
        for value in series:
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass

            try:
                unique_values.add(value)
            except TypeError:
                unique_values.add(str(value))
        return len(unique_values)


def _safe_sample_values(series, limit=5):
    values = []
    has_missing = bool(series.isna().any())

    for value in series:
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        safe_value = make_json_safe(value)
        if safe_value not in values:
            values.append(safe_value)

        if len(values) >= limit:
            break

    if has_missing:
        values.append(None)

    return values


def _empty_numeric_stats():
    return {
        "mean": None,
        "median": None,
        "mode": None,
        "min": None,
        "max": None,
        "range": None,
        "std": None,
        "variance": None,
        "cv": None,
        "coefficient_of_variation": None,
        "skewness": None,
        "kurtosis": None,
        "outlier_count": None,
        "outlier_percentage": None,
        "zero_count": None,
        "negative_count": None,
    }


def _safe_numeric_stats(series, is_numeric):
    if not is_numeric:
        return _empty_numeric_stats()

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return _empty_numeric_stats()

    mean = numeric_series.mean()
    median = numeric_series.median()
    mode_values = numeric_series.mode()
    mode = mode_values.iloc[0] if not mode_values.empty else None
    min_value = numeric_series.min()
    max_value = numeric_series.max()
    value_range = max_value - min_value
    std = numeric_series.std()
    variance = numeric_series.var()

    cv = None
    if mean is not None and not pd.isna(mean) and mean != 0 and not pd.isna(std):
        cv = std / mean

    skewness = numeric_series.skew() if len(numeric_series) > 2 else None
    kurtosis = numeric_series.kurt() if len(numeric_series) > 3 else None

    percentile_25 = numeric_series.quantile(0.25)
    percentile_75 = numeric_series.quantile(0.75)
    iqr = percentile_75 - percentile_25
    lower_bound = percentile_25 - 1.5 * iqr
    upper_bound = percentile_75 + 1.5 * iqr
    outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
    outlier_count = int(outliers.count())
    outlier_percentage = round((outlier_count / int(numeric_series.count())) * 100, 2)

    return make_json_safe(
        {
            "mean": float(mean) if not pd.isna(mean) else None,
            "median": float(median) if not pd.isna(median) else None,
            "mode": float(mode) if mode is not None and not pd.isna(mode) else None,
            "min": float(min_value) if not pd.isna(min_value) else None,
            "max": float(max_value) if not pd.isna(max_value) else None,
            "range": float(value_range) if not pd.isna(value_range) else None,
            "std": float(std) if not pd.isna(std) else None,
            "variance": float(variance) if not pd.isna(variance) else None,
            "cv": float(cv) if cv is not None and not pd.isna(cv) else None,
            "coefficient_of_variation": float(cv) if cv is not None and not pd.isna(cv) else None,
            "skewness": float(skewness) if skewness is not None and not pd.isna(skewness) else None,
            "kurtosis": float(kurtosis) if kurtosis is not None and not pd.isna(kurtosis) else None,
            "outlier_count": outlier_count,
            "outlier_percentage": float(outlier_percentage),
            "zero_count": int((numeric_series == 0).sum()),
            "negative_count": int((numeric_series < 0).sum()),
        }
    )


def detect_column_type(series, column_name, row_count):
    column_name_str = str(column_name)
    column_name_lower = column_name_str.lower()
    raw_dtype = str(series.dtype)

    missing_count = int(series.isnull().sum())
    missing_percentage = round((missing_count / row_count) * 100, 2) if row_count else 0
    unique_count = _safe_unique_count(series)
    unique_percentage = round((unique_count / row_count) * 100, 2) if row_count else 0
    sample_values = _safe_sample_values(series, limit=5)

    is_empty = row_count == 0 or missing_count == row_count
    is_constant = unique_count == 1 and not is_empty

    is_numeric = pd.api.types.is_numeric_dtype(series)
    numeric_stats = _safe_numeric_stats(series, is_numeric)
    is_boolean_dtype = pd.api.types.is_bool_dtype(series)
    is_datetime_dtype = pd.api.types.is_datetime64_any_dtype(series)
    is_text = pd.api.types.is_string_dtype(series)
    is_pandas_categorical = isinstance(series.dtype, pd.CategoricalDtype)

    is_binary = unique_count == 2
    is_boolean = is_boolean_dtype or (
        is_binary and unique_count > 0 and unique_percentage <= 50
    )

    is_id_name = (
        column_name_lower == "id"
        or column_name_lower == "identifier"
        or column_name_lower == "uuid"
        or column_name_lower.endswith("_id")
        or column_name_lower.endswith("id")
        or column_name_lower.startswith("id_")
        or column_name_lower.endswith("_no")
        or column_name_lower.endswith("no")
        or column_name_lower.endswith("_number")
        or column_name_lower.endswith("number")
        or "email" in column_name_lower
        or "phone" in column_name_lower
        or "mobile" in column_name_lower
    )
    is_id_like = (
        not is_empty
        and unique_percentage >= 90
        and unique_count > 10
        and is_id_name
    )

    is_numeric_categorical = (
        is_numeric
        and not is_boolean
        and unique_count <= 10
        and unique_percentage <= 20
    )

    is_possible_datetime = False
    if is_datetime_dtype:
        is_possible_datetime = True
    elif not is_empty and not is_numeric and not is_boolean:
        non_missing_series = series.dropna()
        if len(non_missing_series) > 0:
            try:
                converted_dates = pd.to_datetime(non_missing_series, errors="coerce")
                valid_date_count = int(converted_dates.notna().sum())
                date_success_percentage = (valid_date_count / len(non_missing_series)) * 100
                is_possible_datetime = date_success_percentage >= 80
            except (TypeError, ValueError, OverflowError):
                is_possible_datetime = False

    is_categorical = (
        is_pandas_categorical
        or is_numeric_categorical
        or (is_text and unique_count <= 50 and unique_percentage <= 50)
    )
    is_high_cardinality = (
        not is_id_like
        and not is_empty
        and unique_count > 50
        and unique_percentage > 10
    )
    is_possible_target = (
        not is_empty
        and not is_id_like
        and not is_constant
        and unique_count >= 2
        and unique_percentage <= 50
    )

    if missing_percentage == 0:
        missing_hint = "no_missing"
    elif missing_percentage < 30:
        missing_hint = "low_missing_imputation_candidate"
    elif missing_percentage < 70:
        missing_hint = "medium_missing_imputation_review"
    elif missing_percentage < 85:
        missing_hint = "high_missing_keep_or_drop_review"
    else:
        missing_hint = "extreme_missing_drop_candidate"

    if is_empty:
        detected_type = "empty"
    elif is_id_like:
        detected_type = "id"
    elif is_boolean:
        detected_type = "boolean"
    elif is_possible_datetime:
        detected_type = "datetime"
    elif is_numeric_categorical:
        detected_type = "categorical_numeric"
    elif is_numeric:
        detected_type = "numeric"
    elif is_high_cardinality:
        detected_type = "high_cardinality_categorical"
    elif is_categorical:
        detected_type = "categorical"
    else:
        detected_type = "text"

    if detected_type == "boolean":
        encoding_hint = "binary_encoding_candidate"
    elif detected_type == "categorical":
        if unique_count <= 10:
            encoding_hint = "onehot_encoding_candidate"
        elif unique_count <= 50:
            encoding_hint = "frequency_or_ordinal_encoding_review"
        else:
            encoding_hint = "frequency_target_or_hash_encoding_review"
    elif detected_type == "categorical_numeric":
        encoding_hint = "ordinal_or_keep_numeric_review"
    elif detected_type == "high_cardinality_categorical":
        encoding_hint = "frequency_target_hash_or_embedding_review"
    elif detected_type == "text":
        encoding_hint = "tfidf_or_embedding_review"
    elif detected_type == "datetime":
        encoding_hint = "datetime_feature_extraction_candidate"
    elif detected_type == "id":
        encoding_hint = "do_not_encode_directly"
    else:
        encoding_hint = "not_applicable"

    if detected_type == "numeric":
        scaling_hint = "scaling_candidate"
    elif detected_type == "categorical_numeric":
        scaling_hint = "not_required_or_review"
    else:
        scaling_hint = "not_required"

    if is_empty:
        recommendation = "drop_candidate"
        recommended_role = "review"
        recommendation_reason = "Column is fully empty. Usually not useful, but review before dropping."
    elif is_constant:
        recommendation = "drop_candidate"
        recommended_role = "review"
        recommendation_reason = "Column has only one unique value. Usually gives no ML signal, but review before dropping."
    elif is_id_like:
        recommendation = "review"
        recommended_role = "identifier"
        recommendation_reason = "Column looks like an identifier. Do not use directly for ML unless special handling is planned."
    elif missing_percentage >= 85:
        recommendation = "drop_candidate"
        recommended_role = "review"
        recommendation_reason = "Column has extremely high missing values. It may be better to drop it, but review domain meaning before final removal."
    elif missing_percentage >= 70:
        recommendation = "review"
        recommended_role = "feature"
        recommendation_reason = "Column has very high missing values. Check whether imputation is possible or whether the column should be dropped."
    elif is_high_cardinality:
        recommendation = "review"
        recommended_role = "feature"
        recommendation_reason = "Column has many unique values. Avoid blind one-hot encoding; review frequency, target, hash, or embedding encoding."
    elif missing_percentage > 0:
        recommendation = "keep_with_imputation_review"
        recommended_role = "feature"
        recommendation_reason = "Column has missing values. Choose imputation strategy later."
    else:
        recommendation = "keep"
        recommended_role = "feature"
        recommendation_reason = "Column looks usable as a feature."

    return make_json_safe(
        {
            "column_name": column_name_str,
            "raw_dtype": raw_dtype,
            "detected_type": detected_type,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "missing_hint": missing_hint,
            "unique_count": unique_count,
            "unique_percentage": unique_percentage,
            "sample_values": sample_values,
            "mean": numeric_stats["mean"],
            "median": numeric_stats["median"],
            "mode": numeric_stats["mode"],
            "min": numeric_stats["min"],
            "max": numeric_stats["max"],
            "range": numeric_stats["range"],
            "std": numeric_stats["std"],
            "variance": numeric_stats["variance"],
            "cv": numeric_stats["cv"],
            "coefficient_of_variation": numeric_stats["coefficient_of_variation"],
            "skewness": numeric_stats["skewness"],
            "kurtosis": numeric_stats["kurtosis"],
            "outlier_count": numeric_stats["outlier_count"],
            "outlier_percentage": numeric_stats["outlier_percentage"],
            "zero_count": numeric_stats["zero_count"],
            "negative_count": numeric_stats["negative_count"],
            "is_id_like": is_id_like,
            "is_constant": is_constant,
            "is_high_cardinality": is_high_cardinality,
            "is_possible_datetime": is_possible_datetime,
            "is_possible_target": is_possible_target,
            "recommended_role": recommended_role,
            "encoding_hint": encoding_hint,
            "scaling_hint": scaling_hint,
            "recommendation": recommendation,
            "recommendation_reason": recommendation_reason,
        }
    )


def detect_column_schema(df):
    row_count = len(df)
    schema = []

    for column_name in df.columns:
        schema.append(
            detect_column_type(
                series=df[column_name],
                column_name=column_name,
                row_count=row_count,
            )
        )

    return schema


def build_column_schema(df):
    return detect_column_schema(df)


def build_dataset_profile(df):
    row_count, column_count = get_dataset_shape(df)
    duplicate_row_count = int(df.duplicated().sum())
    total_missing_cells = int(df.isnull().sum().sum())
    total_cells = row_count * column_count
    total_missing_percentage = round((total_missing_cells / total_cells) * 100, 2) if total_cells else 0
    columns = detect_column_schema(df)

    numeric_columns = []
    categorical_columns = []
    datetime_columns = []
    boolean_columns = []
    text_columns = []
    id_like_columns = []
    constant_columns = []
    high_cardinality_columns = []
    possible_target_columns = []

    for column in columns:
        column_name = column["column_name"]
        detected_type = column["detected_type"]

        if detected_type == "numeric":
            numeric_columns.append(column_name)
        elif detected_type in {"categorical", "categorical_numeric"}:
            categorical_columns.append(column_name)
        elif detected_type == "datetime":
            datetime_columns.append(column_name)
        elif detected_type == "boolean":
            boolean_columns.append(column_name)
        elif detected_type == "text":
            text_columns.append(column_name)
        elif detected_type == "id":
            id_like_columns.append(column_name)
        elif detected_type == "high_cardinality_categorical":
            high_cardinality_columns.append(column_name)

        if column["is_id_like"] and column_name not in id_like_columns:
            id_like_columns.append(column_name)
        if column["is_constant"]:
            constant_columns.append(column_name)
        if column["is_high_cardinality"] and column_name not in high_cardinality_columns:
            high_cardinality_columns.append(column_name)
        if column["is_possible_target"]:
            possible_target_columns.append(column_name)

    profile = {
        "row_count": row_count,
        "column_count": column_count,
        "duplicate_row_count": duplicate_row_count,
        "total_missing_cells": total_missing_cells,
        "total_missing_percentage": total_missing_percentage,
        "quality_score": 100,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "boolean_columns": boolean_columns,
        "text_columns": text_columns,
        "id_like_columns": id_like_columns,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "possible_target_columns": possible_target_columns,
        "columns": columns,
    }
    profile["quality_score"] = calculate_quality_score(profile)
    return make_json_safe(profile)


def generate_dataset_profile(df):
    return build_dataset_profile(df)


def generate_dataset_preview(df, limit=20, selected_columns=None):
    return build_preview(df, limit=limit, selected_columns=selected_columns)


def calculate_quality_score(profile):
    score = 100
    row_count = profile.get("row_count", 0)
    column_count = profile.get("column_count", 0)
    duplicate_row_count = profile.get("duplicate_row_count", 0)
    total_missing_percentage = profile.get("total_missing_percentage", 0)
    constant_columns = profile.get("constant_columns", [])
    high_cardinality_columns = profile.get("high_cardinality_columns", [])
    id_like_columns = profile.get("id_like_columns", [])

    if row_count == 0 or column_count == 0:
        return 0

    if total_missing_percentage >= 85:
        score -= 35
    elif total_missing_percentage >= 70:
        score -= 25
    elif total_missing_percentage >= 30:
        score -= 15
    elif total_missing_percentage > 0:
        score -= 5

    duplicate_percentage = (duplicate_row_count / row_count) * 100 if row_count else 0
    if duplicate_percentage >= 50:
        score -= 25
    elif duplicate_percentage >= 20:
        score -= 15
    elif duplicate_percentage > 0:
        score -= 5

    score -= min(len(constant_columns) * 5, 20)
    score -= min(len(high_cardinality_columns) * 3, 15)
    score -= min(len(id_like_columns) * 2, 10)

    return int(max(0, min(100, score)))


def clean_json_value(value):
    return make_json_safe(value)


def dataframe_to_json_safe(data):
    return make_json_safe(data)
