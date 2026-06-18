from apps.datasets.services import make_json_safe
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
)
import re


def _safe_percentage(part, total):
    # MANUAL PANDAS/STATISTICS CODE REQUIRED:
    # Write the real pandas/statistics logic here manually.
    # Do not auto-generate this section.

    if not total:
        return 0
    return round((part / total) * 100, 2)


def build_profile_summary(df):
    # MANUAL PANDAS/STATISTICS CODE REQUIRED:
    # Write the real pandas/statistics logic here manually.
    # Do not auto-generate this section.
    row_count, column_count = df.shape[0], df.shape[1]
    duplicate_row_count = df.duplicated().sum()
    total_missing_cells = df.isnull().sum().sum()
    total_missing_percentage = _safe_percentage(total_missing_cells, row_count * column_count)
    quality_score = 100 - total_missing_percentage - _safe_percentage(duplicate_row_count, row_count)
    quality_score = max(0, min(100, quality_score))

    return {
        "row_count": int(row_count),
        "column_count": int(column_count),
        "duplicate_row_count": int(duplicate_row_count),
        "total_missing_cells": int(total_missing_cells),
        "total_missing_percentage": float(total_missing_percentage),
        "quality_score": float(quality_score),
    }


def build_column_statistics(df):
    # MANUAL PANDAS/STATISTICS CODE REQUIRED:
    # Write the real pandas/statistics logic here manually.
    # Do not auto-generate this section.
    columns = []
    row_count = df.shape[0]


    for column_name in df.columns:
        series = df[column_name]

        missing_stats = calculate_missing_stats(series, row_count)
        unique_stats = calculate_unique_stats(series, row_count)
        detected_type = detect_column_type(column_name, series, row_count)

        num_stats = calculate_numeric_stats(series) if detected_type == "numeric" else {}
        cat_stats = calculate_categorical_stats(series) if detected_type in ["categorical", "numeric_categorical"] else {}

        role = detect_column_role(
            column_name, 
            detected_type, 
            missing_stats.get("missing_percentage"), 
            unique_stats.get("unique_count")
        )
        recommendation = generate_column_recommendation(
            detected_type, 
            missing_stats.get("missing_percentage"), 
            unique_stats.get("unique_count")
        )

        columns.append(
            {
                "column_name": str(column_name),
                "detected_type": detected_type,
                "missing_count": missing_stats.get("missing_count"),
                "missing_percentage": missing_stats.get("missing_percentage"),
                "unique_count": unique_stats.get("unique_count"),
                "unique_ratio": unique_stats.get("unique_ratio"),
                "role": role,
                "recommendation": recommendation,
                "mean": num_stats.get("mean"),
                "median": num_stats.get("median"),
                "mode": num_stats.get("mode") if num_stats.get("mode") is not None else cat_stats.get("mode"),
                "min": num_stats.get("min"),
                "max": num_stats.get("max"),
                "range": num_stats.get("range"),
                "std": num_stats.get("std"),
                "percentile_25": num_stats.get("percentile_25"),
                "percentile_50": num_stats.get("percentile_50"),
                "percentile_75": num_stats.get("percentile_75"),
                "iqr": num_stats.get("iqr"),
                "top_value": cat_stats.get("top_value"),
                "top_frequency": cat_stats.get("top_frequency"),
            }
        )

    return columns



def calculate_profile_quality_score(profile):
    # MANUAL PANDAS/STATISTICS CODE REQUIRED:
    # Write the real pandas/statistics logic here manually.
    # Do not auto-generate this section.
    summary = profile.get("summary", {})

    row_count = summary.get("row_count", 0)
    column_count = summary.get("column_count", 0)
    missing_percentage = summary.get("total_missing_percentage", 0)
    duplicate_row_count = summary.get("duplicate_row_count", 0)

    if row_count == 0 or column_count == 0:
        return 0

    duplicate_percentage = _safe_percentage(duplicate_row_count, row_count)

    score = 100
    score -= missing_percentage
    score -= duplicate_percentage

    return round(max(0, min(100, score)), 2)


def is_id_like_column(column_name, series, row_count):
    column_lower = str(column_name).lower()
    non_null_series = series.dropna()

    unique_count = non_null_series.nunique()
    unique_ratio = unique_count / row_count if row_count else 0

    id_pattern = (
        r"(^|[_\-\s\.])"
        r"(id|uuid|guid|key|code|number|no|phone|mobile|invoice|transaction|order|account|customer|user|serial|reference|ref|postal|postcode|post_code|zipcode|zip_code|pincode|pin_code|zip)"
        r"([_\-\s\.]|$)"
        r"|"
        r"(postal\w*|post\w*code|zip\w*code|pin\w*code|phone\w*|mobile\w*)"
    )

    has_id_name = re.search(id_pattern, column_lower) is not None

    postal_pattern = r"(postal|postcode|post_code|postalcode|zipcode|zip_code|pincode|pin_code|zip)"

    if re.search(postal_pattern, column_lower):
        return True

    mostly_unique = unique_ratio >= 0.8

    avg_length = non_null_series.astype(str).str.len().mean() if len(non_null_series) else 0
    long_code_like = avg_length >= 8 and mostly_unique

    return (has_id_name and unique_ratio >= 0.5) or long_code_like


def is_numeric_column(series):
    return is_numeric_dtype(series)


def is_boolean_column(series):
    return is_bool_dtype(series)


def is_datetime_column(series):
    return is_datetime64_any_dtype(series)

def build_dataset_profile_response(dataset, df):
    # MANUAL PANDAS/STATISTICS CODE REQUIRED:
    # Write the real pandas/statistics logic here manually.
    
    
    # 1. missing value detection
    # 2. duplicate row detection
    # 3. column type detection
    # 4. unique count detection
    # 5. numeric/categorical/date/boolean/text bucket classification
    # 6. constant column detection
    # 7. high-cardinality detection
    # 8. ID-like column detection
    # 9. role/recommendation rules
    # 10. quality score formula
    row_count = df.shape[0]

    numeric_columns = []
    categorical_columns = []
    datetime_columns = []
    boolean_columns = []
    text_columns = []
    id_like_columns = []
    constant_columns = []
    high_cardinality_columns = []
    review_columns = []

    for column_name in df.columns:
        series = df[column_name]

        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / row_count if row_count else 0

        if unique_count == 1:
            constant_columns.append(str(column_name))

        if unique_ratio >= 0.5 and unique_count > 20:
            high_cardinality_columns.append(str(column_name))

        if is_id_like_column(column_name, series, row_count):
            id_like_columns.append(str(column_name))

        elif is_boolean_column(series):
            boolean_columns.append(str(column_name))

        elif is_datetime_column(series):
            datetime_columns.append(str(column_name))

        elif is_numeric_column(series):
            if unique_count <= 2:
                boolean_columns.append(str(column_name))
            elif row_count < 50 and unique_count <= 15:
                review_columns.append(str(column_name))
            elif unique_ratio <= 0.2 and unique_count <= 20:
                categorical_columns.append(str(column_name))
            else:
                numeric_columns.append(str(column_name))

        else:
            categorical_columns.append(str(column_name))

    summary = build_profile_summary(df)

    profile = {
        "dataset_id": dataset.id,
        "summary": summary,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "boolean_columns": boolean_columns,
        "text_columns": text_columns,
        "id_like_columns": id_like_columns,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "review_columns": review_columns,
        "columns": build_column_statistics(df),
    }

    profile["summary"]["quality_score"] = calculate_profile_quality_score(profile)
    return make_json_safe(profile)



def calculate_missing_stats(series, row_count):
    # MANUAL PANDAS/STATISTICS CODE REQUIRED:
    # Calculate missing_count and missing_percentage manually here.
    missing_count = series.isnull().sum()
    missing_percentage = _safe_percentage(missing_count, row_count)
    # Return:
    # {
    #     "missing_count": 0,
    #     "missing_percentage": 0.0
    # }
    return {
        "missing_count": int(missing_count),
        "missing_percentage": missing_percentage,
    }


def calculate_unique_stats(series, row_count):
    # MANUAL PANDAS/STATISTICS CODE REQUIRED:
    # Calculate unique_count and unique_ratio manually here.
    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / row_count if row_count else 0

    return {
        "unique_count": int(unique_count),
        "unique_ratio": unique_ratio,
    }


def detect_column_type(column_name, series, row_count):
    # MANUAL PANDAS/STATISTICS CODE REQUIRED:
    # Detect numeric, categorical, datetime, boolean, text, id_like, review manually here.

    unique_stats = calculate_unique_stats(series, row_count)
    unique_count = unique_stats["unique_count"]
    unique_ratio = unique_stats["unique_ratio"]

    if is_id_like_column(column_name, series, row_count):
        return "id_like"

    if is_boolean_column(series):
        return "boolean"

    if is_datetime_column(series):
        return "datetime"

    if is_numeric_column(series):
        if unique_count <= 2:
            return "boolean"
        elif row_count < 50 and unique_count <= 15:
            return "review"
        elif unique_ratio <= 0.2 and unique_count <= 20:
            return "numeric_categorical"
        else:
            return "numeric"

    return "categorical"
    


def calculate_numeric_stats(series):
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()

    if numeric_series.empty:
        return {
            "mean": None,
            "median": None,
            "mode": None,
            "min": None,
            "max": None,
            "range": None,
            "std": None,
            "percentile_25": None,
            "percentile_50": None,
            "percentile_75": None,
            "iqr": None,
        }

    mean = numeric_series.mean()
    median = numeric_series.median()
    mode = numeric_series.mode().iloc[0] if not numeric_series.mode().empty else None
    min_val = numeric_series.min()
    max_val = numeric_series.max()
    value_range = max_val - min_val
    std = numeric_series.std()

    percentile_25 = numeric_series.quantile(0.25)
    percentile_50 = numeric_series.quantile(0.50)
    percentile_75 = numeric_series.quantile(0.75)
    iqr = percentile_75 - percentile_25

    return {
        "mean": float(mean),
        "median": float(median),
        "mode": float(mode) if mode is not None else None,
        "min": float(min_val),
        "max": float(max_val),
        "range": float(value_range),
        "std": float(std) if not pd.isna(std) else None,
        "percentile_25": float(percentile_25),
        "percentile_50": float(percentile_50),
        "percentile_75": float(percentile_75),
        "iqr": float(iqr),
    }
def calculate_categorical_stats(series):
    
    # Calculate mode, top_value, top_frequency manually here.
    value_counts = series.value_counts(dropna=True)
    mode = value_counts.index[0] if not value_counts.empty else None
    top_value = mode
    top_frequency = value_counts.iloc[0] if not value_counts.empty else None
    return {
        "mode": mode,
        "top_value": top_value,
        "top_frequency": top_frequency,
    }


def detect_column_role(column_name, detected_type, missing_percentage, unique_count):
    column_lower = str(column_name).lower()

    if detected_type == "id_like":
        return "identifier"

    if detected_type == "review":
        return "review"

    if missing_percentage >= 85:
        return "review"

    if unique_count == 1:
        return "exclude"

    target_keywords = ["target", "label", "class", "outcome", "churn", "sales", "price"]

    if any(word in column_lower for word in target_keywords):
        return "target_candidate"

    return "feature"

def generate_column_recommendation(detected_type, missing_percentage, unique_count):
    if detected_type == "id_like":
        return "Possible identifier. Keep for tracking and review before using in ML."

    if detected_type == "review":
        return "Column type is uncertain. Review manually before analysis or ML."

    if unique_count == 1:
        return "Constant column. Usually not useful for analysis or ML."

    if missing_percentage >= 85:
        return "Very high missing values. Review before keeping or dropping."

    if missing_percentage >= 30:
        return "Moderate missing values. Review imputation options later."

    if detected_type == "numeric":
        return "Numeric column. Suitable for descriptive statistics."

    if detected_type == "numeric_categorical":
        return "Numeric values may represent categories. Review before treating as continuous."

    if detected_type == "categorical":
        return "Categorical column. Suitable for frequency and mode analysis."

    if detected_type == "datetime":
        return "Datetime column. Later extract year, month, day, or weekday features."

    if detected_type == "boolean":
        return "Boolean/binary column. Suitable for counts and proportions."

    return "Review column before using in analysis."