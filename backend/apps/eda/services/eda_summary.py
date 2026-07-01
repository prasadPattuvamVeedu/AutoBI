from apps.datasets.models import Dataset
from apps.datasets.services import make_json_safe

from .eda_common import _profile_from_storage, load_eda_dataframe, load_profile_column_groups


SUMMARY_KEYS = [
    "rows",
    "columns",
    "missing_values",
    "duplicates",
    "outliers",
    "datatype_issues",
    "recommendations",
]


def _safe_int(value, default=0):
    try:
        if value in [None, ""]:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default=0):
    try:
        if value in [None, ""]:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _summary_dict(profile_json):
    if not isinstance(profile_json, dict):
        return {}
    summary = profile_json.get("summary")
    return summary if isinstance(summary, dict) else profile_json


def _profile_columns(profile_json):
    if not isinstance(profile_json, dict):
        return []
    columns = profile_json.get("columns") or profile_json.get("column_profiles") or []
    return columns if isinstance(columns, list) else []


def _profile_total_outliers(profile_json):
    total = 0
    for column in _profile_columns(profile_json):
        if isinstance(column, dict):
            total += _safe_int(column.get("outlier_count"))
    return total


def _profile_datatype_issues(profile_json):
    summary = _summary_dict(profile_json)
    explicit = summary.get("datatype_issues")
    if explicit is not None:
        return _safe_int(explicit)

    review_columns = profile_json.get("review_columns") if isinstance(profile_json, dict) else []
    return len(review_columns or [])


def _profile_recommendations(profile_json):
    if not isinstance(profile_json, dict):
        return 0
    recommendations = profile_json.get("recommendations") or profile_json.get("insights") or []
    return len(recommendations) if isinstance(recommendations, list) else _safe_int(recommendations)


def _filter_groups_to_dataframe(column_groups, df):
    available = {str(column) for column in df.columns} if df is not None else set()
    if not available:
        return column_groups
    return {
        key: [column for column in values if str(column) in available]
        for key, values in column_groups.items()
    }


def _build_summary_cards(dataset, df, profile_json, column_groups, recommended_count):
    summary = _summary_dict(profile_json)
    rows = _safe_int(summary.get("row_count"), dataset.row_count or (len(df) if df is not None else 0))
    columns = _safe_int(summary.get("column_count"), dataset.column_count or (len(df.columns) if df is not None else 0))
    missing_values = _safe_int(summary.get("total_missing_cells") or summary.get("missing_cell_count"))
    missing_percentage = _safe_float(summary.get("total_missing_percentage") or summary.get("missing_percentage"))
    duplicates = _safe_int(summary.get("duplicate_row_count") or summary.get("duplicates"))

    return {
        "rows": rows,
        "columns": columns,
        "missing_values": missing_values,
        "missing_percentage": missing_percentage,
        "duplicates": duplicates,
        "outliers": _profile_total_outliers(profile_json),
        "datatype_issues": _profile_datatype_issues(profile_json),
        "recommendations": _profile_recommendations(profile_json),
        "recommended_graphs": recommended_count,
        "numeric_columns": len(column_groups.get("numeric_columns", [])),
        "categorical_columns": len(column_groups.get("categorical_columns", [])),
        "datetime_columns": len(column_groups.get("datetime_columns", [])),
        "text_columns": len(column_groups.get("text_columns", [])),
        "id_like_columns": len(column_groups.get("id_like_columns", [])),
    }


def _all_group_columns(column_groups):
    seen = set()
    result = []
    for values in column_groups.values():
        for column in values:
            key = str(column)
            if key and key not in seen:
                seen.add(key)
                result.append(key)
    return result


def _recommended_graphs(column_groups, summary_cards, target_column=None, mode="advanced"):
    numeric = column_groups.get("numeric_columns", [])
    categorical = (
        column_groups.get("categorical_columns", [])
        + column_groups.get("boolean_columns", [])
        + column_groups.get("high_cardinality_columns", [])
    )
    datetime = column_groups.get("datetime_columns", [])
    all_columns = _all_group_columns(column_groups)
    graphs = []

    if mode == "advanced":
        if summary_cards.get("missing_values", 0) > 0:
            graphs.append({
                "chart_type": "missing_values_bar",
                "title": "Missing Values by Column",
                "description": "Shows stored missing-value quality profile by column.",
                "columns": all_columns,
                "priority": "high",
            })

        for column in numeric[:8]:
            graphs.extend([
                {
                    "chart_type": "histogram",
                    "title": f"Histogram - {column}",
                    "description": "Inspect raw numeric distribution.",
                    "x_column": column,
                    "priority": "high",
                },
                {
                    "chart_type": "boxplot",
                    "title": f"Boxplot - {column}",
                    "description": "Inspect spread and profile-aligned outlier candidates.",
                    "y_column": column,
                    "priority": "high",
                },
                {
                    "chart_type": "qq_plot",
                    "title": f"QQ Plot - {column}",
                    "description": "Check raw numeric normality visually.",
                    "x_column": column,
                    "priority": "medium",
                },
            ])

        if len(numeric) >= 2:
            graphs.append({
                "chart_type": "correlation_heatmap",
                "title": "Correlation Heatmap",
                "description": "Compare relationships among profile numeric columns.",
                "columns": numeric,
                "priority": "high",
            })

        for column in categorical[:8]:
            graphs.append({
                "chart_type": "countplot",
                "title": f"Category Count - {column}",
                "description": "Inspect category balance.",
                "x_column": column,
                "priority": "medium",
            })

        if numeric:
            graphs.append({
                "chart_type": "skewness_kurtosis_table",
                "title": "Skewness and Kurtosis Summary",
                "description": "Summarize numeric distribution shape.",
                "columns": numeric,
                "priority": "medium",
            })

        for date_column in datetime[:2]:
            for value_column in numeric[:3]:
                graphs.append({
                    "chart_type": "line_trend",
                    "title": f"{value_column} Trend by {date_column}",
                    "description": "Inspect raw time trend.",
                    "x_column": date_column,
                    "y_column": value_column,
                    "priority": "medium",
                })
    else:
        graphs.extend([
            {
                "chart_type": "missing_values_comparison",
                "title": "Missing Values: Before vs After",
                "description": "Validate missing-value impact after cleaning or transformation.",
                "columns": all_columns,
                "priority": "high",
            },
            {
                "chart_type": "outlier_comparison",
                "title": "Outliers: Before vs After",
                "description": "Validate outlier impact for profile numeric columns.",
                "columns": numeric,
                "priority": "high",
            },
        ])
        for column in numeric[:8]:
            graphs.append({
                "chart_type": "distribution_change",
                "title": f"Distribution Change - {column}",
                "description": "Compare distribution before and after operations.",
                "x_column": column,
                "priority": "medium",
            })
        if len(numeric) >= 2:
            graphs.append({
                "chart_type": "correlation_shift_heatmap",
                "title": "Correlation Shift Heatmap",
                "description": "Compare numeric correlation structure before and after.",
                "columns": numeric,
                "priority": "medium",
            })

    if target_column and target_column in all_columns:
        graphs.append({
            "chart_type": "target_balance" if mode == "advanced" else "target_balance_after",
            "title": f"Target Balance - {target_column}",
            "description": "Inspect target/class balance.",
            "x_column": target_column,
            "priority": "high",
        })

    return graphs


def build_advanced_eda_summary(dataset_id, dataset_version="raw", target_column=None):
    dataset = Dataset.objects.get(pk=dataset_id)
    df = load_eda_dataframe(dataset_id, dataset_version=dataset_version)
    profile_json = _profile_from_storage(dataset)
    column_groups = _filter_groups_to_dataframe(load_profile_column_groups(dataset_id), df)
    draft_cards = _build_summary_cards(dataset, df, profile_json, column_groups, 0)
    graphs = _recommended_graphs(column_groups, draft_cards, target_column, mode="advanced")
    summary_cards = _build_summary_cards(dataset, df, profile_json, column_groups, len(graphs))

    warnings = []
    if not profile_json:
        warnings.append("Stored dataset profile was not found; column groups are unavailable.")
    if not column_groups.get("numeric_columns"):
        warnings.append("No profile numeric columns are available for numeric EDA graphs.")
    if target_column and target_column not in _all_group_columns(column_groups):
        warnings.append("Selected target column was not found in stored profile column groups.")

    return make_json_safe({
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "eda_mode": "advanced",
        "target_column": target_column,
        "summary_cards": summary_cards,
        "column_groups": column_groups,
        "recommended_graphs": graphs,
        "warnings": warnings,
    })


def build_validation_eda_summary(dataset_id, dataset_version="cleaned", target_column=None):
    dataset = Dataset.objects.get(pk=dataset_id)
    df = load_eda_dataframe(dataset_id, dataset_version=dataset_version)
    profile_json = _profile_from_storage(dataset)
    column_groups = _filter_groups_to_dataframe(load_profile_column_groups(dataset_id), df)
    draft_cards = _build_summary_cards(dataset, df, profile_json, column_groups, 0)
    graphs = _recommended_graphs(column_groups, draft_cards, target_column, mode="validation")
    summary_cards = _build_summary_cards(dataset, df, profile_json, column_groups, len(graphs))

    warnings = []
    if not profile_json:
        warnings.append("Stored dataset profile was not found; validation column groups are unavailable.")
    if dataset_version in ["cleaned", "transformed", "ml_ready"] and df is None:
        warnings.append("Requested validation dataset version was not available.")

    return make_json_safe({
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "eda_mode": "validation",
        "target_column": target_column,
        "summary_cards": summary_cards,
        "column_groups": column_groups,
        "recommended_graphs": graphs,
        "warnings": warnings,
    })
