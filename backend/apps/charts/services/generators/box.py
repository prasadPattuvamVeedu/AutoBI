"""Manual pandas/numpy generator for box plot charts."""

from __future__ import annotations

import pandas as pd

from ..chart_common_operations import build_title, format_rows_for_frontend, numeric_series, validate_column


def _box_stats(series):
    q1 = float(series.quantile(0.25))
    median = float(series.quantile(0.5))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    non_outliers = series[(series >= lower_bound) & (series <= upper_bound)]
    outliers = series[(series < lower_bound) | (series > upper_bound)].head(200).tolist()
    return {
        "min": float(non_outliers.min()) if not non_outliers.empty else float(series.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(non_outliers.max()) if not non_outliers.empty else float(series.max()),
        "mean": float(series.mean()),
        "iqr": float(iqr),
        "lower_whisker": float(non_outliers.min()) if not non_outliers.empty else float(series.min()),
        "upper_whisker": float(non_outliers.max()) if not non_outliers.empty else float(series.max()),
        "outlier_count": int(((series < lower_bound) | (series > upper_bound)).sum()),
        "outliers": [float(v) for v in outliers],
        "count": int(series.count()),
    }


def build_box_plot_data(df, payload, chart_type="box", source_meta=None):
    """Manual implementation for box plot statistics."""
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    value_column = validate_column(df, payload.get("y_column") or payload.get("x_column"), "Box plot value column")
    group_column = validate_column(df, payload.get("x_column") if payload.get("y_column") else None, "Box plot group column", required=False)
    rows = []
    if group_column and group_column != value_column:
        for label, group_df in df[[group_column, value_column]].dropna(subset=[group_column]).groupby(group_column, dropna=False, observed=True):
            series = numeric_series(group_df[value_column]).dropna()
            if not series.empty:
                rows.append({"label": str(label), **_box_stats(series)})
    else:
        series = numeric_series(df[value_column]).dropna()
        if not series.empty:
            rows.append({"label": value_column, **_box_stats(series)})
    if not rows:
        raise ValueError(f"Column '{value_column}' does not contain numeric values for box plot.")
    result_df = pd.DataFrame(rows)
    title = payload.get("title") or build_title(chart_type, group_column, value_column, "")
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": "label",
            "source_x_column": group_column,
            "y_column": value_column,
            "aggregation": "distribution",
        },
        "chart_data_json": {
            "columns": list(result_df.columns),
            "rows": format_rows_for_frontend(result_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(result_df)), "optimized": True},
        },
        "settings_json": {**settings, "source": source_meta},
    }
