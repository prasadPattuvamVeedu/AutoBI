"""Manual pandas generators for heatmap, pivot table, and correlation heatmap."""

from __future__ import annotations

import pandas as pd

from ..chart_common_operations import (
    format_rows_for_frontend,
    normalize_aggregation,
    numeric_series,
    validate_column,
)


def build_heatmap_chart_data(df, payload, chart_type="heatmap", source_meta=None):
    """Manual implementation for heatmap: x category, y category, numeric value."""
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    x_column = validate_column(df, payload.get("x_column"), "Heatmap X column")
    y_column = validate_column(df, payload.get("y_column"), "Heatmap Y column")
    value_column = validate_column(df, payload.get("size_column") or payload.get("value_column") or payload.get("measure"), "Heatmap value column")
    aggregation = normalize_aggregation(payload.get("aggregation") or settings.get("aggregation"))
    work_df = df[[x_column, y_column, value_column]].copy().dropna(subset=[x_column, y_column])
    if aggregation not in {"count", "nunique"}:
        work_df[value_column] = numeric_series(work_df[value_column])
        work_df = work_df.dropna(subset=[value_column])
    pivot = pd.pivot_table(work_df, index=y_column, columns=x_column, values=value_column, aggfunc=aggregation, fill_value=0)
    long_df = pivot.reset_index().melt(id_vars=y_column, var_name=x_column, value_name="value")
    title = payload.get("title") or f"{value_column} heatmap by {x_column} and {y_column}"
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "value_column": value_column,
            "aggregation": aggregation,
        },
        "chart_data_json": {
            "columns": list(long_df.columns),
            "rows": format_rows_for_frontend(long_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(long_df)), "optimized": True},
        },
        "settings_json": {**settings, "aggregation": aggregation, "source": source_meta},
    }


def build_pivot_table_data(df, payload, chart_type="pivot_table", source_meta=None):
    """Manual implementation for pivot table."""
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    index_column = validate_column(df, payload.get("x_column") or settings.get("index_column"), "Pivot index column")
    columns_column = validate_column(df, payload.get("y_column") or settings.get("columns_column"), "Pivot columns column")
    value_column = validate_column(df, payload.get("size_column") or payload.get("value_column") or settings.get("value_column"), "Pivot value column")
    aggregation = normalize_aggregation(payload.get("aggregation") or settings.get("aggregation"))
    work_df = df[[index_column, columns_column, value_column]].copy().dropna(subset=[index_column, columns_column])
    if aggregation not in {"count", "nunique"}:
        work_df[value_column] = numeric_series(work_df[value_column])
        work_df = work_df.dropna(subset=[value_column])
    pivot_df = pd.pivot_table(work_df, index=index_column, columns=columns_column, values=value_column, aggfunc=aggregation, fill_value=0).reset_index()
    pivot_df.columns = [str(column) for column in pivot_df.columns]
    title = payload.get("title") or f"Pivot table: {value_column}"
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": index_column,
            "y_column": columns_column,
            "value_column": value_column,
            "aggregation": aggregation,
        },
        "chart_data_json": {
            "columns": list(pivot_df.columns),
            "rows": format_rows_for_frontend(pivot_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(pivot_df)), "optimized": True},
        },
        "settings_json": {**settings, "aggregation": aggregation, "source": source_meta},
    }


def build_correlation_heatmap_data(df, payload, chart_type="correlation_heatmap", source_meta=None):
    """Manual implementation for correlation heatmap."""
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    selected_columns = payload.get("columns") or payload.get("selected_columns") or settings.get("columns") or []
    if selected_columns:
        columns = [str(column) for column in selected_columns if str(column) in df.columns]
    else:
        columns = list(df.select_dtypes(include="number").columns[:12])
    if len(columns) < 2:
        raise ValueError("Correlation heatmap needs at least two numeric columns.")
    numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(how="all")
    corr = numeric_df.corr(numeric_only=True).fillna(0)
    rows = []
    for y_col in corr.index:
        for x_col in corr.columns:
            rows.append({"x": str(x_col), "y": str(y_col), "value": round(float(corr.loc[y_col, x_col]), 4)})
    result_df = pd.DataFrame(rows)
    title = payload.get("title") or "Correlation heatmap"
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": "x",
            "y_column": "y",
            "value_column": "value",
            "columns": columns,
            "aggregation": "correlation",
        },
        "chart_data_json": {
            "columns": list(result_df.columns),
            "rows": format_rows_for_frontend(result_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(result_df)), "optimized": True},
        },
        "settings_json": {**settings, "columns": columns, "source": source_meta},
    }
