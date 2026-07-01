"""Manual pandas generators for scatter and bubble charts."""

from __future__ import annotations

from ..chart_common_operations import (
    build_title,
    clean_chart_missing_values,
    convert_to_numeric_safe,
    format_rows_for_frontend,
    limit_chart_rows,
    validate_column,
)


def build_scatter_chart_data(df, payload, chart_type="scatter", source_meta=None):
    """Manual implementation for scatter charts: numeric x/y, optional color, no aggregation."""
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    x_column = validate_column(df, payload.get("x_column"), "X-axis column")
    y_column = validate_column(df, payload.get("y_column"), "Y-axis column")
    color_by_column = validate_column(df, payload.get("color_by_column") or settings.get("color_by_column"), "Color by", required=False)
    columns = [x_column, y_column] + ([color_by_column] if color_by_column else [])
    work_df = df[columns].copy()
    work_df = convert_to_numeric_safe(work_df, [x_column, y_column])
    work_df = clean_chart_missing_values(work_df, [x_column, y_column])
    work_df, limit = limit_chart_rows(work_df, payload.get("top_n") or payload.get("limit") or settings.get("top_n") or 1000, max_limit=5000)

    title = payload.get("title") or build_title(chart_type, x_column, y_column, "")
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "color_by_column": color_by_column,
            "top_n": limit,
        },
        "chart_data_json": {
            "columns": list(work_df.columns),
            "rows": format_rows_for_frontend(work_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(work_df)), "optimized": True},
        },
        "settings_json": {**settings, "top_n": limit, "source": source_meta},
    }


def build_bubble_chart_data(df, payload, chart_type="bubble", source_meta=None):
    """Manual implementation for bubble charts: numeric x/y/size, optional color."""
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    x_column = validate_column(df, payload.get("x_column"), "X-axis column")
    y_column = validate_column(df, payload.get("y_column"), "Y-axis column")
    size_column = validate_column(df, payload.get("size_column"), "Size column")
    color_by_column = validate_column(df, payload.get("color_by_column") or settings.get("color_by_column"), "Color by", required=False)
    columns = [x_column, y_column, size_column] + ([color_by_column] if color_by_column else [])
    work_df = df[columns].copy()
    work_df = convert_to_numeric_safe(work_df, [x_column, y_column, size_column])
    work_df = clean_chart_missing_values(work_df, [x_column, y_column, size_column])
    work_df, limit = limit_chart_rows(work_df, payload.get("top_n") or payload.get("limit") or settings.get("top_n") or 1000, max_limit=5000)

    title = payload.get("title") or build_title(chart_type, x_column, y_column, "")
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "size_column": size_column,
            "color_by_column": color_by_column,
            "top_n": limit,
        },
        "chart_data_json": {
            "columns": list(work_df.columns),
            "rows": format_rows_for_frontend(work_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(work_df)), "optimized": True},
        },
        "settings_json": {**settings, "top_n": limit, "source": source_meta},
    }
