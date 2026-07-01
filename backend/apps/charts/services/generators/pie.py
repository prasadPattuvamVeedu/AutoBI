"""Manual pandas generators for pie and donut charts."""

from __future__ import annotations

from ..chart_common_operations import (
    aggregate_chart_data,
    apply_sort,
    build_title,
    format_rows_for_frontend,
    group_pie_other,
    normalize_aggregation,
    validate_column,
)


def build_pie_chart_data(df, payload, chart_type="pie", source_meta=None):
    """Manual implementation for pie/donut charts."""
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    x_column = validate_column(df, payload.get("x_column") or payload.get("dimension"), "Category column")
    aggregation = normalize_aggregation(payload.get("aggregation") or settings.get("aggregation"))
    y_column = payload.get("y_column") or payload.get("measure")
    if aggregation == "count" and not y_column:
        y_column = x_column
    y_column = validate_column(df, y_column, "Value column", required=aggregation != "count") or x_column

    result_df = aggregate_chart_data(df, x_column, y_column, aggregation, output_value_column="value")
    result_df = apply_sort(result_df, "value", "descending")
    result_df = group_pie_other(result_df, x_column, "value", payload.get("top_n") or settings.get("top_n") or 10)
    total = float(result_df["value"].sum()) if not result_df.empty else 0
    result_df["percentage"] = result_df["value"].apply(lambda value: round((float(value) / total) * 100, 2) if total else 0)
    result_df = result_df.rename(columns={x_column: "label"})

    title = payload.get("title") or build_title(chart_type, x_column, y_column, aggregation)
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": "label",
            "source_x_column": x_column,
            "y_column": "value",
            "source_y_column": y_column,
            "aggregation": aggregation,
            "top_n": payload.get("top_n") or settings.get("top_n") or 10,
        },
        "chart_data_json": {
            "columns": list(result_df.columns),
            "rows": format_rows_for_frontend(result_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(result_df)), "optimized": True},
        },
        "settings_json": {**settings, "aggregation": aggregation, "source": source_meta},
    }
