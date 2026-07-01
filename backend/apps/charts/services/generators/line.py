"""Manual pandas generators for line and area charts."""

from __future__ import annotations

import pandas as pd

from ..chart_common_operations import (
    aggregate_chart_data,
    apply_sort,
    build_title,
    format_rows_for_frontend,
    normalize_aggregation,
    validate_column,
)


def _period_series(series, granularity):
    dates = pd.to_datetime(series, errors="coerce")
    granularity = str(granularity or "auto").lower()
    if granularity in {"day", "daily"}:
        return dates.dt.strftime("%Y-%m-%d")
    if granularity in {"month", "monthly", "auto"}:
        return dates.dt.to_period("M").astype(str)
    if granularity in {"quarter", "quarterly"}:
        return dates.dt.to_period("Q").astype(str)
    if granularity in {"year", "yearly"}:
        return dates.dt.to_period("Y").astype(str)
    return dates.dt.strftime("%Y-%m-%d")


def build_line_chart_data(df, payload, chart_type="line", source_meta=None):
    """Manual implementation for line/area charts.

    Flow: date/order column -> period -> group -> aggregate -> sort by period.
    """
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    x_column = validate_column(df, payload.get("x_column") or payload.get("dimension"), "X-axis column")
    y_column = validate_column(df, payload.get("y_column") or payload.get("measure"), "Y-axis column")
    color_by_column = validate_column(df, payload.get("color_by_column") or settings.get("color_by_column"), "Color by", required=False)
    aggregation = normalize_aggregation(payload.get("aggregation") or settings.get("aggregation"))
    granularity = payload.get("date_granularity") or settings.get("date_granularity")

    work_df = df.copy()
    date_values = pd.to_datetime(work_df[x_column], errors="coerce")
    if date_values.notna().any():
        work_df["__period__"] = _period_series(work_df[x_column], granularity)
        group_x = "__period__"
        x_out = "period"
    else:
        group_x = x_column
        x_out = x_column

    group_columns = [group_x] + ([color_by_column] if color_by_column else [])
    result_df = aggregate_chart_data(work_df, group_columns, y_column, aggregation, output_value_column=y_column)
    result_df = apply_sort(result_df, group_x, "ascending")
    if group_x != x_out:
        result_df = result_df.rename(columns={group_x: x_out})

    title = payload.get("title") or build_title(chart_type, x_column, y_column, aggregation)
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": x_out,
            "source_x_column": x_column,
            "y_column": y_column,
            "color_by_column": color_by_column,
            "aggregation": aggregation,
            "date_granularity": granularity or "auto",
        },
        "chart_data_json": {
            "columns": list(result_df.columns),
            "rows": format_rows_for_frontend(result_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(result_df)), "optimized": True},
        },
        "settings_json": {**settings, "aggregation": aggregation, "source": source_meta},
    }
