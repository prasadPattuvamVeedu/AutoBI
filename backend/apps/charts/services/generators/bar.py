"""Manual pandas generators for bar-family charts."""

from __future__ import annotations

from ..chart_common_operations import (
    aggregate_chart_data,
    apply_sort,
    apply_top_n,
    build_title,
    format_rows_for_frontend,
    normalize_aggregation,
    normalize_sort_order,
    unique_keep_order,
    validate_column,
)


def _base_response(df, payload, chart_type, source_meta, result_df, x_column, y_column, aggregation, settings, extra_config=None):
    title = payload.get("title") or build_title(chart_type, x_column, y_column, aggregation)
    config = {
        "title": title,
        "description": payload.get("description") or "",
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "aggregation": aggregation,
        "top_n": payload.get("top_n") or settings.get("top_n"),
        "sort_order": payload.get("sort_order") or settings.get("sort_order"),
        "sort_by": payload.get("sort_by") or settings.get("sort_by"),
        "color_by_column": payload.get("color_by_column") or settings.get("color_by_column"),
        "group_by_column": payload.get("group_by_column") or settings.get("group_by_column"),
        "secondary_y_column": payload.get("secondary_y_column") or settings.get("secondary_y_column"),
    }
    if extra_config:
        config.update(extra_config)
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": config,
        "chart_data_json": {
            "columns": list(result_df.columns),
            "rows": format_rows_for_frontend(result_df),
            "meta": {
                "placeholder": False,
                "source": source_meta,
                "row_count": int(len(result_df)),
                "optimized": True,
            },
        },
        "settings_json": {**settings, "aggregation": aggregation, "source": source_meta},
    }


def build_bar_chart_data(df, payload, chart_type="bar", source_meta=None):
    """Manual implementation for bar and horizontal bar.

    Flow: group by x_column -> aggregate y_column -> sort -> Top N -> rows.
    """
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    x_column = validate_column(df, payload.get("x_column") or payload.get("dimension"), "X-axis column")
    aggregation = normalize_aggregation(payload.get("aggregation") or settings.get("aggregation"))
    y_column = payload.get("y_column") or payload.get("measure")
    if aggregation == "count" and not y_column:
        y_column = x_column
    y_column = validate_column(df, y_column, "Y-axis column", required=aggregation != "count") or x_column
    color_by_column = validate_column(df, payload.get("color_by_column") or settings.get("color_by_column"), "Color by", required=False)

    group_columns = [x_column] + ([color_by_column] if color_by_column else [])
    result_df = aggregate_chart_data(df, group_columns, y_column, aggregation, output_value_column=y_column)
    sort_order = normalize_sort_order(payload.get("sort_order") or settings.get("sort_order"))
    sort_by = payload.get("sort_by") or settings.get("sort_by") or y_column
    result_df = apply_sort(result_df, sort_by if sort_by in result_df.columns else y_column, sort_order)
    result_df = apply_top_n(result_df, y_column, payload.get("top_n") or settings.get("top_n"), sort_order)

    return _base_response(df, payload, chart_type, source_meta, result_df, x_column, y_column, aggregation, settings)


def build_grouped_bar_chart_data(df, payload, chart_type="grouped_bar", source_meta=None):
    """Manual implementation for grouped bar.

    Flow: group by x + color/group -> aggregate -> pivot for frontend series.
    """
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    x_column = validate_column(df, payload.get("x_column") or payload.get("dimension"), "X-axis column")
    y_column = validate_column(df, payload.get("y_column") or payload.get("measure"), "Y-axis column")
    group_column = validate_column(
        df,
        payload.get("color_by_column") or payload.get("group_by_column") or settings.get("color_by_column") or settings.get("group_by_column"),
        "Group/color column",
    )
    aggregation = normalize_aggregation(payload.get("aggregation") or settings.get("aggregation"))

    grouped = aggregate_chart_data(df, [x_column, group_column], y_column, aggregation, output_value_column=y_column)
    pivot_df = grouped.pivot_table(index=x_column, columns=group_column, values=y_column, aggfunc="sum", fill_value=0).reset_index()
    pivot_df.columns = [str(column) for column in pivot_df.columns]
    series = [column for column in pivot_df.columns if column != x_column]
    sort_order = normalize_sort_order(payload.get("sort_order") or settings.get("sort_order"))
    if series:
        pivot_df["__total__"] = pivot_df[series].sum(axis=1)
        pivot_df = apply_sort(pivot_df, "__total__", sort_order)
        pivot_df = apply_top_n(pivot_df, "__total__", payload.get("top_n") or settings.get("top_n"), sort_order)
        pivot_df = pivot_df.drop(columns=["__total__"])

    return _base_response(
        df,
        payload,
        chart_type,
        source_meta,
        pivot_df,
        x_column,
        y_column,
        aggregation,
        settings,
        extra_config={"series": series, "group_by_column": group_column, "color_by_column": group_column},
    )


def build_stacked_bar_chart_data(df, payload, chart_type="stacked_bar", source_meta=None):
    """Manual implementation for stacked bar. Backend shape is same as grouped bar."""
    return build_grouped_bar_chart_data(df, payload, chart_type=chart_type, source_meta=source_meta)


def build_combo_chart_data(df, payload, chart_type="combo", source_meta=None):
    """Combo uses grouped bar base data plus frontend renders mixed bar/line series."""
    return build_grouped_bar_chart_data(df, payload, chart_type=chart_type, source_meta=source_meta)
