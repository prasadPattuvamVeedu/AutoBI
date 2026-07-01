"""Manual pandas generator for table charts."""

from __future__ import annotations

from ..chart_common_operations import apply_sort, format_rows_for_frontend, limit_chart_rows


def build_table_chart_data(df, payload, chart_type="table", source_meta=None):
    """Manual implementation for table data: select columns, sort, limit/paginate."""
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    requested_columns = payload.get("columns") or payload.get("selected_columns") or settings.get("columns") or []
    selected_columns = [str(column) for column in requested_columns if str(column) in df.columns]
    if not selected_columns:
        selected_columns = list(df.columns[:25])
    table_df = df[selected_columns].copy()
    sort_by = payload.get("sort_by") or settings.get("sort_by")
    if sort_by in table_df.columns:
        table_df = apply_sort(table_df, sort_by, payload.get("sort_order") or settings.get("sort_order"))
    table_df, limit = limit_chart_rows(table_df, payload.get("top_n") or payload.get("limit") or settings.get("top_n") or 50, max_limit=500)
    title = payload.get("title") or "Data table"
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": selected_columns[0] if selected_columns else "",
            "y_column": selected_columns[1] if len(selected_columns) > 1 else "",
            "top_n": limit,
            "columns": selected_columns,
        },
        "chart_data_json": {
            "columns": selected_columns,
            "rows": format_rows_for_frontend(table_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(table_df)), "optimized": True},
        },
        "settings_json": {**settings, "top_n": limit, "source": source_meta},
    }
