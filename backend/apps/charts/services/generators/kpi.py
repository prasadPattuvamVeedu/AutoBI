"""Manual pandas generator for KPI cards."""

from __future__ import annotations

from apps.datasets.services import make_json_safe

from ..chart_common_operations import build_title, normalize_aggregation, numeric_series, validate_column


def build_kpi_chart_data(df, payload, chart_type="kpi", source_meta=None):
    """Manual implementation for KPI card calculation."""
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    y_column = validate_column(df, payload.get("y_column") or payload.get("measure"), "KPI value column")
    aggregation = normalize_aggregation(payload.get("aggregation") or settings.get("aggregation"))

    if aggregation == "count":
        value = int(df[y_column].count())
    elif aggregation == "nunique":
        value = int(df[y_column].nunique(dropna=True))
    elif aggregation in {"first", "last"}:
        series = df[y_column].dropna()
        value = series.iloc[0] if aggregation == "first" and not series.empty else series.iloc[-1] if not series.empty else None
    else:
        series = numeric_series(df[y_column]).dropna()
        if series.empty:
            raise ValueError(f"Column '{y_column}' does not contain numeric values for KPI aggregation.")
        value = getattr(series, aggregation)()

    title = payload.get("title") or build_title("kpi", "", y_column, aggregation)
    row = {"metric": title, "value": make_json_safe(value)}
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": "metric",
            "y_column": "value",
            "size_column": "value",
            "source_y_column": y_column,
            "aggregation": aggregation,
        },
        "chart_data_json": {
            "columns": ["metric", "value"],
            "rows": [row],
            "meta": {"placeholder": False, "source": source_meta, "aggregation": aggregation, "optimized": True},
        },
        "settings_json": {**settings, "aggregation": aggregation, "source": source_meta},
    }
