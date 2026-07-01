"""Manual numpy/pandas generator for histogram charts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..chart_common_operations import build_title, format_bin_value, format_rows_for_frontend, numeric_series, safe_int, validate_column


def build_histogram_chart_data(df, payload, chart_type="histogram", source_meta=None):
    """Manual implementation for histogram: x numeric, y automatic count/frequency."""
    source_meta = source_meta or {}
    settings = payload.get("settings_json") or {}
    x_column = validate_column(df, payload.get("x_column") or payload.get("y_column"), "Histogram column")
    bins = safe_int(payload.get("bins") or settings.get("bins"), 12)
    bins = min(max(bins, 2), 100)
    series = numeric_series(df[x_column]).dropna()
    if series.empty:
        raise ValueError(f"Column '{x_column}' does not contain numeric values for a histogram.")
    counts, edges = np.histogram(series, bins=bins)
    rows = []
    for index, count in enumerate(counts):
        start = edges[index]
        end = edges[index + 1]
        rows.append({
            "bin": f"{format_bin_value(start)} - {format_bin_value(end)}",
            "bin_start": float(start),
            "bin_end": float(end),
            "count": int(count),
        })
    result_df = pd.DataFrame(rows)
    title = payload.get("title") or build_title(chart_type, x_column, "count", "count")
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "description": payload.get("description") or "",
            "chart_type": chart_type,
            "x_column": "bin",
            "source_x_column": x_column,
            "y_column": "count",
            "bins": bins,
            "aggregation": "count",
        },
        "chart_data_json": {
            "columns": list(result_df.columns),
            "rows": format_rows_for_frontend(result_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(result_df)), "bins": bins, "optimized": True},
        },
        "settings_json": {**settings, "bins": bins, "source": source_meta},
    }
