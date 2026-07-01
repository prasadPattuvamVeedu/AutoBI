"""Rule-based chart recommendation service for AutoBI."""

from __future__ import annotations

import pandas as pd


def _is_numeric(df, column):
    return column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any()


def _is_datetime(df, column):
    if column not in df.columns:
        return False
    return pd.to_datetime(df[column], errors="coerce").notna().mean() > 0.5


def recommend_chart_types(df, selected_columns=None, column_roles=None):
    selected_columns = [c for c in (selected_columns or []) if c in df.columns]
    numeric = [c for c in selected_columns if _is_numeric(df, c)]
    datetime_cols = [c for c in selected_columns if _is_datetime(df, c)]
    categorical = [c for c in selected_columns if c not in numeric]
    lower = {c.lower(): c for c in selected_columns}
    has_lat_lon = any(k in lower for k in ["lat", "latitude"]) and any(k in lower for k in ["lon", "lng", "longitude"])

    output = []
    if categorical and numeric:
        output += ["bar", "horizontal_bar", "pie", "donut"]
    if len(categorical) >= 2 and numeric:
        output += ["grouped_bar", "stacked_bar", "heatmap"]
    if datetime_cols and numeric:
        output += ["line", "area"]
    if len(numeric) >= 2:
        output += ["scatter", "bubble", "correlation_heatmap"]
    if len(numeric) == 1 and not categorical:
        output += ["histogram", "box", "kpi"]
    if has_lat_lon:
        output += ["symbol_map", "bubble_map", "density_map"]
    if categorical:
        output += ["table"]
    return list(dict.fromkeys(output))


def suggest_chart_improvements(df, chart_type, payload):
    suggestions = []
    if chart_type in {"bar", "horizontal_bar", "pie", "donut", "map"} and not payload.get("top_n"):
        suggestions.append({"title": "Apply Top N", "description": "Limit to the most important categories.", "action": "apply_top_n", "payload_patch": {"top_n": 10}})
    if chart_type in {"bar", "line", "area"} and not payload.get("color_by_column"):
        suggestions.append({"title": "Add Color By", "description": "Split the chart by a category for comparison.", "action": "set_color_by", "payload_patch": {}})
    if chart_type in {"line", "area"}:
        suggestions.append({"title": "Add Forecast", "description": "Use this time series for future forecasting later.", "action": "enable_forecast", "payload_patch": {"forecast_enabled": True}})
    return suggestions
