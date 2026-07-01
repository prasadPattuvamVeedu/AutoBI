"""Chart payload validation rules for AutoBI chart generation."""

from __future__ import annotations

import pandas as pd

from .chart_common_operations import validate_column


class ChartGenerationError(ValueError):
    """Raised when chart data cannot be generated from the selected dataset."""


NUMERIC_REQUIRED_CHARTS = {"scatter", "bubble", "histogram", "box", "box_plot", "correlation_heatmap"}
MAP_CHART_TYPES = {
    "map",
    "region_map",
    "filled_map",
    "choropleth_map",
    "symbol_map",
    "point_map",
    "lat_lon_map",
    "bubble_map",
    "density_map",
    "heat_density_map",
    "flow_map",
    "connection_map",
    "route_map",
}


def validate_numeric_column(df: pd.DataFrame, column: str, label: str) -> str:
    column = validate_column(df, column, label, required=True)
    numeric = pd.to_numeric(df[column], errors="coerce")
    if numeric.dropna().empty:
        raise ChartGenerationError(f"{label} '{column}' must contain numeric values.")
    return column


def validate_chart_payload(df: pd.DataFrame, payload: dict) -> bool:
    chart_type = str(payload.get("chart_type") or "bar").strip().lower()

    try:
        if chart_type in {"bar", "horizontal_bar", "grouped_bar", "stacked_bar", "combo", "pie", "donut"}:
            validate_dimension_measure_chart(df, payload)
        elif chart_type in {"line", "area"}:
            validate_line_chart(df, payload)
        elif chart_type == "histogram":
            validate_histogram_chart(df, payload)
        elif chart_type in {"scatter", "bubble"}:
            validate_scatter_chart(df, payload)
        elif chart_type in {"box", "box_plot"}:
            validate_box_chart(df, payload)
        elif chart_type == "heatmap":
            validate_heatmap_chart(df, payload)
        elif chart_type == "pivot_table":
            validate_pivot_table_chart(df, payload)
        elif chart_type == "correlation_heatmap":
            validate_correlation_heatmap_chart(df, payload)
        elif chart_type == "kpi":
            validate_kpi_chart(df, payload)
        elif chart_type == "table":
            validate_table_chart(df, payload)
        elif chart_type in MAP_CHART_TYPES:
            validate_map_chart(df, payload)
    except ValueError as exc:
        raise ChartGenerationError(str(exc)) from exc
    return True


def validate_dimension_measure_chart(df: pd.DataFrame, payload: dict) -> bool:
    validate_column(df, payload.get("x_column") or payload.get("dimension"), "X-axis column")
    aggregation = str(payload.get("aggregation") or (payload.get("settings_json") or {}).get("aggregation") or "sum").lower()
    y_column = payload.get("y_column") or payload.get("measure")
    if aggregation != "count":
        validate_column(df, y_column, "Y-axis column")
    return True


def validate_line_chart(df: pd.DataFrame, payload: dict) -> bool:
    validate_column(df, payload.get("x_column") or payload.get("dimension"), "X-axis column")
    validate_column(df, payload.get("y_column") or payload.get("measure"), "Y-axis column")
    return True


def validate_histogram_chart(df: pd.DataFrame, payload: dict) -> bool:
    validate_numeric_column(df, payload.get("x_column") or payload.get("y_column"), "Histogram column")
    return True


def validate_scatter_chart(df: pd.DataFrame, payload: dict) -> bool:
    validate_numeric_column(df, payload.get("x_column"), "X-axis column")
    validate_numeric_column(df, payload.get("y_column"), "Y-axis column")
    if str(payload.get("chart_type") or "").lower() == "bubble":
        validate_numeric_column(df, payload.get("size_column"), "Size column")
    elif payload.get("size_column"):
        validate_numeric_column(df, payload.get("size_column"), "Size column")
    return True


def validate_box_chart(df: pd.DataFrame, payload: dict) -> bool:
    validate_numeric_column(df, payload.get("y_column") or payload.get("x_column"), "Box plot value column")
    return True


def validate_heatmap_chart(df: pd.DataFrame, payload: dict) -> bool:
    validate_column(df, payload.get("x_column"), "Heatmap X column")
    validate_column(df, payload.get("y_column"), "Heatmap Y column")
    validate_column(df, payload.get("size_column") or payload.get("value_column") or payload.get("measure"), "Heatmap value column")
    return True


def validate_pivot_table_chart(df: pd.DataFrame, payload: dict) -> bool:
    settings = payload.get("settings_json") or {}
    validate_column(df, payload.get("x_column") or settings.get("index_column"), "Pivot index column")
    validate_column(df, payload.get("y_column") or settings.get("columns_column"), "Pivot columns column")
    validate_column(df, payload.get("size_column") or payload.get("value_column") or settings.get("value_column"), "Pivot value column")
    return True


def validate_correlation_heatmap_chart(df: pd.DataFrame, payload: dict) -> bool:
    settings = payload.get("settings_json") or {}
    columns = payload.get("columns") or payload.get("selected_columns") or settings.get("columns") or []
    if columns:
        for column in columns:
            validate_column(df, column, "Correlation column")
    return True


def validate_kpi_chart(df: pd.DataFrame, payload: dict) -> bool:
    validate_column(df, payload.get("y_column") or payload.get("measure"), "KPI value column")
    return True


def validate_table_chart(df: pd.DataFrame, payload: dict) -> bool:
    settings = payload.get("settings_json") or {}
    columns = payload.get("columns") or payload.get("selected_columns") or settings.get("columns") or []
    for column in columns:
        validate_column(df, column, "Table column", required=False)
    return True


def validate_map_chart(df: pd.DataFrame, payload: dict) -> bool:
    latitude = payload.get("latitude_column") or payload.get("lat_column")
    longitude = payload.get("longitude_column") or payload.get("lon_column") or payload.get("lng_column")
    location = payload.get("location_column") or payload.get("x_column") or payload.get("dimension")
    chart_type = str(payload.get("chart_type") or "map").lower()

    if chart_type in {"flow_map", "connection_map", "route_map"}:
        required = [
            (payload.get("source_latitude_column"), "Source latitude column"),
            (payload.get("source_longitude_column"), "Source longitude column"),
            (payload.get("target_latitude_column"), "Target latitude column"),
            (payload.get("target_longitude_column"), "Target longitude column"),
        ]
        for column, label in required:
            validate_numeric_column(df, column, label)
        return True

    if latitude and longitude:
        validate_numeric_column(df, latitude, "Latitude column")
        validate_numeric_column(df, longitude, "Longitude column")
        if chart_type == "bubble_map":
            validate_numeric_column(df, payload.get("size_column"), "Map size column")
        return True

    if location:
        validate_column(df, location, "Location column")
        return True

    raise ChartGenerationError("Map chart needs either a location column or latitude and longitude columns.")
