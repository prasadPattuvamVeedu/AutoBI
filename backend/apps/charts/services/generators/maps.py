"""Manual pandas generators for map chart family."""

from __future__ import annotations

from ..chart_common_operations import (
    aggregate_chart_data,
    apply_top_n,
    clean_chart_missing_values,
    convert_to_numeric_safe,
    format_rows_for_frontend,
    limit_chart_rows,
    normalize_aggregation,
    validate_column,
)


def build_map_chart_data(df, payload, chart_type=None, source_meta=None):
    """Router for all map chart types."""
    chart_type = chart_type or payload.get("chart_type") or "map"
    if chart_type in {"map", "region_map"}:
        return build_region_map_data(df, payload, chart_type, source_meta)
    if chart_type in {"filled_map", "choropleth_map"}:
        return build_filled_map_data(df, payload, chart_type, source_meta)
    if chart_type in {"symbol_map", "point_map", "lat_lon_map"}:
        return build_symbol_map_data(df, payload, chart_type, source_meta)
    if chart_type == "bubble_map":
        return build_bubble_map_data(df, payload, chart_type, source_meta)
    if chart_type in {"density_map", "heat_density_map"}:
        return build_density_map_data(df, payload, chart_type, source_meta)
    if chart_type in {"flow_map", "connection_map", "route_map"}:
        return build_flow_map_data(df, payload, chart_type, source_meta)
    return build_region_map_data(df, payload, chart_type, source_meta)


def _map_response(df, payload, chart_type, source_meta, result_df, config, settings):
    title = payload.get("title") or config.get("title") or "Map chart"
    config = {"title": title, "description": payload.get("description") or "", "chart_type": chart_type, **config}
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": config,
        "chart_data_json": {
            "columns": list(result_df.columns),
            "rows": format_rows_for_frontend(result_df),
            "meta": {
                "placeholder": False,
                "row_count": int(len(result_df)),
                "source": source_meta or {},
                "optimized": True,
                "note": "Map preview uses location names or coordinates. Add geocoding/topojson on frontend or connector layer if needed.",
            },
        },
        "settings_json": {**settings, "source": source_meta or {}},
    }


def build_region_map_data(df, payload, chart_type="map", source_meta=None):
    """Manual implementation for location/region map: location + value aggregation."""
    settings = payload.get("settings_json") or {}
    aggregation = normalize_aggregation(payload.get("aggregation") or settings.get("aggregation") or "sum")
    location_column = validate_column(df, payload.get("location_column") or payload.get("x_column") or payload.get("dimension"), "Location column")
    value_column = payload.get("y_column") or payload.get("measure") or payload.get("size_column")
    if aggregation == "count" and not value_column:
        value_column = location_column
    value_column = validate_column(df, value_column, "Map value", required=aggregation != "count") or location_column
    result_df = aggregate_chart_data(df, location_column, value_column, aggregation, output_value_column="value")
    result_df = apply_top_n(result_df, "value", payload.get("top_n") or settings.get("top_n") or 500, "descending")
    return _map_response(
        df,
        payload,
        chart_type,
        source_meta,
        result_df,
        {
            "x_column": location_column,
            "y_column": "value",
            "location_column": location_column,
            "value_column": value_column,
            "aggregation": aggregation,
            "top_n": payload.get("top_n") or settings.get("top_n") or 500,
            "map_mode": "location",
        },
        {**settings, "aggregation": aggregation},
    )


def build_filled_map_data(df, payload, chart_type="filled_map", source_meta=None):
    """Manual implementation for filled/choropleth map. Same data shape as region map."""
    return build_region_map_data(df, payload, chart_type, source_meta)


def build_symbol_map_data(df, payload, chart_type="symbol_map", source_meta=None):
    """Manual implementation for symbol/point/lat-lon map."""
    settings = payload.get("settings_json") or {}
    latitude_column = validate_column(df, payload.get("latitude_column") or payload.get("lat_column"), "Latitude column")
    longitude_column = validate_column(df, payload.get("longitude_column") or payload.get("lon_column") or payload.get("lng_column"), "Longitude column")
    value_column = validate_column(df, payload.get("y_column") or payload.get("measure") or payload.get("size_column"), "Map value", required=False)
    label_column = validate_column(df, payload.get("label_column") or payload.get("location_column") or payload.get("x_column"), "Map label", required=False)
    color_by_column = validate_column(df, payload.get("color_by_column") or settings.get("color_by_column"), "Color by", required=False)
    columns = [latitude_column, longitude_column] + [c for c in [value_column, label_column, color_by_column] if c]
    work_df = df[list(dict.fromkeys(columns))].copy()
    work_df = convert_to_numeric_safe(work_df, [latitude_column, longitude_column] + ([value_column] if value_column else []))
    work_df = clean_chart_missing_values(work_df, [latitude_column, longitude_column])
    work_df, limit = limit_chart_rows(work_df, payload.get("top_n") or payload.get("limit") or settings.get("top_n") or 2000, max_limit=5000)
    if work_df.empty:
        raise ValueError("Map chart needs valid latitude and longitude values.")
    return _map_response(
        df,
        payload,
        chart_type,
        source_meta,
        work_df,
        {
            "x_column": longitude_column,
            "y_column": latitude_column,
            "latitude_column": latitude_column,
            "longitude_column": longitude_column,
            "value_column": value_column,
            "label_column": label_column,
            "color_by_column": color_by_column,
            "top_n": limit,
            "map_mode": "coordinates",
        },
        {**settings, "top_n": limit},
    )


def build_bubble_map_data(df, payload, chart_type="bubble_map", source_meta=None):
    """Manual implementation for bubble map: lat/lon + size."""
    result = build_symbol_map_data(df, payload, chart_type, source_meta)
    size_column = validate_column(df, payload.get("size_column"), "Map size column")
    result["chart_config_json"]["size_column"] = size_column
    return result


def build_density_map_data(df, payload, chart_type="density_map", source_meta=None):
    """Manual implementation for density/heat map: lat/lon + optional intensity."""
    result = build_symbol_map_data(df, payload, chart_type, source_meta)
    intensity_column = payload.get("intensity_column") or payload.get("value_column") or payload.get("size_column")
    if intensity_column and intensity_column in df.columns:
        result["chart_config_json"]["intensity_column"] = intensity_column
    return result


def build_flow_map_data(df, payload, chart_type="flow_map", source_meta=None):
    """Manual implementation for source-target flow/connection map."""
    settings = payload.get("settings_json") or {}
    source_latitude = validate_column(df, payload.get("source_latitude_column"), "Source latitude")
    source_longitude = validate_column(df, payload.get("source_longitude_column"), "Source longitude")
    target_latitude = validate_column(df, payload.get("target_latitude_column"), "Target latitude")
    target_longitude = validate_column(df, payload.get("target_longitude_column"), "Target longitude")
    value_column = validate_column(df, payload.get("value_column") or payload.get("y_column") or payload.get("measure"), "Flow value", required=False)
    columns = [source_latitude, source_longitude, target_latitude, target_longitude] + ([value_column] if value_column else [])
    work_df = df[columns].copy()
    work_df = convert_to_numeric_safe(work_df, columns)
    work_df = clean_chart_missing_values(work_df, [source_latitude, source_longitude, target_latitude, target_longitude])
    work_df, limit = limit_chart_rows(work_df, payload.get("top_n") or payload.get("limit") or settings.get("top_n") or 2000, max_limit=5000)
    return _map_response(
        df,
        payload,
        chart_type,
        source_meta,
        work_df,
        {
            "source_latitude_column": source_latitude,
            "source_longitude_column": source_longitude,
            "target_latitude_column": target_latitude,
            "target_longitude_column": target_longitude,
            "value_column": value_column,
            "top_n": limit,
            "map_mode": "flow",
        },
        {**settings, "top_n": limit},
    )
