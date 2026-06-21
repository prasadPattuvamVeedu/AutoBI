import math

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from apps.datasets.models import DatasetVersion
from apps.datasets.services import get_dataset_file_extension, make_json_safe, read_dataset_file


AGGREGATION_MAP = {
    "sum": "sum",
    "mean": "mean",
    "avg": "mean",
    "average": "mean",
    "median": "median",
    "count": "count",
    "nunique": "nunique",
    "unique_count": "nunique",
    "min": "min",
    "max": "max",
}

VERSION_PRIORITY = {
    DatasetVersion.VERSION_TYPE_ML_READY: 4,
    DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED: 3,
    DatasetVersion.VERSION_TYPE_CLEANED: 2,
    DatasetVersion.VERSION_TYPE_ORIGINAL: 1,
}

SUPPORTED_CHART_TYPES = {
    "bar",
    "horizontal_bar",
    "grouped_bar",
    "stacked_bar",
    "line",
    "area",
    "pie",
    "donut",
    "scatter",
    "bubble",
    "histogram",
    "box",
    "heatmap",
    "correlation_heatmap",
    "map",
    "symbol_map",
    "filled_map",
    "kpi",
    "table",
    "pivot_table",
}

GROUPED_CHART_TYPES = {
    "bar",
    "horizontal_bar",
    "grouped_bar",
    "stacked_bar",
    "line",
    "area",
    "pie",
    "donut",
}

DISTRIBUTION_CHART_TYPES = {"histogram", "box"}
MAP_CHART_TYPES = {"map", "symbol_map", "filled_map"}


class ChartGenerationError(ValueError):
    """Raised when chart data cannot be generated from the selected dataset."""


def _is_file_available(file_field):
    return bool(file_field and getattr(file_field, "name", ""))


def get_preferred_dataset_source(dataset):
    """Return the best available source for visualization.

    Priority:
    1. ML-ready version
    2. Feature-engineered version
    3. Cleaned version
    4. Latest active dataset version
    5. Original uploaded dataset file
    """
    try:
        versions = list(dataset.versions.exclude(file=""))
    except Exception:
        versions = []

    versions = [version for version in versions if _is_file_available(version.file)]

    if versions:
        preferred_versions = [
            version
            for version in versions
            if version.version_type
            in {
                DatasetVersion.VERSION_TYPE_ML_READY,
                DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED,
                DatasetVersion.VERSION_TYPE_CLEANED,
            }
        ]

        if preferred_versions:
            selected = sorted(
                preferred_versions,
                key=lambda version: (
                    VERSION_PRIORITY.get(version.version_type, 0),
                    bool(version.is_active),
                    version.version_number,
                    version.created_at,
                ),
                reverse=True,
            )[0]

            return selected.file, {
                "source_kind": "dataset_version",
                "version_id": selected.id,
                "version_number": selected.version_number,
                "version_type": selected.version_type,
                "is_active": selected.is_active,
                "uses_cleaned_or_feature_engineered_data": True,
            }

        active_versions = [version for version in versions if version.is_active]
        selected = sorted(
            active_versions or versions,
            key=lambda version: (version.version_number, version.created_at),
            reverse=True,
        )[0]

        return selected.file, {
            "source_kind": "dataset_version",
            "version_id": selected.id,
            "version_number": selected.version_number,
            "version_type": selected.version_type,
            "is_active": selected.is_active,
            "uses_cleaned_or_feature_engineered_data": selected.version_type
            in {
                DatasetVersion.VERSION_TYPE_ML_READY,
                DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED,
                DatasetVersion.VERSION_TYPE_CLEANED,
            },
        }

    if _is_file_available(dataset.file):
        return dataset.file, {
            "source_kind": "dataset",
            "version_id": None,
            "version_number": None,
            "version_type": "original_upload",
            "is_active": True,
            "uses_cleaned_or_feature_engineered_data": False,
        }

    raise ChartGenerationError("No readable dataset file or dataset version is available for chart generation.")


def _unique_keep_order(values):
    output = []
    for value in values:
        if value is None or value == "":
            continue
        value = str(value)
        if value not in output:
            output.append(value)
    return output


def _required_columns_for_payload(payload, chart_type):
    settings = payload.get("settings_json") or {}

    if chart_type == "table":
        selected = payload.get("columns") or payload.get("selected_columns") or settings.get("columns") or []
        return _unique_keep_order(selected)

    if chart_type == "correlation_heatmap":
        selected = payload.get("columns") or payload.get("selected_columns") or settings.get("columns") or []
        return _unique_keep_order(selected)

    columns = [
        payload.get("x_column") or payload.get("dimension"),
        payload.get("y_column") or payload.get("measure"),
        payload.get("size_column"),
        payload.get("color_by_column"),
        payload.get("group_by_column"),
        payload.get("latitude_column") or payload.get("lat_column"),
        payload.get("longitude_column") or payload.get("lon_column") or payload.get("lng_column"),
        payload.get("location_column"),
    ]
    return _unique_keep_order(columns)


def _read_dataset_file_optimized(file_field, usecols=None):
    """Read only needed columns when possible.

    Falls back to the existing project read_dataset_file() if a column-projection read fails.
    """
    extension = get_dataset_file_extension(file_field.name)
    usecols_set = set(usecols or [])

    if usecols_set:
        try:
            file_field.seek(0)
            if extension == ".csv":
                return pd.read_csv(file_field, usecols=lambda column: str(column) in usecols_set)
            if extension in {".xlsx", ".xls"}:
                return pd.read_excel(file_field, usecols=lambda column: str(column) in usecols_set)
        except (EmptyDataError, ParserError, UnicodeDecodeError, ValueError, TypeError):
            pass

    file_field.seek(0)
    return read_dataset_file(file_field)


def _read_preferred_dataframe(dataset, required_columns=None):
    source_file, source_meta = get_preferred_dataset_source(dataset)
    df = _read_dataset_file_optimized(source_file, usecols=required_columns)
    if df.empty:
        raise ChartGenerationError("The selected dataset source is empty.")
    df.columns = [str(column) for column in df.columns]
    return df, source_meta


def _safe_int(value, default):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_chart_type(chart_type):
    chart_type = (chart_type or "bar").strip().lower()
    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ChartGenerationError(f"Chart type '{chart_type}' is not supported.")
    return chart_type


def _normalize_aggregation(aggregation):
    return AGGREGATION_MAP.get((aggregation or "sum").strip().lower(), "sum")


def _validate_column(df, column_name, label, required=True):
    if not column_name:
        if required:
            raise ChartGenerationError(f"{label} is required for this chart.")
        return ""

    column_name = str(column_name)
    if column_name not in df.columns:
        raise ChartGenerationError(f"{label} '{column_name}' was not found in the selected dataset source.")
    return column_name


def _to_records(df):
    return [
        {str(key): make_json_safe(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def _limit_df(df, limit, default_limit, max_limit):
    limit = _safe_int(limit, default_limit)
    limit = min(max(limit, 1), max_limit)
    return df.head(limit), limit


def _apply_top_n(result_df, value_column, top_n=20, sort_order="descending"):
    """Optimized top/bottom N for chart output.

    Use this after grouping/aggregation, not before aggregation.
    """
    if not top_n or value_column not in result_df.columns:
        return result_df

    top_n = _safe_int(top_n, 20)
    if top_n <= 0 or len(result_df) <= top_n:
        return result_df

    numeric_value = pd.to_numeric(result_df[value_column], errors="coerce")
    sortable_df = result_df.assign(**{value_column: numeric_value}).dropna(subset=[value_column])
    if sortable_df.empty:
        return result_df.head(top_n)

    if sort_order == "ascending":
        return sortable_df.nsmallest(top_n, value_column)
    return sortable_df.nlargest(top_n, value_column)


def _group_pie_other(result_df, x_column, value_column, top_n=10):
    """Limit pie/donut categories and group the rest into Other."""
    top_n = _safe_int(top_n, 10)
    top_n = max(2, min(top_n, 15))
    if len(result_df) <= top_n:
        return result_df

    top_df = result_df.nlargest(top_n, value_column)
    other_df = result_df.drop(index=top_df.index)
    other_value = other_df[value_column].sum()
    if other_value > 0:
        other_row = {column: None for column in result_df.columns}
        other_row[x_column] = "Other"
        other_row[value_column] = other_value
        result_df = pd.concat([top_df, pd.DataFrame([other_row])], ignore_index=True)
    else:
        result_df = top_df
    return result_df


def _build_title(chart_type, x_column, y_column, aggregation):
    if chart_type == "kpi":
        return f"{aggregation.title()} of {y_column}"
    if chart_type == "table":
        return "Data table"
    if chart_type == "histogram":
        return f"Distribution of {x_column or y_column}"
    if chart_type == "box":
        return f"Box plot of {y_column}"
    if x_column and y_column:
        return f"{aggregation.title()} of {y_column} by {x_column}"
    return "Chart"


def _aggregate_grouped_data(df, x_column, value_column, color_by_column, aggregation, top_n, sort_order, chart_type):
    group_columns = [x_column]
    if color_by_column:
        group_columns.append(color_by_column)

    work_columns = list(dict.fromkeys(group_columns + [value_column]))
    work_df = df[work_columns].copy()
    work_df = work_df.dropna(subset=[x_column])

    output_value_column = value_column
    if aggregation == "count":
        grouped_df = work_df.groupby(group_columns, dropna=False, observed=True).size().reset_index(name=output_value_column)
    elif aggregation == "nunique":
        grouped_df = (
            work_df.dropna(subset=[value_column])
            .groupby(group_columns, dropna=False, observed=True)[output_value_column]
            .nunique()
            .reset_index()
        )
    else:
        work_df[output_value_column] = _numeric_series(work_df[output_value_column])
        work_df = work_df.dropna(subset=[output_value_column])
        if work_df.empty:
            raise ChartGenerationError(
                f"Column '{value_column}' does not contain numeric values for {aggregation} aggregation."
            )
        grouped_df = (
            work_df.groupby(group_columns, dropna=False, observed=True)[output_value_column]
            .agg(aggregation)
            .reset_index()
        )

    if chart_type in {"pie", "donut"} and not color_by_column:
        return _group_pie_other(grouped_df, x_column, output_value_column, top_n=min(top_n, 10))

    return _apply_top_n(grouped_df, output_value_column, top_n, sort_order)


def _build_grouped_chart(df, payload, chart_type, source_meta):
    settings = payload.get("settings_json") or {}
    x_column = _validate_column(df, payload.get("x_column") or payload.get("dimension"), "X axis")
    y_column = payload.get("y_column") or payload.get("measure")
    aggregation = _normalize_aggregation(payload.get("aggregation") or settings.get("aggregation"))

    if aggregation == "count" and not y_column:
        value_column = "__row_count__"
        df = df.copy()
        df[value_column] = 1
    else:
        y_column = _validate_column(df, y_column, "Y axis / value")
        value_column = y_column

    size_column = _validate_column(df, payload.get("size_column"), "Size", required=False)
    color_by_column = _validate_column(df, payload.get("color_by_column"), "Color by", required=False)
    default_top_n = 10 if chart_type in {"pie", "donut"} else 20
    top_n = _safe_int(payload.get("top_n") or payload.get("limit") or settings.get("top_n"), default_top_n)
    top_n = min(max(top_n, 1), 100)
    sort_order = (payload.get("sort_order") or settings.get("sort_order") or "descending").lower()
    if sort_order not in {"ascending", "descending"}:
        sort_order = "descending"

    result_df = _aggregate_grouped_data(
        df=df,
        x_column=x_column,
        value_column=value_column,
        color_by_column=color_by_column,
        aggregation=aggregation,
        top_n=top_n,
        sort_order=sort_order,
        chart_type=chart_type,
    )

    if chart_type in {"line", "area"}:
        result_df = result_df.sort_values(by=x_column, kind="mergesort")

    title = payload.get("title") or _build_title(chart_type, x_column, y_column or value_column, aggregation)
    columns = list(result_df.columns)

    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column or value_column,
            "size_column": size_column,
            "color_by_column": color_by_column,
            "aggregation": aggregation,
            "top_n": top_n,
            "sort_order": sort_order,
        },
        "chart_data_json": {
            "columns": columns,
            "rows": _to_records(result_df),
            "meta": {
                "placeholder": False,
                "row_count": int(len(result_df)),
                "source": source_meta,
                "aggregation": aggregation,
                "value_column": value_column,
                "color_by_column": color_by_column,
                "optimized": True,
            },
        },
        "settings_json": {
            **settings,
            "aggregation": aggregation,
            "top_n": top_n,
            "sort_order": sort_order,
            "size_column": size_column,
            "color_by_column": color_by_column,
            "source": source_meta,
        },
    }


def _build_scatter_chart(df, payload, chart_type, source_meta):
    settings = payload.get("settings_json") or {}
    x_column = _validate_column(df, payload.get("x_column"), "X axis")
    y_column = _validate_column(df, payload.get("y_column") or payload.get("measure"), "Y axis")
    size_column = _validate_column(df, payload.get("size_column"), "Size", required=chart_type == "bubble")
    color_by_column = _validate_column(df, payload.get("color_by_column"), "Color by", required=False)
    top_n = _safe_int(payload.get("top_n") or payload.get("limit") or settings.get("top_n"), 500)
    top_n = min(max(top_n, 1), 1000)

    columns = list(dict.fromkeys([column for column in [x_column, y_column, size_column, color_by_column] if column]))
    work_df = df[columns].copy()
    work_df[x_column] = _numeric_series(work_df[x_column])
    work_df[y_column] = _numeric_series(work_df[y_column])
    if size_column:
        work_df[size_column] = _numeric_series(work_df[size_column])
    work_df = work_df.dropna(subset=[x_column, y_column])
    if work_df.empty:
        raise ChartGenerationError("Scatter/bubble chart needs numeric X and Y values.")
    if len(work_df) > top_n:
        work_df = work_df.sample(n=top_n, random_state=42)

    title = payload.get("title") or f"{y_column} vs {x_column}"
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "size_column": size_column,
            "color_by_column": color_by_column,
            "aggregation": "none",
            "top_n": top_n,
        },
        "chart_data_json": {
            "columns": columns,
            "rows": _to_records(work_df),
            "meta": {
                "placeholder": False,
                "row_count": int(len(work_df)),
                "source": source_meta,
                "optimized": True,
                "sampling": "random" if len(work_df) == top_n else "none",
            },
        },
        "settings_json": {
            **settings,
            "top_n": top_n,
            "size_column": size_column,
            "color_by_column": color_by_column,
            "source": source_meta,
        },
    }


def _build_histogram_chart(df, payload, chart_type, source_meta):
    settings = payload.get("settings_json") or {}
    value_column = _validate_column(
        df,
        payload.get("x_column") or payload.get("dimension") or payload.get("measure"),
        "Histogram X axis",
    )
    color_by_column = _validate_column(df, payload.get("color_by_column"), "Color by", required=False)
    bins = _safe_int(payload.get("bins") or settings.get("bins"), 10)
    bins = min(max(bins, 1), 100)

    work_columns = [value_column] + ([color_by_column] if color_by_column else [])
    work_df = df[work_columns].copy()
    work_df[value_column] = _numeric_series(work_df[value_column])
    work_df = work_df.dropna(subset=[value_column])
    if work_df.empty:
        raise ChartGenerationError(f"Column '{value_column}' does not contain numeric values for histogram.")

    binned = pd.cut(work_df[value_column], bins=bins, duplicates="drop")
    interval_categories = list(binned.cat.categories)
    work_df["__histogram_bin__"] = binned

    rows = []
    if color_by_column:
        grouped = work_df.dropna(subset=[color_by_column]).groupby(["__histogram_bin__", color_by_column], observed=True).size()
        for (interval, group), count in grouped.items():
            if pd.isna(interval):
                continue
            rows.append({
                "bin_start": make_json_safe(interval.left),
                "bin_end": make_json_safe(interval.right),
                "label": f"{make_json_safe(interval.left)} - {make_json_safe(interval.right)}",
                "count": int(count),
                "group": make_json_safe(group),
            })
    else:
        counts = work_df["__histogram_bin__"].value_counts(sort=False)
        for interval in interval_categories:
            count = counts.get(interval, 0)
            rows.append({
                "bin_start": make_json_safe(interval.left),
                "bin_end": make_json_safe(interval.right),
                "label": f"{make_json_safe(interval.left)} - {make_json_safe(interval.right)}",
                "count": int(count),
                "group": None,
            })

    title = payload.get("title") or _build_title("histogram", value_column, "count", "count")
    columns = ["label", "count", "bin_start", "bin_end", "group"]
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "chart_type": chart_type,
            "x_column": "label",
            "y_column": "count",
            "x_axis": value_column,
            "y_axis": "auto_count",
            "source_column": value_column,
            "color_by_column": color_by_column,
            "aggregation": "count",
            "bins": bins,
        },
        "chart_data_json": {
            "columns": columns,
            "rows": rows,
            "bins": rows,
            "meta": {
                "placeholder": False,
                "row_count": int(len(rows)),
                "source": source_meta,
                "value_column": value_column,
                "color_by_column": color_by_column,
                "optimized": True,
            },
        },
        "settings_json": {
            **settings,
            "aggregation": "count",
            "bins": bins,
            "color_by_column": color_by_column,
            "source": source_meta,
        },
    }

def _build_box_chart(df, payload, chart_type, source_meta):
    settings = payload.get("settings_json") or {}
    y_column = _validate_column(df, payload.get("y_column") or payload.get("measure"), "Box value")
    x_column = _validate_column(
        df,
        payload.get("x_column") or payload.get("dimension"),
        "Box category",
        required=False,
    )
    top_n = _safe_int(payload.get("top_n") or payload.get("limit") or settings.get("top_n"), 30)
    top_n = min(max(top_n, 1), 100)

    columns = [y_column] + ([x_column] if x_column else [])
    work_df = df[columns].copy()
    work_df[y_column] = _numeric_series(work_df[y_column])
    work_df = work_df.dropna(subset=[y_column])
    if work_df.empty:
        raise ChartGenerationError(f"Column '{y_column}' does not contain numeric values for box chart.")

    if x_column:
        work_df = work_df.dropna(subset=[x_column])
        # Manual business choice: top categories by row count are kept to avoid unreadable box charts.
        category_order = work_df[x_column].value_counts().head(top_n).index
        work_df = work_df[work_df[x_column].isin(category_order)]
        grouped = work_df.groupby(x_column, dropna=False, observed=True)[y_column]
        result_df = grouped.agg(
            min="min",
            q1=lambda s: s.quantile(0.25),
            median="median",
            q3=lambda s: s.quantile(0.75),
            max="max",
            mean="mean",
            count="count",
        ).reset_index()
        result_x_column = x_column
    else:
        series = work_df[y_column]
        result_x_column = "category"
        result_df = pd.DataFrame([
            {
                "category": y_column,
                "min": series.min(),
                "q1": series.quantile(0.25),
                "median": series.median(),
                "q3": series.quantile(0.75),
                "max": series.max(),
                "mean": series.mean(),
                "count": int(series.count()),
            }
        ])

    title = payload.get("title") or _build_title("box", result_x_column, y_column, "distribution")
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "chart_type": chart_type,
            "x_column": result_x_column,
            "y_column": y_column,
            "aggregation": "distribution",
            "top_n": top_n,
        },
        "chart_data_json": {
            "columns": list(result_df.columns),
            "rows": _to_records(result_df),
            "meta": {
                "placeholder": False,
                "row_count": int(len(result_df)),
                "source": source_meta,
                "value_column": y_column,
                "optimized": True,
            },
        },
        "settings_json": {**settings, "top_n": top_n, "source": source_meta},
    }


def _build_heatmap_chart(df, payload, chart_type, source_meta):
    settings = payload.get("settings_json") or {}
    x_column = _validate_column(df, payload.get("x_column") or payload.get("dimension"), "Heatmap X axis")
    y_group_column = _validate_column(
        df,
        payload.get("group_by_column") or payload.get("color_by_column"),
        "Heatmap Y group",
    )
    value_column = payload.get("y_column") or payload.get("measure")
    aggregation = _normalize_aggregation(payload.get("aggregation") or settings.get("aggregation") or "count")

    if aggregation == "count" and not value_column:
        value_column = "__row_count__"
        work_df = df[[x_column, y_group_column]].copy()
        work_df[value_column] = 1
    else:
        value_column = _validate_column(df, value_column, "Heatmap value")
        work_df = df[[x_column, y_group_column, value_column]].copy()

    work_df = work_df.dropna(subset=[x_column, y_group_column])
    if work_df.empty:
        raise ChartGenerationError("Heatmap needs non-empty X and Y group values.")

    if aggregation == "count":
        pivot_df = pd.pivot_table(
            work_df,
            index=y_group_column,
            columns=x_column,
            values=value_column,
            aggfunc="count" if value_column != "__row_count__" else "sum",
            fill_value=0,
        )
    elif aggregation == "nunique":
        pivot_df = pd.pivot_table(
            work_df.dropna(subset=[value_column]),
            index=y_group_column,
            columns=x_column,
            values=value_column,
            aggfunc=pd.Series.nunique,
            fill_value=0,
        )
    else:
        work_df[value_column] = _numeric_series(work_df[value_column])
        work_df = work_df.dropna(subset=[value_column])
        if work_df.empty:
            raise ChartGenerationError(f"Column '{value_column}' does not contain numeric values for heatmap.")
        pivot_df = pd.pivot_table(
            work_df,
            index=y_group_column,
            columns=x_column,
            values=value_column,
            aggfunc=aggregation,
            fill_value=0,
        )

    wide_df = pivot_df.reset_index()
    wide_df.columns = [str(column) for column in wide_df.columns]

    cells_df = pivot_df.stack().reset_index()
    cells_df.columns = [y_group_column, x_column, "value"]

    title = payload.get("title") or f"{aggregation.title()} heatmap"
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_group_column,
            "value_column": value_column,
            "aggregation": aggregation,
        },
        "chart_data_json": {
            "columns": list(wide_df.columns),
            "rows": _to_records(wide_df),
            "cells": _to_records(cells_df),
            "meta": {
                "placeholder": False,
                "row_count": int(len(wide_df)),
                "cell_count": int(len(cells_df)),
                "source": source_meta,
                "aggregation": aggregation,
                "optimized": True,
            },
        },
        "settings_json": {**settings, "aggregation": aggregation, "source": source_meta},
    }


def _build_pivot_table_chart(df, payload, chart_type, source_meta):
    # Manual BI note: this uses the same safe pivot logic as heatmap.
    # Later you can add subtotal/grand-total business rules here if needed.
    result = _build_heatmap_chart(df, payload, chart_type, source_meta)
    result["chart_type"] = "pivot_table"
    result["chart_config_json"]["chart_type"] = "pivot_table"
    result["chart_config_json"]["title"] = payload.get("title") or "Pivot table"
    return result


def _build_correlation_heatmap_chart(df, payload, chart_type, source_meta):
    settings = payload.get("settings_json") or {}
    selected_columns = payload.get("columns") or payload.get("selected_columns") or settings.get("columns") or []

    if selected_columns:
        candidate_columns = [str(column) for column in selected_columns if str(column) in df.columns]
    else:
        candidate_columns = list(df.columns)

    numeric_df = df[candidate_columns].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.shape[1] < 2:
        raise ChartGenerationError("Correlation heatmap needs at least two numeric columns.")

    max_columns = _safe_int(payload.get("max_columns") or settings.get("max_columns"), 20)
    max_columns = min(max(max_columns, 2), 50)
    numeric_df = numeric_df.iloc[:, :max_columns]

    corr_df = numeric_df.corr().fillna(0)
    wide_df = corr_df.reset_index().rename(columns={"index": "column"})
    cells_df = corr_df.stack().reset_index()
    cells_df.columns = ["x_column", "y_column", "value"]

    title = payload.get("title") or "Correlation heatmap"
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "chart_type": chart_type,
            "x_column": "x_column",
            "y_column": "y_column",
            "value_column": "value",
            "aggregation": "correlation",
            "max_columns": max_columns,
        },
        "chart_data_json": {
            "columns": list(wide_df.columns),
            "rows": _to_records(wide_df),
            "cells": _to_records(cells_df),
            "meta": {
                "placeholder": False,
                "row_count": int(len(wide_df)),
                "cell_count": int(len(cells_df)),
                "source": source_meta,
                "optimized": True,
            },
        },
        "settings_json": {**settings, "max_columns": max_columns, "source": source_meta},
    }


def _build_kpi_chart(df, payload, chart_type, source_meta):
    settings = payload.get("settings_json") or {}
    y_column = _validate_column(
        df,
        payload.get("y_column") or payload.get("measure") or payload.get("size_column"),
        "KPI value",
    )
    aggregation = _normalize_aggregation(payload.get("aggregation") or settings.get("aggregation"))

    if aggregation == "count":
        value = int(df[y_column].count())
    elif aggregation == "nunique":
        value = int(df[y_column].nunique(dropna=True))
    else:
        numeric_series = _numeric_series(df[y_column]).dropna()
        if numeric_series.empty:
            raise ChartGenerationError(f"Column '{y_column}' does not contain numeric values for KPI aggregation.")
        value = getattr(numeric_series, aggregation)()

    title = payload.get("title") or _build_title("kpi", "", y_column, aggregation)
    row = {"metric": title, "value": make_json_safe(value)}
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "chart_type": chart_type,
            "x_column": "metric",
            "y_column": "value",
            "size_column": "value",
            "aggregation": aggregation,
        },
        "chart_data_json": {
            "columns": ["metric", "value"],
            "rows": [row],
            "meta": {"placeholder": False, "source": source_meta, "aggregation": aggregation, "optimized": True},
        },
        "settings_json": {**settings, "aggregation": aggregation, "source": source_meta},
    }


def _build_table_chart(df, payload, chart_type, source_meta):
    settings = payload.get("settings_json") or {}
    top_n = _safe_int(payload.get("top_n") or payload.get("limit") or settings.get("top_n"), 50)
    top_n = min(max(top_n, 1), 500)
    requested_columns = payload.get("columns") or payload.get("selected_columns") or settings.get("columns") or []
    selected_columns = [str(column) for column in requested_columns if str(column) in df.columns]
    if not selected_columns:
        selected_columns = list(df.columns[:25])

    table_df = df[selected_columns].head(top_n)
    title = payload.get("title") or "Data table"
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "chart_type": chart_type,
            "x_column": selected_columns[0] if selected_columns else "",
            "y_column": selected_columns[1] if len(selected_columns) > 1 else "",
            "top_n": top_n,
        },
        "chart_data_json": {
            "columns": selected_columns,
            "rows": _to_records(table_df),
            "meta": {"placeholder": False, "source": source_meta, "row_count": int(len(table_df)), "optimized": True},
        },
        "settings_json": {**settings, "top_n": top_n, "source": source_meta},
    }


def _build_map_chart(df, payload, chart_type, source_meta):
    settings = payload.get("settings_json") or {}
    aggregation = _normalize_aggregation(payload.get("aggregation") or settings.get("aggregation") or "sum")
    top_n = _safe_int(payload.get("top_n") or payload.get("limit") or settings.get("top_n"), 500)
    top_n = min(max(top_n, 1), 2000)

    latitude_column = _validate_column(
        df,
        payload.get("latitude_column") or payload.get("lat_column"),
        "Latitude column",
        required=chart_type in {"map", "symbol_map"},
    )
    longitude_column = _validate_column(
        df,
        payload.get("longitude_column") or payload.get("lon_column") or payload.get("lng_column"),
        "Longitude column",
        required=chart_type in {"map", "symbol_map"},
    )
    location_column = _validate_column(
        df,
        payload.get("location_column") or payload.get("x_column") or payload.get("dimension"),
        "Location column",
        required=chart_type == "filled_map",
    )
    value_column = payload.get("y_column") or payload.get("measure") or payload.get("size_column")
    color_by_column = _validate_column(df, payload.get("color_by_column"), "Color by", required=False)

    if chart_type == "filled_map":
        value_column = _validate_column(df, value_column, "Filled map value") if value_column else "__row_count__"
        work_columns = [location_column] + ([] if value_column == "__row_count__" else [value_column])
        if color_by_column:
            work_columns.append(color_by_column)
        work_df = df[list(dict.fromkeys(work_columns))].copy().dropna(subset=[location_column])
        if value_column == "__row_count__" or aggregation == "count":
            result_df = work_df.groupby([location_column], dropna=False, observed=True).size().reset_index(name="value")
        elif aggregation == "nunique":
            result_df = work_df.groupby([location_column], dropna=False, observed=True)[value_column].nunique().reset_index(name="value")
        else:
            work_df[value_column] = _numeric_series(work_df[value_column])
            work_df = work_df.dropna(subset=[value_column])
            if work_df.empty:
                raise ChartGenerationError(f"Column '{value_column}' does not contain numeric values for filled map.")
            result_df = work_df.groupby([location_column], dropna=False, observed=True)[value_column].agg(aggregation).reset_index(name="value")
        result_df = _apply_top_n(result_df, "value", top_n, "descending")
        x_column = location_column
        y_column = "value"
        map_mode = "location"
    else:
        value_column = _validate_column(df, value_column, "Map value", required=False)
        columns = [latitude_column, longitude_column]
        if value_column:
            columns.append(value_column)
        if color_by_column:
            columns.append(color_by_column)
        work_df = df[list(dict.fromkeys(columns))].copy()
        work_df[latitude_column] = _numeric_series(work_df[latitude_column])
        work_df[longitude_column] = _numeric_series(work_df[longitude_column])
        work_df = work_df.dropna(subset=[latitude_column, longitude_column])
        if value_column:
            work_df[value_column] = _numeric_series(work_df[value_column])
        if work_df.empty:
            raise ChartGenerationError("Map chart needs valid latitude and longitude values.")
        if len(work_df) > top_n:
            work_df = work_df.sample(n=top_n, random_state=42)
        result_df = work_df
        x_column = longitude_column
        y_column = latitude_column
        map_mode = "coordinates"

    title = payload.get("title") or "Map chart"
    return {
        "dataset_id": df.attrs.get("dataset_id"),
        "chart_type": chart_type,
        "chart_config_json": {
            "title": title,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_column": y_column,
            "latitude_column": latitude_column,
            "longitude_column": longitude_column,
            "location_column": location_column,
            "value_column": value_column,
            "color_by_column": color_by_column,
            "aggregation": aggregation,
            "top_n": top_n,
            "map_mode": map_mode,
        },
        "chart_data_json": {
            "columns": list(result_df.columns),
            "rows": _to_records(result_df),
            "meta": {
                "placeholder": False,
                "row_count": int(len(result_df)),
                "source": source_meta,
                "optimized": True,
                "map_mode": map_mode,
                "note": "Filled map uses location names only. Add geocoding/topojson on frontend or connector layer if needed.",
            },
        },
        "settings_json": {**settings, "top_n": top_n, "aggregation": aggregation, "source": source_meta},
    }


def generate_chart_data(dataset, payload):
    """Generate chart-ready JSON from a dataset and chart payload.

    MANUAL PANDAS/BI REVIEW AREA:
    - Review business-specific aggregation rules before production.
    - Review how top-N categories should be selected for your customer datasets.
    - Review whether cleaned, feature-engineered, or ML-ready data should be preferred for every chart.
    - Review map/geocoding strategy before enabling customer-facing location-name maps.
    """
    chart_type = _normalize_chart_type(payload.get("chart_type"))
    required_columns = _required_columns_for_payload(payload, chart_type)
    df, source_meta = _read_preferred_dataframe(dataset, required_columns=required_columns)
    df.attrs["dataset_id"] = dataset.id

    if chart_type == "table":
        return _build_table_chart(df, payload, chart_type, source_meta)

    if chart_type == "kpi":
        return _build_kpi_chart(df, payload, chart_type, source_meta)

    if chart_type in {"scatter", "bubble"}:
        return _build_scatter_chart(df, payload, chart_type, source_meta)

    if chart_type in GROUPED_CHART_TYPES:
        return _build_grouped_chart(df, payload, chart_type, source_meta)

    if chart_type == "histogram":
        return _build_histogram_chart(df, payload, chart_type, source_meta)

    if chart_type == "box":
        return _build_box_chart(df, payload, chart_type, source_meta)

    if chart_type == "heatmap":
        return _build_heatmap_chart(df, payload, chart_type, source_meta)

    if chart_type == "pivot_table":
        return _build_pivot_table_chart(df, payload, chart_type, source_meta)

    if chart_type == "correlation_heatmap":
        return _build_correlation_heatmap_chart(df, payload, chart_type, source_meta)

    if chart_type in MAP_CHART_TYPES:
        return _build_map_chart(df, payload, chart_type, source_meta)

    raise ChartGenerationError(f"Chart type '{chart_type}' is not implemented.")


def generate_placeholder_chart_data(dataset, payload):
    """Backward-compatible name used by older views."""
    return generate_chart_data(dataset, payload)

