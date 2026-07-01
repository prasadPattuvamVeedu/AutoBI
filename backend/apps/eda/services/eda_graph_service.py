from .advanced_eda_manual_implementation_guide import (
    generate_boxplot_outlier_chart,
    generate_correlation_heatmap,
    generate_countplot_chart,
    generate_distribution_kde_chart,
    generate_histogram_chart,
    generate_pairplot_matrix,
    generate_missing_value_analysis,
    generate_qq_plot,
    generate_skewness_kurtosis_summary,
    generate_target_balance_chart,
    generate_target_rate_by_category,
    generate_violin_plot,
)
from .eda_common import load_eda_dataframe, load_profile_column_groups
from .validation_eda_manual_implementation_guide import (
    generate_correlation_shift_heatmap,
    generate_distribution_change_chart,
    generate_feature_engineering_validation,
    generate_missing_values_comparison,
    generate_outlier_comparison,
    generate_target_balance_after,
)
from apps.datasets.models import Dataset


def _pick_column(payload):
    return payload.get("column") or payload.get("x_column") or payload.get("y_column") or payload.get("target_column")


def _not_implemented_response(chart_type):
    return {
        "chart_type": chart_type,
        "render_type": "not_implemented",
        "image_base64": None,
        "image_mime_type": None,
        "summary": {},
        "warnings": ["Graph generation is not available yet for this chart."],
    }


def _import_error_response(chart_type):
    return {
        "chart_type": chart_type,
        "render_type": "not_implemented",
        "image_base64": None,
        "image_mime_type": None,
        "summary": {},
        "warnings": ["Install matplotlib and seaborn to generate this EDA chart."],
    }


def _error_response(chart_type, exc):
    return {
        "chart_type": chart_type,
        "render_type": "error",
        "image_base64": None,
        "image_mime_type": None,
        "summary": {},
        "warnings": [f"Graph generation failed: {str(exc)}"],
    }


def _quick_stat_graph_response(df, payload, warning=None):
    """Fast non-image fallback so the UI and LLM still receive graph evidence."""
    chart_type = str(payload.get("chart_type") or "eda_graph").lower()
    column = _pick_column(payload)
    if chart_type in {"missing_values_bar", "missing_values_heatmap", "missing_value_analysis"}:
        return _missing_values_response(df, payload, warning)
    if chart_type == "histogram":
        return _histogram_response(df, payload, warning)
    if chart_type in {"boxplot", "boxplot_outlier", "boxplot_by_target"}:
        return _boxplot_response(df, payload, warning)
    if not column or column not in getattr(df, "columns", []):
        related = payload.get("columns") or []
        column = next((item for item in related if item in getattr(df, "columns", [])), None)
    rows = []
    summary = {}
    warnings = [warning] if warning else []

    try:
        import pandas as pd
        if column and column in df.columns:
            series = df[column]
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            summary = {
                "column": str(column),
                "row_count": int(len(series)),
                "missing_count": int(series.isna().sum()),
                "missing_percentage": round(float(series.isna().mean() * 100), 2),
                "unique_count": int(series.nunique(dropna=True)),
            }
            if not numeric.empty:
                summary.update({
                    "valid_numeric_count": int(len(numeric)),
                    "min": round(float(numeric.min()), 4),
                    "max": round(float(numeric.max()), 4),
                    "mean": round(float(numeric.mean()), 4),
                    "median": round(float(numeric.median()), 4),
                    "std": round(float(numeric.std()), 4) if len(numeric) > 1 else 0,
                    "q1": round(float(numeric.quantile(0.25)), 4),
                    "q3": round(float(numeric.quantile(0.75)), 4),
                })
                iqr_value = float(numeric.quantile(0.75) - numeric.quantile(0.25))
                lower_value = float(numeric.quantile(0.25) - 1.5 * iqr_value)
                upper_value = float(numeric.quantile(0.75) + 1.5 * iqr_value)
                summary["outlier_count"] = int(((numeric < lower_value) | (numeric > upper_value)).sum())
                counts = numeric.value_counts(bins=10, sort=False)
                rows = [{"bin": str(index), "count": int(value)} for index, value in counts.items()]
            else:
                counts = series.fillna("[missing]").astype(str).value_counts().head(15)
                rows = [{"category": str(index), "count": int(value)} for index, value in counts.items()]
        else:
            warnings.append("No valid column was provided for this EDA graph.")
    except Exception as exc:
        warnings.append(f"Fast graph evidence failed: {str(exc)}")

    return {
        "chart_type": chart_type,
        "render_type": "table",
        "column": column,
        "summary": summary,
        "rows": rows,
        "chart_data": {"rows": rows[:50]},
        "warnings": warnings,
    }


def _dedupe_df(df):
    return df.loc[:, ~df.columns.duplicated()].copy()


def _series(df, column):
    import pandas as pd
    if not column or column not in df.columns:
        return None
    value = df[column]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value


def _missing_values_response(df, payload, warning=None):
    df = _dedupe_df(df)
    row_count = max(int(len(df)), 1)
    rows = []
    for column in df.columns:
        series = _series(df, column)
        missing_count = int(series.isna().sum()) if series is not None else 0
        if missing_count:
            rows.append({
                "column": str(column),
                "missing_count": missing_count,
                "missing_percentage": round(float((missing_count / row_count) * 100), 2),
            })
    rows = sorted(rows, key=lambda item: item["missing_count"], reverse=True)
    warnings = [warning] if warning else []
    return {
        "chart_type": str(payload.get("chart_type") or "missing_values_bar").lower(),
        "render_type": "table",
        "summary": {
            "total_missing_cells": int(sum(item["missing_count"] for item in rows)),
            "columns_with_missing": int(len(rows)),
        },
        "rows": rows[:50],
        "chart_data": {"rows": rows[:50]},
        "warnings": warnings,
    }


def _histogram_response(df, payload, warning=None):
    import pandas as pd
    df = _dedupe_df(df)
    column = _pick_column(payload)
    series = _series(df, column)
    warnings = [warning] if warning else []
    if series is None:
        return {
            "chart_type": "histogram",
            "render_type": "table",
            "column": column,
            "summary": {},
            "rows": [],
            "chart_data": {"rows": []},
            "warnings": warnings + [f"Column '{column}' was not found."],
        }

    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    rows = []

    if unique_count <= 30:
        # Numeric-coded categoricals like MSSubClass are more useful as sorted
        # value-count bars than arbitrary bins.
        counts = series.fillna("[missing]").astype(str).value_counts().head(30)
        rows = [{"label": str(index), "value": str(index), "count": int(value)} for index, value in counts.items()]
    elif not numeric.empty:
        counts = numeric.value_counts(bins=10, sort=False)
        rows = [{"label": str(index), "bin": str(index), "count": int(value)} for index, value in counts.items()]
    else:
        counts = series.dropna().astype(str).value_counts().head(30)
        rows = [{"label": str(index), "value": str(index), "count": int(value)} for index, value in counts.items()]

    summary = {
        "column": str(column),
        "row_count": int(len(series)),
        "missing_count": missing_count,
        "missing_percentage": round(float(series.isna().mean() * 100), 2),
        "unique_count": unique_count,
    }
    if not numeric.empty:
        summary.update({
            "min": round(float(numeric.min()), 4),
            "max": round(float(numeric.max()), 4),
            "mean": round(float(numeric.mean()), 4),
            "median": round(float(numeric.median()), 4),
            "std": round(float(numeric.std()), 4) if len(numeric) > 1 else 0,
        })

    return {
        "chart_type": "histogram",
        "render_type": "bar",
        "column": column,
        "summary": summary,
        "rows": rows[:50],
        "chart_data": {"rows": rows[:50]},
        "warnings": warnings,
    }


def _boxplot_response(df, payload, warning=None):
    import pandas as pd
    df = _dedupe_df(df)
    column = payload.get("y_column") or payload.get("column") or payload.get("x_column")
    series = _series(df, column)
    warnings = [warning] if warning else []
    if series is None:
        return {
            "chart_type": "boxplot_outlier",
            "render_type": "table",
            "column": column,
            "summary": {},
            "rows": [],
            "chart_data": {"rows": []},
            "warnings": warnings + [f"Column '{column}' was not found."],
        }

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "chart_type": "boxplot_outlier",
            "render_type": "table",
            "column": column,
            "summary": {"valid_count": 0},
            "rows": [],
            "chart_data": {"rows": []},
            "warnings": warnings + [f"Column '{column}' has no valid numeric values."],
        }

    q1 = float(numeric.quantile(0.25))
    median = float(numeric.quantile(0.5))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = numeric[(numeric < lower) | (numeric > upper)]
    summary = {
        "column": str(column),
        "min": round(float(numeric.min()), 4),
        "q1": round(q1, 4),
        "median": round(median, 4),
        "q3": round(q3, 4),
        "max": round(float(numeric.max()), 4),
        "iqr": round(float(iqr), 4),
        "lower_fence": round(float(lower), 4),
        "upper_fence": round(float(upper), 4),
        "outlier_count": int(outliers.count()),
        "outlier_sample": [round(float(value), 4) for value in outliers.head(20).tolist()],
    }
    rows = [summary]
    return {
        "chart_type": "boxplot_outlier",
        "render_type": "boxplot",
        "column": column,
        "summary": summary,
        "rows": rows,
        "chart_data": {"rows": rows},
        "warnings": warnings,
    }


def _build_standard_graph(df, payload, numeric_columns):
    chart_type = str(payload.get("chart_type") or "").lower()
    column = _pick_column(payload)
    x_column = payload.get("x_column")
    y_column = payload.get("y_column")
    target_column = payload.get("target_column")
    bins = payload.get("bins") or "auto"
    sample_size = payload.get("sample_size")

    if chart_type in {"missing_values_bar", "missing_values_heatmap", "missing_value_analysis"}:
        return generate_missing_value_analysis(df)
    if chart_type in {"distribution_kde", "kde"}:
        return generate_distribution_kde_chart(df, column, bins=bins, sample_size=sample_size)
    if chart_type == "histogram":
        return generate_histogram_chart(df, column, bins=bins, sample_size=sample_size)
    if chart_type in {"boxplot", "boxplot_outlier"}:
        return generate_boxplot_outlier_chart(df, column, group_by=x_column)
    if chart_type == "boxplot_by_target":
        return generate_boxplot_outlier_chart(df, y_column or column, group_by=target_column or x_column)
    if chart_type in {"violin", "violin_plot"}:
        return generate_violin_plot(df, column, group_by=x_column, sample_size=sample_size)
    if chart_type in {"qq_plot", "qq"}:
        return generate_qq_plot(df, column)
    if chart_type in {"correlation_heatmap", "heatmap"}:
        return generate_correlation_heatmap(df, numeric_columns=numeric_columns)
    if chart_type == "target_balance":
        return generate_target_balance_chart(df, target_column or column)
    if chart_type == "target_rate_by_category":
        return generate_target_rate_by_category(df, x_column or column, target_column)
    if chart_type in {"pairplot_matrix", "pairplot"}:
        return generate_pairplot_matrix(df, numeric_columns=numeric_columns, sample_size=sample_size)
    if chart_type in {"skewness_kurtosis", "skewness_kurtosis_summary", "skewness_kurtosis_table"}:
        return generate_skewness_kurtosis_summary(df, numeric_columns=numeric_columns)
    if chart_type in {"countplot", "category_count"}:
        return generate_countplot_chart(df=df, column=column or x_column)

    raise NotImplementedError("USER WILL IMPLEMENT PYTHON GRAPH GENERATION")


def _json_safe_value(value):
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _graph_used_columns(payload):
    columns = []
    for key in ("column", "x_column", "y_column", "target_column", "group_by", "group_by_column"):
        value = payload.get(key)
        if value and value not in columns:
            columns.append(value)
    for value in payload.get("columns") or []:
        if value and value not in columns:
            columns.append(value)
    return columns


def _profile_map(dataset_id):
    try:
        profile_json = Dataset.objects.get(pk=dataset_id).profile.profile_json or {}
    except Exception:
        return {}
    rows = profile_json.get("columns") or profile_json.get("column_profiles") or []
    if isinstance(rows, dict):
        rows = [dict(value, column_name=key) if isinstance(value, dict) else {"column_name": key} for key, value in rows.items()]
    return {
        str(row.get("column_name") or row.get("name") or row.get("column")): row
        for row in rows
        if isinstance(row, dict) and (row.get("column_name") or row.get("name") or row.get("column"))
    }


def _graph_rows(graph_payload):
    if not isinstance(graph_payload, dict):
        return []
    rows = graph_payload.get("chart_data", {}).get("rows") if isinstance(graph_payload.get("chart_data"), dict) else None
    if rows is None:
        rows = graph_payload.get("rows") or graph_payload.get("data") or graph_payload.get("summary", {}).get("rows") or []
    return rows[:50] if isinstance(rows, list) else []


def _source_rows(df, used_columns):
    if not used_columns:
        return []
    try:
        visible_columns = [column for column in used_columns if column in df.columns]
        return [
            {str(key): _json_safe_value(value) for key, value in row.items()}
            for row in df[visible_columns].head(20).to_dict(orient="records")
        ]
    except Exception:
        return []


def _build_eda_ai_context(dataset_id, df, payload, graph_payload):
    """Attach small backend-built evidence for LLM EDA insight generation."""
    df = _dedupe_df(df)
    dataset = None
    try:
        dataset = Dataset.objects.get(pk=dataset_id)
    except Exception:
        dataset = None
    used_columns = [column for column in _graph_used_columns(payload) if column in getattr(df, "columns", [])]
    visible_columns = used_columns[:8] or list(getattr(df, "columns", [])[:8])
    sample_rows = []
    if visible_columns:
        try:
            for row in df[visible_columns].head(8).to_dict(orient="records"):
                sample_rows.append({str(key): _json_safe_value(value) for key, value in row.items()})
        except Exception:
            sample_rows = []

    column_statistics = []
    for column in used_columns[:12]:
        try:
            series = _series(df, column)
            if series is None:
                continue
            item = {
                "column": str(column),
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_percentage": round(float(series.isna().mean() * 100), 2),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": [_json_safe_value(value) for value in series.dropna().head(5).tolist()],
            }
            numeric = None
            try:
                import pandas as pd
                numeric = pd.to_numeric(series, errors="coerce").dropna()
            except Exception:
                numeric = None
            if numeric is not None and not numeric.empty:
                item.update({
                    "min": round(float(numeric.min()), 4),
                    "max": round(float(numeric.max()), 4),
                    "mean": round(float(numeric.mean()), 4),
                    "median": round(float(numeric.median()), 4),
                    "std": round(float(numeric.std()), 4) if len(numeric) > 1 else 0,
                })
            column_statistics.append(item)
        except Exception:
            continue

    profiles = _profile_map(dataset_id)
    column_profiles = []
    for column in used_columns[:12]:
        profile = profiles.get(str(column), {})
        stats = next((item for item in column_statistics if item.get("column") == str(column)), {})
        column_profiles.append({
            "column": str(column),
            "detected_type": profile.get("detected_type") or profile.get("type") or stats.get("dtype"),
            "missing_count": profile.get("missing_count", stats.get("missing_count")),
            "missing_percentage": profile.get("missing_percentage", stats.get("missing_percentage")),
            "unique_count": profile.get("unique_count", stats.get("unique_count")),
            "mean": profile.get("mean", stats.get("mean")),
            "median": profile.get("median", stats.get("median")),
            "min": profile.get("min", stats.get("min")),
            "max": profile.get("max", stats.get("max")),
            "std": profile.get("std", stats.get("std")),
            "skewness": profile.get("skewness"),
            "outlier_count": profile.get("outlier_count"),
            "top_values": (profile.get("top_values") or profile.get("sample_values") or [])[:10],
        })

    chart_config = {
        "chart_type": payload.get("chart_type"),
        "chart_title": payload.get("chart_title") or payload.get("title") or payload.get("chart_type"),
        "x_column": payload.get("x_column") or payload.get("column") or "",
        "y_column": payload.get("y_column") or "",
        "group_column": payload.get("group_column") or payload.get("group_by") or payload.get("group_by_column") or payload.get("target_column") or "",
        "color_by_column": payload.get("color_by_column") or "",
        "size_column": payload.get("size_column") or "",
        "used_columns": used_columns,
        "filters": payload.get("filters") or [],
    }
    rows = _graph_rows(graph_payload)
    warnings = graph_payload.get("warnings", []) if isinstance(graph_payload, dict) else []

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name if dataset else "",
        "page_source": "advanced_eda" if payload.get("eda_mode") != "validation" else "validation_eda",
        "eda_mode": payload.get("eda_mode") or "advanced",
        "dataset_version": payload.get("dataset_version") or "raw",
        "chart_type": payload.get("chart_type"),
        "chart_title": chart_config["chart_title"],
        "x_column": chart_config["x_column"],
        "y_column": chart_config["y_column"],
        "group_column": chart_config["group_column"],
        "color_by_column": chart_config["color_by_column"],
        "size_column": chart_config["size_column"],
        "used_columns": used_columns,
        "chart_config": chart_config,
        "chart_data": {"rows": rows},
        "source_rows": _source_rows(df, used_columns),
        "column_profiles": column_profiles,
        "request_payload": {key: payload.get(key) for key in ("chart_type", "column", "x_column", "y_column", "target_column", "columns", "bins", "sample_size")},
        "graph_summary": graph_payload.get("summary", {}) if isinstance(graph_payload, dict) else {},
        "graph_rows": rows,
        "sample_rows": sample_rows,
        "column_statistics": column_statistics,
        "filters": payload.get("filters") or [],
        "warnings": warnings,
    }


def _with_eda_ai_context(result, dataset_id, df, payload):
    if not isinstance(result, dict):
        return result
    rows = _graph_rows(result)
    if rows and "chart_data" not in result:
        result["chart_data"] = {"rows": rows}
    result.setdefault("ai_context", _build_eda_ai_context(dataset_id, df, payload, result))
    return result


def build_eda_graph_data(dataset_id, payload):
    chart_type = str(payload.get("chart_type") or "").lower()

    try:
        eda_mode = payload.get("eda_mode") or "advanced"
        dataset_version = payload.get("dataset_version") or ("cleaned" if eda_mode == "validation" else "raw")
        column_groups = load_profile_column_groups(dataset_id)
        numeric_columns = payload.get("columns") or column_groups.get("numeric_columns", [])

        if eda_mode == "validation":
            df_before = load_eda_dataframe(dataset_id, dataset_version="raw")
            df_after = load_eda_dataframe(dataset_id, dataset_version=dataset_version)
            column = _pick_column(payload)

            if chart_type == "missing_values_comparison":
                return _with_eda_ai_context(generate_missing_values_comparison(df_before, df_after), dataset_id, df_after, payload)
            if chart_type == "outlier_comparison":
                return _with_eda_ai_context(generate_outlier_comparison(df_before, df_after, columns=numeric_columns), dataset_id, df_after, payload)
            if chart_type == "distribution_change":
                return _with_eda_ai_context(generate_distribution_change_chart(df_before, df_after, column, bins=payload.get("bins") or "auto"), dataset_id, df_after, payload)
            if chart_type == "correlation_shift_heatmap":
                return _with_eda_ai_context(generate_correlation_shift_heatmap(df_before, df_after, numeric_columns=numeric_columns), dataset_id, df_after, payload)
            if chart_type == "feature_engineering_validation":
                return _with_eda_ai_context(generate_feature_engineering_validation(
                    df_before,
                    df_after,
                    feature_columns=payload.get("columns") or [],
                    target_column=payload.get("target_column"),
                ), dataset_id, df_after, payload)
            if chart_type == "target_balance_after":
                return _with_eda_ai_context(generate_target_balance_after(df_after, payload.get("target_column") or column), dataset_id, df_after, payload)

            if chart_type in {"missing_values_bar", "missing_values_heatmap", "missing_value_analysis", "histogram", "boxplot", "boxplot_outlier", "boxplot_by_target"}:
                return _with_eda_ai_context(_quick_stat_graph_response(df_after, payload), dataset_id, df_after, payload)
            return _with_eda_ai_context(_build_standard_graph(df_after, payload, numeric_columns), dataset_id, df_after, payload)

        df = load_eda_dataframe(dataset_id, dataset_version=dataset_version)
        if chart_type in {"missing_values_bar", "missing_values_heatmap", "missing_value_analysis", "histogram", "boxplot", "boxplot_outlier", "boxplot_by_target"}:
            return _with_eda_ai_context(_quick_stat_graph_response(df, payload), dataset_id, df, payload)
        return _with_eda_ai_context(_build_standard_graph(df, payload, numeric_columns), dataset_id, df, payload)
    except NotImplementedError:
        try:
            dataset_version = payload.get("dataset_version") or "raw"
            df = load_eda_dataframe(dataset_id, dataset_version=dataset_version)
            return _with_eda_ai_context(_quick_stat_graph_response(df, payload, "Image graph is not implemented yet; showing fast evidence table for AI insight."), dataset_id, df, payload)
        except Exception:
            return _not_implemented_response(chart_type)
    except ImportError:
        try:
            dataset_version = payload.get("dataset_version") or "raw"
            df = load_eda_dataframe(dataset_id, dataset_version=dataset_version)
            return _with_eda_ai_context(_quick_stat_graph_response(df, payload, "Install matplotlib/seaborn/scipy for Python image graphs; showing fast evidence table for AI insight."), dataset_id, df, payload)
        except Exception:
            return _import_error_response(chart_type)
    except Exception as exc:
        try:
            dataset_version = payload.get("dataset_version") or "raw"
            df = load_eda_dataframe(dataset_id, dataset_version=dataset_version)
            return _with_eda_ai_context(_quick_stat_graph_response(df, payload, f"Image graph failed: {str(exc)}. Showing fast evidence table for AI insight."), dataset_id, df, payload)
        except Exception:
            return _error_response(chart_type, exc)
