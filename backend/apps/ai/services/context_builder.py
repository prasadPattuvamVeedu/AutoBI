from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from apps.datasets.models import Dataset, DatasetProfile
from apps.datasets.services import build_preview, read_dataset_file


MAX_COLUMNS = 80
MAX_SAMPLE_ROWS = 10
MAX_TOP_VALUES = 10
MAX_CONTEXT_CHARS = 18000

SENSITIVE_NAME_RE = re.compile(
    r"(email|e-mail|phone|mobile|address|account|acct|customer[_\s-]*name|"
    r"full[_\s-]*name|transaction[_\s-]*id|user[_\s-]*id|uuid|ssn|aadhaar|pan|passport)",
    re.IGNORECASE,
)


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            if isinstance(item, dict):
                row = dict(item)
                row.setdefault("column_name", key)
                rows.append(row)
            else:
                rows.append({"column_name": key, "detected_type": item})
        return rows
    return []


def _profile_json(dataset: Dataset) -> Dict[str, Any]:
    try:
        return dataset.profile.profile_json or {}
    except DatasetProfile.DoesNotExist:
        return {}


def _safe_number(value):
    if value in ("", None):
        return None
    try:
        return float(value) if isinstance(value, str) and "." in value else int(value)
    except (TypeError, ValueError):
        return value


def _column_name(column: Dict[str, Any]) -> str:
    return str(column.get("column_name") or column.get("name") or column.get("column") or "").strip()


def _is_sensitive_column(column: Dict[str, Any]) -> bool:
    name = _column_name(column)
    detected_type = str(column.get("detected_type") or column.get("type") or "").lower()
    return bool(
        SENSITIVE_NAME_RE.search(name)
        or column.get("is_id_like") is True
        or detected_type == "id"
        or column.get("recommended_role") == "identifier"
    )


def _compact_column(column: Dict[str, Any]) -> Dict[str, Any]:
    top_values = column.get("top_values") or column.get("sample_values") or []
    if not isinstance(top_values, list):
        top_values = []

    return {
        "name": _column_name(column),
        "type": column.get("detected_type") or column.get("type"),
        "missing_count": _safe_number(column.get("missing_count")),
        "missing_percentage": _safe_number(column.get("missing_percentage")),
        "unique_count": _safe_number(column.get("unique_count")),
        "mean": column.get("mean"),
        "median": column.get("median"),
        "min": column.get("min"),
        "max": column.get("max"),
        "std": column.get("std"),
        "skewness": column.get("skewness"),
        "outlier_count": column.get("outlier_count"),
        "outlier_percentage": column.get("outlier_percentage"),
        "top_values": top_values[:MAX_TOP_VALUES] if not _is_sensitive_column(column) else [],
        "is_id_like": bool(column.get("is_id_like") or str(column.get("detected_type", "")).lower() == "id"),
        "is_constant": bool(column.get("is_constant")),
        "is_high_cardinality": bool(column.get("is_high_cardinality")),
        "recommendation": column.get("recommendation"),
        "recommendation_reason": column.get("recommendation_reason"),
    }


def _column_groups(profile: Dict[str, Any], columns: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    groups = {
        "numeric": list(profile.get("numeric_columns") or []),
        "categorical": list(profile.get("categorical_columns") or []),
        "datetime": list(profile.get("datetime_columns") or []),
        "boolean": list(profile.get("boolean_columns") or []),
        "text": list(profile.get("text_columns") or []),
        "id_like": list(profile.get("id_like_columns") or []),
        "geographic": list(profile.get("geographic_columns") or []),
    }

    for column in columns:
        name = column.get("name")
        detected_type = str(column.get("type") or "").lower()
        if not name:
            continue
        if detected_type == "numeric" and name not in groups["numeric"]:
            groups["numeric"].append(name)
        elif "categorical" in detected_type and name not in groups["categorical"]:
            groups["categorical"].append(name)
        elif detected_type == "datetime" and name not in groups["datetime"]:
            groups["datetime"].append(name)
        elif detected_type == "boolean" and name not in groups["boolean"]:
            groups["boolean"].append(name)
        elif detected_type == "text" and name not in groups["text"]:
            groups["text"].append(name)
        elif (detected_type == "id" or column.get("is_id_like")) and name not in groups["id_like"]:
            groups["id_like"].append(name)

    return {key: values[:MAX_COLUMNS] for key, values in groups.items()}


def _latest_preview_rows(dataset: Dataset) -> List[Dict[str, Any]]:
    version = (
        dataset.versions.filter(is_active=True)
        .order_by("-version_number")
        .first()
    )
    if version and isinstance(version.preview_rows, list) and version.preview_rows:
        return version.preview_rows[:MAX_SAMPLE_ROWS]

    if not dataset.file:
        return []

    try:
        dataset.file.open("rb")
        preview = build_preview(read_dataset_file(dataset.file), limit=MAX_SAMPLE_ROWS)
        return preview.get("rows", [])[:MAX_SAMPLE_ROWS]
    except Exception:
        return []
    finally:
        try:
            dataset.file.close()
        except Exception:
            pass


def _masked_sample_rows(rows: Iterable[Dict[str, Any]], columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sensitive_names = {_column_name(column) for column in columns if _is_sensitive_column(column)}
    masked = []
    for row in list(rows)[:MAX_SAMPLE_ROWS]:
        if not isinstance(row, dict):
            continue
        masked.append({
            str(key): ("[masked]" if str(key) in sensitive_names or SENSITIVE_NAME_RE.search(str(key)) else value)
            for key, value in row.items()
        })
    return masked




def _column_sample_values(sample_rows: List[Dict[str, Any]], column_name: str, limit: int = 6) -> List[Any]:
    values = []
    for row in sample_rows or []:
        if not isinstance(row, dict) or column_name not in row:
            continue
        value = row.get(column_name)
        if value in (None, ""):
            continue
        if value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def _attach_column_evidence(payload: Dict[str, Any], sample_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    item = dict(payload)
    column_name = str(item.get("column_name") or "").strip()
    if column_name:
        item["sample_values_from_preview"] = _column_sample_values(sample_rows, column_name)
    return item


def _cleaning_report_context(dataset: Dataset, sample_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build focused cleaning context for LLM prompts without sending full data."""
    try:
        from apps.cleaning.services import build_cleaning_report

        report = build_cleaning_report(dataset)
    except Exception:
        return {"available": False}

    missing_values = [_attach_column_evidence(item, sample_rows) for item in report.get("missing_values", [])]
    imputation_options = [_attach_column_evidence(item, sample_rows) for item in report.get("imputation_options", [])]
    outliers = [_attach_column_evidence(item, sample_rows) for item in report.get("outliers", [])]
    column_outlier_options = [_attach_column_evidence(item, sample_rows) for item in report.get("column_outlier_options", [])]
    recommendations = [_attach_column_evidence(item, sample_rows) for item in report.get("recommendations", [])]

    return {
        "available": True,
        "summary": report.get("summary", {}),
        "missing_values": missing_values[:40],
        "imputation_options": imputation_options[:40],
        "outliers": outliers[:40],
        "column_outlier_options": column_outlier_options[:40],
        "duplicates": report.get("duplicates", {}),
        "datatype_issues": report.get("datatype_issues", [])[:30],
        "cardinality_issues": report.get("cardinality_issues", [])[:30],
        "constant_features": report.get("constant_features", [])[:30],
        "dataset_anomaly": report.get("dataset_anomaly"),
        "recommendations": recommendations[:80],
    }


def _trim_rows(rows: Any, limit: int = 12) -> List[Dict[str, Any]]:
    """Keep a tiny, JSON-safe set of visible graph rows for AI evidence."""
    if not isinstance(rows, list):
        return []
    safe_rows = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        safe_row = {}
        for key, value in list(row.items())[:20]:
            text_key = str(key)
            if SENSITIVE_NAME_RE.search(text_key):
                safe_row[text_key] = "[masked]"
            else:
                safe_row[text_key] = value
        safe_rows.append(safe_row)
    return safe_rows


def _sanitize_frontend_context(value: Any, depth: int = 0) -> Any:
    """Allow clicked chart/EDA context while preventing full dataframe upload."""
    if depth > 5:
        return "[trimmed]"
    if isinstance(value, list):
        return [_sanitize_frontend_context(item, depth + 1) for item in value[:30]]
    if not isinstance(value, dict):
        return value

    blocked_keys = {"dataframe", "csv", "file", "raw_file", "full_rows", "all_rows"}
    row_keys = {"sample_rows", "rows", "preview_rows", "chart_rows", "column_rows", "visible_rows"}
    safe = {}
    for key, item in value.items():
        text_key = str(key)
        if text_key in blocked_keys:
            continue
        if text_key == "chart_data" and isinstance(item, dict):
            safe[text_key] = {
                **{k: _sanitize_frontend_context(v, depth + 1) for k, v in item.items() if k != "rows"},
                "rows": _trim_rows(item.get("rows"), limit=50),
            }
            continue
        if text_key == "source_rows":
            safe[text_key] = _trim_rows(item, limit=20)
            continue
        if text_key in row_keys:
            safe[text_key] = _trim_rows(item, limit=12)
            continue
        safe[text_key] = _sanitize_frontend_context(item, depth + 1)
    return safe


def _extract_used_columns(frontend_context: Dict[str, Any] | None) -> List[str]:
    if not isinstance(frontend_context, dict):
        return []
    used = []

    def add(value):
        if isinstance(value, str) and value and value not in used:
            used.append(value)
        elif isinstance(value, list):
            for item in value:
                add(item)

    containers = [
        frontend_context,
        frontend_context.get("selected_graph_context") or {},
        frontend_context.get("selected_dashboard_card") or {},
        frontend_context.get("eda_graph_context") or {},
    ]
    selected_graph_context = frontend_context.get("selected_graph_context") or {}
    if isinstance(selected_graph_context, dict):
        containers.append(selected_graph_context.get("eda_graph_context") or {})
        containers.append(selected_graph_context.get("graph_response_meta") or {})
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in ("column", "x_column", "y_column", "value_column", "size_column", "color_by_column", "group_column", "group_by_column", "target_column", "latitude_column", "longitude_column"):
            add(container.get(key))
        add(container.get("columns"))
        add(container.get("selected_columns"))
        add(container.get("used_columns"))
        config = container.get("chart_config") or container.get("chart_config_json") or {}
        if isinstance(config, dict):
            for key in ("x_column", "y_column", "value_column", "size_column", "color_by_column", "group_by_column", "target_column"):
                add(config.get(key))
            add(config.get("columns"))
    return used[:20]


def _merge_extra(context: Dict[str, Any], extra_context: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(extra_context, dict) or not extra_context:
        return context
    safe_extra = _sanitize_frontend_context(extra_context)
    context["frontend_context"] = safe_extra

    used_columns = _extract_used_columns(safe_extra)
    available_profiles = {column.get("name"): column for column in context.get("columns", []) if isinstance(column, dict)}
    context["active_graph_evidence"] = {
        "used_columns": [name for name in used_columns if name in available_profiles],
        "column_profiles": [available_profiles[name] for name in used_columns if name in available_profiles],
        "clicked_context_present": bool(safe_extra.get("selected_graph_context") or safe_extra.get("selected_dashboard_card") or safe_extra.get("eda_graph_context")),
    }
    return context


def merge_frontend_context(context: Dict[str, Any], frontend_context: Dict[str, Any] | None) -> Dict[str, Any]:
    return _merge_extra(context, frontend_context)


def trim_context_for_token_limit(context: Dict[str, Any]) -> Dict[str, Any]:
    trimmed = dict(context or {})
    while len(json.dumps(trimmed, default=str, ensure_ascii=False)) > MAX_CONTEXT_CHARS:
        if trimmed.get("sample_rows"):
            trimmed["sample_rows"] = trimmed["sample_rows"][: max(0, len(trimmed["sample_rows"]) - 2)]
        elif len(trimmed.get("columns", [])) > 20:
            trimmed["columns"] = trimmed["columns"][: max(20, len(trimmed["columns"]) - 10)]
        elif trimmed.get("frontend_context"):
            trimmed["frontend_context"] = {"trimmed": True}
        else:
            break
    return trimmed


def build_dataset_ai_context(dataset: Dataset, user, include_sample: bool = True) -> Dict[str, Any]:
    profile = _profile_json(dataset)
    raw_columns = _as_list(profile.get("columns") or profile.get("column_profiles") or dataset.columns_json)
    columns = [_compact_column(column) for column in raw_columns[:MAX_COLUMNS] if _column_name(column)]
    sample_rows = _masked_sample_rows(_latest_preview_rows(dataset), raw_columns) if include_sample else []

    context = {
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "row_count": dataset.row_count or profile.get("row_count"),
            "column_count": dataset.column_count or profile.get("column_count"),
            "source_version": "latest_available",
        },
        "columns": columns,
        "column_groups": _column_groups(profile, columns),
        "quality": {
            "duplicate_row_count": profile.get("duplicate_row_count"),
            "total_missing_cells": profile.get("total_missing_cells"),
            "total_missing_percentage": profile.get("total_missing_percentage"),
            "quality_score": profile.get("quality_score"),
            "constant_columns": profile.get("constant_columns", []),
            "high_cardinality_columns": profile.get("high_cardinality_columns", []),
        },
        "sample_rows": sample_rows,
    }
    return trim_context_for_token_limit(context)


def build_cleaning_ai_context(dataset: Dataset, user, extra_context=None) -> Dict[str, Any]:
    # Cleaning suggestions need profile stats, available method catalogs, issue
    # counts, and a small masked preview so method selection is evidence-based.
    context = build_dataset_ai_context(dataset, user, include_sample=True)
    context["surface"] = "cleaning"
    context["cleaning_report"] = _cleaning_report_context(dataset, context.get("sample_rows", []))
    context["method_policy"] = {
        "missing_values": "Choose selected_method only from imputation_options.available_methods for that column.",
        "outliers": "Choose selected_method only from column_outlier_options.available_methods for that column.",
        "safe_apply": "Detection/review methods should not delete rows. Capping/removal/drop methods require explicit user confirmation.",
    }
    return trim_context_for_token_limit(_merge_extra(context, extra_context))


def build_chart_ai_context(dataset: Dataset, user, extra_context=None) -> Dict[str, Any]:
    context = build_dataset_ai_context(dataset, user, include_sample=True)
    context["surface"] = "charts"
    return trim_context_for_token_limit(_merge_extra(context, extra_context))


def build_dashboard_ai_context(dataset: Dataset, user, extra_context=None) -> Dict[str, Any]:
    context = build_dataset_ai_context(dataset, user, include_sample=False)
    context["surface"] = "dashboard"
    return trim_context_for_token_limit(_merge_extra(context, extra_context))


def build_chat_ai_context(dataset: Dataset, user, question=None, extra_context=None) -> Dict[str, Any]:
    context = build_dataset_ai_context(dataset, user, include_sample=True)
    context["surface"] = "chat"
    context["question"] = question or ""
    return trim_context_for_token_limit(_merge_extra(context, extra_context))
