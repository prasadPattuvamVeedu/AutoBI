"""
Dashboard command-center service helpers.

These functions intentionally avoid implementing manual forecasting, simulation,
or pandas-heavy logic. They validate and echo dashboard command configuration so
the frontend has real backend-backed endpoints while manual implementation can
be added later.
"""

from __future__ import annotations


SUPPORTED_FILTER_OPERATORS = {
    "equals",
    "not_equals",
    "contains",
    "not_contains",
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "between",
    "in",
    "is_null",
    "is_not_null",
}

SUPPORTED_DASHBOARD_THEMES = {
    "executive_light",
    "dark_boardroom",
    "clean_white",
    "violet_premium",
    "midnight",
    "clean_grid",
    "presentation",
}


def _clean_list(value):
    if not isinstance(value, list):
        return []
    return [item for item in value if item not in (None, "")]


def normalize_dashboard_filters(filters):
    """Return backend-safe dashboard filters without touching chart logic."""
    normalized = []
    for item in filters or []:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or item.get("field") or "").strip()
        operator = str(item.get("operator") or "equals").strip().lower()
        if not column or operator not in SUPPORTED_FILTER_OPERATORS:
            continue
        value = item.get("value")
        if operator in {"is_null", "is_not_null"}:
            value = None
        normalized.append({
            "id": item.get("id") or f"filter-{len(normalized) + 1}",
            "column": column,
            "operator": operator,
            "value": value,
            "scope": item.get("scope") or item.get("appliesTo") or "dashboard",
        })
    return normalized


def build_dashboard_command_config(payload):
    """Normalize dashboard command-center config for save/load style endpoints."""
    payload = payload if isinstance(payload, dict) else {}
    theme = str(payload.get("theme") or "executive_light").strip()
    if theme not in SUPPORTED_DASHBOARD_THEMES:
        theme = "executive_light"

    return {
        "theme": theme,
        "filters": normalize_dashboard_filters(payload.get("filters") or []),
        "tooltip_fields": _clean_list(payload.get("tooltip_fields") or payload.get("tooltipFields") or []),
        "drill_path": _clean_list(payload.get("drill_path") or payload.get("drillPath") or []),
        "conditional_rules": _clean_list(payload.get("conditional_rules") or payload.get("conditionalRules") or []),
        "forecast": payload.get("forecast") or {},
        "what_if": payload.get("what_if") or payload.get("whatIf") or {},
    }


def build_dashboard_forecast_config(payload):
    """
    Build a safe forecast configuration response.

    # MANUAL PANDAS/ML CODE REQUIRED
    Real forecasting should be implemented manually later using the selected
    date column, value column, periods, and method.
    """
    payload = payload if isinstance(payload, dict) else {}
    return {
        "date_column": payload.get("date_column") or payload.get("dateColumn") or "",
        "value_column": payload.get("value_column") or payload.get("valueColumn") or "",
        "periods": int(payload.get("periods") or 6),
        "method": payload.get("method") or "moving_average",
        "status": "configuration_saved",
        "message": "Manual forecasting implementation required.",
        "manual_required": True,
    }


def build_dashboard_what_if_config(payload):
    """
    Build a safe what-if configuration response.

    # MANUAL PANDAS/ML CODE REQUIRED
    Real simulation calculations should be implemented manually later.
    """
    payload = payload if isinstance(payload, dict) else {}
    return {
        "variable": payload.get("variable") or "",
        "change_type": payload.get("change_type") or payload.get("changeType") or "percentage",
        "change_value": payload.get("change_value") or payload.get("changeValue") or 0,
        "affected_measure": payload.get("affected_measure") or payload.get("affectedMeasure") or "",
        "status": "configuration_saved",
        "message": "Manual simulation implementation required.",
        "manual_required": True,
    }


def build_dashboard_drill_config(payload):
    """
    Build a safe drill configuration response.

    # MANUAL CODING GUIDE ONLY
    Drill execution can reuse existing chart generation filters on the frontend.
    """
    payload = payload if isinstance(payload, dict) else {}
    return {
        "drill_path": _clean_list(payload.get("drill_path") or payload.get("drillPath") or []),
        "filters": normalize_dashboard_filters(payload.get("filters") or []),
        "status": "configuration_saved",
        "message": "Drill configuration saved. Use existing chart generation filters for execution.",
    }
