from __future__ import annotations

import json
import re
from typing import Any, Dict


SUPPORTED_CHART_TYPES = {"bar", "line", "scatter", "histogram", "donut", "pie", "area", "table", "kpi", "heatmap", "box", "map"}


def _extract_json_candidate(text: str) -> str:
    cleaned = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1].strip()
    return cleaned


def _try_parse_jsonish(value: Any) -> Dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    candidate = _extract_json_candidate(value)
    if not candidate.startswith("{") or not candidate.endswith("}"):
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return []


def _defaults(parsed: Dict[str, Any]) -> Dict[str, Any]:
    parsed.setdefault("summary", "")
    parsed.setdefault("answer", parsed.get("summary") or "")
    parsed["suggestions"] = _coerce_list(parsed.get("suggestions"))
    parsed["insights"] = _coerce_list(parsed.get("insights"))
    parsed["actions"] = _coerce_list(parsed.get("actions"))
    parsed["warnings"] = _coerce_list(parsed.get("warnings"))
    parsed.setdefault("confidence", parsed.get("confidence") or "low")
    parsed.setdefault("related_columns", [])
    parsed.setdefault("used_context", [])
    parsed.setdefault("suggested_actions", _coerce_list(parsed.get("suggested_actions")))
    parsed.setdefault("chart_config", None)
    parsed.setdefault("status", "success")
    parsed.setdefault("source", "gemini")
    return parsed


def _merge_embedded_json(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Gemini sometimes returns a JSON object as a string inside summary/answer."""
    for key in ("summary", "answer", "message", "text"):
        embedded = _try_parse_jsonish(parsed.get(key))
        if embedded:
            merged = {**parsed, **embedded}
            return merged
    return parsed


def _validate_chart_config(parsed: Dict[str, Any]) -> Dict[str, Any]:
    chart_config = parsed.get("chart_config")
    if not isinstance(chart_config, dict):
        parsed["chart_config"] = None
        return parsed

    warnings = parsed.setdefault("warnings", [])
    chart_type = chart_config.get("chart_type")
    if chart_type and chart_type not in SUPPORTED_CHART_TYPES:
        warnings.append(f"Unsupported chart_type '{chart_type}' was replaced with table.")
        chart_config["chart_type"] = "table"

    context_columns = parsed.get("available_columns") or parsed.get("columns") or []
    column_names = set()
    for column in context_columns:
        if isinstance(column, str):
            column_names.add(column)
        elif isinstance(column, dict):
            column_names.add(column.get("name") or column.get("column_name"))
    column_names.discard(None)

    if column_names:
        for key in ("x_column", "y_column", "group_by"):
            value = chart_config.get(key)
            if value and value not in column_names:
                warnings.append(f"Chart config references unavailable column '{value}'.")
                chart_config[key] = "" if key != "group_by" else None
    return parsed



def _extract_string_field(text: str, field: str) -> str:
    """Best-effort extraction for incomplete JSON like {"summary":"...","suggestions":[."""
    if not isinstance(text, str):
        return ""
    pattern = re.compile(rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"', flags=re.IGNORECASE | re.DOTALL)
    match = pattern.search(text)
    if not match:
        return ""
    try:
        return json.loads(f'"{match.group(1)}"')
    except Exception:
        return match.group(1).replace('\\n', ' ').replace('\\"', '"').strip()


def _extract_partial_cards(text: str, key: str) -> list:
    """Extract complete card objects from a partly malformed suggestions/actions list."""
    if not isinstance(text, str) or f'"{key}"' not in text:
        return []
    cards = []
    # This intentionally only extracts simple complete objects. Incomplete objects are ignored.
    for match in re.finditer(r'\{[^{}]*(?:"title"|"name"|"label"|"description"|"reason"|"action")[^{}]*\}', text, flags=re.DOTALL | re.IGNORECASE):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                cards.append(parsed)
        except Exception:
            continue
    return cards[:8]


def _partial_json_fallback(cleaned: str) -> Dict[str, Any] | None:
    summary = _extract_string_field(cleaned, "summary") or _extract_string_field(cleaned, "answer") or _extract_string_field(cleaned, "message")
    suggestions = _extract_partial_cards(cleaned, "suggestions") or _extract_partial_cards(cleaned, "recommendations")
    insights = _extract_partial_cards(cleaned, "insights") or _extract_partial_cards(cleaned, "findings")
    actions = _extract_partial_cards(cleaned, "actions") or _extract_partial_cards(cleaned, "suggested_actions")
    if not (summary or suggestions or insights or actions):
        return None
    return _defaults({
        "summary": summary or "AI response was partially readable.",
        "answer": summary or "AI response was partially readable.",
        "suggestions": suggestions,
        "insights": insights,
        "actions": actions,
        "warnings": ["AI response was not valid JSON, so AutoBI extracted the readable parts."],
        "status": "fallback",
        "source": "parser",
    })

def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse LLM JSON safely. Handles fenced JSON, extra text, and nested JSON strings."""
    if not text:
        return _defaults({"summary": "Empty AI response.", "status": "fallback", "source": "parser"})

    cleaned = (text or "").strip()
    candidate = _extract_json_candidate(cleaned)

    try:
        parsed = json.loads(candidate)
        parsed = parsed if isinstance(parsed, dict) else {"items": parsed}
        parsed = _merge_embedded_json(parsed)
        return _validate_chart_config(_defaults(parsed))
    except json.JSONDecodeError:
        pass

    parsed = _try_parse_jsonish(cleaned)
    if parsed:
        parsed = _merge_embedded_json(parsed)
        return _validate_chart_config(_defaults(parsed))

    partial = _partial_json_fallback(cleaned)
    if partial:
        return _validate_chart_config(partial)

    return _defaults({
        "summary": cleaned[:1000],
        "answer": cleaned[:1000],
        "status": "fallback",
        "source": "parser",
        "parse_warning": "LLM did not return valid JSON.",
        "warnings": ["AI response could not be parsed as JSON."],
    })
