from __future__ import annotations

import re
from typing import Any, Dict, List

from apps.llm.services.llm_client import LLMConfigurationError, LLMServiceError, generate_json
from .prompt_builder import build_autobi_prompt


SUPPORTED_TASKS = {
    "dataset_summary",
    "cleaning_suggestions",
    "outlier_suggestions",
    "chart_suggestions",
    "chart_insights",
    "chart_graph_insight",
    "dashboard_summary",
    "dashboard_graph_insight",
    "eda_graph_insight",
    "feature_engineering_suggestions",
    "chat_assistant",
    "ml_recommendations",
    "natural_language_chart_builder",
}

SUPPORTED_CHART_TYPES = {"bar", "line", "scatter", "histogram", "donut", "pie", "area", "table", "kpi", "heatmap", "box", "map"}


GRAPH_INSIGHT_TASKS = {"chart_insights", "eda_graph_insight", "chart_graph_insight", "dashboard_graph_insight", "dashboard_summary"}


def _safe_len(value):
    return len(value) if isinstance(value, list) else 0


def _as_rows(value):
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        rows = value.get("rows") or value.get("data") or value.get("items") or []
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    return []


def _graph_context_candidates(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return every plausible clicked-graph context shape.

    The frontend has used a few shapes over time: sometimes the clicked graph is
    passed directly as the request context, sometimes under frontend_context, and
    sometimes nested as selected_graph_context.ai_context.  Missing one shape was
    causing eda_graph_insight to fall through to the external LLM and timeout.
    """
    frontend = context.get("frontend_context") if isinstance(context.get("frontend_context"), dict) else {}
    candidates = []

    def add(value):
        if not isinstance(value, dict):
            return
        candidates.append(value)
        for nested_key in (
            "eda_graph_context",
            "selected_graph_context",
            "selected_dashboard_card",
            "ai_context",
            "graph_context",
        ):
            nested = value.get(nested_key)
            if isinstance(nested, dict) and nested is not value:
                candidates.append(nested)

    add(context)
    add(frontend)
    for value in (
        frontend.get("eda_graph_context"),
        frontend.get("selected_graph_context"),
        frontend.get("selected_dashboard_card"),
        frontend.get("graph_response_meta"),
        context.get("eda_graph_context"),
        context.get("selected_graph_context"),
        context.get("selected_dashboard_card"),
        context.get("ai_context"),
    ):
        add(value)

    unique = []
    seen = set()
    for item in candidates:
        marker = id(item)
        if marker not in seen:
            unique.append(item)
            seen.add(marker)
    return unique


def _best_graph_context(context: Dict[str, Any]) -> Dict[str, Any] | None:
    best = None
    best_score = -1
    for candidate in _graph_context_candidates(context):
        rows = _as_rows(candidate.get("chart_data")) or _as_rows(candidate.get("graph_rows")) or _as_rows(candidate.get("rows")) or _as_rows(candidate.get("sample_rows"))
        score = 0
        if candidate.get("chart_type") or candidate.get("chart_title") or candidate.get("title"):
            score += 2
        if candidate.get("used_columns") or candidate.get("columns") or candidate.get("x_column") or candidate.get("y_column"):
            score += 2
        score += min(len(rows), 5)
        if candidate.get("column_statistics") or candidate.get("column_profiles"):
            score += 3
        if candidate.get("graph_summary") or candidate.get("summary"):
            score += 2
        if score > best_score:
            best = candidate
            best_score = score
    return best if best_score > 0 else None


def _graph_rows(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("chart_data", "graph_rows", "rows", "data", "sample_rows", "source_rows"):
        rows = _as_rows(graph.get(key))
        if rows:
            return rows[:50]
    return []


def _graph_used_columns(graph: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
    used = []
    def add(value):
        if isinstance(value, str) and value and value not in used:
            used.append(value)
        elif isinstance(value, list):
            for item in value:
                add(item)
    for key in ("used_columns", "columns", "selected_columns"):
        add(graph.get(key))
    for key in ("column", "x_column", "y_column", "group_column", "group_by_column", "color_by_column", "size_column", "target_column"):
        add(graph.get(key))
    config = graph.get("chart_config") or graph.get("chart_config_json") or {}
    if isinstance(config, dict):
        for key in ("x_column", "y_column", "group_column", "group_by_column", "color_by_column", "size_column", "target_column"):
            add(config.get(key))
    evidence = context.get("active_graph_evidence") or {}
    if isinstance(evidence, dict):
        add(evidence.get("used_columns"))
    return used[:12]


def _column_profile_map(context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    profiles = {}

    def add_profile(column):
        if not isinstance(column, dict):
            return
        name = column.get("name") or column.get("column") or column.get("column_name")
        if name:
            profiles[str(name)] = column

    for column in _columns(context):
        add_profile(column)
    evidence = context.get("active_graph_evidence") or {}
    for column in evidence.get("column_profiles", []) if isinstance(evidence, dict) else []:
        add_profile(column)
    for graph in _graph_context_candidates(context):
        for column in graph.get("column_profiles", []) or []:
            add_profile(column)
        for column in graph.get("column_statistics", []) or []:
            add_profile(column)
    return profiles


def _format_number(value):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer():
        return int(number)
    return round(number, 4)


def _row_extreme_findings(rows: List[Dict[str, Any]], chart_type: str, x_col: str = "", y_col: str = "") -> List[str]:
    findings = []
    if not rows:
        return findings
    if not y_col:
        numeric_keys = []
        for key in rows[0].keys():
            if key == x_col:
                continue
            if any(isinstance(row.get(key), (int, float)) for row in rows):
                numeric_keys.append(key)
        y_col = numeric_keys[0] if numeric_keys else ""
    if y_col:
        numeric_rows = []
        for row in rows:
            value = row.get(y_col)
            if isinstance(value, (int, float)):
                numeric_rows.append((row, value))
            else:
                try:
                    numeric_rows.append((row, float(value)))
                except (TypeError, ValueError):
                    pass
        if numeric_rows:
            high_row, high_value = max(numeric_rows, key=lambda item: item[1])
            low_row, low_value = min(numeric_rows, key=lambda item: item[1])
            high_label = high_row.get(x_col) if x_col else (high_row.get("column") or high_row.get("category") or high_row.get("value"))
            low_label = low_row.get(x_col) if x_col else (low_row.get("column") or low_row.get("category") or low_row.get("value"))
            if high_label is not None:
                findings.append(f"Highest visible {y_col} is {_format_number(high_value)} at {high_label}.")
            if low_label is not None and low_label != high_label:
                findings.append(f"Lowest visible {y_col} is {_format_number(low_value)} at {low_label}.")
    if "hist" in chart_type and rows:
        findings.append(f"The histogram insight is based on {len(rows)} returned bin/value rows from the selected graph.")
    elif "missing" in chart_type and rows:
        findings.append(f"Missing-value insight is based on {len(rows)} returned column-quality rows.")
    elif "box" in chart_type:
        summary_bits = []
        for key in ("min", "q1", "median", "q3", "max", "outlier_count"):
            if key in rows[0]:
                summary_bits.append(f"{key}={_format_number(rows[0].get(key))}")
        if summary_bits:
            findings.append("Boxplot summary: " + ", ".join(summary_bits) + ".")
    return findings[:4]



def _numeric_value(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_profile_list(value):
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            if isinstance(item, dict):
                merged = dict(item)
                merged.setdefault("column", key)
                rows.append(merged)
            else:
                rows.append({"column": key, "value": item})
        return rows
    return []


def _profile_for_column(graph: Dict[str, Any], context: Dict[str, Any], column: str) -> Dict[str, Any]:
    candidates = []
    for source in (graph, context):
        if isinstance(source, dict):
            candidates.extend(_as_profile_list(source.get("column_statistics")))
            candidates.extend(_as_profile_list(source.get("column_profiles")))
    candidates.extend(_column_profile_map(context).values())
    for item in candidates:
        if column and column in {item.get("column"), item.get("name"), item.get("column_name")}:
            return item
    return candidates[0] if candidates else {}


def _deterministic_graph_insight(task: str, context: Dict[str, Any]) -> Dict[str, Any] | None:
    graph = _best_graph_context(context)
    if not graph:
        return None

    chart_type = str(graph.get("chart_type") or graph.get("type") or "selected graph").lower()
    title = str(graph.get("chart_title") or graph.get("title") or graph.get("name") or f"{chart_type} insight")
    config = graph.get("chart_config") or graph.get("chart_config_json") or {}
    x_col = graph.get("x_column") or (config.get("x_column") if isinstance(config, dict) else "") or graph.get("column") or ""
    y_col = graph.get("y_column") or (config.get("y_column") if isinstance(config, dict) else "") or ""
    used_columns = _graph_used_columns(graph, context)
    primary_column = used_columns[0] if used_columns else (x_col or y_col or graph.get("column") or "selected column")
    rows = _graph_rows(graph)
    profile = _profile_for_column(graph, context, primary_column)
    warnings = graph.get("warnings") if isinstance(graph.get("warnings"), list) else []
    graph_summary = graph.get("graph_summary") or graph.get("summary") or {}

    key_findings: List[str] = []
    hidden_insights: List[str] = []
    quality_notes: List[str] = []
    recommended_actions: List[str] = []

    row_count = sum((_numeric_value(row.get("count")) or 0) for row in rows if isinstance(row, dict))

    if any(token in chart_type for token in ("hist", "bar", "count", "missing")):
        numeric_key = None
        if rows and isinstance(rows[0], dict):
            if "count" in rows[0]:
                numeric_key = "count"
            elif "missing_count" in rows[0]:
                numeric_key = "missing_count"
        if numeric_key and rows:
            sorted_rows = sorted(rows, key=lambda row: _numeric_value(row.get(numeric_key)) or 0, reverse=True)
            top = sorted_rows[0]
            second = sorted_rows[1] if len(sorted_rows) > 1 else None
            top_label = top.get("label") or top.get("value") or top.get("bin") or top.get("column") or top.get("category") or "top bucket"
            top_value = _numeric_value(top.get(numeric_key)) or 0
            share = (top_value / row_count * 100) if row_count else None
            key_findings.append(f"{top_label} is the strongest visible bucket with {_format_number(top_value)} records" + (f" ({round(share, 1)}% of returned rows)." if share is not None else "."))
            if second:
                second_value = _numeric_value(second.get(numeric_key)) or 0
                second_label = second.get("label") or second.get("value") or second.get("bin") or second.get("column") or second.get("category") or "second bucket"
                ratio = round(top_value / second_value, 1) if second_value else "many"
                key_findings.append(f"{second_label} is next with {_format_number(second_value)} records, so the top bucket is {ratio}x larger.")
            key_findings.append(f"The preview uses {len(rows)} returned bucket/value rows for {primary_column}.")
            if share is not None and share >= 30:
                hidden_insights.append(f"One bucket controls about {round(share, 1)}% of the visible graph rows. This imbalance can dominate averages, dashboard summaries, and ML splits.")
            if len(rows) <= 30 and any(str(row.get("label") or row.get("value") or "").isdigit() for row in rows):
                hidden_insights.append(f"{primary_column} looks numeric, but repeated code-like values behave like categories. Do not treat it blindly as a continuous measurement.")
        if "missing" in chart_type:
            hidden_insights.append("High missingness can be information, not only a defect. Missing indicators may capture real business conditions.")
            recommended_actions.extend([
                "Use column-specific imputation: mode/missing category for categorical fields and median for skewed numeric fields.",
                "Create missing indicators for columns where missingness may carry meaning.",
                "Before dropping very sparse columns, compare their relationship with the target or key KPI.",
            ])
        else:
            recommended_actions.extend([
                "Compare this distribution against the target or a key category to reveal which bucket drives the outcome.",
                "Group rare buckets into 'Other' if long-tail categories make the chart noisy or ML encoding unstable.",
            ])
    elif "box" in chart_type:
        row = rows[0] if rows and isinstance(rows[0], dict) else graph_summary if isinstance(graph_summary, dict) else {}
        min_v = _numeric_value(row.get("min"))
        q1 = _numeric_value(row.get("q1"))
        median = _numeric_value(row.get("median"))
        q3 = _numeric_value(row.get("q3"))
        max_v = _numeric_value(row.get("max"))
        iqr = _numeric_value(row.get("iqr"))
        if iqr is None and q1 is not None and q3 is not None:
            iqr = q3 - q1
        outlier_count = _numeric_value(row.get("outlier_count")) or 0
        if median is not None and q1 is not None and q3 is not None:
            key_findings.append(f"The middle 50% of {primary_column} lies between {_format_number(q1)} and {_format_number(q3)}, with median {_format_number(median)}.")
        if min_v is not None and max_v is not None:
            key_findings.append(f"The observed range is {_format_number(min_v)} to {_format_number(max_v)}.")
        if outlier_count:
            key_findings.append(f"There are {_format_number(outlier_count)} IQR outlier candidate(s).")
        if max_v is not None and q3 is not None and iqr and iqr > 0 and max_v > q3 + 3 * iqr:
            hidden_insights.append("The upper tail is very long. A few high values can pull the mean upward and hide the normal range.")
        if median is not None and q1 is not None and q3 is not None and iqr and abs((q3 - median) - (median - q1)) > max(1, iqr) * 0.35:
            hidden_insights.append("The box is asymmetric, suggesting skewness. Median/IQR are safer than mean/std for this column.")
        recommended_actions.extend([
            "Review outlier rows with domain rules before deleting anything.",
            "Use IQR capping or winsorization for clear measurement extremes.",
            "If right-skewed, test log transform or robust scaling before ML modeling.",
            "Split the boxplot by target/category to see whether outliers belong to a meaningful segment.",
        ])
    elif "scatter" in chart_type:
        key_findings.append(f"The scatter plot compares {', '.join(used_columns) if used_columns else 'the selected columns'}.")
        hidden_insights.append("Clusters, curved shapes, and extreme points may reveal segments or nonlinear behavior that a simple correlation value can miss.")
        recommended_actions.extend([
            "Calculate Pearson and Spearman correlation.",
            "Add color by target/category to reveal hidden groups.",
            "Inspect extreme points before fitting a model.",
        ])
    else:
        key_findings.append(f"The selected graph returned {len(rows)} row(s) of evidence." if rows else "The selected graph has limited row evidence.")
        hidden_insights.append("Use this graph as an exploration starting point; stronger conclusions need target/group comparison and profile statistics.")
        recommended_actions.append("Generate the graph with clear x/y fields, then compare it before and after cleaning.")

    if profile:
        missing_pct = profile.get("missing_percentage")
        missing_count = profile.get("missing_count")
        unique_count = profile.get("unique_count")
        outlier_count = profile.get("outlier_count")
        try:
            if missing_pct is not None and float(missing_pct) > 0:
                quality_notes.append(f"{primary_column} has {missing_count if missing_count is not None else 'some'} missing values ({missing_pct}%).")
        except (TypeError, ValueError):
            pass
        try:
            if outlier_count is not None and float(outlier_count) > 0:
                quality_notes.append(f"{primary_column} has {outlier_count} profiled outlier candidate(s).")
        except (TypeError, ValueError):
            pass
        if unique_count is not None:
            quality_notes.append(f"{primary_column} has {unique_count} unique value(s); decide whether it is continuous, ordinal, or coded categorical before encoding/scaling.")
    quality_notes.extend([str(warning) for warning in warnings[:2]])
    if not quality_notes:
        quality_notes.append("No major data-quality warning was supplied with this graph context.")

    summary = f"{title} shows how {primary_column} behaves and what operation should come next. " + (key_findings[0] if key_findings else "Graph context was received.")
    confidence = "high" if len(rows) >= 3 and used_columns else "medium" if used_columns else "low"
    insight = {
        "title": title,
        "description": summary,
        "explanation": summary,
        "key_findings": key_findings[:5],
        "hidden_insights": hidden_insights[:5],
        "data_quality_notes": quality_notes[:5],
        "recommended_actions": recommended_actions[:5],
        "confidence": confidence,
        "related_columns": used_columns,
    }
    return _normalize_result(
        {
            "title": title,
            "summary": summary,
            "insights": [insight],
            "actions": [{"title": f"Next operation {index + 1}", "description": action} for index, action in enumerate(recommended_actions[:5])],
            "key_findings": key_findings[:5],
            "hidden_insights": hidden_insights[:5],
            "data_quality_notes": quality_notes[:5],
            "recommended_actions": recommended_actions[:5],
            "used_columns": used_columns,
            "related_columns": used_columns,
            "confidence": confidence,
            "warnings": warnings[:3],
        },
        status="success",
        source="deterministic",
    )

def _columns(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [column for column in context.get("columns", []) if isinstance(column, dict)]


def _column_names(context: Dict[str, Any]) -> List[str]:
    return [column.get("name") for column in _columns(context) if column.get("name")]


def _top_missing(context: Dict[str, Any]):
    columns = [column for column in _columns(context) if column.get("missing_count") not in (None, "")]
    return sorted(columns, key=lambda column: float(column.get("missing_count") or 0), reverse=True)


def _top_outliers(context: Dict[str, Any]):
    columns = [column for column in _columns(context) if column.get("outlier_count") not in (None, "")]
    return sorted(columns, key=lambda column: float(column.get("outlier_count") or 0), reverse=True)


def _find_columns_in_question(context: Dict[str, Any], question: str) -> List[str]:
    lower_question = question.lower()
    return [name for name in _column_names(context) if name and name.lower() in lower_question]


def _first_group(context: Dict[str, Any], group: str) -> str:
    values = context.get("column_groups", {}).get(group) or []
    return values[0] if values else ""


def _infer_chart_config(context: Dict[str, Any], question: str = "") -> Dict[str, Any] | None:
    question_lower = question.lower()
    names_in_question = _find_columns_in_question(context, question)
    groups = context.get("column_groups", {})
    numeric = groups.get("numeric") or []
    categorical = groups.get("categorical") or []
    datetime = groups.get("datetime") or []

    x_column = ""
    y_column = ""
    chart_type = ""

    mentioned_numeric = [name for name in names_in_question if name in numeric]
    mentioned_categorical = [name for name in names_in_question if name in categorical]
    mentioned_datetime = [name for name in names_in_question if name in datetime]

    if mentioned_datetime and (mentioned_numeric or numeric):
        chart_type = "line"
        x_column = mentioned_datetime[0]
        y_column = mentioned_numeric[0] if mentioned_numeric else numeric[0]
    elif len(mentioned_numeric) >= 2:
        chart_type = "scatter"
        x_column, y_column = mentioned_numeric[:2]
    elif (mentioned_categorical or categorical) and (mentioned_numeric or numeric):
        chart_type = "bar"
        x_column = mentioned_categorical[0] if mentioned_categorical else categorical[0]
        y_column = mentioned_numeric[0] if mentioned_numeric else numeric[0]
    elif mentioned_numeric or numeric:
        y_column = mentioned_numeric[0] if mentioned_numeric else numeric[0]
        chart_type = "kpi" if re.search(r"\b(total|sum|average|avg|kpi)\b", question_lower) else "histogram"
    elif mentioned_categorical or categorical:
        x_column = mentioned_categorical[0] if mentioned_categorical else categorical[0]
        chart_type = "donut" if "donut" in question_lower or "pie" in question_lower else "bar"
    else:
        return None

    if "scatter" in question_lower and len(numeric) >= 2:
        chart_type = "scatter"
        x_column = x_column or numeric[0]
        y_column = y_column or numeric[1]
    elif "line" in question_lower and datetime and numeric:
        chart_type = "line"
        x_column = x_column or datetime[0]
        y_column = y_column or numeric[0]
    elif "histogram" in question_lower and numeric:
        chart_type = "histogram"
        x_column = mentioned_numeric[0] if mentioned_numeric else numeric[0]
        y_column = ""

    top_n_match = re.search(r"\btop\s+(\d+)\b", question_lower)
    return {
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "aggregation": "avg" if re.search(r"\b(avg|average|mean)\b", question_lower) else "sum",
        "group_by": None,
        "top_n": int(top_n_match.group(1)) if top_n_match else (10 if "top" in question_lower else None),
        "sort_order": "desc",
    }


def _validate_chart_config(config: Dict[str, Any] | None, context: Dict[str, Any], warnings: List[str] | None = None) -> Dict[str, Any] | None:
    if not isinstance(config, dict):
        return None
    warnings = warnings if warnings is not None else []
    column_names = set(_column_names(context))
    chart_type = config.get("chart_type")
    if chart_type and chart_type not in SUPPORTED_CHART_TYPES:
        warnings.append(f"Unsupported chart type '{chart_type}' was replaced with table.")
    safe = {
        "chart_type": chart_type if chart_type in SUPPORTED_CHART_TYPES else "table",
        "x_column": config.get("x_column") or "",
        "y_column": config.get("y_column") or "",
        "aggregation": config.get("aggregation") or "sum",
        "group_by": config.get("group_by"),
        "top_n": config.get("top_n"),
        "sort_order": config.get("sort_order") or "desc",
    }
    for key in ("x_column", "y_column", "group_by"):
        if safe.get(key) and safe[key] not in column_names:
            warnings.append(f"Unavailable chart column '{safe[key]}' was removed from {key}.")
            safe[key] = "" if key != "group_by" else None
    return safe


def _chat_intent_response(context: Dict[str, Any], question: str) -> Dict[str, Any] | None:
    question_lower = question.lower()

    if any(word in question_lower for word in ("chart", "plot", "graph", "visual")):
        chart_config = _validate_chart_config(_infer_chart_config(context, question), context)
        if chart_config:
            return _normalize_result({
                "answer": "I prepared an advisory chart configuration from the available column metadata. Review it before creating the chart.",
                "confidence": "medium",
                "related_columns": [value for value in [chart_config.get("x_column"), chart_config.get("y_column")] if value],
                "used_context": ["column_groups", "columns"],
                "suggested_actions": [{"label": "Review chart config", "type": "chart_config"}],
                "chart_config": chart_config,
                "warnings": [],
            }, status="fallback", source="deterministic")

    if "missing" in question_lower or "null" in question_lower:
        missing = _top_missing(context)
        if missing:
            top = missing[0]
            return {
                "answer": f"{top.get('name')} has the most missing values: {top.get('missing_count')} missing cells ({top.get('missing_percentage')}%).",
                "confidence": "high",
                "related_columns": [top.get("name")],
                "used_context": ["columns.missing_count", "columns.missing_percentage"],
                "suggested_actions": [{"label": "Review cleaning suggestions", "type": "cleaning_suggestions"}],
                "chart_config": None,
                "warnings": [],
            }

    if "outlier" in question_lower:
        outliers = _top_outliers(context)
        if outliers:
            top = outliers[0]
            return {
                "answer": f"{top.get('name')} has the highest outlier count in the available profile: {top.get('outlier_count')} ({top.get('outlier_percentage')}%).",
                "confidence": "high",
                "related_columns": [top.get("name")],
                "used_context": ["columns.outlier_count", "columns.outlier_percentage"],
                "suggested_actions": [{"label": "Review outlier options", "type": "outlier_suggestions"}],
                "chart_config": None,
                "warnings": [],
            }

    if "id-like" in question_lower or "id like" in question_lower or "identifier" in question_lower:
        id_like = [column.get("name") for column in _columns(context) if column.get("is_id_like")]
        return {
            "answer": "These columns look ID-like or identifier-oriented based on the dataset profile.",
            "confidence": "high" if id_like else "low",
            "related_columns": [name for name in id_like if name],
            "used_context": ["columns.is_id_like", "columns.type"],
            "suggested_actions": [{"label": "Review identifier columns", "type": "manual_review"}],
            "chart_config": None,
            "warnings": ["Advisory only. Do not drop identifier columns without domain review."] if id_like else [],
        }

    if "high-cardinality" in question_lower or "high cardinality" in question_lower:
        high_cardinality = [column.get("name") for column in _columns(context) if column.get("is_high_cardinality")]
        return {
            "answer": "These columns are marked high-cardinality in the available profile.",
            "confidence": "high" if high_cardinality else "low",
            "related_columns": [name for name in high_cardinality if name],
            "used_context": ["columns.is_high_cardinality", "columns.unique_count"],
            "suggested_actions": [{"label": "Review encoding strategy", "type": "manual_review"}],
            "chart_config": None,
            "warnings": [],
        }

    if any(word in question_lower for word in ("remove", "drop", "delete column", "column removal")):
        candidates = [
            column for column in _columns(context)
            if column.get("recommendation") == "drop_candidate"
            or column.get("is_constant")
            or column.get("is_id_like")
            or float(column.get("missing_percentage") or 0) >= 70
        ]
        return {
            "answer": "Columns should not be removed automatically. Based on metadata, review these candidates before deciding.",
            "confidence": "medium" if candidates else "low",
            "related_columns": [column.get("name") for column in candidates[:8] if column.get("name")],
            "used_context": ["missing_percentage", "is_constant", "is_id_like", "recommendation"],
            "suggested_actions": [{"label": "Review column quality", "type": "manual_review"}],
            "chart_config": None,
            "warnings": ["Advisory only. AutoBI will not drop columns automatically."],
        }

    return None


def _normalize_result(result: Dict[str, Any], *, status: str = "success", source: str = "gemini") -> Dict[str, Any]:
    result = result if isinstance(result, dict) else {"summary": str(result or "")}
    result.setdefault("summary", result.get("answer") or "")
    result.setdefault("answer", result.get("summary") or "")
    result.setdefault("suggestions", [])
    result.setdefault("insights", [])
    result.setdefault("actions", [])
    result.setdefault("warnings", [])
    result.setdefault("related_columns", [])
    result.setdefault("suggested_actions", [])
    result.setdefault("chart_config", None)
    result.setdefault("confidence", "low")
    result.setdefault("status", status)
    result.setdefault("source", source)
    return result


def _unsupported_task_error(task: str) -> Dict[str, Any]:
    return _normalize_result(
        {
            "summary": "Unsupported AI task.",
            "answer": "Unsupported AI task.",
            "warnings": [f"Unsupported AI task: {task}"],
        },
        status="error",
        source="backend",
    )


def _fallback_for_task(task: str) -> Dict[str, Any]:
    if task in {"chart_insights", "dashboard_summary", "eda_graph_insight", "chart_graph_insight", "dashboard_graph_insight"}:
        return _normalize_result({"summary": "AI insight unavailable.", "insights": [], "warnings": ["AI response was unavailable."]}, status="fallback", source="fallback")
    if task == "chat_assistant":
        return _normalize_result({"answer": "I could not answer from the available context.", "confidence": "low", "related_columns": [], "used_context": [], "suggested_actions": [], "chart_config": None, "warnings": ["AI response was unavailable."]}, status="fallback", source="fallback")
    return _normalize_result({"summary": "AI suggestions unavailable.", "suggestions": [], "actions": [], "warnings": ["AI response was unavailable."]}, status="fallback", source="fallback")


def _service_error_for_task(task: str, detail: str) -> Dict[str, Any]:
    if task == "chat_assistant":
        return _normalize_result({"detail": detail, "answer": detail, "warnings": [detail]}, status="error", source="backend")
    if task in {"chart_insights", "dashboard_summary", "eda_graph_insight", "chart_graph_insight", "dashboard_graph_insight"}:
        return _normalize_result({"detail": detail, "summary": detail, "insights": [], "warnings": [detail]}, status="error", source="backend")
    return _normalize_result({"detail": detail, "summary": detail, "suggestions": [], "actions": [], "warnings": [detail]}, status="error", source="backend")


def _sanitize_related_columns(result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    available = set(_column_names(context))
    if not available:
        return result

    def keep_real(values):
        return [value for value in values if value in available] if isinstance(values, list) else []

    if "related_columns" in result:
        result["related_columns"] = keep_real(result.get("related_columns"))

    for key in ("suggestions", "insights", "actions", "suggested_actions"):
        if not isinstance(result.get(key), list):
            continue
        for item in result[key]:
            if not isinstance(item, dict):
                continue
            if "related_columns" in item:
                item["related_columns"] = keep_real(item.get("related_columns"))
            action = item.get("action")
            if isinstance(action, dict) and isinstance(action.get("chart_config"), dict):
                    action["chart_config"] = _validate_chart_config(action.get("chart_config"), context, result.setdefault("warnings", []))
    return result


def run_ai_task(task: str, context: Dict[str, Any], question: str | None = None) -> Dict[str, Any]:
    original_task = task
    if task not in SUPPORTED_TASKS:
        return _unsupported_task_error(original_task)
    context = context or {}
    question = question or context.get("question") or context.get("frontend_context", {}).get("question") or ""

    if task == "natural_language_chart_builder":
        chart_config = _validate_chart_config(_infer_chart_config(context, question), context)
        if chart_config:
            return _normalize_result({"summary": "Chart config inferred from dataset metadata.", "chart_config": chart_config, "suggestions": [], "warnings": []}, status="fallback", source="deterministic")

    if task == "chat_assistant":
        response = _chat_intent_response(context, question)
        if response:
            return _normalize_result(response, status="fallback", source="deterministic")

    if task in GRAPH_INSIGHT_TASKS:
        graph_response = _deterministic_graph_insight(task, context)
        if graph_response:
            return graph_response
        # Graph insights must never wait on the external LLM. If context wiring is
        # incomplete, return a fast actionable message so the UI does not stay on
        # Thinking/timeout.
        return _normalize_result(
            {
                "title": "Select a graph first",
                "summary": "AutoBI did not receive clicked graph rows or statistics. Generate/click the graph once, then run Insight again.",
                "insights": [
                    {
                        "title": "Graph context missing",
                        "explanation": "The insight request reached the backend, but selectedGraphContext was empty or not in a supported shape.",
                        "confidence": "low",
                        "related_columns": [],
                    }
                ],
                "key_findings": ["No chart rows were supplied with this insight request."],
                "data_quality_notes": ["Graph data was unavailable, so no data-quality conclusion was made."],
                "recommended_actions": ["Click a graph card and wait for the preview to finish before pressing Insight."],
                "warnings": ["selectedGraphContext missing"],
                "confidence": "low",
            },
            status="fallback",
            source="deterministic",
        )

    prompt = build_autobi_prompt(task=task, context={**context, "question": question})
    try:
        result = generate_json(prompt)
    except LLMConfigurationError as exc:
        return _service_error_for_task(task, str(exc))
    except LLMServiceError:
        return _service_error_for_task(task, "AI service is temporarily unavailable. Please try again later.")
    except Exception:
        return _fallback_for_task(task)

    if task in {"natural_language_chart_builder", "chat_assistant"}:
        result["chart_config"] = _validate_chart_config(result.get("chart_config"), context, result.setdefault("warnings", []))
    return _normalize_result(_sanitize_related_columns(result, context), status=result.get("status", "success"), source=result.get("source", result.get("provider", "gemini")))
