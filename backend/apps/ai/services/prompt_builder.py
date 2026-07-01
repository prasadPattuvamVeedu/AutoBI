from __future__ import annotations

import json
from typing import Any, Dict


BASE_RULES = """
Rules:
- Use only the provided context.
- Do not invent columns.
- Do not invent values.
- If context is insufficient, say what is missing.
- Return valid frontend-safe JSON only.
- Include warnings when confidence is low or context is insufficient.
- related_columns must contain only real column names present in context.columns.
- chart_config must use only valid chart types and existing columns from context.columns.
- Suggestions are advisory only. Do not apply, delete, clean, transform, or train automatically.
""".strip()


def _context_json(context: Dict[str, Any]) -> str:
    return json.dumps(context or {}, ensure_ascii=False, default=str)[:16000]


def _prompt(task: str, context: Dict[str, Any], expected_json: str, focus: str) -> str:
    return f"""
You are AutoBI's dataset-aware analytics assistant.
Task: {task}
Focus: {focus}

{BASE_RULES}

Expected JSON shape:
{expected_json}

Context:
{_context_json(context)}
""".strip()


CLEANING_JSON = """
{
  "summary": "short summary",
  "suggestions": [
    {
      "title": "Column name + issue",
      "column_name": "existing column name or Dataset",
      "issue_code": "MISSING_VALUES or OUTLIERS_DETECTED or other provided issue",
      "selected_method": "one method from the available methods for that exact column",
      "available_methods": [],
      "reason": "explain using missing_count/missing_percentage/outlier_count/outlier_percentage/type/statistics/sample values",
      "confidence": 0.86,
      "safety": "Safe or Review",
      "related_columns": [],
      "action": {
        "method": "same as selected_method",
        "requires_user_confirmation": true
      }
    }
  ],
  "warnings": [],
  "actions": []
}
""".strip()

SUGGESTION_JSON = """
{
  "summary": "short summary",
  "suggestions": [
    {
      "title": "...",
      "reason": "...",
      "confidence": 0.86,
      "related_columns": [],
      "action": {}
    }
  ],
  "warnings": [],
  "actions": []
}
""".strip()

INSIGHT_JSON = """
{
  "summary": "main insight summary",
  "insights": [
    {
      "title": "...",
      "explanation": "...",
      "evidence": "...",
      "confidence": 0.82,
      "related_columns": []
    }
  ],
  "warnings": []
}
""".strip()

GRAPH_INSIGHT_JSON = """
{
  "title": "...",
  "summary": "...",
  "key_findings": [],
  "data_quality_notes": [],
  "recommended_actions": [],
  "used_columns": [],
  "confidence": "high"
}
""".strip()

CHAT_JSON = """
{
  "answer": "...",
  "confidence": "low",
  "related_columns": [],
  "used_context": [],
  "suggested_actions": [],
  "chart_config": null,
  "warnings": []
}
""".strip()

CHART_JSON = """
{
  "summary": "chart recommendation summary",
  "chart_config": {
    "chart_type": "bar",
    "x_column": "",
    "y_column": "",
    "aggregation": "sum",
    "group_by": null,
    "top_n": null,
    "sort_order": "desc"
  },
  "suggestions": [],
  "warnings": []
}
""".strip()


def build_dataset_summary_prompt(context):
    return _prompt("dataset_summary", context, SUGGESTION_JSON, "Summarize dataset shape, quality, useful column groups, and next safe analysis steps.")


def build_cleaning_suggestions_prompt(context):
    return _prompt(
        "cleaning_suggestions",
        context,
        CLEANING_JSON,
        "For each missing-value or cleaning issue, select the best method only from that column's available_methods. Use missing_count, missing_percentage, detected type, profile statistics, and sample_values_from_preview as evidence. Do not suggest a method that is not listed for that exact column.",
    )


def build_outlier_suggestions_prompt(context):
    return _prompt(
        "outlier_suggestions",
        context,
        CLEANING_JSON,
        "For each outlier issue, select the best method only from that column's column_outlier_options.available_methods. Use outlier_count, outlier_percentage, lower/upper bounds, skewness, kurtosis, and sample_values_from_preview. Prefer detection/review unless the user explicitly asks to cap/remove rows.",
    )


def build_chart_suggestions_prompt(context):
    return _prompt("chart_suggestions", context, SUGGESTION_JSON, "Suggest chart ideas using existing numeric, categorical, datetime, and geographic columns.")


def build_chart_insights_prompt(context):
    return _prompt(
        "chart_insights",
        context,
        INSIGHT_JSON,
        "Generate insights from the clicked chart or EDA graph when frontend_context.selected_graph_context, frontend_context.eda_graph_context, or active_graph_evidence is present. Use only chart rows/sample rows, chart_config, used columns, and profile statistics. Mention visible highs/lows/distribution/quality issues only when supported by provided rows or summaries. Do not invent values beyond the supplied graph rows and summaries.",
    )


def build_dashboard_summary_prompt(context):
    return _prompt(
        "dashboard_summary",
        context,
        INSIGHT_JSON,
        "Summarize dashboard state and, when a selected_dashboard_card is provided, prioritize that clicked graph. Use dashboard card chart_config, sample rows, global filters, tooltip fields, and column profiles as evidence. Do not invent totals or trends not present in the card rows/summaries.",
    )


def build_graph_insight_prompt(context):
    return _prompt(
        context.get("task") or "graph_insight",
        context,
        GRAPH_INSIGHT_JSON,
        "Generate insight for the selected/clicked graph only. Mention exact chart_type and columns. Use chart_data rows, source_rows, column_profiles, chart_config, filters, and warnings only. If chart_data is empty, say graph data is insufficient.",
    )


def build_feature_engineering_suggestions_prompt(context):
    return _prompt("feature_engineering_suggestions", context, SUGGESTION_JSON, "Suggest advisory feature engineering ideas using existing columns, possible targets, and saved feature rules. Do not create or apply transformations.")


def build_ml_recommendations_prompt(context):
    return _prompt("ml_recommendations", context, SUGGESTION_JSON, "Suggest ML readiness review steps using quality, column roles, missingness, leakage risk, target candidates, and preprocessing metadata. Do not train models.")


def build_chat_assistant_prompt(context):
    return _prompt("chat_assistant", context, CHAT_JSON, "Answer the user's question using retrieved context and any precomputed intent facts.")


def build_natural_language_chart_prompt(context):
    return _prompt("natural_language_chart_builder", context, CHART_JSON, "Return one chart_config from the user question and available column metadata.")


PROMPT_BUILDERS = {
    "dataset_summary": build_dataset_summary_prompt,
    "cleaning_suggestions": build_cleaning_suggestions_prompt,
    "outlier_suggestions": build_outlier_suggestions_prompt,
    "chart_suggestions": build_chart_suggestions_prompt,
    "chart_insights": build_chart_insights_prompt,
    "chart_graph_insight": build_graph_insight_prompt,
    "dashboard_summary": build_dashboard_summary_prompt,
    "dashboard_graph_insight": build_graph_insight_prompt,
    "eda_graph_insight": build_graph_insight_prompt,
    "feature_engineering_suggestions": build_feature_engineering_suggestions_prompt,
    "chat_assistant": build_chat_assistant_prompt,
    "ml_recommendations": build_ml_recommendations_prompt,
    "natural_language_chart_builder": build_natural_language_chart_prompt,
}


def build_autobi_prompt(task: str, context: Dict[str, Any]) -> str:
    builder = PROMPT_BUILDERS.get(task, build_dataset_summary_prompt)
    return builder(context)
