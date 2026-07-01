"""Rule-based chart insight service for AutoBI."""

from __future__ import annotations


def generate_chart_insight(rows, chart_type, payload):
    """Generate simple chart insight from returned rows."""
    rows = rows or []
    if not rows:
        return {"summary": "No rows available for insight.", "bullets": []}
    x_key = payload.get("x_column") or "label"
    y_key = payload.get("y_column") or "value"
    numeric_rows = []
    for row in rows:
        try:
            numeric_rows.append((row, float(row.get(y_key))))
        except (TypeError, ValueError):
            continue
    if not numeric_rows:
        return {"summary": f"{len(rows)} rows are available for this chart.", "bullets": []}
    top_row, top_value = max(numeric_rows, key=lambda item: item[1])
    low_row, low_value = min(numeric_rows, key=lambda item: item[1])
    return {
        "summary": f"Highest value is {top_value:,.2f}.",
        "bullets": [
            f"Top category: {top_row.get(x_key, top_row.get('label', 'N/A'))}",
            f"Lowest category: {low_row.get(x_key, low_row.get('label', 'N/A'))}",
            f"Rows analyzed: {len(rows)}",
        ],
    }
