"""Common chart settings used by every chart response.

This file is intentionally UI/metadata focused. It is safe for Codex/assistant
implementation because it does not decide statistical or ML behavior.

Manual pandas/numpy graph logic should stay in chart_manual_implementation_guide.py
or in chart_data_service.py sections marked MANUAL.
"""

DEFAULT_CHART_THEME = {
    "theme_name": "autobi_premium_light",
    "background": "#FFFFFF",
    "plot_background": "#FFFFFF",
    "primary_color": "#2563EB",
    "accent_color": "#6D35F5",
    "grid_color": "#E5EAF3",
    "text_color": "#0F172A",
    "muted_text_color": "#64748B",
    "font_family": "Inter, Segoe UI, system-ui, sans-serif",
}

COMMON_SETTING_DEFAULTS = {
    "theme": DEFAULT_CHART_THEME,
    "chart_size": "medium",
    "show_grid": True,
    "legend": True,
    "show_labels": True,
    "label_position": "top",
    "element_size": "normal",
    "bar_gap": "normal",
    "line_width": "normal",
    "point_size": "normal",
    "pie_size": "normal",
    "number_format": "compact",
    "currency_prefix": "",
    "percentage_suffix": "",
    "animation": "subtle",
}


def normalize_common_chart_settings(settings=None):
    """Merge frontend style settings with AutoBI defaults.

    This common normalizer applies to all chart types. It should only contain
    presentation settings such as theme, colors, labels, size, grid, legend,
    and number formatting. Data-science decisions such as aggregation, Top N,
    filtering, grouping, sampling, and outlier logic remain in manual chart
    logic areas.
    """
    incoming = dict(settings or {})
    theme = {**DEFAULT_CHART_THEME, **(incoming.get("theme") or {})}
    merged = {**COMMON_SETTING_DEFAULTS, **incoming, "theme": theme}

    # Keep old frontend keys working.
    if "color" in incoming and incoming["color"]:
        merged["theme"]["primary_color"] = incoming["color"]
    else:
        merged["color"] = merged["theme"]["primary_color"]

    merged["show_grid"] = bool(merged.get("show_grid", True))
    merged["legend"] = bool(merged.get("legend", True))
    merged["show_labels"] = bool(merged.get("show_labels", True))
    return merged


def apply_common_chart_settings(response_payload):
    """Attach normalized common style settings to a chart response."""
    payload = dict(response_payload or {})
    payload["settings_json"] = normalize_common_chart_settings(payload.get("settings_json"))
    return payload
