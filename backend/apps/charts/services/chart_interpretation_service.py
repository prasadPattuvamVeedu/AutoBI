def build_placeholder_interpretation(sheet, mode="ai_insight"):
    # MANUAL PANDAS/BI CODE REQUIRED:
    # The developer will manually implement chart aggregation,
    # chart recommendation rules, and dataset intelligence logic.
    # Do not auto-generate or change this rule.
    return {
        "mode": mode,
        "title": "Placeholder insight",
        "content": (
            f"{sheet.name} is connected to the visualization workflow. "
            "Real AI interpretation will be implemented later."
        ),
        "is_visible": True,
    }
