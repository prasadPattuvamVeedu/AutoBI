def build_chart_suggestions(dataset):
    # MANUAL PANDAS/BI CODE REQUIRED:
    # The developer will manually implement chart aggregation,
    # chart recommendation rules, and dataset intelligence logic.
    # Do not auto-generate or change this rule.
    columns = []
    profile = getattr(dataset, "profile", None)
    profile_json = getattr(profile, "profile_json", {}) if profile else {}

    if isinstance(dataset.columns_json, list) and dataset.columns_json:
        columns = dataset.columns_json
    elif isinstance(profile_json, dict):
        columns = profile_json.get("columns") or profile_json.get("column_profiles") or []

    column_names = [
        str(column.get("column_name") or column.get("name") or column.get("column"))
        for column in columns
        if isinstance(column, dict) and (column.get("column_name") or column.get("name") or column.get("column"))
    ]

    if not column_names and isinstance(dataset.columns_json, list):
        column_names = [str(column) for column in dataset.columns_json if column]

    first_column = column_names[0] if column_names else ""
    second_column = column_names[1] if len(column_names) > 1 else first_column

    return {
        "dataset_id": dataset.id,
        "suggestions": [
            {
                "id": "placeholder-bar",
                "title": "Column overview",
                "chart_type": "bar",
                "x_column": first_column,
                "y_column": second_column,
                "reason": "Placeholder suggestion based on available schema columns.",
            },
            {
                "id": "placeholder-line",
                "title": "Trend placeholder",
                "chart_type": "line",
                "x_column": first_column,
                "y_column": second_column,
                "reason": "Connection-only placeholder for Day 10 visualization flow.",
            },
        ],
        "available_columns": column_names,
    }
