from __future__ import annotations

import ast
import math
import re
from typing import Any

import numpy as np
import pandas as pd


ALLOWED_FUNCTIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "safe_divide",
    "round",
    "abs",
    "log",
    "sqrt",
    "year",
    "month",
    "day",
    "concat",
    "contains",
    "equals",
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "and_condition",
    "or_condition",
    "if_else",
}

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Compare,
    ast.BoolOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.And,
    ast.Or,
)


def _safe_int(value: Any, default: int = 20) -> int:
    try:
        value = int(value)

        if value <= 0:
            return default

        return value

    except (TypeError, ValueError):
        return default


def _quote_simple_literal(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return '""'
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
        return text
    if text.lower() in {"true", "false", "null", "none"}:
        return text.lower().replace("none", "null")
    return '"' + text.replace('"', '\\"') + '"'


def _normalize_user_friendly_expression(expression: str) -> str:
    """
    Accept a small human-friendly rule syntax and convert it to the
    existing safe AST expression language. This keeps backend execution safe
    while letting users write simple BI rules such as:
        if SalePrice > 200000 then High else Low
    """

    text = str(expression or "").strip()
    if not text:
        return text

    simple_if_match = re.match(
        r"^if\s+(.+?)\s+then\s+(.+?)\s+else\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if simple_if_match:
        condition, true_value, false_value = simple_if_match.groups()
        return (
            f"if_else({condition.strip()}, "
            f"{_quote_simple_literal(true_value)}, "
            f"{_quote_simple_literal(false_value)})"
        )

    return text


def _replace_column_names_with_safe_tokens(
    expression: str,
    columns: list[str],
) -> tuple[str, dict[str, str]]:
    """
    Replace real dataframe column names with safe Python identifiers.

    Example:
        "Profit / Sales"
        becomes:
        "__col_0__ / __col_1__"

        "year(Order Date)"
        becomes:
        "year(__col_2__)"
    """

    safe_expression = expression
    column_map = {}

    sorted_columns = sorted(columns, key=len, reverse=True)

    for index, column in enumerate(sorted_columns):
        token = f"__col_{index}__"

        pattern = re.escape(column)

        safe_expression = re.sub(
            rf"(?<![A-Za-z0-9_]){pattern}(?![A-Za-z0-9_])",
            token,
            safe_expression,
        )

        column_map[token] = column

    return safe_expression, column_map


def _validate_ast_node(node: ast.AST) -> None:
    """
    Validate that parsed expression contains only safe AST nodes.
    """

    for child in ast.walk(node):
        if not isinstance(child, ALLOWED_AST_NODES):
            raise ValueError(
                f"Unsafe expression element not allowed: {type(child).__name__}"
            )

        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name):
                raise ValueError("Only simple function calls are allowed.")

            function_name = child.func.id

            if function_name not in ALLOWED_FUNCTIONS:
                raise ValueError(f"Function '{function_name}' is not allowed.")


def _extract_referenced_columns(
    tree: ast.AST,
    column_map: dict[str, str],
) -> list[str]:
    referenced_columns = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in column_map:
                referenced_columns.append(column_map[node.id])

    return referenced_columns


def _evaluate_ast_node(
    node: ast.AST,
    df: pd.DataFrame,
    column_map: dict[str, str],
) -> Any:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in column_map:
            raise ValueError(f"Unknown column or variable: {node.id}")

        column_name = column_map[node.id]

        return df[column_name]

    if isinstance(node, ast.UnaryOp):
        value = _evaluate_ast_node(node.operand, df, column_map)

        if isinstance(node.op, ast.USub):
            return -pd.to_numeric(value, errors="coerce")

        if isinstance(node.op, ast.UAdd):
            return pd.to_numeric(value, errors="coerce")

    if isinstance(node, ast.BinOp):
        left = _evaluate_ast_node(node.left, df, column_map)
        right = _evaluate_ast_node(node.right, df, column_map)

        left_number = pd.to_numeric(left, errors="coerce")
        right_number = pd.to_numeric(right, errors="coerce")

        if isinstance(node.op, ast.Add):
            return left_number + right_number

        if isinstance(node.op, ast.Sub):
            return left_number - right_number

        if isinstance(node.op, ast.Mult):
            return left_number * right_number

        if isinstance(node.op, ast.Div):
            return _safe_divide(left_number, right_number)

        if isinstance(node.op, ast.Mod):
            return left_number % right_number

        if isinstance(node.op, ast.Pow):
            return left_number ** right_number

    if isinstance(node, ast.Compare):
        left = _evaluate_ast_node(node.left, df, column_map)

        final_mask = pd.Series(True, index=df.index)

        current_left = left

        for operator, comparator_node in zip(node.ops, node.comparators):
            right = _evaluate_ast_node(comparator_node, df, column_map)

            mask = _apply_compare_operator(current_left, right, operator)

            final_mask = final_mask & mask

            current_left = right

        return final_mask

    if isinstance(node, ast.BoolOp):
        values = [
            _evaluate_ast_node(value_node, df, column_map)
            for value_node in node.values
        ]

        if isinstance(node.op, ast.And):
            result = pd.Series(True, index=df.index)
            for value in values:
                result = result & value
            return result

        if isinstance(node.op, ast.Or):
            result = pd.Series(False, index=df.index)
            for value in values:
                result = result | value
            return result

    if isinstance(node, ast.Call):
        function_name = node.func.id

        args = [
            _evaluate_ast_node(arg, df, column_map)
            for arg in node.args
        ]

        return _apply_safe_function(function_name, args, df)

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _apply_compare_operator(left: Any, right: Any, operator: ast.cmpop) -> pd.Series:
    left_number = pd.to_numeric(left, errors="coerce")
    right_number = pd.to_numeric(right, errors="coerce")

    if isinstance(operator, ast.Eq):
        return left == right

    if isinstance(operator, ast.NotEq):
        return left != right

    if isinstance(operator, ast.Gt):
        return left_number > right_number

    if isinstance(operator, ast.GtE):
        return left_number >= right_number

    if isinstance(operator, ast.Lt):
        return left_number < right_number

    if isinstance(operator, ast.LtE):
        return left_number <= right_number

    raise ValueError("Unsupported comparison operator.")


def _apply_safe_function(
    function_name: str,
    args: list[Any],
    df: pd.DataFrame,
) -> Any:
    if function_name == "add":
        return pd.to_numeric(args[0], errors="coerce") + pd.to_numeric(args[1], errors="coerce")

    if function_name == "subtract":
        return pd.to_numeric(args[0], errors="coerce") - pd.to_numeric(args[1], errors="coerce")

    if function_name == "multiply":
        return pd.to_numeric(args[0], errors="coerce") * pd.to_numeric(args[1], errors="coerce")

    if function_name in {"divide", "safe_divide"}:
        return _safe_divide(
            pd.to_numeric(args[0], errors="coerce"),
            pd.to_numeric(args[1], errors="coerce"),
        )

    if function_name == "round":
        value = pd.to_numeric(args[0], errors="coerce")
        decimals = 0

        if len(args) > 1:
            decimals = _safe_int(args[1], default=0)

        return value.round(decimals)

    if function_name == "abs":
        return pd.to_numeric(args[0], errors="coerce").abs()

    if function_name == "log":
        value = pd.to_numeric(args[0], errors="coerce")
        return np.log(value.where(value > 0, np.nan))

    if function_name == "sqrt":
        value = pd.to_numeric(args[0], errors="coerce")
        return np.sqrt(value.where(value >= 0, np.nan))

    if function_name == "year":
        dates = pd.to_datetime(args[0], errors="coerce")
        return dates.dt.year

    if function_name == "month":
        dates = pd.to_datetime(args[0], errors="coerce")
        return dates.dt.month

    if function_name == "day":
        dates = pd.to_datetime(args[0], errors="coerce")
        return dates.dt.day

    if function_name == "concat":
        result = pd.Series("", index=df.index)

        for arg in args:
            if isinstance(arg, pd.Series):
                result = result + arg.fillna("").astype(str)
            else:
                result = result + str(arg)

        return result

    if function_name == "contains":
        text = args[0].fillna("").astype(str)
        search_value = str(args[1])
        return text.str.contains(search_value, case=False, na=False)

    if function_name == "equals":
        return args[0] == args[1]

    if function_name == "greater_than":
        return pd.to_numeric(args[0], errors="coerce") > pd.to_numeric(args[1], errors="coerce")

    if function_name == "greater_than_or_equal":
        return pd.to_numeric(args[0], errors="coerce") >= pd.to_numeric(args[1], errors="coerce")

    if function_name == "less_than":
        return pd.to_numeric(args[0], errors="coerce") < pd.to_numeric(args[1], errors="coerce")

    if function_name == "less_than_or_equal":
        return pd.to_numeric(args[0], errors="coerce") <= pd.to_numeric(args[1], errors="coerce")

    if function_name == "and_condition":
        return pd.Series(args[0], index=df.index).astype(bool) & pd.Series(args[1], index=df.index).astype(bool)

    if function_name == "or_condition":
        return pd.Series(args[0], index=df.index).astype(bool) | pd.Series(args[1], index=df.index).astype(bool)

    if function_name == "if_else":
        condition = args[0]
        true_value = args[1]
        false_value = args[2]

        return np.where(condition, true_value, false_value)

    raise ValueError(f"Unsupported function: {function_name}")


def _safe_divide(left: Any, right: Any) -> Any:
    return np.where(right != 0, left / right, np.nan)


def _convert_output_type(value: Any, output_type: str) -> Any:
    output_type = (output_type or "number").lower().strip()

    if output_type in {"number", "numeric"}:
        converted = pd.to_numeric(value, errors="coerce")
        try:
            original = pd.Series(value)
            converted_series = pd.Series(converted)
            if original.notna().any() and converted_series.notna().sum() == 0:
                return value
        except (TypeError, ValueError):
            pass
        return converted

    if output_type in {"text", "category", "categorical", "string"}:
        if isinstance(value, pd.Series):
            return value.fillna("").astype(str)

        if isinstance(value, (np.ndarray, list, tuple)):
            return pd.Series(value).fillna("").astype(str).to_numpy()

        return str(value)

    if output_type == "boolean":
        if isinstance(value, pd.Series):
            return value.astype(bool)

        if isinstance(value, (np.ndarray, list, tuple)):
            return pd.Series(value).astype(bool).to_numpy()

        return bool(value)

    if output_type == "date":
        return pd.to_datetime(value, errors="coerce")

    return value


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except ValueError:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def _format_rows_for_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")

    safe_records = []

    for row in records:
        safe_row = {}

        for key, value in row.items():
            safe_row[key] = _json_safe_value(value)

        safe_records.append(safe_row)

    return safe_records


SAFE_FIELD_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_ ]{0,63}$")


def get_supported_calculated_field_functions() -> list[dict[str, str]]:
    """
    Return safe functions allowed in calculated fields.
    """

    return [
        {
            "name": "add",
            "description": "Add two numeric values or columns.",
            "input_type": "number",
            "example": "add(Sales, Profit)",
        },
        {
            "name": "subtract",
            "description": "Subtract second value from first value.",
            "input_type": "number",
            "example": "subtract(Sales, Cost)",
        },
        {
            "name": "multiply",
            "description": "Multiply two numeric values or columns.",
            "input_type": "number",
            "example": "multiply(Quantity, Price)",
        },
        {
            "name": "divide",
            "description": "Divide first value by second value.",
            "input_type": "number",
            "example": "divide(Profit, Sales)",
        },
        {
            "name": "safe_divide",
            "description": "Divide safely. Returns null when denominator is zero.",
            "input_type": "number",
            "example": "safe_divide(Profit, Sales)",
        },
        {
            "name": "round",
            "description": "Round numeric value.",
            "input_type": "number",
            "example": "round(Profit / Sales, 2)",
        },
        {
            "name": "abs",
            "description": "Absolute numeric value.",
            "input_type": "number",
            "example": "abs(Profit)",
        },
        {
            "name": "log",
            "description": "Natural log of numeric value.",
            "input_type": "number",
            "example": "log(Sales)",
        },
        {
            "name": "sqrt",
            "description": "Square root of numeric value.",
            "input_type": "number",
            "example": "sqrt(Sales)",
        },
        {
            "name": "year",
            "description": "Extract year from date column.",
            "input_type": "date",
            "example": "year(Order Date)",
        },
        {
            "name": "month",
            "description": "Extract month from date column.",
            "input_type": "date",
            "example": "month(Order Date)",
        },
        {
            "name": "day",
            "description": "Extract day from date column.",
            "input_type": "date",
            "example": "day(Order Date)",
        },
        {
            "name": "concat",
            "description": "Combine text values.",
            "input_type": "text",
            "example": "concat(First Name, Last Name)",
        },
        {
            "name": "contains",
            "description": "Check whether text contains another text.",
            "input_type": "text",
            "example": "contains(Product, 'phone')",
        },
        {
            "name": "equals",
            "description": "Check whether a column equals a value.",
            "input_type": "mixed",
            "example": "equals(Region, 'East')",
        },
        {
            "name": "greater_than",
            "description": "Check whether a numeric column is greater than a value.",
            "input_type": "number",
            "example": "greater_than(Sales, 1000)",
        },
        {
            "name": "greater_than_or_equal",
            "description": "Check whether a numeric column is greater than or equal to a value.",
            "input_type": "number",
            "example": "greater_than_or_equal(Sales, 1000)",
        },
        {
            "name": "less_than",
            "description": "Check whether a numeric column is less than a value.",
            "input_type": "number",
            "example": "less_than(Sales, 1000)",
        },
        {
            "name": "less_than_or_equal",
            "description": "Check whether a numeric column is less than or equal to a value.",
            "input_type": "number",
            "example": "less_than_or_equal(Sales, 1000)",
        },
        {
            "name": "and_condition",
            "description": "Combine two boolean conditions with AND.",
            "input_type": "boolean",
            "example": "and_condition(greater_than(Sales, 1000), equals(Region, 'East'))",
        },
        {
            "name": "or_condition",
            "description": "Combine two boolean conditions with OR.",
            "input_type": "boolean",
            "example": "or_condition(equals(Region, 'East'), equals(Region, 'West'))",
        },
        {
            "name": "if_else",
            "description": "Return one value if condition is true, otherwise another value.",
            "input_type": "mixed",
            "example": "if_else(Sales > 1000, 'High', 'Low')",
        },
    ]


def validate_calculated_field_name(
    df: pd.DataFrame,
    field_name: str,
) -> bool:
    """
    Validate new calculated field name.
    """

    if not field_name:
        raise ValueError("Calculated field name is required.")

    field_name = str(field_name).strip()

    if not field_name:
        raise ValueError("Calculated field name cannot be empty.")

    if field_name in df.columns:
        raise ValueError(f"Column '{field_name}' already exists.")

    if not SAFE_FIELD_NAME_PATTERN.match(field_name):
        raise ValueError(
            "Calculated field name can contain only letters, numbers, spaces, and underscore. "
            "It must start with a letter."
        )

    return True


def validate_calculated_field_expression(
    df: pd.DataFrame,
    expression: str,
    allowed_columns: list[str] | None = None,
) -> bool:
    """
    Validate calculated field expression before running.

    Important:
        This does not use raw eval().
        It parses expression using ast and only allows safe nodes/functions.
    """

    if not expression:
        raise ValueError("Calculated field expression is required.")

    expression = _normalize_user_friendly_expression(expression)

    if not expression:
        raise ValueError("Calculated field expression cannot be empty.")

    allowed_columns = allowed_columns or list(df.columns)

    safe_expression, column_map = _replace_column_names_with_safe_tokens(
        expression=expression,
        columns=allowed_columns,
    )

    try:
        tree = ast.parse(safe_expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid calculated field expression: {expression}") from exc

    _validate_ast_node(tree)

    referenced_columns = _extract_referenced_columns(tree, column_map)

    missing_columns = [
        column for column in referenced_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Calculated field expression references missing columns: {missing_columns}"
        )

    return True


def get_calculated_field_dependency_columns(
    calculated_fields: list[dict[str, Any]] | None,
    available_columns: list[str],
) -> list[str]:
    """
    Return dataframe columns referenced by calculated field expressions.

    This is used before optimized dataset loading so expressions such as
    "Sales * Quantity" load Sales and Quantity even when the chart only
    selects the calculated output field.
    """

    dependencies = []

    for field in calculated_fields or []:
        if not isinstance(field, dict):
            continue

        expression = field.get("expression") or field.get("formula")
        if not expression:
            continue

        expression = _normalize_user_friendly_expression(expression)
        safe_expression, column_map = _replace_column_names_with_safe_tokens(
            expression=str(expression),
            columns=available_columns,
        )

        try:
            tree = ast.parse(safe_expression, mode="eval")
            _validate_ast_node(tree)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid calculated field expression: {expression}") from exc

        for column in _extract_referenced_columns(tree, column_map):
            if column not in dependencies:
                dependencies.append(column)

    return dependencies


def preview_calculated_field(
    df: pd.DataFrame,
    field_name: str,
    expression: str,
    output_type: str = "number",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Preview calculated field result before saving.
    """

    validate_calculated_field_name(df, field_name)
    validate_calculated_field_expression(df, expression)

    working_df = apply_calculated_field(
        df=df,
        field_name=field_name,
        expression=expression,
        output_type=output_type,
    )

    limit = _safe_int(limit, default=20)

    preview_df = working_df.head(limit)

    return _format_rows_for_json(preview_df)


def apply_calculated_field(
    df: pd.DataFrame,
    field_name: str,
    expression: str,
    output_type: str = "number",
) -> pd.DataFrame:
    """
    Add calculated field column to dataframe.
    """

    expression = _normalize_user_friendly_expression(expression)

    validate_calculated_field_name(df, field_name)
    validate_calculated_field_expression(df, expression)

    working_df = df.copy()

    safe_expression, column_map = _replace_column_names_with_safe_tokens(
        expression=expression,
        columns=list(working_df.columns),
    )

    tree = ast.parse(safe_expression, mode="eval")

    result = _evaluate_ast_node(
        node=tree.body,
        df=working_df,
        column_map=column_map,
    )

    result = _convert_output_type(result, output_type)

    working_df[field_name] = result

    return working_df


def apply_calculated_fields(
    df: pd.DataFrame,
    calculated_fields: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """
    Apply multiple calculated fields before chart generation.
    """

    if df.empty:
        return df

    if not calculated_fields:
        return df

    working_df = df.copy()

    for field in calculated_fields:
        if not isinstance(field, dict):
            continue

        field_name = field.get("field_name") or field.get("name")
        expression = field.get("expression") or field.get("formula")
        output_type = field.get("output_type") or field.get("dataType") or field.get("data_type") or "number"

        if not field_name or not expression:
            continue

        working_df = apply_calculated_field(
            df=working_df,
            field_name=field_name,
            expression=expression,
            output_type=output_type,
        )

    return working_df
