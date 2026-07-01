# backend/apps/charts/services/chart_common_operations.py

"""
AutoBI Chart Common Operations

This file contains common reusable chart-data operation function definitions.

IMPORTANT:
- Do not implement frontend style here.
- Do not implement chart-specific logic here.
- Implement only common pandas/numpy operations manually.
- Chart-specific logic should go inside generators/*.py.

Frontend-only settings:
theme, base color, palette, label position, legend position, grid visibility,
font size, number format, bar radius, line width, point size.

Backend common operations:
filters, aggregation, sorting, Top N, Bottom N, row limit, missing value cleanup,
numeric conversion, date filtering, JSON-safe formatting.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

import numpy as np
# -------------------------------------------------------------------
# BASIC HELPERS
# -------------------------------------------------------------------

def safe_int(value: Any, default: int) -> int:
    """
    Purpose:
        Safely convert value to integer.

    Used by:
        top_n, bottom_n, row limit, group limit, bins.

    Manual implementation:
        - If value is None or empty, return default.
        - Try int(value).
        - If failed, return default.
    """
    try:
        if value is None:
            return default
        if value =="":
            return default
        return int(value)
    
    except (TypeError, ValueError):
        return default

def normalize_aggregation(aggregation: Any) -> str:
    """
    Purpose:
        Convert frontend aggregation names into backend/pandas names.

    Example:
        avg -> mean
        average -> mean
        count_distinct -> nunique
        unique_count -> nunique
        variance -> var

    Used by:
        bar, pie, line, map, KPI, heatmap, pivot table.

    Manual implementation:
        - Convert aggregation to lowercase string.
        - Match it to supported aggregation map.
        - Return default 'sum' if unknown.
    """
    aggregation_map = {
        "sum": "sum",

        "mean": "mean",
        "avg": "mean",
        "average": "mean",

        "median": "median",

        "count": "count",

        "nunique": "nunique",
        "unique_count": "nunique",
        "count_distinct": "nunique",
        "distinct_count": "nunique",

        "std": "std",
        "stddev": "std",
        "standard_deviation": "std",

        "var": "var",
        "variance": "var",

        "min": "min",
        "max": "max",

        "first": "first",
        "last": "last",
    }
    value = str(aggregation,"sum").strip().lower()
    return aggregation_map.get(value,"sum")

def normalize_sort_order(sort_order: Any, default: str = "descending") -> str:
    """
    Purpose:
        Normalize sort order.

    Accepted frontend values:
        asc
        ascending
        desc
        descending
        smallest
        largest
        low_to_high
        high_to_low

    Output:
        ascending or descending

    Manual implementation:
        - Convert value to lowercase.
        - Return 'ascending' for asc-like values.
        - Otherwise return 'descending'.
    """
    value = str(sort_order or default).strip().lower()

    ascending_values={
        "asc",
        "ascending",
        "smallest",
        "low_to_high",
        "low to high",
        "a-z",
        "oldest",
    }
    descending_values = {
        "desc",
        "descending",
        "largest",
        "high_to_low",
        "high to low",
        "z-a",
        "newest",
    }
    
    if value in ascending_values:
        return "ascending"
    
    if value in descending_values:
        return "descending"

    return default

def unique_keep_order(values: Iterable[Any]) -> list[str]:
    """
    Purpose:
        Remove duplicate column names while keeping order.

    Used by:
        selecting dataframe columns before aggregation.

    Manual implementation:
        - Loop through values.
        - Skip None or empty.
        - Convert to string.
        - Add only if not already added.
    """
    output = []

    for value in values:
        if value is None:
            continue

        if value == "":
             continue
        
        value = str(value)

        if value not in output:
            output.append(value)
    return output


def validate_column(
    df: pd.DataFrame,
    column_name: Any,
    label: str,
    required: bool = True,
) -> str:
    """
    Purpose:
        Validate that selected column exists in dataframe.
    """

    if not column_name:
        if required:
            raise ValueError(f"{label} is required.")
        return ""

    column_name = str(column_name)

    if column_name not in df.columns:
        raise ValueError(f"{label} '{column_name}' was not found in the dataset.")

    return column_name


def numeric_series(series: pd.Series) -> pd.Series:
    """
    Purpose:
        Convert pandas Series to numeric safely.

    Used by:
        numeric charts, aggregation, histogram, box, scatter, KPI.
    """

    return pd.to_numeric(series, errors="coerce")

# -------------------------------------------------------------------
# DATA CLEANING HELPERS FOR CHARTS
# -------------------------------------------------------------------

def convert_to_numeric_safe(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Purpose:
        Convert selected columns to numeric.

    Used by:
        scatter, bubble, histogram, box, heatmap, KPI, numeric aggregation.

    Manual implementation:
        - Copy dataframe.
        - Loop selected columns.
        - If column exists, convert using pd.to_numeric.
        - Return copied dataframe.
    """
    output = df.copy()
    for column in columns:
          output[columns] = pd.to_numeric(df[column],errors = "coerce")


def clean_chart_missing_values(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> pd.DataFrame:
    """
    Purpose:
        Drop rows where required chart columns are missing.

    Used by:
        all chart generators before aggregation/rendering.

    Manual implementation:
        - Keep only columns that exist in df.
        - If no valid columns, return df.
        - Use df.dropna(subset=columns).
        - Return cleaned df.
    """
    columns = []

    for column in required_columns or []:
        if column in df.columns:
            columns.append(column)

    if not columns:
        return df

    return df.dropna(subset=columns)


# -------------------------------------------------------------------
# FILTER OPERATIONS
# -------------------------------------------------------------------

def apply_date_range_filter(
    df: pd.DataFrame,
    column: str,
    start_date: Any = None,
    end_date: Any = None,
) -> pd.DataFrame:
    """
    Purpose:
        Filter dataframe using a date column.

    Used by:
        line chart, area chart, dashboard global filters, table, map.

    Manual implementation:
        - Check column exists.
        - Convert column to datetime using pd.to_datetime.
        - Create boolean mask.
        - If start_date exists, keep rows >= start_date.
        - If end_date exists, keep rows <= end_date.
        - Return filtered df.
    """
    if column not in df.columns:
        return df
    
    date_series = pd.to_datetime(df[column],errors="coerce")

    mask = date_series.notna()

    if start_date is not None:
        start_date = pd.to_datetime(start_date,errors = "coerce")
        if pd.notna(start_date):
             mask = mask & (date_series >= start_date)

    if end_date is not None:
        end_date = pd.to_datetime(end_date, errors="coerce")
        if pd.notna(end_date):
            mask = mask & (date_series <= end_date)

    return df[mask]


def apply_chart_filters(
    df: pd.DataFrame,
    filters: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """
    Purpose:
        Apply chart-level, dashboard-level, and drill-down filters.

    Used by:
        all charts.

    Supported operators:
        equals
        not_equals
        contains
        not_contains
        starts_with
        ends_with
        greater_than
        less_than
        greater_equal
        less_equal
        between
        in
        not_in
        is_null
        not_null
        date_range

    Example filter:
        {
            "column": "Region",
            "operator": "equals",
            "value": "South"
        }

    Manual implementation:
        - If filters empty, return df.
        - Loop through filter rules.
        - Read column, operator, value, value2.
        - Skip invalid column.
        - For text operators, use astype(str).
        - For numeric operators, use numeric_series().
        - For date_range, call apply_date_range_filter().
        - Return filtered dataframe.
    """
    if not filters:
        return df
    
    filtered_df = df.copy()

    for filter_rule in filters:
        column = filter_rule.get("column")
        operator = filter_rule.get("operator")
        value = filter_rule.get("value")
        value2 = filter_rule.get("value2")

        if not column or column not in filtered_df.column:
            continue
        if not operator:
            continue
        series = filtered_df[column]

        if operator == "equals":
            mask = series==value
            filtered_df = filtered_df[mask]

        elif operator == "not equals":
             mask = series != value
             filtered_df =  filtered_df[mask]

        elif operator == "contain":
            text_series= series.astype(str)
            mask = text_series.str.contains(str(value),case = False,na=False)
            filtered_df = filtered_df[mask]
        elif operator == "not_contain":
            text_series = series.astype(str)
            mask = ~text_series.str.contain(str(value),case = False,na=False)
            filtered_df = filtered_df[mask]

        elif operator == "starts_with":
            text_series = series.astype(str)
            mask = text_series.str.startswith(str(value),na= False)
            filtered_df = filtered_df[mask]
        elif operator == "ends_with":
            text_series = series.astype(str)
            mask = text_series.str.endswith(str(value),na= False)
            filtered_df = filtered_df[mask]  
        
        elif operator == "greater_than":
            numbers = numeric_series(series)
            compare_value = pd.to_numeric(value, errors="coerce")

            if pd.notna(compare_value):
                mask = numbers > compare_value
                filtered_df = filtered_df[mask]

        elif operator == "less_than":
            numbers = numeric_series(series)
            compare_value = pd.to_numeric(value, errors="coerce")

            if pd.notna(compare_value):
                mask = numbers < compare_value
                filtered_df = filtered_df[mask]

        elif operator == "greater_equal":
            numbers = numeric_series(series)
            compare_value = pd.to_numeric(value, errors="coerce")

            if pd.notna(compare_value):
                mask = numbers >= compare_value
                filtered_df = filtered_df[mask]

        elif operator == "less_equal":
            numbers = numeric_series(series)
            compare_value = pd.to_numeric(value, errors="coerce")

            if pd.notna(compare_value):
                mask = numbers <= compare_value
                filtered_df = filtered_df[mask]
        
        elif operator == "between":
            numbers = numeric_series(series)
            start_value = pd.to_numeric(value, errors="coerce")
            end_value = pd.to_numeric(value2, errors="coerce")

            if pd.notna(start_value) and pd.notna(end_value):
                mask = (numbers >= start_value) & (numbers <= end_value)
                filtered_df = filtered_df[mask]
        

        elif operator == "in":
            if isinstance(value, list):
                allowed_values = value
            else:
                allowed_values = [value]

            mask = series.isin(allowed_values)
            filtered_df = filtered_df[mask]

        elif operator == "not_in":
            if isinstance(value, list):
                blocked_values = value
            else:
                blocked_values = [value]

            mask = ~series.isin(blocked_values)
            filtered_df = filtered_df[mask]

        elif operator == "is_null":
            mask = series.isna()
            filtered_df = filtered_df[mask]

        elif operator == "not_null":
            mask = series.notna()
            filtered_df = filtered_df[mask]

        elif operator == "date_range":
            filtered_df = apply_date_range_filter(
                filtered_df,
                column=column,
                start_date=value,
                end_date=value2,
            )

    return filtered_df
        
# -------------------------------------------------------------------
# AGGREGATION OPERATIONS
# -------------------------------------------------------------------

def aggregate_chart_data(
    df: pd.DataFrame,
    group_columns: str | list[str],
    value_column: str,
    aggregation: str = "sum",
    output_value_column: str | None = None,
) -> pd.DataFrame:
    """
    Purpose:
        Group dataframe and calculate selected aggregation.

    Used by:
        bar, horizontal bar, grouped bar, stacked bar,
        line, area, pie, donut, map, heatmap, pivot table, KPI.

    Supported aggregations:
        sum
        mean
        median
        count
        nunique
        std
        var
        min
        max
        first
        last

    Manual implementation:
        - Normalize aggregation.
        - Convert group_columns to list.
        - Validate group columns.
        - Select only needed columns.
        - For count, use groupby().size().
        - For nunique, use groupby()[value].nunique().
        - For numeric aggregations, convert value_column to numeric.
        - Drop missing numeric values.
        - Apply groupby aggregation.
        - Reset index.
        - Rename output value column.
        - Return dataframe.
    """
    aggregation = (aggregation or "sum").lower().strip()

    if isinstance(group_columns,str):
        group_columns=[group_columns]

    valid_group_columns=[]

    for column in group_columns:
        if column in df.columns:
            valid_group_columns.append(column)

    if not valid_group_columns:
        return pd.DataFrame()
    
    if value_column not in df.columns and aggregation not in ["count"]:
        return pd.DataFrame()

    output_value_column = output_value_column or value_column

    if aggregation == "count":
        result = (
            df.groupby(valid_group_columns, dropna=False)
            .size()
            .reset_index(name=output_value_column)
        )
        return result

    needed_columns = valid_group_columns + [value_column]
    working_df = df[needed_columns].copy()

    if aggregation == "nunique":
        result = (
            working_df.groupby(valid_group_columns, dropna=False)[value_column]
            .nunique()
            .reset_index(name=output_value_column)
        )
        return result

    allowed_aggregations = {
        "sum",
        "mean",
        "median",
        "std",
        "var",
        "min",
        "max",
        "first",
        "last",
    }

    if aggregation not in allowed_aggregations:
        aggregation = "sum"

    numeric_aggregations = {
        "sum",
        "mean",
        "median",
        "std",
        "var",
        "min",
        "max",
    }

    if aggregation in numeric_aggregations:
        working_df[value_column] = pd.to_numeric(
            working_df[value_column],
            errors="coerce",
        )

        working_df = working_df.dropna(subset=[value_column])

    result = (
        working_df.groupby(valid_group_columns, dropna=False)[value_column]
        .agg(aggregation)
        .reset_index()
    )

    result = result.rename(columns={value_column: output_value_column})

    return result
# -------------------------------------------------------------------
# SORT / TOP N / BOTTOM N
# -------------------------------------------------------------------

def apply_sort(
    df: pd.DataFrame,
    sort_column: str | None,
    sort_order: str = "descending",
) -> pd.DataFrame:
    """
    Purpose:
        Sort chart output rows.

    Used by:
        almost all chart generators.

    Manual implementation:
        - If df empty or sort column missing, return df.
        - Normalize sort order.
        - Try numeric sorting first.
        - If numeric conversion fails, sort as text.
        - Return sorted dataframe.
    """
    if df.empty():
        return df
    
    if not sort_column or sort_column not in df.columns:
        return df
    
    sort_order = (sort_order or "descending").lower().strip()
    ascending = sort_order in ["ascending", "asc"]

    working_df = df.copy()

    numeric_values = pd.to_numeric(
        working_df[sort_column],
        errors="coerce",
    )

    if numeric_values.notna().any():
        sort_values = numeric_values
    else:
        sort_values = working_df[sort_column].astype(str)

    working_df["_sort_value"] = sort_values

    working_df = working_df.sort_values(
        by="_sort_value",
        ascending=ascending,
        na_position="last",
    )

    working_df = working_df.drop(columns=["_sort_value"])

    return working_df.reset_index(drop=True)







def safe_int(value: Any, default: int | None = None) -> int | None:
    """
    Safely convert value to int.
    If conversion fails, return default.
    """
    try:
        if value is None:
            return default

        value = int(value)

        if value <= 0:
            return default

        return value

    except (TypeError, ValueError):
        return default

    
   


def apply_top_n(
    df: pd.DataFrame,
    value_column: str,
    top_n: Any = None,
    sort_order: str = "descending",
) -> pd.DataFrame:
    """
    Purpose:
        Keep Top N rows after aggregation.

    Used by:
        bar, horizontal bar, pie, donut, map, table.

    Important:
        Apply Top N after aggregation, not before.

    Manual implementation:
        - If top_n missing, return df.
        - Convert top_n using safe_int().
        - Convert value_column to numeric.
        - If sort_order ascending, use nsmallest().
        - Else use nlargest().
        - Return top rows.
    """
    if df.empty():
        return df
    
    if not top_n:
        return df
    
    if value_column not in df.column:
        return df
    
    top_n_value = safe_int(top_n)

    if top_n_value is None:
        return df
    
    working_df = df.copy()

    working_df[value_column] = pd.to_numeric(working_df[value_column],errors="coerce")
    working_df = working_df.dropna(subset=[value_column])

    if sort_order in ["ascending","asc"]:
        result = working_df.nsmallest(top_n_value,columns=value_column)
    else:
        result = working_df.nlargest(top_n_value,columns=value_column)
    
    return result.reset_index(drop=True)




def apply_bottom_n(
    df: pd.DataFrame,
    value_column: str,
    bottom_n: Any = None,
) -> pd.DataFrame:
    """
    Purpose:
        Keep Bottom N rows by selected value column.

    Used by:
        bar, table, map, analysis suggestions.

    Manual implementation:
        - If bottom_n missing, return df.
        - Convert bottom_n using safe_int().
        - Convert value_column to numeric.
        - Use nsmallest().
        - Return bottom rows.
    """
    if df.empty:
        return df

    if bottom_n is None:
        return df

    if value_column not in df.columns:
        return df

    bottom_n_value = safe_int(bottom_n)

    if bottom_n_value is None:
        return df

    working_df = df.copy()

    working_df[value_column] = pd.to_numeric(
        working_df[value_column],
        errors="coerce",
    )

    working_df = working_df.dropna(subset=[value_column])

    result = working_df.nsmallest(
        bottom_n_value,
        columns=value_column,
    )

    return result.reset_index(drop=True)


def apply_groupby_limit(
    df: pd.DataFrame,
    group_column: str,
    limit_per_group: int = 10,
    value_column: str | None = None,
) -> pd.DataFrame:
    """
    Purpose:
        Keep top rows inside each group.

    Example:
        Top 5 products inside each region.

    Used by:
        grouped bar, stacked bar, table.

    Manual implementation:
        - Validate group_column.
        - Convert limit_per_group to int.
        - If value_column exists, sort each group by value descending.
        - Use groupby(group_column).head(limit_per_group).
        - Return limited dataframe.
    """
    if df.empty:
        return df

    if not group_column or group_column not in df.columns:
        return df
    
    limit_per_group = safe_int(limit_per_group, default=10)
    working_df = df.copy()

    if value_column and value_column in working_df.columns:
        working_df[value_column] = pd.to_numeric( working_df[value_column],
            errors="coerce",
        )
    working_df = working_df.sort_values(by=[group_column,value_column],ascending=[True,False],na_position="last")
    result = (working_df.groupby(group_column,dropna=True).head(limit_per_group).reset_index(drop=True))
    return result




def limit_chart_rows(
    df: pd.DataFrame,
    limit: Any = 1000,
    max_limit: int = 5000,
) -> tuple[pd.DataFrame, int]:
    """
    Purpose:
        Limit rows sent to frontend for performance.

    Used by:
        scatter, bubble, table, maps, raw preview.

    Manual implementation:
        - Convert limit using safe_int().
        - Clamp between 1 and max_limit.
        - Return df.head(limit), limit.
    """
    if df.empty():
        return df ,0

    max_limit = safe_int(max_limit,default = 5000)

    limit_value = safe_int(limit,default =1000)
    if limit_value > max_limit:
        limit_value = max_limit
    if limit_value < 1:
        limit_value =1
    limited_df = df.head(limit_value)

    return limited_df, limit_value


# -------------------------------------------------------------------
# PIE / HISTOGRAM HELPERS
# -------------------------------------------------------------------

def group_pie_other(
    result_df: pd.DataFrame,
    label_column: str,
    value_column: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Purpose:
        Keep top N pie slices and group remaining rows as 'Other'.

    Used by:
        pie, donut.

    Manual implementation:
        - If rows <= top_n, return result.
        - Use nlargest(top_n, value_column).
        - Sum remaining values.
        - Add new row with label 'Other'.
        - Return combined dataframe.
    """
    if result_df.empty():
        return result_df
    
    if label_column not in result_df.columns:
        return result_df
    if value_column not in result_df.columns:
        return result_df

    try:
        top_n = int(top_n)
    except (TypeError,ValueError):
        top_n =10

    if top_n <= 0:
        top_n =10
    if len(result_df) <= top_n:
        return result_df
    working_df = result_df.copy()
    working_df[value_column]=pd.to_numeric(working_df[value_column],errors="coerce")
    working_df = working_df.dropna(subset=[value_column])
    if len(working_df)<=top_n:
        return working_df.reset_index(drop=True)
    top_df = working_df.nlargest(top_n,column=value_column)
    remaining_df = working_df.drop(index = top_df.index)
    other_total = remaining_df[value_column].sum()
    other_row = {
         label_column: "Other",
        value_column: other_total,
    }

    final_df = pd.concat(
        [
            top_df,
            pd.DataFrame([other_row]),
        ],
        ignore_index=True,
    )

    return final_df.reset_index(drop=True)
    
    
import math

def format_bin_value(value: Any) -> Any:
    """
    Purpose:
        Format histogram bin start/end values.

    Used by:
        histogram.

    Manual implementation:
        - Try converting value to float.
        - If large number, format without decimals.
        - If small number, keep 1 or 2 decimals.
        - If not numeric, return as JSON-safe value.
    """
    try:
        number = float(value)
    except (TypeError,ValueError):
        if pd.isna(value):
            return None
        return str(value)

    if math.isnan(number) or math.isinf(number):
        return None
    absolute_number = abs(number)

    if absolute_number >= 100:
        return round(number)

    if absolute_number >= 10:
        return round(number, 1)

    return round(number, 2)
# -------------------------------------------------------------------
# FRONTEND OUTPUT HELPERS
# -------------------------------------------------------------------
def to_json_safe_value(value: Any) -> Any:
    """
    Convert one pandas/numpy/python value into JSON-safe value.
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()

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

    return value







def format_rows_for_frontend(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Purpose:
        Convert pandas dataframe into JSON-safe row dictionaries.

    Used by:
        all chart generators.

    Must handle:
        NaN
        None
        Timestamp
        numpy int/float/bool
        Infinity

    Manual implementation:
        - Replace np.nan with None.
        - Convert df to records.
        - Convert each value into JSON-safe format.
        - Return list of dictionaries.
    """
    if df.empty:
        return []

    records = df.to_dict(orient="records")

    safe_records = []

    for row in records:
        safe_row = {}

        for key, value in row.items():
            safe_row[key] = to_json_safe_value(value)

        safe_records.append(safe_row)

    return safe_records


def build_title(
    chart_type: str,
    x_column: str | None,
    y_column: str | None,
    aggregation: str | None,
) -> str:
    """
    Purpose:
        Generate fallback chart title.

    Used by:
        chart generators when user did not enter custom title.

    Manual implementation:
        - KPI: '{Aggregation} of {Y}'
        - Table: 'Data table'
        - Histogram: 'Distribution of {X}'
        - Box: 'Box plot of {Y}'
        - X + Y chart: '{Aggregation} of {Y} by {X}'
        - Else: 'Chart'
    """
    chart_type = (chart_type or "").lower().strip()
    aggregation = (aggregation or "sum").lower().strip()

    # Make aggregation title-friendly
    aggregation_label = aggregation.replace("_", " ").title()

    if chart_type == "kpi":
        if y_column:
            return f"{aggregation_label} of {y_column}"
        return "KPI"

    if chart_type == "table":
        return "Data table"

    if chart_type == "histogram":
        if x_column:
            return f"Distribution of {x_column}"
        return "Distribution"

    if chart_type in ["box", "boxplot", "box_plot"]:
        if y_column:
            return f"Box plot of {y_column}"
        return "Box plot"

    if x_column and y_column:
        return f"{aggregation_label} of {y_column} by {x_column}"

    return "Chart"

from typing import Any


def normalize_filter_rule(rule: dict[str, Any]) -> dict[str, Any]:
    """
    Purpose:
        Convert frontend filter rule into standard backend format.

    Input example:
        {
            "field": "Region",
            "operator": "equals",
            "value": "South",
            "scope": "visual"
        }

    Output example:
        {
            "column": "Region",
            "operator": "equals",
            "value": "South",
            "value2": None,
            "scope": "visual"
        }
    """

    if not isinstance(rule, dict):
        return {
            "column": None,
            "operator": None,
            "value": None,
            "value2": None,
            "scope": "visual",
        }

    column = (
        rule.get("column")
        or rule.get("field")
        or rule.get("name")
    )

    operator = (
        rule.get("operator")
        or rule.get("op")
        or rule.get("condition")
    )

    value = rule.get("value")
    value2 = rule.get("value2") or rule.get("end_value")

    scope = rule.get("scope") or "visual"

    operator = normalize_filter_operator(operator)

    return {
        "column": column,
        "operator": operator,
        "value": value,
        "value2": value2,
        "scope": scope,
    }


def normalize_filter_operator(operator: Any) -> str | None:
    """
    Convert different frontend operator names into one backend format.
    """

    if operator is None:
        return None

    operator = str(operator).lower().strip()

    operator = operator.replace(" ", "_")
    operator = operator.replace("-", "_")

    operator_map = {
        "=": "equals",
        "==": "equals",
        "eq": "equals",
        "equals": "equals",

        "!=": "not_equals",
        "<>": "not_equals",
        "ne": "not_equals",
        "not_equals": "not_equals",

        "contains": "contains",
        "not_contains": "not_contains",

        "starts_with": "starts_with",
        "startswith": "starts_with",

        "ends_with": "ends_with",
        "endswith": "ends_with",

        ">": "greater_than",
        "gt": "greater_than",
        "greater_than": "greater_than",

        "<": "less_than",
        "lt": "less_than",
        "less_than": "less_than",

        ">=": "greater_equal",
        "gte": "greater_equal",
        "greater_equal": "greater_equal",

        "<=": "less_equal",
        "lte": "less_equal",
        "less_equal": "less_equal",

        "between": "between",

        "in": "in",
        "not_in": "not_in",

        "is_null": "is_null",
        "null": "is_null",

        "not_null": "not_null",
        "is_not_null": "not_null",

        "date_range": "date_range",
        "daterange": "date_range",
    }

    return operator_map.get(operator, operator)




def validate_filter_rule(df: pd.DataFrame, rule: dict[str, Any]) -> bool:
    """
    Purpose:
        Validate one filter rule before applying it.

    Manual implementation:
        - Check column exists.
        - Check operator is supported.
        - For numeric operators, check value can be numeric.
        - For between, check value and value2 exist.
        - Return True or raise ValueError.
    """

    if not isinstance(rule, dict):
        raise ValueError("Filter rule must be a dictionary.")

    column = rule.get("column")
    operator = rule.get("operator")
    value = rule.get("value")
    value2 = rule.get("value2")

    if not column:
        raise ValueError("Filter rule missing column.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in dataframe.")

    if not operator:
        raise ValueError("Filter rule missing operator.")

    supported_operators = {
        "equals",
        "not_equals",
        "contains",
        "not_contains",
        "starts_with",
        "ends_with",
        "greater_than",
        "less_than",
        "greater_equal",
        "less_equal",
        "between",
        "in",
        "not_in",
        "is_null",
        "not_null",
        "date_range",
    }

    if operator not in supported_operators:
        raise ValueError(f"Unsupported filter operator: {operator}")

    numeric_operators = {
        "greater_than",
        "less_than",
        "greater_equal",
        "less_equal",
    }

    if operator in numeric_operators:
        numeric_value = pd.to_numeric(value, errors="coerce")

        if pd.isna(numeric_value):
            raise ValueError(
                f"Operator '{operator}' requires a numeric value."
            )

    if operator == "between":
        if value is None or value2 is None:
            raise ValueError("Between filter requires value and value2.")

        start_value = pd.to_numeric(value, errors="coerce")
        end_value = pd.to_numeric(value2, errors="coerce")

        if pd.isna(start_value) or pd.isna(end_value):
            raise ValueError("Between filter values must be numeric.")

    if operator in {"in", "not_in"}:
        if value is None:
            raise ValueError(f"Operator '{operator}' requires a value.")

        if not isinstance(value, list):
            raise ValueError(f"Operator '{operator}' requires a list value.")

    if operator == "date_range":
        if value is None and value2 is None:
            raise ValueError("Date range filter requires start date or end date.")

        if value is not None:
            start_date = pd.to_datetime(value, errors="coerce")
            if pd.isna(start_date):
                raise ValueError("Invalid start date for date range filter.")

        if value2 is not None:
            end_date = pd.to_datetime(value2, errors="coerce")
            if pd.isna(end_date):
                raise ValueError("Invalid end date for date range filter.")

    return True






def apply_single_filter_rule(
    df: pd.DataFrame,
    rule: dict[str, Any],
) -> pd.DataFrame:
    """
    Purpose:
        Apply one filter rule to dataframe.

    Manual implementation:
        - Normalize rule.
        - Validate rule.
        - Apply pandas boolean mask.
        - Return filtered dataframe.
    """

    if df.empty:
        return df

    rule = normalize_filter_rule(rule)

    validate_filter_rule(df, rule)

    column = rule.get("column")
    operator = rule.get("operator")
    value = rule.get("value")
    value2 = rule.get("value2")

    series = df[column]

    if operator == "equals":
        mask = series == value
        return df[mask]

    if operator == "not_equals":
        mask = series != value
        return df[mask]

    if operator == "contains":
        text_series = series.astype(str)
        mask = text_series.str.contains(
            str(value),
            case=False,
            na=False,
        )
        return df[mask]

    if operator == "not_contains":
        text_series = series.astype(str)
        mask = ~text_series.str.contains(
            str(value),
            case=False,
            na=False,
        )
        return df[mask]

    if operator == "starts_with":
        text_series = series.astype(str)
        mask = text_series.str.startswith(
            str(value),
            na=False,
        )
        return df[mask]

    if operator == "ends_with":
        text_series = series.astype(str)
        mask = text_series.str.endswith(
            str(value),
            na=False,
        )
        return df[mask]

    if operator == "greater_than":
        numbers = pd.to_numeric(series, errors="coerce")
        compare_value = pd.to_numeric(value, errors="coerce")

        mask = numbers > compare_value
        return df[mask]

    if operator == "less_than":
        numbers = pd.to_numeric(series, errors="coerce")
        compare_value = pd.to_numeric(value, errors="coerce")

        mask = numbers < compare_value
        return df[mask]

    if operator == "greater_equal":
        numbers = pd.to_numeric(series, errors="coerce")
        compare_value = pd.to_numeric(value, errors="coerce")

        mask = numbers >= compare_value
        return df[mask]

    if operator == "less_equal":
        numbers = pd.to_numeric(series, errors="coerce")
        compare_value = pd.to_numeric(value, errors="coerce")

        mask = numbers <= compare_value
        return df[mask]

    if operator == "between":
        numbers = pd.to_numeric(series, errors="coerce")
        start_value = pd.to_numeric(value, errors="coerce")
        end_value = pd.to_numeric(value2, errors="coerce")

        mask = (numbers >= start_value) & (numbers <= end_value)
        return df[mask]

    if operator == "in":
        mask = series.isin(value)
        return df[mask]

    if operator == "not_in":
        mask = ~series.isin(value)
        return df[mask]

    if operator == "is_null":
        mask = series.isna()
        return df[mask]

    if operator == "not_null":
        mask = series.notna()
        return df[mask]

    if operator == "date_range":
        return apply_date_range_filter(
            df=df,
            column=column,
            start_date=value,
            end_date=value2,
        )

    return df



def apply_chart_filters(
    df: pd.DataFrame,
    filters: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    """
    Purpose:
        Apply multiple chart/dashboard filter rules.

    Manual implementation:
        - If no filters, return df.
        - Loop filters.
        - Call apply_single_filter_rule().
        - Return final filtered dataframe.
    """

    if df.empty:
        return df

    if not filters:
        return df

    filtered_df = df.copy()

    for rule in filters:
        filtered_df = apply_single_filter_rule(filtered_df, rule)

    return filtered_df.reset_index(drop=True)
