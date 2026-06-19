"""
Placeholder service catalog for AutoBI Manual Feature Builder.

This module intentionally does not implement feature engineering logic.
It only exposes stable response shapes and a comprehensive rule catalog
that can be wired and implemented manually later.
"""

import re
from io import BytesIO
from pathlib import Path
from datetime import datetime

import pandas as pd

from django.core.files.base import ContentFile
from django.db import transaction

from apps.datasets.models import DatasetVersion
from apps.datasets.services import build_preview, make_json_safe, read_dataset_file



PLACEHOLDER_STATUS = "placeholder"
PLACEHOLDER_MESSAGE = (
    "Manual feature engineering logic is not implemented yet. "
    "This returns supported rule options only."
)


def _rule(
    rule_id,
    label,
    category,
    operation,
    description,
    output_type,
    requires_user_input=False,
    user_inputs=None,
    parameters=None,
):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.
    return {
        "rule_id": rule_id,
        "label": label,
        "category": category,
        "operation": operation,
        "description": description,
        "requires_user_input": requires_user_input,
        "user_inputs": user_inputs or [],
        "rule_json": {
            "operation": operation,
            "output_type": output_type,
            "parameters": parameters or {},
        },
    }


RULE_GROUPS = {
    "datetime_extraction": [
        _rule("datetime_extract_date_yyyy_mm_dd", "Extract date YYYY-MM-DD", "datetime_extraction", "extract_date_yyyy_mm_dd", "Extract a date matching YYYY-MM-DD format.", "date"),
        _rule("datetime_extract_date_dd_mm_yyyy", "Extract date DD-MM-YYYY", "datetime_extraction", "extract_date_dd_mm_yyyy", "Extract a date matching DD-MM-YYYY format.", "date"),
        _rule("datetime_extract_date_mm_dd_yyyy", "Extract date MM/DD/YYYY", "datetime_extraction", "extract_date_mm_dd_yyyy", "Extract a date matching MM/DD/YYYY format.", "date"),
        _rule("datetime_extract_year_only", "Extract year only", "datetime_extraction", "extract_year", "Extract the year component from a date-like value.", "numeric"),
        _rule("datetime_extract_month_only", "Extract month only", "datetime_extraction", "extract_month", "Extract the month component from a date-like value.", "numeric"),
        _rule("datetime_extract_day_only", "Extract day only", "datetime_extraction", "extract_day", "Extract the day component from a date-like value.", "numeric"),
        _rule("datetime_extract_time", "Extract time HH:MM or HH:MM:SS", "datetime_extraction", "extract_time", "Extract a time component in HH:MM or HH:MM:SS format.", "text"),
        _rule("datetime_extract_quarter", "Extract quarter from date", "datetime_extraction", "extract_quarter", "Derive the quarter from a date-like value.", "categorical"),
        _rule("datetime_extract_weekday", "Extract weekday from date", "datetime_extraction", "extract_weekday", "Derive the weekday from a date-like value.", "categorical"),
        _rule("datetime_weekend_flag", "Extract weekend flag from date", "datetime_extraction", "extract_weekend_flag", "Create a boolean flag for weekend dates.", "boolean"),
        _rule(
            "datetime_age_duration_reference",
            "Extract age/duration from reference date placeholder",
            "datetime_extraction",
            "extract_age_duration_from_reference_date",
            "Create an age or duration feature using a user-provided reference date.",
            "numeric",
            True,
            [{"name": "reference_date", "type": "date", "required": True}],
        ),
    ],
    "string_split_position": [
        _rule("string_extract_first_character", "Extract first character", "string_split_position", "extract_first_character", "Extract the first character from a text value.", "text"),
        _rule("string_extract_last_character", "Extract last character", "string_split_position", "extract_last_character", "Extract the last character from a text value.", "text"),
        _rule("string_extract_first_n_characters", "Extract first N characters", "string_split_position", "extract_first_n_characters", "Extract the first N characters from a text value.", "text", True, [{"name": "n", "type": "integer", "required": True}], {"n": 1}),
        _rule("string_extract_last_n_characters", "Extract last N characters", "string_split_position", "extract_last_n_characters", "Extract the last N characters from a text value.", "text", True, [{"name": "n", "type": "integer", "required": True}], {"n": 1}),
        _rule("string_remove_first_character", "Remove first character", "string_split_position", "remove_first_character", "Remove the first character from a text value.", "text"),
        _rule("string_remove_last_character", "Remove last character", "string_split_position", "remove_last_character", "Remove the last character from a text value.", "text"),
        _rule("string_remove_first_n_characters", "Remove first N characters", "string_split_position", "remove_first_n_characters", "Remove the first N characters from a text value.", "text", True, [{"name": "n", "type": "integer", "required": True}], {"n": 1}),
        _rule("string_remove_last_n_characters", "Remove last N characters", "string_split_position", "remove_last_n_characters", "Remove the last N characters from a text value.", "text", True, [{"name": "n", "type": "integer", "required": True}], {"n": 1}),
        _rule("string_split_slash_first", "Split by slash and take first part", "string_split_position", "split_slash_first", "Split text on slash and keep the first token.", "text"),
        _rule("string_split_slash_last", "Split by slash and take last part", "string_split_position", "split_slash_last", "Split text on slash and keep the last token.", "text"),
        _rule("string_split_comma_first", "Split by comma and take first part", "string_split_position", "split_comma_first", "Split text on comma and keep the first token.", "text"),
        _rule("string_split_comma_last", "Split by comma and take last part", "string_split_position", "split_comma_last", "Split text on comma and keep the last token.", "text"),
        _rule("string_split_underscore_first", "Split by underscore and take first part", "string_split_position", "split_underscore_first", "Split text on underscore and keep the first token.", "text"),
        _rule("string_split_underscore_last", "Split by underscore and take last part", "string_split_position", "split_underscore_last", "Split text on underscore and keep the last token.", "text"),
        _rule("string_split_hyphen_first", "Split by hyphen and take first part", "string_split_position", "split_hyphen_first", "Split text on hyphen and keep the first token.", "text"),
        _rule("string_split_hyphen_last", "Split by hyphen and take last part", "string_split_position", "split_hyphen_last", "Split text on hyphen and keep the last token.", "text"),
        _rule(
            "string_split_slash_selected",
            "Split by slash and take selected part",
            "string_split_position",
            "split_slash_selected_part",
            "Split text on slash and keep the selected token.",
            "text",
            True,
            [{"name": "part_index", "type": "integer", "required": True}],
        ),
        _rule("string_split_space_first", "Split by space and take first word", "string_split_position", "split_space_first_word", "Split text on spaces and keep the first word.", "text"),
        _rule("string_split_space_last", "Split by space and take last word", "string_split_position", "split_space_last_word", "Split text on spaces and keep the last word.", "text"),
        _rule(
            "string_prefix_before_delimiter",
            "Extract prefix before delimiter",
            "string_split_position",
            "extract_prefix_before_delimiter",
            "Extract the text before a user-provided delimiter.",
            "text",
            True,
            [{"name": "delimiter", "type": "text", "required": True}],
        ),
        _rule(
            "string_suffix_after_delimiter",
            "Extract suffix after delimiter",
            "string_split_position",
            "extract_suffix_after_delimiter",
            "Extract the text after a user-provided delimiter.",
            "text",
            True,
            [{"name": "delimiter", "type": "text", "required": True}],
        ),
        _rule(
            "string_middle_token_by_index",
            "Extract middle token by index placeholder",
            "string_split_position",
            "extract_middle_token_by_index",
            "Extract a token from a split string by user-provided index.",
            "text",
            True,
            [
                {"name": "delimiter", "type": "text", "required": True},
                {"name": "token_index", "type": "integer", "required": True},
            ],
        ),
    ],
    "regex_extraction": [
        _rule("regex_extract_first_number", "Extract first number", "regex_extraction", "extract_first_number", "Extract the first number-like token from text.", "numeric"),
        _rule("regex_extract_all_digits", "Extract all digits", "regex_extraction", "extract_all_digits", "Extract all digit characters from text.", "text"),
        _rule("regex_extract_alphabetic_text", "Extract alphabetic text", "regex_extraction", "extract_alphabetic_text", "Extract alphabetic characters from text.", "text"),
        _rule("regex_extract_alphanumeric_code", "Extract alphanumeric code", "regex_extraction", "extract_alphanumeric_code", "Extract an alphanumeric code-like token.", "text"),
        _rule("regex_extract_email_domain", "Extract email domain", "regex_extraction", "extract_email_domain", "Extract the domain portion from an email-like value.", "categorical"),
        _rule("regex_extract_url_domain", "Extract URL domain", "regex_extraction", "extract_url_domain", "Extract the domain portion from a URL-like value.", "categorical"),
        _rule("regex_extract_postal_zip", "Extract postal/zip-like code", "regex_extraction", "extract_postal_zip_code", "Extract a postal or ZIP-like code.", "text"),
        _rule("regex_extract_phone_like", "Extract phone-like number placeholder", "regex_extraction", "extract_phone_like_number", "Extract a phone-like number pattern.", "text"),
        _rule(
            "regex_custom_extraction",
            "Custom regex extraction",
            "regex_extraction",
            "custom_regex_extraction",
            "Extract text using a user-provided regular expression.",
            "text",
            True,
            [{"name": "regex_pattern", "type": "text", "required": True}],
        ),
    ],
    "numeric_derivation": [
        _rule("numeric_text_number_to_numeric", "Convert text number to numeric placeholder", "numeric_derivation", "convert_text_number_to_numeric", "Convert number-like text into a numeric feature.", "numeric"),
        _rule("numeric_extract_amount_from_text", "Extract numeric amount from text", "numeric_derivation", "extract_numeric_amount_from_text", "Extract an amount-like number from text.", "numeric"),
        _rule("numeric_extract_percentage_from_text", "Extract percentage from text", "numeric_derivation", "extract_percentage_from_text", "Extract a percentage-like value from text.", "numeric"),
        _rule("numeric_absolute_value", "Create absolute value", "numeric_derivation", "create_absolute_value", "Create an absolute-value numeric feature.", "numeric"),
        _rule(
            "numeric_rounded_value",
            "Create rounded value",
            "numeric_derivation",
            "create_rounded_value",
            "Create a rounded numeric feature.",
            "numeric",
            True,
            [{"name": "decimal_places", "type": "integer", "required": False}],
        ),
        _rule(
            "numeric_binned_group",
            "Create binned numeric group",
            "numeric_derivation",
            "create_binned_numeric_group",
            "Create a categorical bin from numeric ranges.",
            "categorical",
            True,
            [{"name": "bins", "type": "list", "required": True}],
        ),
        _rule("numeric_high_medium_low_bucket", "Create high/medium/low bucket placeholder", "numeric_derivation", "create_high_medium_low_bucket", "Create a high, medium, or low bucket from numeric values.", "categorical", True, [{"name": "thresholds", "type": "object", "required": True}]),
        _rule("numeric_ratio_two_columns", "Create ratio from two columns placeholder", "numeric_derivation", "create_ratio_from_two_columns", "Create a ratio feature using two source columns.", "numeric", True, [{"name": "denominator_column", "type": "column", "required": True}]),
        _rule("numeric_difference_two_columns", "Create difference between two columns placeholder", "numeric_derivation", "create_difference_between_two_columns", "Create a difference feature using two source columns.", "numeric", True, [{"name": "comparison_column", "type": "column", "required": True}]),
        _rule("numeric_sum_multiple_columns", "Create sum from multiple columns placeholder", "numeric_derivation", "create_sum_from_multiple_columns", "Create a sum feature using multiple source columns.", "numeric", True, [{"name": "additional_columns", "type": "columns", "required": True}]),
    ],
    "text_features": [
        _rule("text_lowercase", "Lowercase text", "text_features", "lowercase_text", "Convert text to lowercase.", "text"),
        _rule("text_uppercase", "Uppercase text", "text_features", "uppercase_text", "Convert text to uppercase.", "text"),
        _rule("text_titlecase", "Title case text", "text_features", "titlecase_text", "Convert text to title case.", "text"),
        _rule("text_trim_whitespace", "Trim whitespace", "text_features", "trim_whitespace", "Trim leading and trailing whitespace.", "text"),
        _rule("text_remove_all_spaces", "Remove all spaces", "text_features", "remove_all_spaces", "Remove all whitespace characters.", "text"),
        _rule("text_remove_special_characters", "Remove special characters", "text_features", "remove_special_characters", "Remove non-alphanumeric characters from text.", "text"),
        _rule("text_length", "Text length", "text_features", "text_length", "Count the total length of a text value.", "numeric"),
        _rule("text_word_count", "Word count", "text_features", "word_count", "Count words in a text value.", "numeric"),
        _rule("text_character_count", "Character count", "text_features", "character_count", "Count characters in a text value.", "numeric"),
        _rule("text_digit_count", "Digit count", "text_features", "digit_count", "Count digit characters in text.", "numeric"),
        _rule("text_alphabet_count", "Alphabet count", "text_features", "alphabet_count", "Count alphabetic characters in text.", "numeric"),
        _rule("text_special_character_count", "Special character count", "text_features", "special_character_count", "Count special characters in text.", "numeric"),
        _rule("text_uppercase_count", "Uppercase count", "text_features", "uppercase_count", "Count uppercase letters in text.", "numeric"),
        _rule("text_lowercase_count", "Lowercase count", "text_features", "lowercase_count", "Count lowercase letters in text.", "numeric"),
        _rule("text_contains_keyword_flag", "Contains keyword flag", "text_features", "contains_keyword_flag", "Create a flag for text containing a user-provided keyword.", "boolean", True, [{"name": "keyword", "type": "text", "required": True}]),
        _rule("text_starts_with_keyword_flag", "Starts with keyword flag", "text_features", "starts_with_keyword_flag", "Create a flag for text starting with a user-provided keyword.", "boolean", True, [{"name": "keyword", "type": "text", "required": True}]),
        _rule("text_ends_with_keyword_flag", "Ends with keyword flag", "text_features", "ends_with_keyword_flag", "Create a flag for text ending with a user-provided keyword.", "boolean", True, [{"name": "keyword", "type": "text", "required": True}]),
        _rule("text_is_empty_flag", "Is empty text flag", "text_features", "is_empty_text_flag", "Create a flag for empty text values.", "boolean"),
    ],
    "categorical_features": [
        _rule("categorical_rare_grouping", "Rare category grouping placeholder", "categorical_features", "rare_category_grouping", "Group rare categories into a placeholder category.", "categorical", True, [{"name": "minimum_frequency", "type": "number", "required": True}]),
        _rule("categorical_frequency_encoding", "Frequency encoding placeholder", "categorical_features", "frequency_encoding", "Create a frequency-based encoded feature.", "numeric"),
        _rule("categorical_user_defined_mapping", "Group by user-defined mapping placeholder", "categorical_features", "group_by_user_defined_mapping", "Group categories using a user-defined mapping.", "categorical", True, [{"name": "mapping", "type": "object", "required": True}]),
        _rule("categorical_extract_prefix", "Extract category prefix", "categorical_features", "extract_category_prefix", "Extract a prefix from category text.", "categorical", True, [{"name": "delimiter", "type": "text", "required": False}]),
        _rule("categorical_extract_suffix", "Extract category suffix", "categorical_features", "extract_category_suffix", "Extract a suffix from category text.", "categorical", True, [{"name": "delimiter", "type": "text", "required": False}]),
        _rule("categorical_normalize_lowercase", "Normalize category text lowercase placeholder", "categorical_features", "normalize_category_lowercase", "Normalize category text to lowercase.", "categorical"),
        _rule("categorical_trim_whitespace", "Trim whitespace placeholder", "categorical_features", "trim_whitespace", "Trim leading and trailing whitespace from category text.", "categorical"),
    ],
    "boolean_flags": [
        _rule("boolean_is_missing", "Is missing flag", "boolean_flags", "is_missing_flag", "Create a flag for missing values.", "boolean"),
        _rule("boolean_is_not_missing", "Is not missing flag", "boolean_flags", "is_not_missing_flag", "Create a flag for present values.", "boolean"),
        _rule("boolean_contains_value", "Contains value flag", "boolean_flags", "contains_value_flag", "Create a flag for values containing user-provided text.", "boolean", True, [{"name": "value", "type": "text", "required": True}]),
        _rule("boolean_equals_value", "Equals value flag", "boolean_flags", "equals_value_flag", "Create a flag for values equal to user-provided text.", "boolean", True, [{"name": "value", "type": "text", "required": True}]),
        _rule("boolean_greater_than_threshold", "Greater than threshold flag", "boolean_flags", "greater_than_threshold_flag", "Create a flag for values greater than a threshold.", "boolean", True, [{"name": "threshold", "type": "number", "required": True}]),
        _rule("boolean_less_than_threshold", "Less than threshold flag", "boolean_flags", "less_than_threshold_flag", "Create a flag for values less than a threshold.", "boolean", True, [{"name": "threshold", "type": "number", "required": True}]),
        _rule("boolean_between_range", "Between range flag", "boolean_flags", "between_range_flag", "Create a flag for values within a user-provided range.", "boolean", True, [{"name": "min_value", "type": "number", "required": True}, {"name": "max_value", "type": "number", "required": True}]),
        _rule("boolean_is_valid_date", "Is valid date flag", "boolean_flags", "is_valid_date_flag", "Create a flag for date-like values.", "boolean"),
        _rule("boolean_is_valid_number", "Is valid number flag", "boolean_flags", "is_valid_number_flag", "Create a flag for number-like values.", "boolean"),
    ],
    "id_code_decomposition": [
        _rule("id_extract_prefix", "Extract ID prefix", "id_code_decomposition", "extract_id_prefix", "Extract a prefix from an ID or code.", "text"),
        _rule("id_extract_numeric_part", "Extract ID numeric part", "id_code_decomposition", "extract_id_numeric_part", "Extract the numeric portion from an ID or code.", "numeric"),
        _rule("id_extract_date_part", "Extract ID date part", "id_code_decomposition", "extract_id_date_part", "Extract a date-like portion from an ID or code.", "date"),
        _rule("id_extract_region_code", "Extract region code placeholder", "id_code_decomposition", "extract_region_code", "Extract a region code from an ID or code.", "categorical", True, [{"name": "region_pattern", "type": "text", "required": False}]),
        _rule("id_extract_branch_code", "Extract branch code placeholder", "id_code_decomposition", "extract_branch_code", "Extract a branch code from an ID or code.", "categorical", True, [{"name": "branch_pattern", "type": "text", "required": False}]),
        _rule("id_extract_product_code", "Extract product code placeholder", "id_code_decomposition", "extract_product_code", "Extract a product code from an ID or code.", "categorical", True, [{"name": "product_pattern", "type": "text", "required": False}]),
        _rule("id_keep_original_tracking", "Keep original column for tracking only", "id_code_decomposition", "keep_original_column_for_tracking", "Keep the original source column for traceability.", "text"),
    ],
}


def _copy_rule_groups():
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.
    return {
        category: [dict(rule, rule_json=dict(rule["rule_json"])) for rule in rules]
        for category, rules in RULE_GROUPS.items()
    }


def _flatten_rule_groups():
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.
    rules = []
    for category_rules in RULE_GROUPS.values():
        rules.extend(category_rules)
    return rules


def _normalize_compare_value(value):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.
    if value is None:
        return ""
    return str(value).strip().lower()


def _is_blank(value):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.
    return value is None or str(value).strip() == ""


def _safe_positive_int(value, default=1):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return parsed if parsed > 0 else default


def _try_parse_date(text):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.
    if _is_blank(text):
        return None

    text = str(text).strip()
    date_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]

    for date_format in date_formats:
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue

    return None


def _apply_manual_rule_to_value(value, rule_json):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.

    operation = (rule_json or {}).get("operation")
    parameters = (rule_json or {}).get("parameters") or {}

    if value is None:
        if operation == "is_missing_flag":
            return 1
        if operation == "is_not_missing_flag":
            return 0
        return None

    text = str(value)

    if operation == "manual_mapping_from_examples":
        mapping = parameters.get("mapping") or {}
        return mapping.get(text)

    if operation == "extract_first_character":
        return text[0] if text else None

    if operation == "extract_last_character":
        return text[-1] if text else None

    if operation == "extract_first_n_characters":
        n = _safe_positive_int(parameters.get("n"), 1)
        return text[:n] if text else None

    if operation == "extract_last_n_characters":
        n = _safe_positive_int(parameters.get("n"), 1)
        return text[-n:] if text else None

    if operation == "remove_first_character":
        return text[1:] if text else None

    if operation == "remove_last_character":
        return text[:-1] if text else None

    if operation == "remove_first_n_characters":
        n = _safe_positive_int(parameters.get("n"), 1)
        return text[n:] if text else None

    if operation == "remove_last_n_characters":
        n = _safe_positive_int(parameters.get("n"), 1)
        return text[:-n] if text and n <= len(text) else ""

    if operation == "extract_date_yyyy_mm_dd":
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        return match.group(0) if match else None

    if operation == "extract_date_dd_mm_yyyy":
        match = re.search(r"\d{2}-\d{2}-\d{4}", text)
        return match.group(0) if match else None

    if operation == "extract_date_mm_dd_yyyy":
        match = re.search(r"\d{2}/\d{2}/\d{4}", text)
        return match.group(0) if match else None

    if operation == "extract_year":
        match = re.search(r"\b(19|20)\d{2}\b", text)
        return match.group(0) if match else None

    if operation == "extract_month":
        date_text = _apply_manual_rule_to_value(value, {"operation": "extract_date_yyyy_mm_dd"})
        parsed_date = _try_parse_date(date_text)
        return parsed_date.month if parsed_date else None

    if operation == "extract_day":
        date_text = _apply_manual_rule_to_value(value, {"operation": "extract_date_yyyy_mm_dd"})
        parsed_date = _try_parse_date(date_text)
        return parsed_date.day if parsed_date else None

    if operation == "extract_time":
        match = re.search(r"\b\d{2}:\d{2}(:\d{2})?\b", text)
        return match.group(0) if match else None

    if operation == "extract_quarter":
        date_text = _apply_manual_rule_to_value(value, {"operation": "extract_date_yyyy_mm_dd"})
        parsed_date = _try_parse_date(date_text)
        if not parsed_date:
            return None
        return f"Q{((parsed_date.month - 1) // 3) + 1}"

    if operation == "extract_weekday":
        date_text = _apply_manual_rule_to_value(value, {"operation": "extract_date_yyyy_mm_dd"})
        parsed_date = _try_parse_date(date_text)
        return parsed_date.strftime("%A") if parsed_date else None

    if operation == "extract_weekend_flag":
        date_text = _apply_manual_rule_to_value(value, {"operation": "extract_date_yyyy_mm_dd"})
        parsed_date = _try_parse_date(date_text)
        if not parsed_date:
            return None
        return 1 if parsed_date.weekday() >= 5 else 0

    if operation == "split_underscore_first":
        parts = text.split("_")
        return parts[0] if parts else None

    if operation == "split_underscore_last":
        parts = text.split("_")
        return parts[-1] if parts else None

    if operation == "split_hyphen_first":
        parts = text.split("-")
        return parts[0] if parts else None

    if operation == "split_hyphen_last":
        parts = text.split("-")
        return parts[-1] if parts else None

    if operation == "split_space_first_word":
        parts = text.split()
        return parts[0] if parts else None

    if operation == "split_space_last_word":
        parts = text.split()
        return parts[-1] if parts else None

    if operation == "split_slash_first":
        parts = text.split("/")
        return parts[0] if parts else None

    if operation == "split_slash_last":
        parts = text.split("/")
        return parts[-1] if parts else None

    if operation == "split_comma_first":
        parts = text.split(",")
        return parts[0].strip() if parts else None

    if operation == "split_comma_last":
        parts = text.split(",")
        return parts[-1].strip() if parts else None

    if operation == "extract_first_number":
        match = re.search(r"-?\d+(\.\d+)?", text)
        return match.group(0) if match else None

    if operation == "extract_all_digits":
        digits = re.findall(r"\d", text)
        return "".join(digits) if digits else None

    if operation == "extract_alphabetic_text":
        letters = re.findall(r"[A-Za-z]+", text)
        return " ".join(letters) if letters else None

    if operation == "extract_alphanumeric_code":
        match = re.search(r"[A-Za-z0-9]+", text)
        return match.group(0) if match else None

    if operation == "extract_email_domain":
        match = re.search(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
        return match.group(1) if match else None

    if operation == "extract_url_domain":
        match = re.search(r"https?://([^/\s]+)", text)
        return match.group(1) if match else None

    if operation == "lowercase_text" or operation == "normalize_category_lowercase":
        return text.lower()

    if operation == "uppercase_text":
        return text.upper()

    if operation == "titlecase_text":
        return text.title()

    if operation == "trim_whitespace":
        return text.strip()

    if operation == "remove_all_spaces":
        return re.sub(r"\s+", "", text)

    if operation == "remove_special_characters":
        return re.sub(r"[^A-Za-z0-9\s]", "", text)

    if operation == "text_length":
        return len(text)

    if operation == "word_count":
        return len(text.split())

    if operation == "character_count":
        return len(text)

    if operation == "digit_count":
        return sum(char.isdigit() for char in text)

    if operation == "alphabet_count":
        return sum(char.isalpha() for char in text)

    if operation == "special_character_count":
        return sum(not char.isalnum() and not char.isspace() for char in text)

    if operation == "uppercase_count":
        return sum(char.isupper() for char in text)

    if operation == "lowercase_count":
        return sum(char.islower() for char in text)

    if operation == "contains_keyword_flag":
        keyword = parameters.get("keyword", "")
        if not keyword:
            return None
        return 1 if keyword.lower() in text.lower() else 0

    if operation == "starts_with_keyword_flag":
        keyword = parameters.get("keyword", "")
        if not keyword:
            return None
        return 1 if text.lower().startswith(keyword.lower()) else 0

    if operation == "ends_with_keyword_flag":
        keyword = parameters.get("keyword", "")
        if not keyword:
            return None
        return 1 if text.lower().endswith(keyword.lower()) else 0

    if operation == "is_empty_text_flag":
        return 1 if text.strip() == "" else 0

    if operation == "is_missing_flag":
        return 1 if text.strip() == "" else 0

    if operation == "is_not_missing_flag":
        return 0 if text.strip() == "" else 1

    if operation == "is_valid_date_flag":
        date_text = _apply_manual_rule_to_value(value, {"operation": "extract_date_yyyy_mm_dd"})
        return 1 if _try_parse_date(date_text) else 0

    if operation == "is_valid_number_flag":
        try:
            float(text)
            return 1
        except ValueError:
            return 0

    return None


def _score_rule_against_examples(source_values, expected_outputs, rule):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.

    generated_outputs = []
    matched_count = 0
    total_compared = 0

    source_values_list = list(source_values or [])
    expected_outputs_list = list(expected_outputs or [])

    for index, source_value in enumerate(source_values_list):
        expected_output = expected_outputs_list[index] if index < len(expected_outputs_list) else None
        generated_output = _apply_manual_rule_to_value(source_value, rule.get("rule_json"))
        generated_outputs.append(generated_output)

        if _is_blank(expected_output):
            continue

        total_compared += 1
        if _normalize_compare_value(generated_output) == _normalize_compare_value(expected_output):
            matched_count += 1

    match_score = matched_count / total_compared if total_compared else 0

    return {
        "match_score": match_score,
        "matched_count": matched_count,
        "total_compared": total_compared,
        "generated_outputs": generated_outputs,
    }


def _typed_example_pairs(source_values, expected_outputs):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.
    pairs = []
    source_values_list = list(source_values or [])
    expected_outputs_list = list(expected_outputs or [])

    for index, source_value in enumerate(source_values_list):
        expected_output = expected_outputs_list[index] if index < len(expected_outputs_list) else None
        if _is_blank(expected_output) or source_value is None:
            continue
        pairs.append((str(source_value), str(expected_output).strip()))

    return pairs


def _make_dynamic_rule(rule_id, label, category, operation, output_type, parameters, description):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.
    return {
        "rule_id": rule_id,
        "label": label,
        "category": category,
        "operation": operation,
        "description": description,
        "requires_user_input": False,
        "user_inputs": [],
        "rule_json": {
            "operation": operation,
            "output_type": output_type,
            "parameters": parameters or {},
        },
    }


def _infer_dynamic_rules_from_examples(source_values, expected_outputs, output_type=None):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.
    pairs = _typed_example_pairs(source_values, expected_outputs)
    dynamic_rules = []

    if not pairs:
        return dynamic_rules

    mapping = {}
    for source_value, expected_output in pairs:
        mapping[source_value] = expected_output

    dynamic_rules.append(
        _make_dynamic_rule(
            "manual_mapping_from_examples",
            "Manual mapping from examples",
            "categorical_features",
            "manual_mapping_from_examples",
            output_type or "text",
            {"mapping": mapping},
            "Map exact source values to user-provided example outputs.",
        )
    )

    first_n_values = []
    last_n_values = []
    remove_first_n_values = []
    remove_last_n_values = []

    for source_value, expected_output in pairs:
        if source_value.startswith(expected_output) and expected_output:
            first_n_values.append(len(expected_output))
        if source_value.endswith(expected_output) and expected_output:
            last_n_values.append(len(expected_output))
        if source_value.endswith(expected_output) and len(source_value) > len(expected_output):
            remove_first_n_values.append(len(source_value) - len(expected_output))
        if source_value.startswith(expected_output) and len(source_value) > len(expected_output):
            remove_last_n_values.append(len(source_value) - len(expected_output))

    if first_n_values and len(set(first_n_values)) == 1:
        n = first_n_values[0]
        dynamic_rules.append(
            _make_dynamic_rule(
                f"dynamic_extract_first_{n}_characters",
                f"Extract first {n} character{'s' if n != 1 else ''}",
                "string_split_position",
                "extract_first_n_characters",
                output_type or "text",
                {"n": n},
                "Extract a fixed number of characters from the left side.",
            )
        )

    if last_n_values and len(set(last_n_values)) == 1:
        n = last_n_values[0]
        dynamic_rules.append(
            _make_dynamic_rule(
                f"dynamic_extract_last_{n}_characters",
                f"Extract last {n} character{'s' if n != 1 else ''}",
                "string_split_position",
                "extract_last_n_characters",
                output_type or "text",
                {"n": n},
                "Extract a fixed number of characters from the right side.",
            )
        )

    if remove_first_n_values and len(set(remove_first_n_values)) == 1:
        n = remove_first_n_values[0]
        dynamic_rules.append(
            _make_dynamic_rule(
                f"dynamic_remove_first_{n}_characters",
                f"Remove first {n} character{'s' if n != 1 else ''}",
                "string_split_position",
                "remove_first_n_characters",
                output_type or "text",
                {"n": n},
                "Remove a fixed number of characters from the left side.",
            )
        )

    if remove_last_n_values and len(set(remove_last_n_values)) == 1:
        n = remove_last_n_values[0]
        dynamic_rules.append(
            _make_dynamic_rule(
                f"dynamic_remove_last_{n}_characters",
                f"Remove last {n} character{'s' if n != 1 else ''}",
                "string_split_position",
                "remove_last_n_characters",
                output_type or "text",
                {"n": n},
                "Remove a fixed number of characters from the right side.",
            )
        )

    return dynamic_rules


def infer_manual_feature_rules(
    source_values,
    expected_outputs=None,
    output_type=None,
    source_column=None,
    new_feature_name=None,
):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.

    inferred_rules = []

    if expected_outputs is not None:
        candidate_rules = []
        candidate_rules.extend(_flatten_rule_groups())
        candidate_rules.extend(_infer_dynamic_rules_from_examples(source_values, expected_outputs, output_type))

        seen_rule_keys = set()
        for rule in candidate_rules:
            rule_key = (
                rule.get("operation"),
                str((rule.get("rule_json") or {}).get("parameters") or {}),
            )
            if rule_key in seen_rule_keys:
                continue
            seen_rule_keys.add(rule_key)

            score = _score_rule_against_examples(source_values, expected_outputs, rule)
            if score["total_compared"] <= 0:
                continue

            inferred_rule = dict(rule)
            inferred_rule["rule_json"] = dict(rule["rule_json"])
            inferred_rule.update(score)
            inferred_rules.append(inferred_rule)

    inferred_rules.sort(
        key=lambda item: (
            item.get("match_score", 0),
            item.get("matched_count", 0),
            -len(str((item.get("rule_json") or {}).get("parameters") or {})),
        ),
        reverse=True,
    )

    return {
        "status": "rules_inferred" if inferred_rules else PLACEHOLDER_STATUS,
        "message": "Rules were scored against user examples." if inferred_rules else PLACEHOLDER_MESSAGE,
        "source_column": source_column,
        "new_feature_name": new_feature_name,
        "output_type": output_type,
        "source_values": list(source_values or []),
        "expected_outputs": list(expected_outputs or []) if expected_outputs is not None else None,
        "inferred_rules": inferred_rules,
        "rule_groups": _copy_rule_groups(),
    }


def preview_manual_feature_rule(source_values, rule_json):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.

    preview_rows = []
    for value in list(source_values or []):
        preview_rows.append({"source_value": value, "preview_output": _apply_manual_rule_to_value(value, rule_json)})

    return {
        "status": "preview_ready",
        "message": "Manual preview generated for sample values only. Dataset was not modified.",
        "rule_json": rule_json,
        "preview_rows": preview_rows,
    }


def validate_manual_feature_rule(source_values, rule_json):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.

    preview = preview_manual_feature_rule(source_values, rule_json)
    preview_rows = preview.get("preview_rows", [])
    total_rows = len(preview_rows)
    empty_output_count = sum(1 for row in preview_rows if row.get("preview_output") is None or str(row.get("preview_output")).strip() == "")
    generated_count = total_rows - empty_output_count
    warnings = []

    if total_rows == 0:
        warnings.append("No source values were provided for validation.")
    if total_rows > 0 and empty_output_count > 0:
        warnings.append(f"{empty_output_count} out of {total_rows} preview rows returned empty output.")
    if total_rows > 0 and generated_count == 0:
        warnings.append("Selected rule did not generate output for any preview row.")

    return {
        "status": "validation_ready",
        "message": "Manual validation generated for sample values only. Dataset was not modified.",
        "rule_json": rule_json,
        "total_rows": total_rows,
        "generated_count": generated_count,
        "empty_output_count": empty_output_count,
        "warnings": warnings,
        "preview_rows": preview_rows,
    }


def build_manual_feature_mapping(source_column, new_feature_name, rule_json):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.

    operation = (rule_json or {}).get("operation")
    output_type = (rule_json or {}).get("output_type")
    parameters = (rule_json or {}).get("parameters") or {}

    return {
        "status": "mapping_ready",
        "message": "Manual feature mapping generated. Dataset was not modified.",
        "source_column": source_column,
        "new_feature_name": new_feature_name,
        "rule_json": rule_json,
        "mapping": {
            "original_column": source_column,
            "transformed_feature": new_feature_name,
            "operation": operation,
            "output_type": output_type,
            "parameters": parameters,
            "created_by": "manual_feature_builder",
        },
    }


def _latest_active_version(dataset, version_type):
    return (
        DatasetVersion.objects.filter(
            dataset=dataset,
            version_type=version_type,
            is_active=True,
            file__isnull=False,
        )
        .exclude(file="")
        .order_by("-version_number")
        .first()
    )


def get_latest_feature_engineered_dataset_version(dataset):
    return _latest_active_version(dataset, DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED)


def get_feature_engineering_source_version(dataset):
    feature_version = get_latest_feature_engineered_dataset_version(dataset)
    if feature_version is not None:
        return feature_version

    cleaned_version = _latest_active_version(dataset, DatasetVersion.VERSION_TYPE_CLEANED)
    if cleaned_version is not None:
        return cleaned_version

    return None


def load_feature_engineering_dataframe(dataset):
    source_version = get_feature_engineering_source_version(dataset)

    if source_version is not None:
        with source_version.file.open("rb") as source_file:
            return read_dataset_file(source_file), source_version

    if not dataset.file:
        raise ValueError("Dataset does not have an associated file.")

    with dataset.file.open("rb") as source_file:
        return read_dataset_file(source_file), None


def _numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def _apply_manual_rule_to_row(row, source_column, rule_json):
    operation = (rule_json or {}).get("operation")
    parameters = (rule_json or {}).get("parameters") or {}
    source_value = row.get(source_column)

    if operation == "create_absolute_value":
        numeric_value = pd.to_numeric(source_value, errors="coerce")
        return abs(numeric_value) if pd.notna(numeric_value) else None

    if operation == "create_rounded_value":
        numeric_value = pd.to_numeric(source_value, errors="coerce")
        decimal_places = int(parameters.get("decimal_places") or 0)
        return round(float(numeric_value), decimal_places) if pd.notna(numeric_value) else None

    if operation == "create_difference_between_two_columns":
        comparison_column = parameters.get("comparison_column")
        if not comparison_column:
            return None
        left = pd.to_numeric(source_value, errors="coerce")
        right = pd.to_numeric(row.get(comparison_column), errors="coerce")
        return float(left - right) if pd.notna(left) and pd.notna(right) else None

    if operation == "create_ratio_from_two_columns":
        denominator_column = parameters.get("denominator_column")
        if not denominator_column:
            return None
        numerator = pd.to_numeric(source_value, errors="coerce")
        denominator = pd.to_numeric(row.get(denominator_column), errors="coerce")
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return None
        return float(numerator / denominator)

    if operation == "create_sum_from_multiple_columns":
        source_columns = [source_column] + list(parameters.get("additional_columns") or [])
        values = [pd.to_numeric(row.get(column), errors="coerce") for column in source_columns]
        valid_values = [value for value in values if pd.notna(value)]
        return float(sum(valid_values)) if valid_values else None

    return _apply_manual_rule_to_value(source_value, rule_json)


def apply_manual_feature_rule_to_dataframe(df, source_column, new_feature_name, rule_json):
    if not source_column:
        raise ValueError("A source column is required.")
    if source_column not in df.columns:
        raise ValueError(f"Source column '{source_column}' does not exist in the active dataset.")
    if not new_feature_name:
        raise ValueError("A new feature name is required.")

    output_df = df.copy()
    output_df[new_feature_name] = output_df.apply(
        lambda row: _apply_manual_rule_to_row(row, source_column, rule_json),
        axis=1,
    )
    return output_df


def _get_feature_engineering_plan(source_version):
    if source_version is None:
        return {"feature_engineering_rules": []}

    plan = source_version.transformation_plan_json or {}
    rules = plan.get("feature_engineering_rules") or []
    if not isinstance(rules, list):
        rules = []
    return {"feature_engineering_rules": list(rules)}


def _next_version_number(dataset):
    last_version = DatasetVersion.objects.filter(dataset=dataset).order_by("-version_number").first()
    return (last_version.version_number + 1) if last_version else 1


def _safe_dataset_name(dataset):
    return "".join(
        char if char.isalnum() or char in {"_", "-", "."} else "_"
        for char in (dataset.name or "dataset")
    )


def _dataframe_content_file(dataset, df, version_number, source_version=None):
    source_name = (
        source_version.file.name
        if source_version is not None and source_version.file
        else dataset.file.name if dataset.file else dataset.name or "dataset"
    )
    extension = Path(source_name).suffix.lower() or ".csv"
    buffer = BytesIO()

    if extension in {".xlsx", ".xls"}:
        df.to_excel(buffer, index=False)
        output_extension = ".xlsx"
    else:
        df.to_csv(buffer, index=False)
        output_extension = ".csv"

    buffer.seek(0)
    return ContentFile(
        buffer.read(),
        name=f"{_safe_dataset_name(dataset)}_feature_engineered_v{version_number}{output_extension}",
    )


def save_manual_feature_rule(dataset, payload):
    source_column = payload.get("source_column")
    new_feature_name = payload.get("new_feature_name")
    rule_json = payload.get("rule_json") or {}

    with transaction.atomic():
        source_df, source_version = load_feature_engineering_dataframe(dataset)
        output_df = apply_manual_feature_rule_to_dataframe(
            source_df,
            source_column=source_column,
            new_feature_name=new_feature_name,
            rule_json=rule_json,
        )

        plan = _get_feature_engineering_plan(source_version)
        feature_mapping = build_manual_feature_mapping(
            source_column=source_column,
            new_feature_name=new_feature_name,
            rule_json=rule_json,
        )
        next_rule = {
            "source_column": source_column,
            "new_feature_name": new_feature_name,
            "output_type": payload.get("output_type") or (rule_json or {}).get("output_type"),
            "rule_json": rule_json,
            "mapping": feature_mapping.get("mapping", {}),
        }
        plan["feature_engineering_rules"].append(next_rule)

        DatasetVersion.objects.filter(
            dataset=dataset,
            version_type=DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED,
            is_active=True,
        ).update(is_active=False)

        version_number = _next_version_number(dataset)
        preview = build_preview(output_df, limit=20)
        feature_file = _dataframe_content_file(dataset, output_df, version_number, source_version)

        version = DatasetVersion.objects.create(
            dataset=dataset,
            version_number=version_number,
            file=feature_file,
            is_cleaned=False,
            version_type=DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED,
            parent_version=source_version,
            preview_rows=make_json_safe(preview.get("rows", [])),
            columns=make_json_safe(preview.get("columns", [])),
            transformation_plan_json=make_json_safe(plan),
            is_active=True,
            transformation_log=make_json_safe({
                "action": "manual_feature_engineering",
                "active_version_type": DatasetVersion.VERSION_TYPE_FEATURE_ENGINEERED,
                "source_version_id": source_version.id if source_version else None,
                "source_version_type": source_version.version_type if source_version else "original",
                "feature_engineering_plan_json": plan,
                "latest_rule": next_rule,
                "shape": list(output_df.shape),
            }),
        )

    return {
        "status": "saved",
        "message": "Manual feature rule applied to the active feature-engineered working dataset.",
        "saved": True,
        "feature_mapping": feature_mapping,
        "active_version_type": version.version_type,
        "active_version_id": version.id,
        "active_preview_rows": version.preview_rows,
        "active_columns": version.columns,
        "feature_engineering_plan_json": version.transformation_plan_json,
    }


def save_manual_feature_rule_placeholder(payload):
    return {
        "status": PLACEHOLDER_STATUS,
        "message": "Manual feature rule persistence requires a dataset. Use save_manual_feature_rule(dataset, payload).",
        "saved": False,
        "payload": payload,
    }


def get_supported_manual_feature_rule_catalog(output_type=None):
    # MANUAL PANDAS/ML CODE REQUIRED:
    # The developer will manually implement feature engineering inference,
    # regex extraction, string parsing, numeric derivation, datetime extraction,
    # categorical grouping, boolean flag creation, validation, feature mapping,
    # and pandas transformation logic.
    # Do not auto-generate or change this rule.

    return {
        "status": PLACEHOLDER_STATUS,
        "message": PLACEHOLDER_MESSAGE,
        "output_type": output_type,
        "rule_groups": _copy_rule_groups(),
    }
