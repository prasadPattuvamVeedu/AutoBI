def validate_dataset_file(file):
    """
    Validate an uploaded dataset file.

    Later this will check supported CSV/Excel file types and file size limits.
    """
    return None


def read_dataset_file(file_path):
    """
    Read a dataset file from disk.

    Later this will read CSV/Excel files using pandas and return a DataFrame.
    """
    return None


def get_dataset_shape(df):
    """
    Get the number of rows and columns in a dataset.

    Later this will return row_count and column_count from a DataFrame.
    """
    return None, None


def generate_dataset_preview(df, limit=20):
    """
    Generate a small preview of a dataset.

    Later this will return column names and the first rows up to the limit.
    """
    return {
        "columns": [],
        "rows": [],
    }


def generate_dataset_profile(df):
    """
    Generate profile information for a dataset.

    Later this will create a JSON-safe summary of dataset columns and values.
    """
    return {}


def clean_json_value(value):
    """
    Convert one value into a JSON-safe value.

    Later this will handle pandas and numpy values that JSON cannot serialize.
    """
    return value


def dataframe_to_json_safe(data):
    """
    Recursively convert data into JSON-safe values.

    Later this will clean nested dictionaries and lists returned from pandas.
    """
    return data
