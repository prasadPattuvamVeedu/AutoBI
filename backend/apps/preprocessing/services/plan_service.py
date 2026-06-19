from django.core.exceptions import ObjectDoesNotExist

from apps.datasets.models import Dataset, DatasetVersion
from apps.datasets.services import get_dataset_shape, make_json_safe, read_dataset_file
from apps.preprocessing.models import PreprocessingPlan


# MANUAL PANDAS/ML CODE REQUIRED:
# The developer will manually implement preprocessing,
# transformation reuse, validation, and prediction logic.
# Do not auto-generate or change this rule.

def create_preprocessing_plan(owner, dataset_id, dataset_version_id, name, plan_json, target_column=None):
    if dataset_id is None or dataset_version_id is None:
        raise ValueError("Dataset and dataset version are required to create a preprocessing plan.")

    try:
        dataset = Dataset.objects.get(pk=dataset_id, owner=owner)
    except Dataset.DoesNotExist:
        raise ValueError("Dataset not found.")

    try:
        dataset_version = DatasetVersion.objects.get(pk=dataset_version_id, dataset=dataset)
    except DatasetVersion.DoesNotExist:
        raise ValueError("Dataset version not found.")

    required_columns = get_required_columns_from_plan(plan_json)
    feature_mapping = build_feature_mapping_placeholder(plan_json)

    plan = PreprocessingPlan.objects.create(
        owner=owner,
        dataset=dataset,
        dataset_version=dataset_version,
        name=name,
        target_column=target_column,
        plan_json=make_json_safe(plan_json),
        required_columns_json=make_json_safe(required_columns),
        feature_mapping_json=make_json_safe(feature_mapping),
    )

    return plan


# MANUAL PANDAS/ML CODE REQUIRED:
# The developer will manually implement preprocessing,
# transformation reuse, validation, and prediction logic.
# Do not auto-generate or change this rule.

def get_preprocessing_plan(owner, plan_id):
    try:
        return PreprocessingPlan.objects.get(pk=plan_id, owner=owner)
    except PreprocessingPlan.DoesNotExist:
        return None


# MANUAL PANDAS/ML CODE REQUIRED:
# The developer will manually implement preprocessing,
# transformation reuse, validation, and prediction logic.
# Do not auto-generate or change this rule.

def get_required_columns_from_plan(plan_json):
    if not isinstance(plan_json, dict):
        return []

    required_columns = []
    for step in plan_json.get("steps", []):
        if isinstance(step, dict) and step.get("column"):
            required_columns.append(step["column"])
        elif isinstance(step, dict) and step.get("columns"):
            required_columns.extend(step["columns"])

    return list({str(column) for column in required_columns})


# MANUAL PANDAS/ML CODE REQUIRED:
# The developer will manually implement preprocessing,
# transformation reuse, validation, and prediction logic.
# Do not auto-generate or change this rule.

def build_feature_mapping_placeholder(plan_json):
    if not isinstance(plan_json, dict):
        return {}

    mapping = {"source_columns": plan_json.get("steps", [])}
    return make_json_safe(mapping)
