from django.core.exceptions import ObjectDoesNotExist

from apps.datasets.models import Dataset, DatasetVersion
from apps.datasets.services import get_dataset_shape, make_json_safe, read_dataset_file, validate_dataset_file
from apps.preprocessing.models import PredictionDataset, PredictionPreparationJob, PreprocessingPlan


# MANUAL PANDAS/ML CODE REQUIRED:
# The developer will manually implement preprocessing,
# transformation reuse, validation, and prediction logic.
# Do not auto-generate or change this rule.

def validate_prediction_dataset_against_plan(owner, plan_id, prediction_dataset_id):
    try:
        plan = PreprocessingPlan.objects.get(pk=plan_id, owner=owner)
    except PreprocessingPlan.DoesNotExist:
        raise ValueError("Preprocessing plan not found.")

    try:
        prediction_dataset = PredictionDataset.objects.get(pk=prediction_dataset_id, owner=owner)
    except PredictionDataset.DoesNotExist:
        raise ValueError("Prediction dataset not found.")

    required_columns = plan.required_columns_json or []
    available_columns = [str(c) for c in prediction_dataset.columns_json or []]
    missing_columns = [col for col in required_columns if col not in available_columns]

    errors = []
    if missing_columns:
        errors.append({"missing_columns": missing_columns})

    prediction_dataset.validation_status = (
        PredictionDataset.VALIDATION_PASSED if not errors else PredictionDataset.VALIDATION_FAILED
    )
    prediction_dataset.validation_errors_json = make_json_safe(errors)
    prediction_dataset.save(update_fields=["validation_status", "validation_errors_json"])

    job = PredictionPreparationJob.objects.create(
        owner=owner,
        preprocessing_plan=plan,
        prediction_dataset=prediction_dataset,
        status=PredictionPreparationJob.STATUS_VALIDATED if not errors else PredictionPreparationJob.STATUS_PENDING,
        validation_result_json={
            "required_columns": required_columns,
            "available_columns": available_columns,
            "missing_columns": missing_columns,
        },
        prepared_preview_json={},
    )

    return job


# MANUAL PANDAS/ML CODE REQUIRED:
# The developer will manually implement preprocessing,
# transformation reuse, validation, and prediction logic.
# Do not auto-generate or change this rule.

def prepare_prediction_dataset(owner, plan_id, prediction_dataset_id):
    job = validate_prediction_dataset_against_plan(owner, plan_id, prediction_dataset_id)

    if job.prediction_dataset.validation_status != PredictionDataset.VALIDATION_PASSED:
        raise ValueError("Prediction dataset did not pass validation.")

    # Build placeholder preview for a prediction dataset before any real transformation logic.
    job.prepared_preview_json = {
        "message": "Prediction will be available after ML model training is implemented.",
        "columns": job.prediction_dataset.columns_json,
        "rows": [],
    }
    job.status = PredictionPreparationJob.STATUS_PREPARED
    job.save(update_fields=["status", "prepared_preview_json"])

    return job


# MANUAL PANDAS/ML CODE REQUIRED:
# The developer will manually implement preprocessing,
# transformation reuse, validation, and prediction logic.
# Do not auto-generate or change this rule.

def create_prediction_dataset(owner, source_dataset_id, file):
    try:
        source_dataset = Dataset.objects.get(pk=source_dataset_id, owner=owner)
    except Dataset.DoesNotExist:
        raise ValueError("Source dataset not found.")

    error = validate_dataset_file(file)
    if error:
        raise ValueError(error)

    try:
        df = read_dataset_file(file)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    row_count, column_count = get_dataset_shape(df)
    columns_json = [str(column) for column in df.columns]
    file_type = file.name.rsplit(".", 1)[-1].lower() if "." in file.name else ""
    file.seek(0)

    prediction_dataset = PredictionDataset.objects.create(
        owner=owner,
        source_dataset=source_dataset,
        uploaded_file=file,
        file_type=file_type,
        file_size=file.size,
        row_count=row_count,
        column_count=column_count,
        columns_json=columns_json,
    )

    return prediction_dataset
