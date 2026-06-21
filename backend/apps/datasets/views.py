import logging

from django.conf import settings
from django.db import IntegrityError, connection, transaction
from django.db.models.deletion import ProtectedError
from rest_framework import generics
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ColumnSchema, Dataset, DatasetProfile, DatasetVersion
from .serializers import (
    DatasetDetailSerializer,
    DatasetSerializer,
    DatasetUploadSerializer,
    TransformedDatasetSerializer,
)

from apps.profiling.services import build_dataset_profile_response
from apps.datasets.services import read_dataset_file
from apps.profiling.serializers import DatasetProfileSummarySerializer

logger = logging.getLogger(__name__)


def _table_exists(table_name):
    """Return False when a model was added but its migration/table was not created yet."""
    try:
        return table_name in connection.introspection.table_names()
    except Exception:
        logger.warning("Could not inspect database tables.", exc_info=True)
        return False


def _execute_if_table_exists(cursor, table_name, sql, params):
    if _table_exists(table_name):
        cursor.execute(sql, params)


def _storage_file_tuple(file_field):
    if file_field and getattr(file_field, "name", ""):
        return (file_field.storage, file_field.name)
    return None

class DatasetUploadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = DatasetUploadSerializer(
            data=request.data,
            context={"request": request},
        )
        serializer.is_valid(raise_exception=True)

        uploaded_file = serializer.validated_data["file"]
        confirm_large_file = serializer.validated_data["confirm_large_file"]

        if (
            uploaded_file
            and uploaded_file.size > settings.DATASET_LARGE_FILE_WARNING_BYTES
            and not confirm_large_file
        ):
            return Response(
                {
                    "code": "large_file_warning",
                    "detail": (
                        "This file is large. Processing it locally may use high RAM "
                        "and may be slow or unstable. Do you want to continue locally "
                        "or use customer cloud storage instead?"
                    ),
                },
                status=status.HTTP_409_CONFLICT,
            )

        dataset = serializer.save()
        return Response(DatasetDetailSerializer(dataset).data, status=status.HTTP_201_CREATED)


class DatasetListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        queryset = Dataset.objects.filter(owner=request.user).order_by("-created_at")
        serializer = DatasetSerializer(queryset, many=True)
        return Response(serializer.data)


class DatasetDetailView(generics.RetrieveDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = DatasetDetailSerializer
    lookup_url_kwarg = "id"

    def get_queryset(self):
        return Dataset.objects.filter(owner=self.request.user)

    def destroy(self, request, *args, **kwargs):
        dataset = self.get_object()

        try:
            files_to_delete = self._delete_dataset_generated_records(dataset)
        except ProtectedError as exc:
            return Response(
                {"detail": f"Dataset delete blocked by related records: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except IntegrityError as exc:
            logger.exception("Dataset delete integrity error for dataset_id=%s", dataset.pk)
            return Response(
                {"detail": f"Dataset delete failed because related data could not be removed: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            logger.exception("Dataset delete failed for dataset_id=%s", dataset.pk)
            return Response(
                {"detail": f"Dataset delete failed: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        self._delete_files_best_effort(files_to_delete)
        return Response(status=status.HTTP_204_NO_CONTENT)

    def _collect_file(self, file_field, files_to_delete):
        item = _storage_file_tuple(file_field)
        if item:
            files_to_delete.append(item)

    def _delete_dataset_generated_records(self, dataset):
        """
        Delete the dataset without letting missing optional app tables block deletion.

        Your current DB has preprocessing models imported in Django, but the table
        preprocessing_predictiondataset is missing. A normal Django cascade delete tries
        to query that missing reverse relation and fails before deleting the dataset.
        This method deletes known related rows with guarded raw SQL, skips optional
        tables that do not exist, then deletes the dataset row itself.
        """
        files_to_delete = []
        self._collect_file(dataset.file, files_to_delete)

        for version in DatasetVersion.objects.filter(dataset=dataset).only("id", "file"):
            self._collect_file(version.file, files_to_delete)

        if _table_exists("preprocessing_predictiondataset"):
            try:
                from apps.preprocessing.models import PredictionDataset

                for prediction_dataset in PredictionDataset.objects.filter(source_dataset=dataset).only("id", "uploaded_file"):
                    self._collect_file(prediction_dataset.uploaded_file, files_to_delete)
            except Exception:
                logger.warning("Could not collect prediction dataset files.", exc_info=True)

        with transaction.atomic():
            with connection.cursor() as cursor:
                # Dashboard and chart rows must go before the dataset row because they FK to it.
                _execute_if_table_exists(
                    cursor,
                    "dashboards_dashboarditem",
                    "DELETE FROM dashboards_dashboarditem WHERE dashboard_id IN (SELECT id FROM dashboards_dashboard WHERE dataset_id = %s)",
                    [dataset.pk],
                )
                _execute_if_table_exists(
                    cursor,
                    "dashboards_dashboarditem",
                    "DELETE FROM dashboards_dashboarditem WHERE sheet_id IN (SELECT id FROM charts_chartsheet WHERE dataset_id = %s)",
                    [dataset.pk],
                )
                _execute_if_table_exists(
                    cursor,
                    "dashboards_dashboard",
                    "DELETE FROM dashboards_dashboard WHERE dataset_id = %s",
                    [dataset.pk],
                )
                _execute_if_table_exists(
                    cursor,
                    "charts_chartinterpretation",
                    "DELETE FROM charts_chartinterpretation WHERE sheet_id IN (SELECT id FROM charts_chartsheet WHERE dataset_id = %s)",
                    [dataset.pk],
                )
                _execute_if_table_exists(
                    cursor,
                    "charts_chartsheet",
                    "DELETE FROM charts_chartsheet WHERE dataset_id = %s",
                    [dataset.pk],
                )

                # Optional preprocessing tables may not exist yet; skip safely if missing.
                _execute_if_table_exists(
                    cursor,
                    "preprocessing_predictionpreparationjob",
                    """
                    DELETE FROM preprocessing_predictionpreparationjob
                    WHERE preprocessing_plan_id IN (SELECT id FROM preprocessing_preprocessingplan WHERE dataset_id = %s)
                       OR prediction_dataset_id IN (SELECT id FROM preprocessing_predictiondataset WHERE source_dataset_id = %s)
                    """,
                    [dataset.pk, dataset.pk],
                ) if _table_exists("preprocessing_predictiondataset") and _table_exists("preprocessing_preprocessingplan") else None
                _execute_if_table_exists(
                    cursor,
                    "preprocessing_preprocessingplan",
                    "DELETE FROM preprocessing_preprocessingplan WHERE dataset_id = %s",
                    [dataset.pk],
                )
                _execute_if_table_exists(
                    cursor,
                    "preprocessing_predictiondataset",
                    "DELETE FROM preprocessing_predictiondataset WHERE source_dataset_id = %s",
                    [dataset.pk],
                )

                # Dataset app tables.
                cursor.execute("DELETE FROM datasets_datasetprofile WHERE dataset_id = %s", [dataset.pk])
                cursor.execute("DELETE FROM datasets_columnschema WHERE dataset_id = %s", [dataset.pk])
                cursor.execute("DELETE FROM datasets_datasetversion WHERE dataset_id = %s", [dataset.pk])
                cursor.execute("DELETE FROM datasets_dataset WHERE id = %s AND owner_id = %s", [dataset.pk, self.request.user.pk])

        return files_to_delete

    def _delete_files_best_effort(self, files_to_delete):
        for storage, file_name in files_to_delete:
            try:
                storage.delete(file_name)
            except Exception:
                logger.warning("Could not delete dataset file %s", file_name, exc_info=True)


class TransformedDatasetListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        versions = (
            DatasetVersion.objects.filter(dataset__owner=request.user)
            .exclude(version_type=DatasetVersion.VERSION_TYPE_ORIGINAL)
            .select_related("dataset")
            .order_by("-created_at")
        )
        serializer = TransformedDatasetSerializer(versions, many=True)
        return Response(serializer.data)


class DatasetTransformedVersionsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, id):
        try:
            dataset = Dataset.objects.get(pk=id, owner=request.user)
        except Dataset.DoesNotExist:
            return Response({"detail": "Dataset not found."}, status=status.HTTP_404_NOT_FOUND)

        versions = (
            DatasetVersion.objects.filter(dataset=dataset)
            .exclude(version_type=DatasetVersion.VERSION_TYPE_ORIGINAL)
            .select_related("dataset")
            .order_by("-version_number")
        )
        serializer = TransformedDatasetSerializer(versions, many=True)
        return Response(serializer.data)


class TransformedDatasetDetailView(generics.DestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = TransformedDatasetSerializer
    lookup_url_kwarg = "version_id"

    def get_queryset(self):
        return (
            DatasetVersion.objects.filter(dataset__owner=self.request.user)
            .exclude(version_type=DatasetVersion.VERSION_TYPE_ORIGINAL)
            .select_related("dataset")
        )

    def destroy(self, request, *args, **kwargs):
        version = self.get_object()
        file_to_delete = _storage_file_tuple(version.file)

        try:
            with transaction.atomic():
                with connection.cursor() as cursor:
                    # Delete optional preprocessing plans only if that app table exists.
                    _execute_if_table_exists(
                        cursor,
                        "preprocessing_preprocessingplan",
                        "DELETE FROM preprocessing_preprocessingplan WHERE dataset_version_id = %s",
                        [version.pk],
                    )
                    cursor.execute(
                        "DELETE FROM datasets_datasetversion WHERE id = %s AND dataset_id IN (SELECT id FROM datasets_dataset WHERE owner_id = %s)",
                        [version.pk, request.user.pk],
                    )
        except Exception as exc:
            logger.exception("Transformed dataset delete failed for version_id=%s", version.pk)
            return Response(
                {"detail": f"Unable to delete transformed dataset: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if file_to_delete:
            self._delete_file_best_effort(*file_to_delete)
        return Response(status=status.HTTP_204_NO_CONTENT)

    def _delete_file_best_effort(self, storage, file_name):
        try:
            storage.delete(file_name)
        except Exception:
            logger.warning("Could not delete transformed dataset file %s", file_name, exc_info=True)


class TransformedDatasetPreviewView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, version_id):
        try:
            version = (
                DatasetVersion.objects.select_related("dataset")
                .exclude(version_type=DatasetVersion.VERSION_TYPE_ORIGINAL)
                .get(pk=version_id, dataset__owner=request.user)
            )
        except DatasetVersion.DoesNotExist:
            return Response({"detail": "Transformed dataset not found."}, status=status.HTTP_404_NOT_FOUND)

        preview = {"columns": version.columns or [], "rows": version.preview_rows or []}
        if (not preview["columns"] and not preview["rows"]) and version.file:
            try:
                version.file.open("rb")
                df = read_dataset_file(version.file)
                preview = {"columns": list(df.columns), "rows": df.head(20).where(df.notna(), None).to_dict(orient="records")}
            except ValueError as exc:
                return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
            finally:
                version.file.close()

        return Response({
            "id": version.id,
            "name": TransformedDatasetSerializer(version).data.get("name"),
            "source_dataset_id": version.dataset_id,
            "source_dataset_name": version.dataset.name,
            "preview_json": preview,
            "columns": preview.get("columns", []),
            "rows": preview.get("rows", []),
        })


class UseTransformedDatasetForMLView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, version_id):
        try:
            version = (
                DatasetVersion.objects.select_related("dataset")
                .exclude(version_type=DatasetVersion.VERSION_TYPE_ORIGINAL)
                .get(pk=version_id, dataset__owner=request.user)
            )
        except DatasetVersion.DoesNotExist:
            return Response({"detail": "Transformed dataset not found."}, status=status.HTTP_404_NOT_FOUND)

        version.version_type = DatasetVersion.VERSION_TYPE_ML_READY
        version.is_active = True
        log = version.transformation_log or {}
        log["use_for_ml"] = True
        version.transformation_log = log
        version.save(update_fields=["version_type", "is_active", "transformation_log"])
        return Response(TransformedDatasetSerializer(version).data)


class DatasetPreviewView(APIView):
    """Placeholder for future dataset preview logic."""

    def get(self, request, id):
        return Response({"detail": f"Dataset {id} preview endpoint placeholder."})


class DatasetProfileView(APIView):
    """
    Generates and returns the detailed profile for a dataset.
    This view also updates the stored DatasetProfile in the database.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, id):
        try:
            dataset = Dataset.objects.get(pk=id, owner=request.user)
        except Dataset.DoesNotExist:
            return Response(
                {"detail": "Dataset not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            df = read_dataset_file(dataset.file)
        except ValueError as exc:
            return Response(
                {"detail": f"Error reading dataset file: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        profile_data = build_dataset_profile_response(dataset, df)
        dataset_profile, created = DatasetProfile.objects.get_or_create(dataset=dataset)
        dataset_profile.profile_json = profile_data
        dataset_profile.save()

        serializer = DatasetProfileSummarySerializer(profile_data)
        return Response(serializer.data)
