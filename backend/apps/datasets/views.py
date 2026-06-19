from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.models.deletion import ProtectedError
from rest_framework import generics
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import ColumnSchema, Dataset, DatasetProfile, DatasetVersion
from .serializers import DatasetDetailSerializer, DatasetSerializer, DatasetUploadSerializer

from apps.profiling.services import build_dataset_profile_response
from apps.datasets.services import read_dataset_file
from apps.profiling.serializers import DatasetProfileSummarySerializer

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
            self._delete_dataset_generated_records(dataset)
        except ProtectedError as exc:
            return Response(
                {"detail": f"Dataset delete blocked by related records: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except IntegrityError as exc:
            return Response(
                {"detail": f"Dataset delete failed because related data could not be removed: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(status=status.HTTP_204_NO_CONTENT)

    def _delete_dataset_generated_records(self, dataset):
        with transaction.atomic():
            DatasetProfile.objects.filter(dataset=dataset).delete()
            ColumnSchema.objects.filter(dataset=dataset).delete()

            if hasattr(dataset, "preprocessing_plans"):
                dataset.preprocessing_plans.all().delete()

            if hasattr(dataset, "prediction_datasets"):
                dataset.prediction_datasets.all().delete()

            for version in DatasetVersion.objects.filter(dataset=dataset):
                version.delete()

            dataset.delete()


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
