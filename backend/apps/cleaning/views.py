from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.cleaning.serializers import (
    CleaningApplySerializer,
    DatasetRollbackSerializer,
    OutlierPreviewRequestSerializer,
)
from apps.cleaning.services import (
    apply_cleaning_actions,
    build_outlier_preview,
    build_cleaning_report,
    rollback_to_version,
)
from apps.datasets.models import Dataset, DatasetVersion
from apps.datasets.serializers import DatasetVersionSerializer


class BaseDatasetCleaningView(APIView):
    permission_classes = [IsAuthenticated]

    def get_dataset(self, id, user):
        try:
            return Dataset.objects.get(pk=id, owner=user)
        except Dataset.DoesNotExist:
            return None


class DatasetCleaningReportView(BaseDatasetCleaningView):
    def get(self, request, id):
        dataset = self.get_dataset(id, request.user)
        if dataset is None:
            return Response({"detail": "Dataset not found."}, status=status.HTTP_404_NOT_FOUND)

        report = build_cleaning_report(dataset)
        return Response(report)


class DatasetCleaningApplyView(BaseDatasetCleaningView):
    def post(self, request, id):
        dataset = self.get_dataset(id, request.user)
        if dataset is None:
            return Response({"detail": "Dataset not found."}, status=status.HTTP_404_NOT_FOUND)

        serializer = CleaningApplySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        validated = serializer.validated_data

        try:
            version = apply_cleaning_actions(
                dataset,
                actions=validated.get("actions", []),
                apply_safe=validated.get("apply_safe", False),
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        serialized_version = DatasetVersionSerializer(version)
        return Response(serialized_version.data, status=status.HTTP_201_CREATED)


class DatasetOutlierPreviewView(BaseDatasetCleaningView):
    def post(self, request, id):
        dataset = self.get_dataset(id, request.user)
        if dataset is None:
            return Response({"detail": "Dataset not found."}, status=status.HTTP_404_NOT_FOUND)

        serializer = OutlierPreviewRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        column_name = serializer.validated_data["column_name"]
        method = serializer.validated_data["method"]
        params = serializer.validated_data.get("params", {})

        try:
            preview = build_outlier_preview(
                dataset, 
                column_name=column_name, 
                method=method, 
                params=params
            )
            return Response(preview)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


class DatasetVersionsView(BaseDatasetCleaningView):
    def get(self, request, id):
        dataset = self.get_dataset(id, request.user)
        if dataset is None:
            return Response({"detail": "Dataset not found."}, status=status.HTTP_404_NOT_FOUND)

        versions = DatasetVersion.objects.filter(dataset=dataset).order_by("-version_number")
        serializer = DatasetVersionSerializer(versions, many=True)
        return Response({"versions": serializer.data})


class DatasetRollbackView(BaseDatasetCleaningView):
    def post(self, request, id):
        dataset = self.get_dataset(id, request.user)
        if dataset is None:
            return Response({"detail": "Dataset not found."}, status=status.HTTP_404_NOT_FOUND)

        serializer = DatasetRollbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        version_id = serializer.validated_data["version_id"]

        try:
            version = rollback_to_version(dataset, version_id)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        serialized_version = DatasetVersionSerializer(version)
        return Response(serialized_version.data, status=status.HTTP_201_CREATED)
