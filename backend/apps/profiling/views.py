from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.datasets.models import Dataset
from apps.datasets.services import read_dataset_file

from .serializers import DatasetProfileSummarySerializer
from .services import build_dataset_profile_response


class DatasetProfileSummaryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, dataset_id):
        dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)

        if not dataset.file:
            return Response(
                {"detail": "Dataset file is not available."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            dataset.file.open("rb")
            df = read_dataset_file(dataset.file)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        finally:
            dataset.file.close()

        profile = build_dataset_profile_response(dataset, df)
        serializer = DatasetProfileSummarySerializer(profile)
        return Response(serializer.data)
