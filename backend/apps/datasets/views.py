from django.conf import settings
from rest_framework import generics
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Dataset
from .serializers import DatasetDetailSerializer, DatasetSerializer, DatasetUploadSerializer


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


class DatasetPreviewView(APIView):
    """Placeholder for future dataset preview logic."""

    def get(self, request, id):
        return Response({"detail": f"Dataset {id} preview endpoint placeholder."})


class DatasetProfileView(APIView):
    """Placeholder for future dataset profile logic."""

    def get(self, request, id):
        return Response({"detail": f"Dataset {id} profile endpoint placeholder."})
