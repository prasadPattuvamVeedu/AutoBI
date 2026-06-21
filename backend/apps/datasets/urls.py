from django.urls import include, path

from .views import (
    DatasetDetailView,
    DatasetListView,
    DatasetPreviewView,
    DatasetProfileView,
    DatasetUploadView,
    DatasetTransformedVersionsView,
    TransformedDatasetDetailView,
    TransformedDatasetListView,
    TransformedDatasetPreviewView,
    UseTransformedDatasetForMLView,
)


urlpatterns = [
    path("upload/", DatasetUploadView.as_view(), name="dataset-upload"),
    path("transformed/", TransformedDatasetListView.as_view(), name="transformed-dataset-list"),
    path("transformed/<int:version_id>/", TransformedDatasetDetailView.as_view(), name="transformed-dataset-detail"),
    path("transformed/<int:version_id>/preview/", TransformedDatasetPreviewView.as_view(), name="transformed-dataset-preview"),
    path("transformed/<int:version_id>/use-for-ml/", UseTransformedDatasetForMLView.as_view(), name="transformed-dataset-use-for-ml"),
    path("", DatasetListView.as_view(), name="dataset-list"),
    path("<int:id>/", DatasetDetailView.as_view(), name="dataset-detail"),
    path("<int:id>/transformed/", DatasetTransformedVersionsView.as_view(), name="dataset-transformed-versions"),
    path("<int:id>/preview/", DatasetPreviewView.as_view(), name="dataset-preview"),
    path("<int:id>/profile/", DatasetProfileView.as_view(), name="dataset-profile"),
    path("<int:id>/cleaning/", include("apps.cleaning.urls")),
]
