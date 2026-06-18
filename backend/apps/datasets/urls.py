from django.urls import include, path

from .views import (
    DatasetDetailView,
    DatasetListView,
    DatasetPreviewView,
    DatasetProfileView,
    DatasetUploadView,
)


urlpatterns = [
    path("upload/", DatasetUploadView.as_view(), name="dataset-upload"),
    path("", DatasetListView.as_view(), name="dataset-list"),
    path("<int:id>/", DatasetDetailView.as_view(), name="dataset-detail"),
    path("<int:id>/preview/", DatasetPreviewView.as_view(), name="dataset-preview"),
    path("<int:id>/profile/", DatasetProfileView.as_view(), name="dataset-profile"),
    path("<int:id>/cleaning/", include("apps.cleaning.urls")),
]
