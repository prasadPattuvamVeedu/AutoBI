from django.urls import path

from .views import DatasetProfileSummaryView


urlpatterns = [
    path("<int:dataset_id>/summary/", DatasetProfileSummaryView.as_view(), name="profile-summary"),
]
