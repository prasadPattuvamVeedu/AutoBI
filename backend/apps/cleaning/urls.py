from django.urls import path

from .views import (
    DatasetCleaningApplyView,
    DatasetCleaningReportView,
    DatasetRollbackView,
    DatasetVersionsView,
)


urlpatterns = [
    path("report/", DatasetCleaningReportView.as_view(), name="dataset-cleaning-report"),
    path("apply/", DatasetCleaningApplyView.as_view(), name="dataset-cleaning-apply"),
    path("versions/", DatasetVersionsView.as_view(), name="dataset-versions"),
    path("rollback/", DatasetRollbackView.as_view(), name="dataset-rollback"),
]