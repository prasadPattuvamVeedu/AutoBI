from django.urls import path

from .views import AdvancedEdaSummaryView, EdaGraphView, ValidationEdaSummaryView


urlpatterns = [
    path("advanced/<int:dataset_id>/summary/", AdvancedEdaSummaryView.as_view(), name="advanced-eda-summary"),
    path("validation/<int:dataset_id>/summary/", ValidationEdaSummaryView.as_view(), name="validation-eda-summary"),
    path("<int:dataset_id>/graph/", EdaGraphView.as_view(), name="eda-graph"),
]
