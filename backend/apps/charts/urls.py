from django.urls import path

from .views import (
    ChartNoteDetailView,
    ChartSheetAiInsightView,
    ChartSheetDetailView,
    ChartSheetListCreateView,
    ChartSheetNotesView,
    DatasetChartGenerateView,
    DatasetChartSuggestionsView,
)


urlpatterns = [
    path("datasets/<int:dataset_id>/suggestions/", DatasetChartSuggestionsView.as_view(), name="chart-suggestions"),
    path("datasets/<int:dataset_id>/generate/", DatasetChartGenerateView.as_view(), name="chart-generate"),
    path("sheets/", ChartSheetListCreateView.as_view(), name="chart-sheet-list-create"),
    path("sheets/<int:pk>/", ChartSheetDetailView.as_view(), name="chart-sheet-detail"),
    path("sheets/<int:id>/ai-insight/", ChartSheetAiInsightView.as_view(), name="chart-sheet-ai-insight"),
    path("sheets/<int:id>/notes/", ChartSheetNotesView.as_view(), name="chart-sheet-notes"),
    path("notes/<int:pk>/", ChartNoteDetailView.as_view(), name="chart-note-detail"),
]
