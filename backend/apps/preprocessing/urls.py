from django.urls import path

from .views import (
    ManualFeatureInferRulesView,
    ManualFeaturePreviewView,
    ManualFeatureSavePlaceholderView,
    PreprocessingPlanListCreateView,
    PreprocessingPlanDetailView,
    PredictionDatasetUploadView,
    ValidatePredictionDatasetView,
    PreparePredictionDatasetView,
)

urlpatterns = [
    path("plans/", PreprocessingPlanListCreateView.as_view(), name="preprocessing-plan-list-create"),
    path("plans/<int:plan_id>/", PreprocessingPlanDetailView.as_view(), name="preprocessing-plan-detail"),
    path("prediction-datasets/upload/", PredictionDatasetUploadView.as_view(), name="prediction-dataset-upload"),
    path("datasets/<int:dataset_id>/manual-feature/infer-rules/", ManualFeatureInferRulesView.as_view(), name="manual-feature-infer-rules"),
    path("datasets/<int:dataset_id>/manual-feature/preview/", ManualFeaturePreviewView.as_view(), name="manual-feature-preview"),
    path("datasets/<int:dataset_id>/manual-feature/save-placeholder/", ManualFeatureSavePlaceholderView.as_view(), name="manual-feature-save-placeholder"),
    path("plans/<int:plan_id>/validate-prediction-dataset/", ValidatePredictionDatasetView.as_view(), name="validate-prediction-dataset"),
    path("plans/<int:plan_id>/prepare-prediction-dataset/", PreparePredictionDatasetView.as_view(), name="prepare-prediction-dataset"),
]
