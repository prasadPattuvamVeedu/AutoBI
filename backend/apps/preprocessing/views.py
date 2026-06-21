from rest_framework import status
from django.shortcuts import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.datasets.models import Dataset

from .serializers import (
    ManualFeatureInferRulesSerializer,
    ManualFeaturePreviewSerializer,
    ManualFeatureSavePlaceholderSerializer,
    PreprocessingPlanSerializer,
    PredictionDatasetUploadSerializer,
    PredictionDatasetSerializer,
    PredictionPreparationJobSerializer,
    ValidatePredictionDatasetSerializer,
    PreparePredictionDatasetSerializer,
)
from .services.plan_service import create_preprocessing_plan, get_preprocessing_plan
from .services.prediction_prepare_service import (
    create_prediction_dataset,
    prepare_prediction_dataset,
    validate_prediction_dataset_against_plan,
)
from .services.manual_feature_builder import (
    infer_manual_feature_rules,
    preview_manual_feature_rule,
    save_manual_feature_rule,
)
from .models import PreprocessingPlan, PredictionDataset, PredictionPreparationJob


class PreprocessingPlanListCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        plans = PreprocessingPlan.objects.filter(owner=request.user).order_by("-created_at")
        serializer = PreprocessingPlanSerializer(plans, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = PreprocessingPlanSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        plan = create_preprocessing_plan(
            owner=request.user,
            dataset_id=serializer.validated_data["dataset"].id,
            dataset_version_id=serializer.validated_data["dataset_version"].id,
            name=serializer.validated_data["name"],
            target_column=serializer.validated_data.get("target_column"),
            plan_json=serializer.validated_data.get("plan_json", {}),
        )

        return Response(PreprocessingPlanSerializer(plan).data, status=status.HTTP_201_CREATED)


class PreprocessingPlanDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, plan_id):
        plan = get_preprocessing_plan(request.user, plan_id)
        if plan is None:
            return Response({"detail": "Preprocessing plan not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(PreprocessingPlanSerializer(plan).data)


class PredictionDatasetUploadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = PredictionDatasetUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            prediction_dataset = create_prediction_dataset(
                owner=request.user,
                source_dataset_id=serializer.validated_data["source_dataset_id"],
                file=serializer.validated_data["file"],
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(PredictionDatasetSerializer(prediction_dataset).data, status=status.HTTP_201_CREATED)


class ValidatePredictionDatasetView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, plan_id):
        serializer = ValidatePredictionDatasetSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            job = validate_prediction_dataset_against_plan(
                owner=request.user,
                plan_id=plan_id,
                prediction_dataset_id=serializer.validated_data["prediction_dataset_id"],
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(PredictionPreparationJobSerializer(job).data)


class PreparePredictionDatasetView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, plan_id):
        serializer = PreparePredictionDatasetSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            job = prepare_prediction_dataset(
                owner=request.user,
                plan_id=plan_id,
                prediction_dataset_id=serializer.validated_data["prediction_dataset_id"],
            )
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(PredictionPreparationJobSerializer(job).data)


class ManualFeatureInferRulesView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)

        serializer = ManualFeatureInferRulesSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        response = infer_manual_feature_rules(
            source_values=data.get("source_values", []),
            expected_outputs=data.get("expected_outputs", []),
            output_type=data.get("output_type"),
            source_column=data["source_column"],
            new_feature_name=data.get("new_feature_name"),
        )
        return Response(response)


class ManualFeaturePreviewView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        get_object_or_404(Dataset, id=dataset_id, owner=request.user)

        serializer = ManualFeaturePreviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        response = preview_manual_feature_rule(
            source_values=data.get("source_values", []),
            rule_json=data["rule_json"],
        )
        return Response(response)


class ManualFeatureSavePlaceholderView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)

        serializer = ManualFeatureSavePlaceholderSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            save_result = save_manual_feature_rule(dataset, data)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        return Response(
            {
                "save_result": save_result,
                "feature_mapping": save_result.get("feature_mapping"),
                "active_version_type": save_result.get("active_version_type"),
                "active_version_id": save_result.get("active_version_id"),
                "active_preview_rows": save_result.get("active_preview_rows", []),
                "active_columns": save_result.get("active_columns", []),
                "feature_engineering_plan_json": save_result.get("feature_engineering_plan_json", {}),
            }
        )
