from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.datasets.models import Dataset
from .serializers import AiTaskRequestSerializer
from .services.ai_router import run_ai_task
from .services.context_builder import (
    build_chart_ai_context,
    build_chat_ai_context,
    build_cleaning_ai_context,
    build_dashboard_ai_context,
    build_dataset_ai_context,
    merge_frontend_context,
    trim_context_for_token_limit,
)


def build_context_for_task(task, dataset, user, frontend_context=None, question=None):
    if dataset is None:
        return frontend_context or {}

    if task in {"cleaning_suggestions", "outlier_suggestions"}:
        return build_cleaning_ai_context(dataset, user, frontend_context)
    if task in {"chart_suggestions", "chart_insights", "chart_graph_insight", "eda_graph_insight", "natural_language_chart_builder"}:
        return build_chart_ai_context(dataset, user, frontend_context)
    if task in {"dashboard_summary", "dashboard_graph_insight"}:
        return build_dashboard_ai_context(dataset, user, frontend_context)
    if task == "chat_assistant":
        return build_chat_ai_context(dataset, user, question=question, extra_context=frontend_context)
    context = build_dataset_ai_context(dataset, user, include_sample=True)
    return trim_context_for_token_limit(merge_frontend_context(context, frontend_context))


class AiTaskView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = AiTaskRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        dataset_id = serializer.validated_data.get("dataset_id")
        dataset = None

        if dataset_id:
            try:
                dataset = Dataset.objects.get(id=dataset_id, owner=request.user)
            except Dataset.DoesNotExist:
                return Response(
                    {"detail": "Dataset not found.", "suggestions": [], "actions": []},
                    status=status.HTTP_404_NOT_FOUND,
                )

        task = serializer.validated_data["task"]
        question = serializer.validated_data.get("question") or ""
        context = build_context_for_task(
            task=task,
            dataset=dataset,
            user=request.user,
            frontend_context=serializer.validated_data.get("context") or {},
            question=question,
        )
        try:
            result = run_ai_task(
                task=task,
                context=context,
                question=question,
            )
        except Exception:
            return Response(
                {"detail": "AI service is temporarily unavailable. Please try again later.", "suggestions": [], "actions": []},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        return Response(result)
