from django.shortcuts import get_object_or_404
from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.datasets.models import Dataset

from .models import ChartInterpretation, ChartSheet
from .serializers import (
    ChartGenerateSerializer,
    ChartInterpretationSerializer,
    ChartNoteSerializer,
    ChartSheetSerializer,
    DashboardCommandConfigSerializer,
    DashboardDrillSerializer,
    DashboardForecastSerializer,
    DashboardWhatIfSerializer,
)
from .services.chart_data_service import ChartGenerationError, generate_chart_data
from .services.chart_interpretation_service import build_placeholder_interpretation
from apps.ai.services.ai_router import run_ai_task
from apps.ai.services.context_builder import build_chart_ai_context
from .services.chart_suggestion_service import build_chart_suggestions
from .services.dashboard_service import (
    build_dashboard_command_config,
    build_dashboard_drill_config,
    build_dashboard_forecast_config,
    build_dashboard_what_if_config,
)


class DatasetChartSuggestionsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, dataset_id):
        dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
        return Response(build_chart_suggestions(dataset))


class DatasetChartGenerateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
        serializer = ChartGenerateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            return Response(generate_chart_data(dataset, serializer.validated_data))
        except ChartGenerationError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)




class DatasetDashboardCommandConfigView(APIView):
    """Backend-backed dashboard command-center config echo/save endpoint."""
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        get_object_or_404(Dataset, id=dataset_id, owner=request.user)
        serializer = DashboardCommandConfigSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(build_dashboard_command_config(serializer.validated_data))


class DatasetDashboardForecastView(APIView):
    """Safe forecast configuration endpoint; manual forecasting comes later."""
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        get_object_or_404(Dataset, id=dataset_id, owner=request.user)
        serializer = DashboardForecastSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(build_dashboard_forecast_config(serializer.validated_data))


class DatasetDashboardWhatIfView(APIView):
    """Safe what-if configuration endpoint; manual simulation comes later."""
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        get_object_or_404(Dataset, id=dataset_id, owner=request.user)
        serializer = DashboardWhatIfSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(build_dashboard_what_if_config(serializer.validated_data))


class DatasetDashboardDrillView(APIView):
    """Safe drill configuration endpoint; execution reuses chart filters."""
    permission_classes = [IsAuthenticated]

    def post(self, request, dataset_id):
        get_object_or_404(Dataset, id=dataset_id, owner=request.user)
        serializer = DashboardDrillSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(build_dashboard_drill_config(serializer.validated_data))

class ChartSheetListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ChartSheetSerializer

    def get_queryset(self):
        queryset = ChartSheet.objects.filter(owner=self.request.user).select_related("dataset")
        dataset_id = self.request.query_params.get("dataset_id")
        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id, dataset__owner=self.request.user)
        return queryset

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class ChartSheetDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ChartSheetSerializer

    def get_queryset(self):
        return ChartSheet.objects.filter(owner=self.request.user, dataset__owner=self.request.user)


class ChartSheetAiInsightView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, id):
        sheet = get_object_or_404(
            ChartSheet,
            id=id,
            owner=request.user,
            dataset__owner=request.user,
        )
        config = sheet.chart_config_json or {}
        rows = (sheet.chart_data_json or {}).get("rows") or []
        ai_context = build_chart_ai_context(sheet.dataset, request.user, {
            "surface": "saved_sheet_ai_insight",
            "selected_graph_context": {
                "sheet_id": sheet.id,
                "title": sheet.name,
                "chart_type": sheet.chart_type or config.get("chart_type"),
                "chart_config": config,
                "x_column": config.get("x_column") or "",
                "y_column": config.get("y_column") or "",
                "group_by_column": config.get("group_by_column") or "",
                "color_by_column": config.get("color_by_column") or "",
                "size_column": config.get("size_column") or "",
                "sample_rows": rows[:12],
                "row_count": len(rows),
                "chart_data_meta": (sheet.chart_data_json or {}).get("meta") or {},
            },
        })
        ai_result = run_ai_task("chart_insights", ai_context)
        insight_lines = []
        for item in ai_result.get("insights") or []:
            if isinstance(item, dict):
                title = item.get("title") or "Insight"
                explanation = item.get("explanation") or item.get("description") or item.get("evidence") or ""
                insight_lines.append(f"{title}: {explanation}".strip())
        payload = {
            "mode": "ai_insight",
            "title": ai_result.get("summary") or f"AI insight for {sheet.name}",
            "content": "\n".join(insight_lines) or ai_result.get("answer") or ai_result.get("summary") or build_placeholder_interpretation(sheet, mode="ai_insight").get("content", ""),
            "is_visible": True,
        }
        interpretation = ChartInterpretation.objects.create(
            owner=request.user,
            sheet=sheet,
            **payload,
        )
        sheet.ai_interpretation_json = {
            "latest_interpretation_id": interpretation.id,
            "title": interpretation.title,
            "content": interpretation.content,
        }
        sheet.save(update_fields=["ai_interpretation_json", "updated_at"])
        return Response(ChartInterpretationSerializer(interpretation).data)


class ChartSheetNotesView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, id):
        sheet = get_object_or_404(
            ChartSheet,
            id=id,
            owner=request.user,
            dataset__owner=request.user,
        )
        serializer = ChartNoteSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        note = ChartInterpretation.objects.create(
            owner=request.user,
            sheet=sheet,
            mode=serializer.validated_data.get("mode") or "note",
            title=serializer.validated_data.get("title", ""),
            content=serializer.validated_data.get("content", ""),
            is_visible=serializer.validated_data.get("is_visible", True),
        )
        sheet.user_notes_json = list(sheet.user_notes_json or []) + [
            {"id": note.id, "title": note.title, "content": note.content}
        ]
        sheet.save(update_fields=["user_notes_json", "updated_at"])
        return Response(ChartInterpretationSerializer(note).data, status=status.HTTP_201_CREATED)


class ChartNoteDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ChartInterpretationSerializer

    def get_queryset(self):
        return ChartInterpretation.objects.filter(
            owner=self.request.user,
            sheet__owner=self.request.user,
            sheet__dataset__owner=self.request.user,
        )
