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
)
from .services.chart_data_service import ChartGenerationError, generate_chart_data
from .services.chart_interpretation_service import build_placeholder_interpretation
from .services.chart_suggestion_service import build_chart_suggestions


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
        payload = build_placeholder_interpretation(sheet, mode="ai_insight")
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
