from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.datasets.models import Dataset

from .serializers import EdaGraphRequestSerializer
from .services.eda_graph_service import build_eda_graph_data
from .services.eda_summary import build_advanced_eda_summary, build_validation_eda_summary


def _get_owned_dataset(request, dataset_id):
    return get_object_or_404(Dataset, pk=dataset_id, owner=request.user)


class AdvancedEdaSummaryView(APIView):
    def get(self, request, dataset_id):
        _get_owned_dataset(request, dataset_id)
        payload = build_advanced_eda_summary(
            dataset_id,
            dataset_version=request.query_params.get("dataset_version") or "raw",
            target_column=request.query_params.get("target_column") or None,
        )
        return Response(payload)


class ValidationEdaSummaryView(APIView):
    def get(self, request, dataset_id):
        _get_owned_dataset(request, dataset_id)
        payload = build_validation_eda_summary(
            dataset_id,
            dataset_version=request.query_params.get("dataset_version") or "cleaned",
            target_column=request.query_params.get("target_column") or None,
        )
        return Response(payload)


class EdaGraphView(APIView):
    def post(self, request, dataset_id):
        _get_owned_dataset(request, dataset_id)
        serializer = EdaGraphRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            payload = build_eda_graph_data(dataset_id, serializer.validated_data)
        except NotImplementedError:
            payload = {
                "ok": True,
                "render_type": "not_implemented",
                "warnings": ["Graph generation is not available yet for this chart."],
            }
        except Exception as exc:
            chart_type = serializer.validated_data.get("chart_type") or "eda_graph"
            payload = {
                "ok": False,
                "chart_type": chart_type,
                "render_type": "error",
                "rows": [],
                "chart_data": {"rows": []},
                "summary": {},
                "warnings": [f"Graph fallback failed: {str(exc)}"],
                "ai_context": {
                    "dataset_id": dataset_id,
                    "page_source": "advanced_eda",
                    "chart_type": chart_type,
                    "chart_title": chart_type,
                    "used_columns": [],
                    "chart_config": {"chart_type": chart_type},
                    "chart_data": {"rows": []},
                    "source_rows": [],
                    "column_profiles": [],
                    "warnings": [f"Graph fallback failed: {str(exc)}"],
                },
            }

        return Response(payload)
