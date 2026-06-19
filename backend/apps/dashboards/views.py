from django.shortcuts import get_object_or_404
from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Dashboard, DashboardItem
from .serializers import DashboardItemSerializer, DashboardLayoutSerializer, DashboardSerializer


class DashboardListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = DashboardSerializer

    def get_queryset(self):
        queryset = Dashboard.objects.filter(owner=self.request.user, dataset__owner=self.request.user)
        dataset_id = self.request.query_params.get("dataset_id")
        if dataset_id:
            queryset = queryset.filter(dataset_id=dataset_id)
        return queryset

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class DashboardDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = DashboardSerializer

    def get_queryset(self):
        return Dashboard.objects.filter(owner=self.request.user, dataset__owner=self.request.user)


class DashboardItemCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, id):
        dashboard = get_object_or_404(
            Dashboard,
            id=id,
            owner=request.user,
            dataset__owner=request.user,
        )
        serializer = DashboardItemSerializer(
            data=request.data,
            context={"request": request, "dashboard": dashboard},
        )
        serializer.is_valid(raise_exception=True)
        item = serializer.save(dashboard=dashboard)
        return Response(DashboardItemSerializer(item, context={"request": request}).data, status=status.HTTP_201_CREATED)


class DashboardLayoutUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def put(self, request, id):
        dashboard = get_object_or_404(
            Dashboard,
            id=id,
            owner=request.user,
            dataset__owner=request.user,
        )
        serializer = DashboardLayoutSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        for item_payload in serializer.validated_data.get("items", []):
            item_id = item_payload.get("id")
            if not item_id:
                continue
            DashboardItem.objects.filter(
                id=item_id,
                dashboard=dashboard,
                dashboard__owner=request.user,
            ).update(
                x=item_payload.get("x", 0),
                y=item_payload.get("y", 0),
                w=item_payload.get("w", 6),
                h=item_payload.get("h", 4),
                config_json=item_payload.get("config_json", {}),
            )

        dashboard.layout_json = serializer.validated_data.get("layout_json", dashboard.layout_json)
        dashboard.save(update_fields=["layout_json", "updated_at"])
        return Response(DashboardSerializer(dashboard, context={"request": request}).data)
