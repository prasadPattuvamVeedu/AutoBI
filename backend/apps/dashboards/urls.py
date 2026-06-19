from django.urls import path

from .views import (
    DashboardDetailView,
    DashboardItemCreateView,
    DashboardLayoutUpdateView,
    DashboardListCreateView,
)


urlpatterns = [
    path("", DashboardListCreateView.as_view(), name="dashboard-list-create"),
    path("<int:pk>/", DashboardDetailView.as_view(), name="dashboard-detail"),
    path("<int:id>/items/", DashboardItemCreateView.as_view(), name="dashboard-items"),
    path("<int:id>/layout/", DashboardLayoutUpdateView.as_view(), name="dashboard-layout"),
]
