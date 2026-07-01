from django.urls import path
from .views import AiTaskView

urlpatterns = [
    path("task/", AiTaskView.as_view(), name="ai-task"),
]
