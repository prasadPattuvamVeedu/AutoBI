from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("api/auth/", include("apps.accounts.urls")),
    path("api/ai/", include("apps.ai.urls")),
    path("api/datasets/", include("apps.datasets.urls")),
    path("api/profiling/", include("apps.profiling.urls")),
    path("api/cleaning/<int:id>/", include("apps.cleaning.urls")),
    path("api/eda/", include("apps.eda.urls")),
    path("api/preprocessing/", include("apps.preprocessing.urls")),
    path("api/charts/", include("apps.charts.urls")),
    path("api/dashboards/", include("apps.dashboards.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
