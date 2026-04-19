from django.urls import path

from . import views


urlpatterns = [
    path("", views.upload_predict, name="upload_predict"),
    path("pipeline/", views.pipeline_explainability, name="pipeline_explainability"),
    path("reasoning/", views.ai_reasoning, name="ai_reasoning"),
    path("api/predict/", views.api_predict_proxy, name="api_predict_proxy"),
    path("api/jobs/start/", views.api_job_start, name="api_job_start"),
    path("api/jobs/start-image/", views.api_job_start_image, name="api_job_start_image"),
    path("api/jobs/<uuid:job_id>/", views.api_job_status, name="api_job_status"),
]

