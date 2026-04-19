import os
import tempfile
import threading
import uuid
import requests

from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods


@ensure_csrf_cookie
def upload_predict(request):
    return render(request, "ui/upload_predict.html")


def pipeline_explainability(request):
    return render(request, "ui/pipeline_explainability.html")


def ai_reasoning(request):
    return render(request, "ui/ai_reasoning.html")

def _job_key(job_id: uuid.UUID) -> str:
    return f"dfdc:job:{job_id}"


def _run_job(job_id: uuid.UUID, video_path: str, filename: str, frames: str | None, use_dual_stream: str | None = None):
    cache.set(
        _job_key(job_id),
        {"job_id": str(job_id), "status": "running", "filename": filename, "frames": frames},
        timeout=60 * 60,
    )
    try:
        with open(video_path, "rb") as f:
            files = {"file": (filename, f, "application/octet-stream")}
            params = {}
            if frames:
                params["frames"] = frames
            if use_dual_stream:
                params["use_dual_stream"] = use_dual_stream
            r = requests.post(
                f"{settings.FASTAPI_BASE_URL}/predict",
                files=files,
                params=params,
                timeout=60 * 20,
            )

        try:
            data = r.json()
        except ValueError:
            cache.set(
                _job_key(job_id),
                {
                    "job_id": str(job_id),
                    "status": "error",
                    "error": "FastAPI returned non-JSON response.",
                    "raw": r.text,
                },
                timeout=60 * 60,
            )
            return

        if r.status_code >= 400:
            cache.set(
                _job_key(job_id),
                {
                    "job_id": str(job_id),
                    "status": "error",
                    "error": data.get("detail") if isinstance(data, dict) else "FastAPI error.",
                    "fastapi_status": r.status_code,
                    "data": data,
                },
                timeout=60 * 60,
            )
            return

        cache.set(
            _job_key(job_id),
            {"job_id": str(job_id), "status": "done", "result": data},
            timeout=60 * 60,
        )
    except Exception as exc:
        cache.set(
            _job_key(job_id),
            {"job_id": str(job_id), "status": "error", "error": str(exc)},
            timeout=60 * 60,
        )
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


@require_http_methods(["POST"])
def api_job_start(request):
    """
    Start an analysis job and immediately return a job_id.
    This prevents client navigation from cancelling long inference requests.
    """
    if "file" not in request.FILES:
        return JsonResponse({"detail": "Missing file upload."}, status=400)

    video = request.FILES["file"]
    frames = request.POST.get("frames") or request.GET.get("frames")
    use_dual_stream = request.POST.get("use_dual_stream") or request.GET.get("use_dual_stream")

    job_id = uuid.uuid4()
    cache.set(
        _job_key(job_id),
        {
            "job_id": str(job_id),
            "status": "queued",
            "filename": video.name,
            "frames": frames,
        },
        timeout=60 * 60,
    )

    suffix = os.path.splitext(video.name)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(prefix="dfdc_", suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as out:
        for chunk in video.chunks():
            out.write(chunk)

    t = threading.Thread(
        target=_run_job,
        args=(job_id, tmp_path, video.name, frames, use_dual_stream),
        daemon=True,
    )
    t.start()

    return JsonResponse({"job_id": str(job_id), "status": "queued"}, status=202)


@require_http_methods(["GET"])
def api_job_status(request, job_id):
    job = cache.get(_job_key(job_id))
    if not job:
        return JsonResponse({"detail": "Job not found or expired."}, status=404)
    return JsonResponse(job, status=200)


@require_http_methods(["POST"])
def api_predict_proxy(request):
    """
    Django proxy endpoint so the frontend always calls same-origin.
    It forwards the uploaded file to FastAPI /predict and returns the JSON.
    """
    if "file" not in request.FILES:
        return JsonResponse({"detail": "Missing file upload."}, status=400)

    video = request.FILES["file"]
    frames = request.POST.get("frames") or request.GET.get("frames")
    use_dual_stream = request.POST.get("use_dual_stream") or request.GET.get("use_dual_stream")

    files = {"file": (video.name, video.read(), video.content_type or "application/octet-stream")}
    params = {}
    if frames:
        params["frames"] = frames
    if use_dual_stream:
        params["use_dual_stream"] = use_dual_stream


    try:
        r = requests.post(
            f"{settings.FASTAPI_BASE_URL}/predict",
            files=files,
            params=params,
            timeout=60 * 20,
        )
    except requests.RequestException as exc:
        return JsonResponse({"detail": f"Backend request failed: {exc}"}, status=502)

    try:
        data = r.json()
    except ValueError:
        return JsonResponse(
            {"detail": "Backend returned non-JSON response.", "raw": r.text},
            status=502,
        )

    return JsonResponse(data, status=r.status_code)


@require_http_methods(["POST"])
def api_job_start_image(request):
    """
    Start an image analysis job using the dual-stream model and return a job_id immediately.
    """
    if "file" not in request.FILES:
        return JsonResponse({"detail": "Missing file upload."}, status=400)

    image = request.FILES["file"]
    job_id = uuid.uuid4()
    cache.set(
        _job_key(job_id),
        {"job_id": str(job_id), "status": "queued", "filename": image.name, "frames": 1},
        timeout=60 * 60,
    )

    suffix = os.path.splitext(image.name)[1] or ".jpg"
    fd, tmp_path = tempfile.mkstemp(prefix="dfdc_img_", suffix=suffix)
    os.close(fd)
    with open(tmp_path, "wb") as out:
        for chunk in image.chunks():
            out.write(chunk)

    def _run_image_job():
        try:
            cache.set(
                _job_key(job_id),
                {"job_id": str(job_id), "status": "running", "filename": image.name, "frames": 1},
                timeout=60 * 60,
            )
            with open(tmp_path, "rb") as f:
                files = {"file": (image.name, f, image.content_type or "image/jpeg")}
                r = requests.post(
                    f"{settings.FASTAPI_BASE_URL}/predict-image",
                    files=files,
                    timeout=60 * 5,
                )
            try:
                data = r.json()
            except ValueError:
                cache.set(
                    _job_key(job_id),
                    {"job_id": str(job_id), "status": "error", "error": "FastAPI returned non-JSON."},
                    timeout=60 * 60,
                )
                return
            if r.status_code >= 400:
                cache.set(
                    _job_key(job_id),
                    {"job_id": str(job_id), "status": "error",
                     "error": data.get("detail") if isinstance(data, dict) else "FastAPI error."},
                    timeout=60 * 60,
                )
                return
            cache.set(
                _job_key(job_id),
                {"job_id": str(job_id), "status": "done", "result": data},
                timeout=60 * 60,
            )
        except Exception as exc:
            cache.set(
                _job_key(job_id),
                {"job_id": str(job_id), "status": "error", "error": str(exc)},
                timeout=60 * 60,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    t = threading.Thread(target=_run_image_job, daemon=True)
    t.start()

    return JsonResponse({"job_id": str(job_id), "status": "queued"}, status=202)
