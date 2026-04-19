import os
import sys
import uuid
import traceback
import subprocess
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure we can import from inference as a package even if run from inside the inference directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.inference import predict as pred_infer

app = FastAPI(title="Deepfake Video Detection API")


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

FRONTEND_DIR = os.path.dirname(__file__)

DEFAULT_FRAMES = 5
THRESHOLD = 0.5


class PredictionResponse(BaseModel):
    filename: str
    frames_used: int
    probability_fake: float
    stacked_pred: Optional[list] = None
    frames: list
    grad_cams: Optional[list] = None
    is_fake: bool

def run_inference(video_path: str, frames: int = DEFAULT_FRAMES, use_dual_stream: bool = False) -> tuple:
    return pred_infer(video_path, frames, use_dual_stream)

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Video file to analyze"),
    frames: Optional[int] = None,
    use_dual_stream: Optional[bool] = False,
):
    """
    Upload a video, run the deepfake detector, and return JSON with the result.
    """
    frames_to_use = frames or DEFAULT_FRAMES
    if frames_to_use <= 0:
        raise HTTPException(status_code=400, detail="frames must be a positive integer.")
    if frames_to_use>=10:
        frames_to_use=10
    _, ext = os.path.splitext(file.filename or "")
    if not ext:
        ext = ".mp4"

    temp_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_name)

    try:
        contents = await file.read()
        with open(temp_path, "wb") as out_file:
            out_file.write(contents)

        probability,stacked_probablity,frame_images,grad_cams = run_inference(temp_path, frames_to_use, use_dual_stream)

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    if stacked_probablity is None:
        return PredictionResponse(
            filename=file.filename,
            frames_used=frames_to_use,
            probability_fake=probability,
            frames=frame_images if 'frame_images' in locals() else [],
            grad_cams=grad_cams if 'grad_cams' in locals() else [],
            is_fake=probability >= THRESHOLD,
        )
    
    return PredictionResponse(
        filename=file.filename,
        frames_used=frames_to_use,
        probability_fake=probability,
        stacked_pred=stacked_probablity,
        frames=frame_images,
        grad_cams=grad_cams,
        is_fake=probability >= THRESHOLD,
    )


@app.post("/predict-image", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(..., description="Image file to analyze"),
):
    """
    Upload a single image (jpg/png/webp), run through the Dual-Stream model,
    and return JSON with the result.
    """
    import io
    import cv2
    import torch
    import numpy as np
    from PIL import Image as PILImage
    from torchvision import transforms
    from inference.inference import _load_dual_stream_model, DEVICE

    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="Only image files are supported (jpg, png, webp, bmp).")

    contents = await file.read()

    try:
        pil_img = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not open image: {exc}")

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    try:
        model = _load_dual_stream_model()
        tensor = test_transform(pil_img).unsqueeze(0).to(DEVICE)
        tensor.requires_grad = True

        import torch.nn.functional as F
        from inference.helpers.gradcam import PureGradCAM

        # Model must be in eval mode, but we need gradients for Grad-CAM
        model.eval()
        for param in model.parameters():
            param.requires_grad = True

        target_layer = model.rgb_stream[-5]
        cam_extractor = PureGradCAM(model, target_layer)

        raw_cam = cam_extractor.generate(tensor)

        # The prediction was obtained during generate since it does a forward pass
        # But we can also just get the logits again
        with torch.no_grad():
            logits = model(tensor)
            prob_fake = (1.0 - torch.sigmoid(logits)).item()

        # Generate Grad-CAM image overlay
        cam_resized = F.interpolate(raw_cam, size=(256, 256), mode='bilinear', align_corners=False)
        cam_np = cam_resized.squeeze().detach().cpu().numpy()

        original_resized = pil_img.resize((256, 256))
        red_overlay = PILImage.new('RGB', (256, 256), (255, 0, 0))
        mask = PILImage.fromarray(np.uint8(255 * cam_np), mode='L')
        heatmap_img = PILImage.composite(red_overlay, original_resized, mask)

        # Convert simple image to hex
        np_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        np_img = cv2.resize(np_img, (224, 224))
        frame_hex = cv2.imencode(".jpg", np_img)[1].tobytes().hex()

        # Convert heatmap to hex
        np_heatmap = cv2.cvtColor(np.array(heatmap_img), cv2.COLOR_RGB2BGR)
        np_heatmap = cv2.resize(np_heatmap, (224, 224))
        heatmap_hex = cv2.imencode(".jpg", np_heatmap)[1].tobytes().hex()

        frame_preds = [{"frame": 0, "fake_prob": prob_fake}]

        return PredictionResponse(
            filename=file.filename,
            frames_used=1,
            probability_fake=prob_fake,
            stacked_pred=frame_preds,
            frames=[frame_hex],
            grad_cams=[heatmap_hex],
            is_fake=prob_fake >= THRESHOLD,
        )

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


async def root():
    return JSONResponse(
        {
            "message": "Deepfake detection API is running.",
            "endpoints": {
                "GET /ui": "Open the Deepfake Insight Studio web interface.",
                "POST /predict": "Upload a video file and get deepfake probability.",
            },
        }
    )


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """
    Serve the frontend UI from the same origin as the API.
    """
    index_path = os.path.join(FRONTEND_DIR, "frontend.html")
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        raise HTTPException(
            status_code=500,
            detail="frontend.html not found. Make sure it exists next to api.py.",
        )

