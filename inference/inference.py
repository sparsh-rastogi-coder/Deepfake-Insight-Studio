import os, sys, time
import cv2
import numpy as np
np.int = int
import pandas as pd
import random
from random import randint
from PIL import ImageFilter, Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from imutils.video import FileVideoStream 
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import Normalize

from .helpers.weigths_cfg import raw_data_stack, meta_models
from .helpers.functions import disable_grad, weight_preds, predict_on_video, predict_on_video_dual_stream
from .helpers.blazeface import BlazeFace
from .helpers.read_video_1 import VideoReader
from .helpers.face_extract_1 import FaceExtractor

from .MetaModel import MetaModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_dual_stream_cache = None

def _load_dual_stream_model():
    """Load (and cache) the DualStream_CBAM_CNN model weights."""
    global _dual_stream_cache
    if _dual_stream_cache is None:
        from .dual_stream_model import DualStream_CBAM_CNN
        model_path = os.path.join(BASE_DIR, "..", "best_dual_stream_model.pth")
        model = DualStream_CBAM_CNN().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        model = disable_grad(model)
        _dual_stream_cache = model
    return _dual_stream_cache


def predict(video_path: str, frames: int, use_dual_stream: bool = False):
    # move the main logic here
    WEIGTHS_PATH = os.path.join(BASE_DIR, "pretrained")
    WEIGTHS_EXT = ".pth"

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    input_size = 256
    frames_per_video = int(frames)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = Normalize(mean, std)

    facedet = BlazeFace().to(device)
    facedet.load_weights(os.path.join(BASE_DIR, "helpers", "blazeface.pth"))
    facedet.load_anchors(os.path.join(BASE_DIR, "helpers", "anchors.npy"))
    _ = facedet.train(False)

    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn, facedet)

    if use_dual_stream:
        # Dual-stream required normalization
        normalize_transform_dual = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        from .dual_stream_model import DualStream_CBAM_CNN
        model_path = os.path.join(BASE_DIR, "..", "best_dual_stream_model.pth")
        model = DualStream_CBAM_CNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model = disable_grad(model)
        
        prob, frame_pred, frame_images, grad_cam = predict_on_video_dual_stream(
            video_read_fn,
            normalize_transform_dual,
            model,
            video_path,
            frames,
            input_size,
            device
        )
        return prob, frame_pred, frame_images, grad_cam

    ''' Load and initialize models '''

    models = []
    weigths = []
    stack_models = []

    for raw_model in raw_data_stack:
        checkpoint = torch.load(
            os.path.join(WEIGTHS_PATH, raw_model[0] + WEIGTHS_EXT),
            map_location=device,
        )

        if "-" in raw_model[1]:
            model = EfficientNet.from_name(raw_model[1])
            model._fc = nn.Linear(model._fc.in_features, 1)
        else:
            model = timm.create_model(raw_model[1], pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, 1)

        model.load_state_dict(checkpoint)
        _ = model.eval()
        _ = disable_grad(model)
        model = model.to(device)
        stack_models.append(model)

        del checkpoint, model

    for meta_raw in meta_models:
        checkpoint = torch.load(
            os.path.join(WEIGTHS_PATH, meta_raw[0] + WEIGTHS_EXT),
            map_location=device,
        )

        model = MetaModel(models=raw_data_stack[meta_raw[1]], extended=meta_raw[2]).to(
            device
        )

        model.load_state_dict(checkpoint)
        _ = model.eval()
        _ = disable_grad(model)
        model.to(device)
        models.append(model)
        weigths.append(meta_raw[3])

        del model, checkpoint

    total = sum([1 - score for score in weigths])
    weigths_norm = [(1 - score) / total for score in weigths]


    prob,frame_pred,frame_images,grad_cams = predict_on_video(
        face_extractor,
        normalize_transform,
        stack_models,
        models,
        meta_models,
        weigths_norm,
        video_path,
        frames,
        input_size,
        device,
    )
    return prob,frame_pred,frame_images,grad_cams

