import torch
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class MockExtractor:
    def process_video(self, video_path):
        return [{"faces": [np.zeros((256, 256, 3), dtype=np.uint8)]}]
    def keep_only_best_face(self, faces): pass

from inference.helpers.functions import predict_on_video_dual_stream
from inference.dual_stream_model import DualStream_CBAM_CNN
from torchvision.transforms import Normalize
import cv2

normalize_transform_dual = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
model = DualStream_CBAM_CNN().to(device)
print(predict_on_video_dual_stream(MockExtractor(), normalize_transform_dual, model, 'dummy', 5, 256, device))
