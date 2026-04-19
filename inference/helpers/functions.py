import os, sys, time
import traceback
import cv2
import numpy as np
np.int = int
import torch
import torch.nn as nn
import torch.nn.functional as F
from imutils.video import FileVideoStream 
from torchvision.transforms import Normalize

def disable_grad(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    return model
        

def weight_preds(preds, weights):
    final_preds = []
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            if len(final_preds) != len(preds[i]):
                final_preds.append(preds[i][j] * weights[i])
            else:
                final_preds[j] += preds[i][j] * weights[i]
                
    return torch.FloatTensor(final_preds)


def predict_on_video(face_extractor, normalize_transform, stack_models, models, meta_models, weigths, video_path, batch_size, input_size, device):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)
        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

            frame_images = []
            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    resized_face = cv2.resize(face, (input_size, input_size))
                    
                    if n < batch_size:
                        x[n] = resized_face
                        frame_images.append(resized_face)
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))

                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    #x[n] = cv2.flip(resized_face, 1)
                    #n += 1

            del faces

            if n > 0:
                x = torch.tensor(x, device=device).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                # Make a prediction
                with torch.no_grad():
                    y_pred = 0
                    stacked_preds = []
                    preds = []
                    
                    for i in range(len(stack_models)):
                        stacked_preds.append(stack_models[i](x).squeeze()[:n].unsqueeze(dim=1))
                    
                    for i in range(len(models)):
                        preds.append(models[i](stacked_preds[meta_models[i][1]]))
                
                    del x
                    
                    frame_preds = torch.sigmoid(weight_preds(preds, weigths))

                    y_pred = frame_preds.mean().item()

                    frame_preds = frame_preds.detach().cpu().tolist()
                    frame_preds = [
                        {"frame": i, "fake_prob": float(p)}
                        for i, p in enumerate(frame_preds)
                    ]
                    frame_images = [
                        cv2.imencode(".jpg", img)[1].tobytes().hex()
                        for img in frame_images
                    ]
                    del preds

                    return y_pred, frame_preds , frame_images, []

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))
        traceback.print_exc()
    
    return 0.5, [], [], []


def predict_on_video_dual_stream(video_read_fn, normalize_transform, model, video_path, batch_size, input_size, device):
    try:
        from PIL import Image
        from torchvision import transforms
        
        # EXACT transforms from Kaggle training
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        video_data = video_read_fn(video_path)
        if video_data is None:
            return 0.5, [], []
            
        frames, idxs = video_data
        
        if len(frames) > 0:
            batch_tensors = []
            frame_images = []
            
            for frame in frames: # frame is RGB numpy array 
                if len(batch_tensors) < batch_size:
                    # Use PIL matching kaggle logic 
                    tensor_frame = test_transform(frame)
                    batch_tensors.append(tensor_frame)
                    # Also keep for visual output (needs cv2 BGR to encode)
                    cv2_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (input_size, input_size))
                    frame_images.append(cv2_frame)

            if len(batch_tensors) > 0:
                x = torch.stack(batch_tensors).to(device)
                x.requires_grad = True

                from .gradcam import PureGradCAM
                model.eval()
                for param in model.parameters():
                    param.requires_grad = True

                target_layer = model.rgb_stream[-5]
                cam_extractor = PureGradCAM(model, target_layer)

                raw_cam = cam_extractor.generate(x)
                
                # Retrieve logits without computing new gradients
                with torch.no_grad():
                    logits = model(x)
                    # Model outputs P(Real). API expects P(Fake). So invert it:
                    frame_preds = 1.0 - torch.sigmoid(logits)
                    y_pred = frame_preds.mean().item()

                    frame_preds = frame_preds.detach().cpu().tolist()
                    frame_preds = [
                        {"frame": i, "fake_prob": float(p[0]) if isinstance(p, list) else float(p)}
                        for i, p in enumerate(frame_preds)
                    ]
                    frame_images_hex = [
                        cv2.imencode(".jpg", img)[1].tobytes().hex()
                        for img in frame_images
                    ]
                    
                    import torch.nn.functional as F
                    gradcam_images = []
                    cam_resized = F.interpolate(raw_cam, size=(256, 256), mode='bilinear', align_corners=False)
                    for i in range(len(batch_tensors)):
                        cam_np = cam_resized[i].squeeze().detach().cpu().numpy()
                        # Convert original frame tensor to PIL for masking
                        original_pil = transforms.ToPILImage()(((x[i] * 0.5) + 0.5).cpu())
                        red_overlay = Image.new('RGB', (256, 256), (255, 0, 0))
                        mask = Image.fromarray(np.uint8(255 * cam_np), mode='L')
                        heatmap_img = Image.composite(red_overlay, original_pil, mask)

                        np_heatmap = cv2.cvtColor(np.array(heatmap_img), cv2.COLOR_RGB2BGR)
                        np_heatmap = cv2.resize(np_heatmap, (224, 224))
                        gradcam_images.append(cv2.imencode(".jpg", np_heatmap)[1].tobytes().hex())

                    return y_pred, frame_preds, frame_images_hex, gradcam_images

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))
        traceback.print_exc()
    
    return 0.5, [], [], []
