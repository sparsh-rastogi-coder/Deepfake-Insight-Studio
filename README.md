# Deepfake Insight Studio

Welcome to the **Deepfake Insight Studio**. This is a comprehensive, production-ready full-stack application built to identify deepfakes in both videos and spatial images using state-of-the-art Deep Learning models and Novel deep learning Architectures. 

This repository heavily relies on ensembled architectures such as EfficientNets, as well as a dedicated *Dual-Stream Architectures* using visual self-attention and PyTorch natively.

> **Kaggle Notebooks (Training & Testing)**
> - Training Notebook: `https://www.kaggle.com/code/sparshrastogicsv/fork-of-deepfakedetection-training`
> - Evaluation/Testing Notebook: `https://www.kaggle.com/code/sparshrastogicsv/deepfakedetection`
> - Walkthrough: ESE presentation.pdf
---

## Executive Summary

Our project focuses on Deepfake Detection using an array of architectures ranging from **EfficientNet ensemble models** on the DFDC benchmark to two highly customized deep learning architectures targeting the 140K Real/Fake dataset: 
- **Dual-Stream CBAM CNN**
- **WGC-Net** (Wavelet Guided Cross Attention Network)


> **We achieved 99.47% accuracy using a Dual-Stream CBAM CNN, but cross-dataset tests revealed a ~41% generalization drop, proving that current deepfake detectors heavily overfit to dataset-specific artifacts.**

---

## Benchmark & Leaderboard Results

### 1. DFDC Benchmark Results (Industry Dataset)
- **Best Raw Single Model (`tf_efficientnet_b6_ns`)**:
  - Accuracy: `0.8733`
  - Log Loss: `0.3157`
- **Best Meta Model** (Using stacked ensemble with 10 models):
  - Log Loss: `0.2630`
- **Final Ensemble Result** (Combined 14 raw + 5 meta-models):
  - Final Log Loss: `0.2630`
  - **Rank 14 on DFDC Leaderboard**

### 2. Dual-Stream CBAM CNN Results (140k Dataset)
**Final Test Results**: `99.47%` Accuracy | `0.0157` Loss

| Class | Precision | Recall | F1 Score |
|---|---|---|---|
| Fake | 0.9984 | 0.9910 | 0.9947 |
| Real | 0.9911 | 0.9984 | 0.9947 |

*Macro Average*: Precision: 0.9947 | Recall: 0.9947 | F1: 0.9947

### 3. WGC-Net Results (140k Dataset)
- **Final Accuracy**: `97.45%` utilizing customized wavelet-guided cross attention mechanisms.

### 4. Baseline Architecture Comparison
| Model | Accuracy |
|---|---|
| EfficientNet-B0 | 98.19% |
| **Dual-Stream CBAM** | **99.47%** |
| WGC-Net | 97.45% |

### 5. Cross-Dataset Generalization Test (Extremely Important)
When tested on **out-of-distribution, real-world compressed datasets**, a massive performance drop occurs:

| Model | In-Distribution Accuracy | Out-Distribution Architecture |
|---|---|---|
| EfficientNet-B0 | 98.19% | ~58% |
| Dual-Stream CBAM | 99.47% | ~59% |
| WGC-Net | 97.45% | ~58% |

*Average Performance Drop*: **~41%**

---

## Main Conclusions & Findings

1. **Frequency Features Command Clean Datasets**: Wavelet and Laplacian based models achieved near-perfect performance on highly predictable, uncompressed artifacts.
2. **Compression Destroys Fake Artifacts**: Real-world H.264 and standard social media compressions effectively wash out high-frequency manipulation clues.
3. **High Benchmark Accuracy ≠ Real-World Robustness**: A model can predictably score 99%+ in synthetic lab conditions but fail rapidly on unseen or compressed web-crawled videos.
4. **The Power of Dual-Stream Architecture**: The Dual-Stream CBAM model confidently stood as the premier architecture because it uniquely separated the input data into an **RGB Semantic stream** and a **Laplacian Noise stream**, significantly improving classification robustness.

---

## Future Scope
- **Vision Transformers** for enhanced localized global contexts.
- **Cross-generator datasets** integrating classical GANs, Diffusion models, and generic FaceSwap variants.
- **Real-time spatial video** deepfake detection mechanisms.
- **Compression-robust training frameworks** capable of handling heavy noise augmentation.

---

## Key Features
- **Video & Image Support**: Fully analyzes whole MP4/WebM videos (by extracting face crops frame-by-frame utilizing BlazeFace) or processes single targeted images directly.
- **Model Ensembling**: Comes thoroughly integrated with older ensemble pipelines (stacking robust models over multiple frames).
- **Dual-Stream CBAM Integration**: Dedicated pipeline utilizing a customized Dual-Stream Convolutional Block Attention Module architecture, achieving extremely high (~99%) spatial mapping accuracies based on Kaggle training.
- **Grad-CAM Explainability**: Visual spatial mappings are available for the *AI Reasoning* step. Uses Deep Learning back-propagation to highlight *what pixels* (i.e., visual artifacts, mismatched lighting, facial boundary distortions) triggered the Fake vs Real classification.
- **FastAPI + Django Architecture**: Uses an ASGI local FastAPI backend for heavily parallelized tensor operations and PyTorch hooks, while passing cleanly to a Django-based neon-themed web UI.

---

## Project Architecture
The project is split into two independent services that communicate locally.

1. **`inference/` (FastAPI Server)**: 
   - Handles the actual Neural Network predictions.
   - Loads the massive `.pth` PyTorch model states globally into GPU (if CUDA is accessible) or CPU.
   - Computes Deepfake Confidence %, Extracts Faces, and runs pure PyTorch Grad-CAM hooks. 
   - Exposes REST endpoints (`/predict` and `/predict-image`).

2. **`django_ui/` (Django Server)**: 
   - Stores the neon-themed HTML/TailwindCSS frontend views.
   - Acts as a proxy interface, keeping model-load stutters perfectly separate from the frontend user experience.
   - Displays prediction results, Pipeline statistics, and rendering of the Grad-CAM heatmap visualizations.

---

## Installation & Requirements

Ensure that you have `Python 3.10+` installed on your system.

### Option 1: GPU (Recommended for speed)
When dealing with `torch`, if you have an NVIDIA card, ensure you have the appropriate CUDA Toolkit installed. 

### Create your Virtual Environment
It is heavily recommended to use a standard Virtual Environment.

```bash
# Clone the codebase
git clone <your-repo-link>
cd dfdc-kaggle-solution

# Create and activate a virtual environment
python -m venv inference/myenv
inference\myenv\Scripts\activate   # On Windows
# source inference/myenv/bin/activate # On Mac/Linux
```

### Install Dependencies
Install dependencies to your environment (including PyTorch, FastAPI, OpenCV, and Django).
*(Note: Please ensure `torch` is configured to your specific CUDA requirements if necessary).*

```bash
pip install -r django_ui/requirements.txt
```

---

## How to Run the Application

Because this uses a detached backend and frontend, you must run **two separate server processes** simultaneously in two separate terminals.

### 1. Launch the FastAPI Backend (Terminal 1)
This service must be started first to mount PyTorch and prepare the Dual Stream models.
```bash
# Ensure your virtual environment is activated
inference\myenv\Scripts\python.exe -m uvicorn inference.api:app --host 127.0.0.1 --port 8000
```
*API is now running on `http://localhost:8000`. You can test endpoints via `http://localhost:8000/docs`.*

### 2. Launch the Django Frontend View (Terminal 2)
```bash
# Navigate to the django UI folder
cd django_ui

# Start the built-in development server
python manage.py runserver 8080
```
*Web dashboard is now visually accessible at `http://localhost:8080`.*

---

## Functionality & Using the Dashboard

Once both servers are running optimally, navigate to `http://localhost:8080` in your web browser.

1. **Upload & Predict**: Your main dashboard.
   - **Video Mode**: Drag and drop `.mp4` video files. You can choose how many frames to extract, and optionally check "Use Dual-Stream Model".
   - **Image Mode**: Checking "Use Dual-Stream Model" exposes a second checkbox allowing static image uploads. You can pass in a single image frame / cropped face to get a targeted verdict.
   
2. **Analysis Process**: Depending on the resolution and CPU/GPU processing unit, wait while the FastAPI backend handles extraction, PIL transforms, and model inference loops. Wait for the redirect.

3. **Pipeline & Explainability**: See a unified breakdown of which specific frames were utilized or extracted. 

4. **AI Reasoning (Grad-CAM)**: If you utilized the "Dual-Stream Model", navigate to the `AI Reasoning` tab. The model will parse gradients globally to calculate spatial reasoning, rendering a bright red-overlay indicating the exact regions that appeared manipulative or inorganic.

---

## Modifying Pre-Trained Weights
If you choose to retrain the Dual Stream Model externally via Kaggle, locate your output `best_dual_stream_model.pth`. Simply drop and overwrite the weights file within the base-path of the application directory to automatically update the web suite.
