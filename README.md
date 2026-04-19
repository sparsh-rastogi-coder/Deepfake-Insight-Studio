# Deepfake Insight Studio 🕵️‍♂️🤖

Welcome to the **Deepfake Insight Studio**. This is a comprehensive, production-ready full-stack application built to identify deepfakes in both videos and spatial images using state-of-the-art Deep Learning models and Novel deep learning Architectures. 

This repository heavily relies on ensembled architectures such as EfficientNets, as well as a dedicated *Dual-Stream Architectures* using visual self-attention and PyTorch natively.

> **Kaggle Notebooks (Training & Testing)**
> - Training Notebook: `https://www.kaggle.com/code/sparshrastogicsv/fork-of-deepfakedetection-training`
> - Evaluation/Testing Notebook: `https://www.kaggle.com/code/sparshrastogicsv/deepfakedetection`

---

## 🌟 Key Features
- **Video & Image Support**: Fully analyzes whole MP4/WebM videos (by extracting face crops frame-by-frame utilizing BlazeFace) or processes single targeted images directly.
- **Model Ensembling**: Comes thoroughly integrated with older ensemble pipelines (stacking robust models over multiple frames).
- **Dual-Stream CBAM Integration**: Dedicated pipeline utilizing a customized Dual-Stream Convolutional Block Attention Module architecture, achieving extremely high (~99%) spatial mapping accuracies based on Kaggle training.
- **Grad-CAM Explainability**: Visual spatial mappings are available for the *AI Reasoning* step. Uses Deep Learning back-propagation to highlight *what pixels* (i.e., visual artifacts, mismatched lighting, facial boundary distortions) triggered the Fake vs Real classification.
- **FastAPI + Django Architecture**: Uses an ASGI local FastAPI backend for heavily parallelized tensor operations and PyTorch hooks, while passing cleanly to a Django-based neon-themed web UI.

---

## 🏗️ Project Architecture
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

## ⚙️ Installation & Requirements

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

## 🚀 How to Run the Application

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

## 📂 Functionality & Using the Dashboard

Once both servers are running optimally, navigate to `http://localhost:8080` in your web browser.

1. **Upload & Predict**: Your main dashboard.
   - **Video Mode**: Drag and drop `.mp4` video files. You can choose how many frames to extract, and optionally check "Use Dual-Stream Model".
   - **Image Mode**: Checking "Use Dual-Stream Model" exposes a second checkbox allowing static image uploads. You can pass in a single image frame / cropped face to get a targeted verdict.
   
2. **Analysis Process**: Depending on the resolution and CPU/GPU processing unit, wait while the FastAPI backend handles extraction, PIL transforms, and model inference loops. Wait for the redirect.

3. **Pipeline & Explainability**: See a unified breakdown of which specific frames were utilized or extracted. 

4. **AI Reasoning (Grad-CAM)**: If you utilized the "Dual-Stream Model", navigate to the `AI Reasoning` tab. The model will parse gradients globally to calculate spatial reasoning, rendering a bright red-overlay indicating the exact regions that appeared manipulative or inorganic.

---

## 🛠️ Modifying Pre-Trained Weights
If you choose to retrain the Dual Stream Model externally via Kaggle, locate your output `best_dual_stream_model.pth`. Simply drop and overwrite the weights file within the base-path of the application directory to automatically update the web suite.
