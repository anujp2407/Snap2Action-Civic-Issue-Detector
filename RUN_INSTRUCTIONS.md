# Pothole Detection — Run & Setup Guide

This project combines **YOLOv8 Deep Learning Pothole Detection** (from [noorkhokhar99's repository](https://github.com/noorkhokhar99/Pothole-Detection-Pothole-Detection-using-python-and-deep-learning)) with a **CustomTkinter Desktop Application** and **OpenCV Physical Verification**.

---

## 1. Prerequisites & Installation

Open PowerShell or Command Prompt, navigate to the project directory, and install requirements:

```powershell
cd "C:\Users\Anuj\Desktop\Projects\Pothhole Detection"
pip install -r requirements.txt
```

> **Note:** If PyTorch is not yet installed on your system, install it via:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> *(Or use CPU mode if you don't have an NVIDIA GPU: `pip install torch torchvision`)*

---

## 2. Model Weights (`y8best.pt`)

The pre-trained pothole model weights (`y8best.pt`) are hosted on Google Drive by the original author:

1. **Download link:** [Google Drive - y8best.pt Weights](https://drive.google.com/drive/folders/1T1h2WH3_wrNb0foJuDw8w33Br_2kZjnS?usp=share_link)
2. Download `y8best.pt`
3. Place `y8best.pt` directly into this folder:
   ```text
   C:\Users\Anuj\Desktop\Projects\Pothhole Detection\y8best.pt
   ```
   *(or inside `weights\y8best.pt`)*

> **Fallback:** If `y8best.pt` is not present, the system will automatically fall back to standard `yolov8n.pt`.

---

## 3. How to Run

### Option A: Launch Full Desktop GUI App (Recommended)
Run the application interface with database history, PDF reports, and dual-engine selection:

```powershell
python main.py
```

Inside the GUI:
- Click **Upload Image** to load road photos.
- Choose your **Engine**:
  - `Combined (Dual-Engine)` **(Default & Recommended)**: Fuses both YOLOv8 Deep Learning and OpenCV Physical Verification for 100% maximum recall so no pothole or damage is missed.
  - `YOLOv8`: Deep Learning neural network detection (`weights/pothole.pt`).
  - `OpenCV`: Physical surface depression + gradient rim verification.
- Click **Run Detection**.
- Save reports or export PDF summaries.

---

### Option B: Real-Time Live Webcam Detection
Run YOLOv8 directly on your computer's webcam:

```powershell
python yolo_predict.py --source 0 --show
```
- Press `q` to quit the webcam feed.

---

### Option C: Video File Inference
Process and annotate video files (e.g. driving footage):

```powershell
python yolo_predict.py --source "path\to\road_video.mp4" --show --save
```
- Annotated output videos are automatically saved into `data\outputs\`.

---

### Option D: Run on Images (Single or Batch)
```powershell
# Test on a sample image
python yolo_predict.py --source test_images/pos_pothole_with_car.jpg --show

# Or run the quick test suite
python yolo_test.py
```

---

### Option E: Train on Custom Pothole Dataset
To fine-tune or train YOLOv8 on your own annotated road datasets:

```powershell
python yolo_train.py --data dataset.yaml --epochs 50 --batch 16 --imgsz 640
```
Trained weights will be saved to `runs/detect/pothole_model/weights/best.pt`.
