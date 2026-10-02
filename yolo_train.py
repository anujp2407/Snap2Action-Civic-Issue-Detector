"""
yolo_train.py
-------------
Train / fine-tune YOLOv8 on a custom pothole dataset.
Replaces the outdated train.py / val.py from the original repository with
the modern, stable Ultralytics YOLOv8 Python API.

Usage:
    python yolo_train.py --data data.yaml --epochs 50 --batch 16 --imgsz 640
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] 'ultralytics' is not installed. Run: pip install ultralytics")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Pothole Dataset")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to dataset yaml configuration")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base pretrained weights (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--device", type=str, default="", help="Device: '0' or '0,1' for GPU, or 'cpu'")
    parser.add_argument("--name", type=str, default="pothole_model", help="Experiment name")
    return parser.parse_args()


def train():
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"[!] Dataset config file '{args.data}' not found.")
        print("Please provide a valid dataset YAML file (e.g., exported from Roboflow or annotated dataset).")
        print("\nExample data.yaml structure:")
        print("---------------------------------")
        print("path: ./dataset")
        print("train: images/train")
        print("val: images/val")
        print("names:")
        print("  0: pothole")
        print("---------------------------------")
        return

    print(f"[*] Starting YOLOv8 training using base model: {args.model}")
    model = YOLO(args.model)

    device = args.device if args.device else (0 if torch_available_gpu() else "cpu")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        name=args.name,
        save=True,
    )

    print("\n[*] Evaluating on validation split...")
    metrics = model.val()
    print(f"[+] mAP50-95: {metrics.box.map:.4f}")
    print(f"[+] Best weights saved to: runs/detect/{args.name}/weights/best.pt")


def torch_available_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    train()
