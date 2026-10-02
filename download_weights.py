"""
download_weights.py
-------------------
Helper script to verify and obtain YOLOv8 pothole weights for the project.

Notes on 'y8best.pt' vs 'pothole.pt':
- The 'y8best.pt' file in Noor Khokhar's Google Drive is actually trained on
  GARBAGE / WASTE OVERFLOW (classes: ['garbage']). It mistakenly classifies broken
  asphalt inside potholes as loose garbage/trash!
- To get accurate pothole bounding boxes labeled as 'pothole', this script can
  automatically download a dedicated YOLOv8 Pothole model ('pothole.pt') trained
  specifically on road damage datasets.
"""

import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

POTHOLE_WEIGHTS_URL = "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
POTHOLE_TARGET = WEIGHTS_DIR / "pothole.pt"

KHOKHAR_WEIGHTS = [
    PROJECT_ROOT / "y8best.pt",
    WEIGHTS_DIR / "y8best.pt",
]

GDRIVE_LINK = "https://drive.google.com/drive/folders/1T1h2WH3_wrNb0foJuDw8w33Br_2kZjnS?usp=share_link"


def download_genuine_pothole_model(force=False):
    """Downloads a dedicated YOLOv8 pothole detection model from Hugging Face."""
    if POTHOLE_TARGET.exists() and POTHOLE_TARGET.stat().st_size > 1000 and not force:
        return str(POTHOLE_TARGET)

    print(f"[*] Downloading dedicated YOLOv8 Pothole weights from Hugging Face...")
    print(f"    Source: {POTHOLE_WEIGHTS_URL}")
    print(f"    Destination: {POTHOLE_TARGET}")
    try:
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = f"\r--> {percent:5.1f}% [{readsofar / (1024*1024):.1f}MB / {totalsize / (1024*1024):.1f}MB]"
                sys.stdout.write(s)
                sys.stdout.flush()

        urllib.request.urlretrieve(POTHOLE_WEIGHTS_URL, str(POTHOLE_TARGET), reporthook=reporthook)
        print("\n[+] Successfully downloaded genuine pothole weights!")
        return str(POTHOLE_TARGET)
    except Exception as e:
        print(f"\n[!] Failed to download from Hugging Face: {e}")
        return None


def get_weights_path():
    """
    Returns best available weights:
    1. weights/pothole.pt (Dedicated pothole model)
    2. y8best.pt (Khokhar's model)
    3. yolov8n.pt (COCO base fallback)
    """
    # 1. Check if dedicated pothole model exists
    if POTHOLE_TARGET.exists() and POTHOLE_TARGET.stat().st_size > 1000:
        return str(POTHOLE_TARGET)

    # 2. Check if Khokhar's weights exist
    for p in KHOKHAR_WEIGHTS:
        if p.exists() and p.stat().st_size > 1000:
            return str(p)

    # 3. Check fallback
    fallback = PROJECT_ROOT / "yolov8n.pt"
    if fallback.exists():
        return str(fallback)

    return "yolov8n.pt"


def check_and_instruct():
    weights_path = get_weights_path()
    print("=" * 70)
    print(" YOLOv8 Pothole Detection — Model Weights Inspector")
    print("=" * 70)

    if Path(weights_path).name == "pothole.pt":
        print(f"[OK] Using dedicated Pothole Model: {weights_path}")
        print("     Classes: ['pothole']")
    elif Path(weights_path).name == "y8best.pt":
        print(f"[CAUTION] Currently using '{weights_path}' (from Noor Khokhar's Google Drive).")
        print("          This file was trained on the 'garbage' dataset (classes: ['garbage']).")
        print("          It detects the broken rocks inside potholes as 'GARBAGE'.")
        print("")
        print("To switch to a true Pothole model with class 'pothole':")
        print("Run: python download_weights.py --download-pothole")
    else:
        print(f"[NOTICE] Using fallback: {weights_path}")
        print("Run: python download_weights.py --download-pothole")

    print("=" * 70)
    return weights_path


if __name__ == "__main__":
    if "--download-pothole" in sys.argv:
        download_genuine_pothole_model(force=True)
    else:
        check_and_instruct()
