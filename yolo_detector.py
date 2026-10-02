"""
yolo_detector.py
----------------
Bridge module integrating YOLOv8 Deep Learning Pothole & Infrastructure
Damage Detection with the Snap2Action desktop application and database.

Strictly filters for civic infrastructure damage (potholes, road cracks)
and ignores non-infrastructure objects (vehicles, pedestrians, signs, clutter).
"""

from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

from download_weights import get_weights_path

_LOADED_MODEL = None
_LOADED_MODEL_PATH = None

# Permitted civic infrastructure damage keywords
INFRASTRUCTURE_KEYWORDS = [
    "pothole", "potholes", "crack", "cracks", "road damage", "road defect",
    "defect", "manhole", "depression", "rutting", "alligator", "garbage",
    "0", 0, "damage"
]


def is_infrastructure_class(name, num_classes=1):
    """Verify if detection label belongs to road / civic infrastructure damage."""
    # If the custom model only has 1 class (dedicated pothole detector like pothole.pt),
    # all its detections are genuine road defects regardless of whether the label is '0' or 'pothole'.
    if num_classes == 1:
        return True
    clean = str(name).lower().strip()
    return any(kw in clean for kw in INFRASTRUCTURE_KEYWORDS)


def normalize_damage_label(raw_name):
    """Normalize label strictly to 'Pothole' or 'Crack'."""
    clean = str(raw_name).lower().strip()
    if "crack" in clean:
        return "Crack"
    return "Pothole"


def is_yolo_available():
    return _YOLO_AVAILABLE


def get_yolo_model(force_reload=False):
    global _LOADED_MODEL, _LOADED_MODEL_PATH
    if not _YOLO_AVAILABLE:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

    target_path = get_weights_path()
    if force_reload or _LOADED_MODEL is None or _LOADED_MODEL_PATH != target_path:
        _LOADED_MODEL = YOLO(target_path)
        _LOADED_MODEL_PATH = target_path
    return _LOADED_MODEL


def detect_potholes_yolo(image_path, conf=0.25):
    """
    Run YOLOv8 inference and strictly extract pothole & infrastructure damage.
    Non-infrastructure objects (cars, pedestrians, etc.) are discarded.
    """
    if not _YOLO_AVAILABLE:
        return {
            "issue_type": "unknown",
            "confidence": 0.0,
            "bbox": None,
            "all_boxes": [],
            "pothole_count": 0,
            "reason": "Ultralytics package not installed. Run 'pip install ultralytics'.",
            "engine": "YOLOv8",
        }

    model = get_yolo_model()
    results = model.predict(source=str(image_path), conf=conf, verbose=False)

    all_boxes = []
    max_conf = 0.0
    primary_bbox = None

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            score = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            raw_cls_name = model.names.get(cls_id, "pothole")

            # STRICT FILTER: Discard anything not road/infrastructure damage
            if not is_infrastructure_class(raw_cls_name, num_classes=len(model.names)):
                continue

            cls_name = normalize_damage_label(raw_cls_name)

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            bbox = (x1, y1, w, h)

            all_boxes.append({
                "bbox": bbox,
                "confidence": score,
                "class_name": cls_name,
                "raw_class_name": raw_cls_name,
            })

            if score > max_conf:
                max_conf = score
                primary_bbox = bbox

    pothole_count = len(all_boxes)
    weights_filename = Path(_LOADED_MODEL_PATH).name

    if pothole_count > 0:
        return {
            "issue_type": "pothole",
            "confidence": round(max_conf, 3),
            "bbox": primary_bbox,
            "all_boxes": all_boxes,
            "pothole_count": pothole_count,
            "reason": f"Detected {pothole_count} road infrastructure damage region(s) (peak conf: {max_conf:.2f})",
            "engine": "YOLOv8 Deep Learning",
            "weights": weights_filename,
        }
    else:
        return {
            "issue_type": "unknown",
            "confidence": 0.0,
            "bbox": None,
            "all_boxes": [],
            "pothole_count": 0,
            "reason": "No potholes or infrastructure damage detected above threshold.",
            "engine": "YOLOv8 Deep Learning",
            "weights": weights_filename,
        }


def draw_yolo_overlay(image_bgr, detection):
    """Draw bounding boxes and labels for infrastructure damage detections."""
    annotated = image_bgr.copy()
    all_boxes = detection.get("all_boxes", [])

    for item in all_boxes:
        bbox = item["bbox"]
        conf = item["confidence"]
        name = item.get("class_name", "Pothole")

        x, y, w, h = bbox
        # Draw high-visibility red box around damage
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

        label = f"{name}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x, max(0, y - th - 6)), (x + tw + 6, max(th + 6, y)), (0, 0, 255), -1)
        cv2.putText(annotated, label, (x + 3, max(th + 2, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated
