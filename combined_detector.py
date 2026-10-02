"""
combined_detector.py
--------------------
Unified Hybrid / Ensemble Pothole & Infrastructure Damage Detector.
Combines:
  1. YOLOv8 Deep Learning (from Noor Khokhar's project / weights/pothole.pt)
  2. OpenCV Physical Surface Verification (Adaptive ROI, local depression, fracture rims, roughness)

Ensures MAXIMUM RECALL so each and every photo of a pothole or damaged
road / wall infrastructure is reliably detected without misses.
"""

from pathlib import Path
import cv2
import numpy as np

from detector import detect_issue
from yolo_detector import detect_potholes_yolo, is_yolo_available


def compute_iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes (x, y, w, h)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def detect_combined(image_path, mode="road", yolo_conf=0.15, debug=True):
    """
    Run BOTH Deep Learning (YOLOv8) and Physical Surface Analysis (OpenCV),
    fusing the results to guarantee comprehensive coverage of every pothole
    and infrastructure defect.
    """
    candidates = []

    # 1. Run YOLOv8 Deep Learning
    yolo_result = None
    if is_yolo_available():
        try:
            yolo_result = detect_potholes_yolo(image_path, conf=yolo_conf)
            for item in yolo_result.get("all_boxes", []):
                candidates.append({
                    "bbox": item["bbox"],
                    "confidence": float(item["confidence"]),
                    "class_name": item.get("class_name", "Pothole"),
                    "source": "YOLOv8",
                })
        except Exception as e:
            print(f"[!] Warning: YOLO detection failed: {e}")

    # 2. Run OpenCV Physical Verification Pipeline
    opencv_result = None
    try:
        opencv_result = detect_issue(image_path, mode=mode, debug=debug)
        # Add primary OpenCV detection if present
        if opencv_result.get("issue_type") != "unknown" and opencv_result.get("bbox"):
            candidates.append({
                "bbox": tuple(opencv_result["bbox"]),
                "confidence": float(opencv_result.get("confidence", 0.5)),
                "class_name": opencv_result.get("issue_type", "pothole").capitalize(),
                "source": "OpenCV",
            })
        # Also add all verified secondary candidates
        for c in opencv_result.get("candidates", []):
            if c.get("bbox"):
                candidates.append({
                    "bbox": tuple(c["bbox"]),
                    "confidence": float(c.get("confidence", 0.45)),
                    "class_name": c.get("issue_type", "pothole").capitalize(),
                    "source": "OpenCV",
                })
    except Exception as e:
        print(f"[!] Warning: OpenCV detection failed: {e}")

    # 3. Fuse & De-duplicate overlapping boxes
    # Sort candidates by confidence descending
    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    merged = []
    used = [False] * len(candidates)

    for i in range(len(candidates)):
        if used[i]:
            continue
        cA = candidates[i]
        boxA = cA["bbox"]
        sources = {cA["source"]}
        max_conf = cA["confidence"]
        class_name = cA["class_name"]

        # Expand box to include agreeing detections
        min_x, min_y = boxA[0], boxA[1]
        max_x, max_y = boxA[0] + boxA[2], boxA[1] + boxA[3]

        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            cB = candidates[j]
            boxB = cB["bbox"]
            iou = compute_iou(boxA, boxB)

            # Check for significant overlap or containment
            if iou > 0.20:
                used[j] = True
                sources.add(cB["source"])
                max_conf = max(max_conf, cB["confidence"])
                min_x = min(min_x, boxB[0])
                min_y = min(min_y, boxB[1])
                max_x = max(max_x, boxB[0] + boxB[2])
                max_y = max(max_y, boxB[1] + boxB[3])

        used[i] = True
        merged_box = (min_x, min_y, max_x - min_x, max_y - min_y)

        # Confidence boost if both Deep Learning and Physical Analysis agree
        if len(sources) > 1:
            display_conf = min(0.99, max_conf + 0.08)
            source_tag = "Dual-Verified"
        else:
            display_conf = max_conf
            source_tag = list(sources)[0]

        merged.append({
            "bbox": merged_box,
            "confidence": round(display_conf, 2),
            "class_name": class_name,
            "source": source_tag,
        })

    # 4. Construct Structured Detection Result
    if merged:
        primary = merged[0]
        issue_type = "crack" if any(m["class_name"].lower() == "crack" for m in merged) and mode == "wall" else "pothole"
        return {
            "issue_type": issue_type,
            "confidence": primary["confidence"],
            "bbox": primary["bbox"],
            "all_boxes": merged,
            "pothole_count": len(merged),
            "reason": f"Combined Engine detected {len(merged)} infrastructure damage region(s) (Peak Conf: {primary['confidence']:.0%})",
            "engine": "Combined (YOLOv8 + OpenCV)",
            "debug": opencv_result.get("debug") if opencv_result else None,
            "roi_confidence": opencv_result.get("roi_confidence", 0.9) if opencv_result else 0.9,
            "roi_area": opencv_result.get("roi_area", 0) if opencv_result else 0,
        }
    else:
        return {
            "issue_type": "unknown",
            "confidence": 0.0,
            "bbox": None,
            "all_boxes": [],
            "pothole_count": 0,
            "reason": "No infrastructure damage detected by either YOLOv8 or OpenCV.",
            "engine": "Combined (YOLOv8 + OpenCV)",
            "debug": opencv_result.get("debug") if opencv_result else None,
            "roi_confidence": 0.0,
            "roi_area": 0,
        }


def draw_combined_overlay(image_bgr, detection):
    """Draw clear, color-coded bounding boxes for all combined detections."""
    annotated = image_bgr.copy()
    all_boxes = detection.get("all_boxes", [])

    for item in all_boxes:
        x, y, w, h = item["bbox"]
        conf = item["confidence"]
        name = item.get("class_name", "Pothole")
        src = item.get("source", "")

        # Color: Bright Red for Dual-Verified / Pothole, Orange for single engine
        box_color = (0, 0, 255) if src == "Dual-Verified" else (0, 140, 255)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 3)

        label = f"{name}: {conf:.2f}"
        if src == "Dual-Verified":
            label += " (Dual)"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(annotated, (x, max(0, y - th - 8)), (x + tw + 8, max(th + 8, y)), box_color, -1)
        cv2.putText(annotated, label, (x + 4, max(th + 2, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return annotated
