"""
yolo_predict.py
---------------
Real-time and batch YOLOv8 Pothole & Infrastructure Damage inference script.
Supports images, videos, and live webcam feeds.
Strictly detects potholes and infrastructure damage, ignoring vehicles,
pedestrians, and other non-infrastructure objects.

Usage:
    # 1. Run on a video file
    python yolo_predict.py --source tested.mp4 --show --save

    # 2. Run on a single image
    python yolo_predict.py --source test_images/road_pothole_city.jpg --show

    # 3. Run on live webcam
    python yolo_predict.py --source 0 --show
"""

import argparse
import sys
import time
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] 'ultralytics' library is not installed.")
    print("Please install it using: pip install ultralytics")
    sys.exit(1)

from download_weights import get_weights_path, check_and_instruct
from yolo_detector import is_infrastructure_class, normalize_damage_label


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Pothole & Infrastructure Damage Inference")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLOv8 weights (e.g. weights/pothole.pt)")
    parser.add_argument("--source", type=str, default="0", help="Video file, image path, or webcam index (0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence detection threshold (0.0 - 1.0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640)")
    parser.add_argument("--show", action="store_true", default=True, help="Display window with real-time detections")
    parser.add_argument("--save", action="store_true", default=False, help="Save annotated output to disk")
    parser.add_argument("--output-dir", type=str, default="data/outputs", help="Directory to save output results")
    return parser.parse_args()


def run_inference(model_path, source, conf=0.25, imgsz=640, show=True, save=False, output_dir="data/outputs"):
    if model_path is None or not Path(model_path).exists():
        model_path = get_weights_path()

    print(f"[*] Loading model from: {model_path}")
    model = YOLO(model_path)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Convert numeric source string (e.g. '0') to integer for webcam
    source_val = int(source) if source.isdigit() else source

    print(f"[*] Starting inference on source: {source}")
    print(f"[*] Infrastructure damage confidence threshold: {conf}")
    print("Press 'q' in the display window to exit early.")

    results_generator = model.predict(
        source=source_val,
        conf=conf,
        imgsz=imgsz,
        stream=True,
        show=False,
    )

    writer = None
    save_filename = None

    if save:
        timestamp = int(time.time())
        save_filename = out_path / f"pothole_result_{timestamp}.mp4"

    prev_time = time.time()
    total_frames = 0

    for result in results_generator:
        total_frames += 1
        frame = result.orig_img.copy()

        # Strict infrastructure filtering
        infra_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                score = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                raw_name = model.names.get(cls_id, "pothole")

                if not is_infrastructure_class(raw_name, num_classes=len(model.names)):
                    continue

                label_name = normalize_damage_label(raw_name)
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                infra_boxes.append((xyxy, score, label_name))

        pothole_count = len(infra_boxes)

        # Draw only infrastructure damage detections
        annotated_frame = frame.copy()
        for (x1, y1, x2, y2), score, label_name in infra_boxes:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            tag = f"{label_name}: {score:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), (0, 0, 255), -1)
            cv2.putText(annotated_frame, tag, (x1 + 3, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Overlay statistics: Pothole count and FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            annotated_frame,
            f"Potholes / Road Defects: {pothole_count} | FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255) if pothole_count > 0 else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Video saving handler
        if save and writer is None and isinstance(source_val, (int, str)):
            h, w = annotated_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(save_filename), fourcc, 25.0, (w, h))

        if writer is not None:
            writer.write(annotated_frame)

        if show:
            cv2.imshow("Pothole & Infrastructure Damage Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[*] Inference stopped by user.")
                break

    if writer is not None:
        writer.release()
        print(f"[+] Output video saved to: {save_filename}")

    cv2.destroyAllWindows()
    print(f"[+] Completed processing {total_frames} frames.")


def main():
    args = parse_args()
    check_and_instruct()
    run_inference(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        show=args.show,
        save=args.save,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
