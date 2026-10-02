"""
yolo_test.py
------------
Quick test script to verify YOLOv8 pothole detection on local test images.
Replaces the incomplete test.py from the original repository.
"""

import sys
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] 'ultralytics' is not installed. Run: pip install ultralytics")
    sys.exit(1)

from download_weights import get_weights_path, check_and_instruct


def run_test():
    check_and_instruct()
    weights = get_weights_path()
    print(f"[*] Testing YOLOv8 model loading from: {weights}")
    model = YOLO(weights)

    test_dir = Path(__file__).parent / "test_images"
    test_images = list(test_dir.glob("*.jpg"))

    if not test_images or not (test_dir / "road_pothole_city.jpg").exists():
        print("[*] Updating test suite with realistic test images...")
        import make_test_images
        make_test_images.generate_all()
        test_images = list(test_dir.glob("*.jpg"))

    # Prioritize realistic road photos first in test run
    test_images.sort(key=lambda p: (0 if "pothole" in p.name.lower() else 1, p.name))
    print(f"[*] Found {len(test_images)} test images. Running evaluation...")

    from yolo_detector import is_infrastructure_class, normalize_damage_label

    for img_path in test_images[:5]:
        print(f"\n--- Testing: {img_path.name} ---")
        results = model.predict(source=str(img_path), conf=0.25, verbose=False)
        infra_detections = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    raw_name = model.names.get(cls_id, "pothole")
                    if not is_infrastructure_class(raw_name, num_classes=len(model.names)):
                        continue
                    clean_name = normalize_damage_label(raw_name)
                    conf = float(box.conf[0].item())
                    xyxy = [round(x, 1) for x in box.xyxy[0].tolist()]
                    infra_detections.append((clean_name, conf, xyxy))

        if infra_detections:
            print(f"    [+] Infrastructure Damage: {len(infra_detections)} region(s)")
            for name, conf, xyxy in infra_detections:
                print(f"        -> {name} | Confidence: {conf:.2f} | BBox: {xyxy}")
        else:
            print("    [-] No infrastructure damage detected with confidence >= 0.25.")

    print("\n[+] YOLOv8 test completed successfully!")


if __name__ == "__main__":
    run_test()
