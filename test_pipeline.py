"""
test_pipeline.py
----------------
Automated validation suite for Snap2Action_CV detection and localization.

Verifies:
1. True Positives: Pothole on road (with car present) is localized; car is ignored.
2. True Positives: Wall crack is localized; brick pattern is ignored.
3. False Positive Rejection: Shadows, road markings, clean road, clean wall,
   and windows are correctly classified as UNKNOWN.
4. Database insertion & statistics retrieval.
5. PDF report generation.
"""

import sys
from pathlib import Path
import make_test_images
from detector import detect_issue
from severity import estimate_severity
import database as db
from report_generator import generate_pdf_report


def run_tests():
    print("=" * 70)
    print(" SNAP2ACTION_CV — AUTOMATED DETECTION & LOCALIZATION TEST SUITE")
    print("=" * 70)

    # 1. Generate test images
    make_test_images.generate_all()
    test_dir = Path(__file__).parent / "test_images"

    test_cases = [
        {
            "name": "Pothole with Car on Road",
            "file": "pos_pothole_with_car.jpg",
            "mode": "road",
            "expected_type": "pothole",
            "bbox_check": lambda b: b is not None and b[0] >= 420 and b[1] >= 350,  # Pothole is in right half
            "disallow_car": lambda b: b is not None and (b[0] < 300 or b[2] < 300), # Car is at x: 70..270
            "description": "Must detect pothole in right half of road; CAR MUST BE IGNORED",
        },
        {
            "name": "Multiple Potholes on Road",
            "file": "pos_multiple_potholes.jpg",
            "mode": "road",
            "expected_type": "pothole",
            "bbox_check": lambda b: b is not None,
            "description": "Must detect potholes and expose candidate regions",
        },
        {
            "name": "Road with Soft Shadow Only",
            "file": "neg_shadow_only.jpg",
            "mode": "road",
            "expected_type": "unknown",
            "description": "Must NOT detect shadow as pothole (penumbra & texture check)",
        },
        {
            "name": "Road with Lane Markings",
            "file": "neg_road_marking.jpg",
            "mode": "road",
            "expected_type": "unknown",
            "description": "Must NOT detect bright painted lines / markings as damage",
        },
        {
            "name": "Clean Normal Road",
            "file": "neg_normal_road.jpg",
            "mode": "road",
            "expected_type": "unknown",
            "description": "Must return unknown on clean road surface",
        },
        {
            "name": "Wall with Actual Crack",
            "file": "pos_wall_crack.jpg",
            "mode": "wall",
            "expected_type": "crack",
            "bbox_check": lambda b: b is not None,
            "description": "Must localize meandering fracture on wall",
        },
        {
            "name": "Wall with Window and Pipe",
            "file": "neg_wall_window.jpg",
            "mode": "wall",
            "expected_type": "unknown",
            "description": "Must NOT detect window frame or straight pipe as crack",
        },
        {
            "name": "Clean Normal Brick Wall",
            "file": "neg_normal_wall.jpg",
            "mode": "wall",
            "expected_type": "unknown",
            "description": "Must NOT detect regular brick mortar grid as cracks",
        },
    ]

    all_passed = True
    results_summary = []

    print("\n--- Running Detection Pipeline Tests ---")
    for tc in test_cases:
        img_path = str(test_dir / tc["file"])
        res = detect_issue(img_path, mode=tc["mode"], debug=False)
        sev = estimate_severity(res, res.get("roi_area", 0))

        actual_type = res.get("issue_type")
        conf = res.get("confidence", 0.0)
        bbox = res.get("bbox")
        sev_label = sev.get("severity", "Unknown")

        passed = (actual_type == tc["expected_type"])

        # Check localization constraints
        if passed and "bbox_check" in tc:
            if not tc["bbox_check"](bbox):
                passed = False
        if passed and "disallow_car" in tc:
            if tc["disallow_car"](bbox):
                print(f"  [!] FAILED: Detected the CAR region {bbox} instead of the pothole!")
                passed = False

        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        results_summary.append({
            "name": tc["name"],
            "expected": tc["expected_type"],
            "actual": actual_type,
            "confidence": f"{conf:.0%}",
            "severity": sev_label,
            "status": status_str,
            "bbox": str(bbox) if bbox else "None",
        })

        print(f"[{status_str}] {tc['name']}")
        print(f"       Expected: {tc['expected_type']} | Actual: {actual_type} ({conf:.0%}) | Severity: {sev_label}")
        if bbox:
            print(f"       Localized BBox: {bbox}")
        if res.get("reason"):
            print(f"       Reason: {res.get('reason')}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Test Case':<32} {'Expected':<10} {'Actual':<10} {'Conf':<8} {'Severity':<10} {'Status'}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['name']:<32} {r['expected']:<10} {r['actual']:<10} {r['confidence']:<8} {r['severity']:<10} {r['status']}")
    print("=" * 80)

    # 2. Test Database Integration
    print("\n--- Testing Database & Report Integration ---")
    db.init_db()
    sample_res = detect_issue(str(test_dir / "pos_pothole_with_car.jpg"), mode="road")
    sample_sev = estimate_severity(sample_res, sample_res.get("roi_area", 0))
    report_id = db.insert_report(
        str(test_dir / "pos_pothole_with_car.jpg"),
        None, "road", sample_res, sample_sev, notes="Test automated report"
    )
    print(f"✓ Inserted test report #{report_id} into SQLite database")

    history = db.fetch_history(limit=5)
    assert len(history) > 0, "Failed to retrieve history"
    print(f"✓ Retrieved {len(history)} record(s) from history")

    stats = db.dashboard_stats()
    print(f"✓ Dashboard stats: Total={stats['total']}, ByType={stats['by_type']}, BySev={stats['by_severity']}")

    # 3. Test PDF Report Generation
    test_pdf = Path(__file__).parent / "data" / "reports" / f"report_{report_id}.pdf"
    pdf_out = generate_pdf_report(
        db.fetch_report(report_id),
        output_path=str(test_pdf)
    )
    assert Path(pdf_out).exists(), f"PDF report file was not created at {pdf_out}"
    print(f"✓ Generated PDF report successfully at {pdf_out} ({Path(pdf_out).stat().st_size} bytes)")

    print("\n" + "=" * 70)
    if all_passed:
        print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (100% Accuracy)")
    else:
        print(" SOME TESTS FAILED — Review the output above.")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
