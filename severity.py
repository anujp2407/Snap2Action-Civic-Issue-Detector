"""
severity.py
-----------
Severity estimation. Deliberately kept separate from detection: detection
answers "what/where is the damage", severity answers "how bad is it",
using the already-localized region rather than raw full-image contour
counts.
"""


def estimate_severity(detection, roi_area_px):
    """
    detection: the primary detection dict returned by detector.detect_issue
               (must have issue_type, area, bbox, confidence)
    roi_area_px: area (in pixels, original image scale) of the ROI the
                 detection was found within. Pass None/0 if unknown.

    Returns: {"severity": "Low"|"Medium"|"High"|"Unknown", "score": float}
    """
    if detection.get("issue_type") in (None, "unknown") or not detection.get("area"):
        return {"severity": "Unknown", "score": 0.0}

    area = detection["area"]
    issue_type = detection["issue_type"]
    confidence = detection.get("confidence", 0.5)
    bbox = detection.get("bbox") or [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)

    feats = detection.get("features", {})
    surface_contrast = feats.get("surface_score", 0.5)
    edge_gradient = feats.get("edge_gradient_score", 0.5)

    if issue_type == "pothole":
        area_ratio = (area / roi_area_px) if roi_area_px and roi_area_px > 0 else 0.0
        # Weight area ratio, crater contrast depth, and edge severity
        area_score = min(1.0, area_ratio * 8.5)
        depth_score = min(1.0, surface_contrast * 1.2)
        score = (0.60 * area_score + 0.20 * depth_score + 0.10 * edge_gradient + 0.10 * confidence)
    elif issue_type == "crack":
        # Use bbox diagonal as a proxy for crack length, and the shorter
        # side as a proxy for width
        length = (w ** 2 + h ** 2) ** 0.5
        width = min(w, h)
        length_ratio = length / (roi_area_px ** 0.5) if roi_area_px and roi_area_px > 0 else 0.0
        width_score = min(1.0, width / 20.0)
        length_score = min(1.0, length_ratio * 1.5)
        score = (0.55 * length_score + 0.25 * width_score + 0.10 * edge_gradient + 0.10 * confidence)
    else:
        score = confidence

    score = float(max(0.0, min(1.0, score)))

    if score >= 0.62:
        label = "High"
    elif score >= 0.32:
        label = "Medium"
    else:
        label = "Low"

    return {"severity": label, "score": round(score, 3)}
