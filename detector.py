"""
detector.py
-----------
Structured detection pipeline for Snap2Action_CV.

    INPUT IMAGE
        -> preprocessing (image_processor)
        -> adaptive surface ROI (image_processor)
        -> candidate generation (blob depressions & thin fractures)
        -> per-candidate feature extraction (contrast, edge sharpness, texture,
           shape irregularity, chroma, tortuosity)
        -> rigorous false-positive rejection (shadows, vehicles, tyres, road
           markings, manholes, wires, pipes, windows, mortar lines)
        -> classification (pothole / crack / unknown)
        -> exact bbox + contour localization
        -> severity estimation (severity.py)
        -> structured result

Key principle: A contour alone is NEVER sufficient evidence for damage.
Every candidate must be verified across multiple independent physical
criteria. If no candidate clears the confidence bar, the system reports
'unknown' with a clear diagnostic explanation.
"""

import cv2
import numpy as np

from image_processor import (
    load_image,
    resize_for_processing,
    preprocess,
    local_texture_map,
    gradient_orientation_map,
    estimate_surface_roi,
)

CONFIDENCE_THRESHOLD = 0.48          # primary detection must reach this
SECONDARY_THRESHOLD = 0.32           # secondary candidates kept for analysis


# --------------------------------------------------------------------------
# Candidate Generation
# --------------------------------------------------------------------------

def _blob_candidates(gray, roi_mask):
    """Generate candidate pothole depression regions inside the surface ROI.
    Uses wide-kernel local background subtraction with a physical contrast floor."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    local_baseline = cv2.blur(blurred.astype(np.float32), (65, 65))
    diff = local_baseline - blurred.astype(np.float32)

    dark_mask = (diff >= 14.0).astype(np.uint8) * 255
    dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=roi_mask)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= 100]


def _crack_candidates(gray, roi_mask):
    """Generate candidate thin elongated crack structures using blackhat
    morphology with noise filtering and gap bridging."""
    denoised = cv2.medianBlur(gray, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.bitwise_and(blackhat, blackhat, mask=roi_mask)

    crack_mask = (blackhat >= 24).astype(np.uint8) * 255

    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bridged = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, bridge_kernel, iterations=2)
    bridged = cv2.dilate(bridged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= 60]


# --------------------------------------------------------------------------
# Multi-Feature Extraction
# --------------------------------------------------------------------------

def _location_score(contour, roi_mask):
    """Fraction of candidate area that lies within the verified surface ROI."""
    mask = np.zeros(roi_mask.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    inside = cv2.bitwise_and(mask, roi_mask)
    total = np.count_nonzero(mask)
    return float(np.count_nonzero(inside) / total) if total > 0 else 0.0


def _shape_features(contour):
    """Geometric properties of the candidate contour."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / max(1, min(w, h))
    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = (area / hull_area) if hull_area > 0 else 0.0

    if len(contour) >= 5:
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        rotated_ar = max(rw, rh) / max(1.0, min(rw, rh))
    else:
        rotated_ar = aspect_ratio

    thinness = (perimeter ** 2) / (4.0 * np.pi * area) if area > 0 else 0.0

    tortuosity = 1.0
    is_axis_aligned = False
    if len(contour) >= 6:
        pts = contour.reshape(-1, 2)
        dist_mat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        max_dist = np.max(dist_mat)
        if max_dist > 5:
            tortuosity = (perimeter * 0.5) / max_dist

        if len(pts) >= 5:
            [vx, vy, _, _] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.degrees(np.arctan2(abs(float(vy)), abs(float(vx))))
            is_axis_aligned = (angle < 7.0) or (angle > 83.0)

    return {
        "area": area,
        "perimeter": perimeter,
        "bbox": (int(x), int(y), int(w), int(h)),
        "aspect_ratio": max(aspect_ratio, rotated_ar),
        "circularity": circularity,
        "solidity": solidity,
        "thinness": thinness,
        "tortuosity": tortuosity,
        "is_axis_aligned": is_axis_aligned,
    }


def _surface_contrast_score(gray, contour, bbox):
    """Compare candidate interior intensity against a surrounding surface ring."""
    x, y, w, h = bbox
    pad = max(8, int(0.35 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)

    inner_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(inner_mask, [contour], -1, 255, thickness=-1)

    outer_mask = np.zeros(gray.shape, dtype=np.uint8)
    outer_mask[y0:y1, x0:x1] = 255
    ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

    inner_vals = gray[inner_mask == 255]
    ring_vals = gray[ring_mask == 255]
    if inner_vals.size == 0 or ring_vals.size == 0:
        return 0.0, 128.0, 128.0

    inner_mean = float(np.mean(inner_vals))
    ring_mean = float(np.mean(ring_vals))

    depression_depth = ring_mean - inner_mean
    if depression_depth <= 0:
        return 0.0, inner_mean, ring_mean

    score = float(np.clip(depression_depth / 45.0, 0.0, 1.0))
    return score, inner_mean, ring_mean


def _edge_gradient_sharpness(gray, contour):
    """Physical rim sharpness discriminator:
    Samples the gradient magnitude across a narrow transition band around the contour.
    A true crater edge produces a sharp gradient ridge (ridge_grad >= 25);
    A soft shadow penumbra has no sharp ridge anywhere (ridge_grad < 14)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    boundary_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(boundary_mask, [contour], -1, 255, thickness=5)

    boundary_grads = mag[boundary_mask == 255]
    if boundary_grads.size == 0:
        return 0.0, 0.0

    # Strongest 40% of gradient samples in the transition band
    if len(boundary_grads) > 10:
        top_k = max(5, int(len(boundary_grads) * 0.40))
        ridge_grad = float(np.mean(np.partition(boundary_grads, -top_k)[-top_k:]))
    else:
        ridge_grad = float(np.mean(boundary_grads))

    sharpness_score = float(np.clip((ridge_grad - 14.0) / 22.0, 0.0, 1.0))
    return sharpness_score, ridge_grad


def _texture_score(texture_map, contour, bbox):
    """Compare local standard deviation (roughness) inside vs outside the candidate."""
    x, y, w, h = bbox
    pad = max(8, int(0.35 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(texture_map.shape[1], x + w + pad), min(texture_map.shape[0], y + h + pad)

    inner_mask = np.zeros(texture_map.shape, dtype=np.uint8)
    cv2.drawContours(inner_mask, [contour], -1, 255, thickness=-1)

    outer_mask = np.zeros(texture_map.shape, dtype=np.uint8)
    outer_mask[y0:y1, x0:x1] = 255
    ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))

    inner_vals = texture_map[inner_mask == 255]
    ring_vals = texture_map[ring_mask == 255]
    if inner_vals.size == 0 or ring_vals.size == 0:
        return 0.0

    roughness_diff = float(np.mean(inner_vals)) - float(np.mean(ring_vals))
    return float(np.clip(roughness_diff / 10.0, 0.0, 1.0))


def _color_features(image, contour):
    """Sample color and chroma inside candidate to reject vehicles and paint."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat_vals = hsv[:, :, 1][mask == 255]
    val_vals = hsv[:, :, 2][mask == 255]

    b, g, r = cv2.split(image)
    max_c = np.maximum(np.maximum(b, g), r)[mask == 255]
    min_c = np.minimum(np.minimum(b, g), r)[mask == 255]
    chroma_vals = max_c - min_c

    if sat_vals.size == 0:
        return 0.0, 128.0, 0.0

    return float(np.mean(sat_vals)), float(np.mean(val_vals)), float(np.mean(chroma_vals))


# --------------------------------------------------------------------------
# Rigorous False-Positive Penalties
# --------------------------------------------------------------------------

def _false_positive_penalty(shape_feats, contrast_depth, mean_grad, inner_sat, inner_val, inner_chroma, kind):
    """Explicit penalties for non-damage visual structures."""
    penalty = 0.0
    ar = shape_feats["aspect_ratio"]
    solidity = shape_feats["solidity"]
    circularity = shape_feats["circularity"]
    tortuosity = shape_feats.get("tortuosity", 1.0)
    thinness = shape_feats.get("thinness", 1.0)
    is_axis_aligned = shape_feats.get("is_axis_aligned", False)

    if kind == "pothole":
        # 1. SHADOW: soft penumbra with low boundary gradient ridge
        if mean_grad < 14.0:
            penalty += (14.0 - mean_grad) / 14.0 * 0.55

        # 2. VEHICLE PAINT / COLORED OBJECT: significant chroma
        if inner_chroma > 35.0:
            penalty += min(0.70, (inner_chroma - 30.0) / 30.0 * 0.70)

        # 3. MANHOLE / PERFECT CIRCLE: geometrically clean manufactured disk
        if circularity > 0.92 and solidity > 0.96:
            penalty += 0.60

        # 4. ROAD MARKING / RECTANGULAR STRIPE
        x, y, w, h = shape_feats["bbox"]
        box_fill = shape_feats["area"] / float(max(1, w * h))
        if inner_val > 175.0 or (box_fill > 0.88 and solidity > 0.94 and circularity < 0.50):
            penalty += 0.75

        # 5. EXTREME ELONGATION
        if ar > 5.0:
            penalty += 0.45

        # 6. EXTREME FRAGMENTATION (gravel noise)
        if solidity < 0.20:
            penalty += 0.35

    else:  # kind == "crack"
        # 1. NON-CRACK / COMPACT BLOBS
        if thinness < 2.8 or circularity > 0.22:
            penalty += 0.75

        # 2. STRAIGHT WIRES / PIPES (lack meandering irregularity)
        if tortuosity < 1.15:
            penalty += 0.55

        # 3. AXIS-ALIGNED LINES (mortar joints, window sills, vertical pipes)
        if is_axis_aligned:
            penalty += 0.50

        # 4. WEAK CONTRAST (shallow surface texture)
        if contrast_depth < 25.0:
            penalty += 0.45

        # 5. BRIGHT LINES
        if inner_val > 185.0:
            penalty += 0.70

    return float(np.clip(penalty, 0.0, 0.85))


# --------------------------------------------------------------------------
# Candidate Evaluation
# --------------------------------------------------------------------------

def _score_candidate(contour, gray, texture_map, image_bgr, roi_mask, roi_area, kind):
    shape_feats = _shape_features(contour)
    if shape_feats["area"] <= 0:
        return None

    if kind == "crack":
        if shape_feats["circularity"] > 0.22:
            return None
        x, y, w, h = shape_feats["bbox"]
        if max(w, h) < 65:
            return None

    location = _location_score(contour, roi_mask)
    if location < 0.60:
        return None

    frac_area = shape_feats["area"] / roi_area if roi_area > 0 else 0.0
    if frac_area < 0.0008 or frac_area > 0.35:
        return None

    ideal_size = 0.03
    size_score = float(np.clip(1.0 - abs(frac_area - ideal_size) / 0.32, 0.0, 1.0))

    contrast_score, inner_mean, ring_mean = _surface_contrast_score(gray, contour, shape_feats["bbox"])
    depression_depth = ring_mean - inner_mean
    if kind == "pothole" and contrast_score <= 0.05:
        return None

    sharpness_score, mean_grad = _edge_gradient_sharpness(gray, contour)
    texture = _texture_score(texture_map, contour, shape_feats["bbox"])
    inner_sat, inner_val, inner_chroma = _color_features(image_bgr, contour)

    if kind == "pothole":
        circ = shape_feats["circularity"]
        shape_score = float(np.clip(1.0 - abs(circ - 0.55) * 1.5, 0.0, 1.0))
    else:
        thinness = shape_feats.get("thinness", 1.0)
        tortuosity = shape_feats.get("tortuosity", 1.0)
        thinness_score = float(np.clip((thinness - 2.0) / 5.0, 0.0, 1.0))
        tort_score = float(np.clip((tortuosity - 1.15) / 0.35, 0.0, 1.0))
        shape_score = 0.5 * thinness_score + 0.5 * tort_score

    penalty = _false_positive_penalty(
        shape_feats, depression_depth, mean_grad, inner_sat, inner_val, inner_chroma, kind
    )

    raw_score = (
        0.24 * location +
        0.24 * contrast_score +
        0.20 * sharpness_score +
        0.16 * texture +
        0.16 * shape_score -
        penalty
    )
    confidence = float(np.clip(raw_score, 0.0, 1.0))

    x, y, w, h = shape_feats["bbox"]
    return {
        "issue_type": "pothole" if kind == "pothole" else "crack",
        "confidence": round(confidence, 3),
        "bbox": [int(x), int(y), int(x + w), int(y + h)],
        "area": int(shape_feats["area"]),
        "contour": contour,
        "features": {
            "location_score": round(location, 3),
            "surface_score": round(contrast_score, 3),
            "edge_gradient_score": round(sharpness_score, 3),
            "texture_score": round(texture, 3),
            "shape_score": round(shape_score, 3),
            "size_score": round(size_score, 3),
            "false_positive_penalty": round(penalty, 3),
            "mean_boundary_gradient": round(mean_grad, 2),
            "inner_chroma": round(inner_chroma, 1),
        },
    }


# --------------------------------------------------------------------------
# Master Detection Function
# --------------------------------------------------------------------------

def detect_issue(image_path, mode="road", debug=False):
    """Run the complete structured detection pipeline on an image."""
    image = load_image(image_path)
    image, inverse_scale = resize_for_processing(image, max_dim=900)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_map = local_texture_map(gray)
    roi_mask, roi_confidence = estimate_surface_roi(image, mode=mode)
    roi_area = float(np.count_nonzero(roi_mask))

    all_candidates = []

    if roi_area > 0 and roi_confidence >= 0.20:
        if mode == "road":
            blob_contours = _blob_candidates(gray, roi_mask)
            for c in blob_contours:
                cand = _score_candidate(c, gray, texture_map, image, roi_mask, roi_area, kind="pothole")
                if cand:
                    all_candidates.append(cand)

            crack_contours = _crack_candidates(gray, roi_mask)
            for c in crack_contours:
                cand = _score_candidate(c, gray, texture_map, image, roi_mask, roi_area, kind="crack")
                if cand:
                    all_candidates.append(cand)

        else:  # mode == "wall"
            crack_contours = _crack_candidates(gray, roi_mask)
            for c in crack_contours:
                cand = _score_candidate(c, gray, texture_map, image, roi_mask, roi_area, kind="crack")
                if cand:
                    all_candidates.append(cand)

    if roi_confidence < 0.35:
        for c in all_candidates:
            c["confidence"] = round(c["confidence"] * 0.65, 3)

    all_candidates.sort(key=lambda d: d["confidence"], reverse=True)

    kept = [c for c in all_candidates if c["confidence"] >= SECONDARY_THRESHOLD]
    kept = _suppress_overlaps(kept)

    diagnostic_reason = "No sufficiently reliable infrastructure damage detected"
    if all_candidates and all_candidates[0]["confidence"] < CONFIDENCE_THRESHOLD:
        top_pen = all_candidates[0]["features"].get("false_positive_penalty", 0.0)
        if top_pen >= 0.40:
            diagnostic_reason = "Candidates rejected as likely shadow, vehicle, or surface marking"

    result = {
        "roi_confidence": round(roi_confidence, 3),
        "roi_area": int(roi_area * (inverse_scale ** 2)),
        "candidates": [_public_candidate(c, inverse_scale) for c in kept],
    }

    if not kept or kept[0]["confidence"] < CONFIDENCE_THRESHOLD:
        result.update({
            "issue_type": "unknown",
            "confidence": round(kept[0]["confidence"], 3) if kept else 0.0,
            "bbox": None,
            "area": None,
            "reason": diagnostic_reason,
        })
    else:
        primary = kept[0]
        bbox = [int(v * inverse_scale) for v in primary["bbox"]]
        result.update({
            "issue_type": primary["issue_type"],
            "confidence": primary["confidence"],
            "bbox": bbox,
            "area": int(primary["area"] * (inverse_scale ** 2)),
        })

    if debug:
        stages = preprocess(image)
        stages["roi_mask"] = roi_mask
        stages["texture_map"] = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cand_canvas = image.copy()
        for c in all_candidates:
            cv2.drawContours(cand_canvas, [c["contour"]], -1, (0, 255, 255), 1)
        stages["candidates"] = cand_canvas

        result["debug"] = stages

    return result


def _suppress_overlaps(candidates, iou_thresh=0.35):
    """Greedy NMS de-duplication."""
    kept = []
    for cand in candidates:
        overlap = False
        for existing in kept:
            if _iou(cand["bbox"], existing["bbox"]) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(cand)
    return kept


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _public_candidate(c, inverse_scale):
    """Clean candidate dict for UI/database consumption."""
    bbox = [int(v * inverse_scale) for v in c["bbox"]]
    return {
        "issue_type": c["issue_type"],
        "confidence": c["confidence"],
        "bbox": bbox,
        "area": int(c["area"] * (inverse_scale ** 2)),
        "features": c["features"],
    }
