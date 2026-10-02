"""
image_processor.py
-------------------
Image loading, preprocessing, and ADAPTIVE region-of-interest (ROI)
estimation for Snap2Action_CV.

Design goals:
- Segment the image into true infrastructure surface regions (road pavement
  or wall facade).
- Fill interior holes in the surface mask so surface defects (potholes/cracks)
  are guaranteed to be inside the surface ROI.
- Actively reject and suppress non-infrastructure objects:
  vehicles (cars, bikes, tyres, undercarriages), sky, vegetation, and specular glare.
- Use chroma = max(BGR) - min(BGR) to avoid low-light HSV saturation noise.
- Pure functions for testability.
"""

import cv2
import numpy as np


def load_image(path):
    """Load an image from disk as BGR. Raises ValueError if unreadable."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def resize_for_processing(image, max_dim=900):
    """Downscale large images for speed; preserve aspect ratio.
    Returns (resized_image, scale_factor)."""
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return image, (1.0 / scale if scale != 0 else 1.0)


def preprocess(image):
    """Run classic preprocessing stages and return every intermediate
    result for debug visualization."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    return {
        "original": image,
        "gray": gray,
        "blurred": blurred,
        "otsu": otsu,
        "edges": edges,
        "morphed": morphed,
    }


def local_texture_map(gray, ksize=9):
    """Local standard-deviation map for texture/roughness estimation."""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (ksize, ksize))
    sq_mean = cv2.blur(gray_f * gray_f, (ksize, ksize))
    variance = np.clip(sq_mean - mean * mean, 0, None)
    return np.sqrt(variance)


def gradient_orientation_map(gray):
    """Sobel gradient magnitude + orientation."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    orientation = cv2.phase(gx, gy, angleInDegrees=True)
    return magnitude, orientation


def detect_non_surface_mask(image):
    """Detect regions that cannot physically be asphalt or concrete road surfaces:
    - High-chroma painted objects (vehicles, colored panels, taillights).
      Requires chroma >= 35, saturation >= 50, and val >= 60 so dark asphalt
      noise is not falsely flagged as vehicle paint.
    - Specular highlights on vehicle bodies or windshields.
    - Attached tyres and vehicle undercarriages (dilated downwards from vehicle bodies).
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Chroma = max(B,G,R) - min(B,G,R)
    b, g, r = cv2.split(image)
    max_c = np.maximum(np.maximum(b, g), r)
    min_c = np.minimum(np.minimum(b, g), r)
    chroma = max_c - min_c

    # Vehicles: significant chroma (>= 35) AND high saturation (>= 50) AND visible brightness (V >= 60)
    high_sat = ((chroma >= 35) & (sat >= 50) & (val >= 60)).astype(np.uint8) * 255

    # Specular reflections (bright glare on vehicle glass/chrome)
    specular = ((val >= 238) & (sat < 35)).astype(np.uint8) * 255

    combined_obstacles = cv2.bitwise_or(high_sat, specular)

    # Bridge vehicle body parts (cabin, windshield, hood, bumpers)
    k_body = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    obstacles_closed = cv2.morphologyEx(combined_obstacles, cv2.MORPH_CLOSE, k_body, iterations=2)

    # Dilate vehicle mask slightly downwards to encompass attached tyres and undercarriage
    k_down = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 25))
    obstacles_expanded = cv2.dilate(obstacles_closed, k_down, iterations=1)

    # Filter out tiny specks (< 0.05% of frame)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obstacles_expanded, connectivity=8)
    filtered = np.zeros((h, w), dtype=np.uint8)
    min_area = 0.0005 * (h * w)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    return filtered


def estimate_surface_roi(image, mode="road"):
    """Adaptively estimate the region of interest for 'road' or 'wall' analysis."""
    h, w = image.shape[:2]
    total_area = float(h * w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mode == "road":
        obstacles = detect_non_surface_mask(image)

        sky_mask = np.zeros((h, w), dtype=np.uint8)
        upper_limit = int(h * 0.32)
        upper_gray = gray[:upper_limit, :]
        upper_texture = local_texture_map(upper_gray, ksize=7)
        sky_candidates = (upper_gray > 135) & (upper_texture < 16.0)
        sky_mask[:upper_limit, :][sky_candidates] = 255

        y_start, y_end = int(h * 0.55), int(h * 0.95)
        x_start, x_end = int(w * 0.15), int(w * 0.85)
        sample_patch = image[y_start:y_end, x_start:x_end]
        sample_obs = obstacles[y_start:y_end, x_start:x_end]

        valid_patch_mask = sample_obs == 0
        if np.count_nonzero(valid_patch_mask) > 100:
            ref_bgr = np.median(sample_patch[valid_patch_mask], axis=0)
        else:
            ref_bgr = np.array([105.0, 105.0, 107.0])

        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(np.uint8([[ref_bgr]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

        lab_diff = image_lab - ref_lab
        color_dist = np.sqrt(np.sum(lab_diff ** 2, axis=2))

        b, g, r = cv2.split(image)
        max_c = np.maximum(np.maximum(b, g), r)
        min_c = np.minimum(np.minimum(b, g), r)
        chroma = max_c - min_c

        road_seed = (color_dist < 55.0) & (chroma < 30) & (obstacles == 0) & (sky_mask == 0)
        road_seed[:int(h * 0.28), :] = False

        road_binary = road_seed.astype(np.uint8) * 255

        bridge_size = max(11, int(min(h, w) * 0.04)) | 1
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_size, bridge_size))
        road_closed = cv2.morphologyEx(road_binary, cv2.MORPH_CLOSE, k_close, iterations=2)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(road_closed, connectivity=8)
        final_mask = np.zeros((h, w), dtype=np.uint8)
        best_area = 0
        best_idx = None

        for i in range(1, n_labels):
            c_area = stats[i, cv2.CC_STAT_AREA]
            cy = centroids[i][1]
            if c_area >= 0.04 * total_area and cy >= (h * 0.35):
                if c_area > best_area:
                    best_area = c_area
                    best_idx = i

        if best_idx is not None:
            comp_mask = (labels == best_idx).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                largest_cnt = max(cnts, key=cv2.contourArea)
                cv2.drawContours(final_mask, [largest_cnt], -1, 255, thickness=-1)
        else:
            final_mask[int(h * 0.35):, :] = 255

        final_mask[obstacles > 0] = 0
        final_mask[:int(h * 0.28), :] = 0

        area_frac = float(np.count_nonzero(final_mask)) / total_area
        roi_confidence = float(np.clip(0.55 + area_frac * 0.5, 0.45, 0.95)) if area_frac >= 0.10 else 0.25

        return final_mask, roi_confidence

    else:  # mode == "wall"
        wall_prior = np.zeros((h, w), dtype=np.uint8)
        wall_prior[int(h * 0.10):int(h * 0.88), :] = 255

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        k_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k_rect, iterations=2)

        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        window_mask = np.zeros((h, w), dtype=np.uint8)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.02 * total_area or area > 0.45 * total_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            bbox_area = cw * ch
            solidity = area / bbox_area if bbox_area > 0 else 0
            if solidity > 0.82:
                window_mask[y:y + ch, x:x + cw] = 255

        wall_mask = cv2.bitwise_and(wall_prior, cv2.bitwise_not(window_mask))
        roi_confidence = 0.88 if np.count_nonzero(wall_mask) > 0.20 * total_area else 0.35

        return wall_mask, roi_confidence
