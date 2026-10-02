"""
visualization.py
-----------------
Drawing helpers shared by the GUI and the report generator:
- Overlay the localized detection (and secondary candidates) on the image.
- Render multi-stage debug contact sheet.

Visual rules:
- PRIMARY detection: bold RED bounding box and label (e.g. "Pothole 87%").
- SECONDARY candidates: muted AMBER bounding box.
- Ignored non-damage / false-positive regions: left untouched or neutral.
"""

import cv2
import numpy as np

PRIMARY_COLOR = (0, 0, 220)       # Red (BGR)
SECONDARY_COLOR = (0, 165, 255)   # Amber (BGR)
TEXT_BG_COLOR = (20, 20, 25)


def draw_detection_overlay(image_bgr, detection):
    """Return a copy of image_bgr with bounding boxes + labels drawn for
    the primary detection (red) and any secondary candidates (amber)."""
    out = image_bgr.copy()
    candidates = detection.get("candidates", [])
    primary_bbox = detection.get("bbox")

    for cand in candidates:
        bbox = cand.get("bbox")
        if not bbox:
            continue
        is_primary = (bbox == primary_bbox)
        color = PRIMARY_COLOR if is_primary else SECONDARY_COLOR
        thickness = 3 if is_primary else 2
        x1, y1, x2, y2 = bbox

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label = f"{cand.get('issue_type', '?').title()} {cand.get('confidence', 0):.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(th + 6, y1 - 6)

        # Label background
        cv2.rectangle(out, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)

    if not candidates or detection.get("issue_type") == "unknown":
        msg = f"Status: {detection.get('reason', 'No reliable damage detected')}"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (16, 16), (24 + tw, 26 + th), (25, 25, 30), -1)
        cv2.putText(out, msg, (20, 20 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 190, 255), 2, cv2.LINE_AA)

    return out


def render_debug_grid(stages: dict, tile_size=(260, 190)):
    """Compose a contact-sheet grid image from a dict of stage_name -> image array."""
    tiles = []
    labels = []
    for name, img in stages.items():
        if img is None:
            continue
        if len(img.shape) == 2:
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            disp = img.copy()
        disp = cv2.resize(disp, tile_size, interpolation=cv2.INTER_AREA)
        tiles.append(disp)
        labels.append(name.replace("_", " ").title())

    if not tiles:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    cols = min(4, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    pad = 8
    label_h = 24
    grid_w = cols * tile_size[0] + (cols + 1) * pad
    grid_h = rows * (tile_size[1] + label_h) + (rows + 1) * pad
    canvas = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)

    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        r, c = divmod(idx, cols)
        x = pad + c * (tile_size[0] + pad)
        y = pad + r * (tile_size[1] + label_h + pad)
        canvas[y:y + tile_size[1], x:x + tile_size[0]] = tile
        cv2.putText(canvas, label, (x + 4, y + tile_size[1] + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    return canvas
