"""
make_test_images.py
-------------------
Generates synthetic positive and negative test images covering realistic
civic damage and non-damage visual challenges:
- pos_pothole_with_car: Pothole on road + Car present (must localize pothole, ignore car)
- pos_multiple_potholes: Multiple potholes on road surface
- neg_shadow_only: Soft shadow on road (must NOT detect pothole)
- neg_road_marking: Painted lane lines and markings (must NOT detect damage)
- neg_normal_road: Clean intact asphalt road
- pos_wall_crack: Irregular meandering branching fracture on wall
- neg_wall_window: Wall with rectangular window and vertical pipe (must NOT detect crack)
- neg_normal_wall: Intact brick wall pattern
"""

import os
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "test_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def road_bg(w=700, h=500, tone=110):
    img = np.full((h, w, 3), tone, dtype=np.uint8)
    noise = np.random.randint(-5, 6, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Sky band at top 28%
    img[:int(h * 0.28), :] = (175, 155, 125)

    # Road surface in lower 72%
    cv2.rectangle(img, (0, int(h * 0.28)), (w, h), (105, 105, 107), -1)
    noise2 = np.random.randint(-7, 8, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise2, 0, 255).astype(np.uint8)

    # Subtle horizontal asphalt texture
    for y in range(int(h * 0.35), h, 7):
        cv2.line(img, (0, y), (w, y), (98, 98, 100), 1)

    return img


def wall_bg(w=700, h=500, tone=180):
    img = np.full((h, w, 3), tone, dtype=np.uint8)
    # Brick pattern
    for y in range(0, h, 30):
        offset = 0 if (y // 30) % 2 == 0 else 30
        for x in range(-30, w, 60):
            cv2.rectangle(img, (x + offset, y), (x + offset + 55, y + 25), (165, 165, 168), 2)
    noise = np.random.randint(-5, 5, (h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def draw_realistic_pothole(img, center=(500, 420), r=(50, 26)):
    """Draws an irregular, dark depression with broken jagged rim texture."""
    cx, cy = center
    rx, ry = r
    # Outer dark crater depression
    cv2.ellipse(img, (cx, cy), (rx, ry), 8, 0, 360, (38, 33, 30), -1)
    # Inner deeper jagged pit
    cv2.ellipse(img, (cx + 2, cy + 2), (int(rx * 0.75), int(ry * 0.70)), -6, 0, 360, (22, 19, 18), -1)
    # Broken jagged asphalt stone edges (sharp physical gradient)
    for _ in range(16):
        ang = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(0.7, 1.05)
        px = int(cx + rx * dist * np.cos(ang))
        py = int(cy + ry * dist * np.sin(ang))
        shade = int(np.random.randint(25, 65))
        cv2.circle(img, (px, py), np.random.randint(2, 5), (shade, shade - 2, shade - 4), -1)


def generate_all():
    print("Generating test images...")

    # 1. POSITIVE: Pothole + Car (must localize pothole, ignore car)
    img1 = road_bg()
    draw_realistic_pothole(img1, center=(520, 420), r=(55, 26))
    # Car body (high saturation blue + red taillights + tyres)
    cv2.rectangle(img1, (70, 240), (270, 340), (190, 40, 30), -1)   # saturated blue car body
    cv2.rectangle(img1, (100, 195), (240, 240), (220, 90, 80), -1)  # windshield & cabin
    cv2.circle(img1, (110, 345), 24, (12, 12, 12), -1)              # black tyre
    cv2.circle(img1, (230, 345), 24, (12, 12, 12), -1)              # black tyre
    cv2.circle(img1, (110, 345), 10, (110, 110, 110), -1)          # wheel rim
    cv2.circle(img1, (230, 345), 10, (110, 110, 110), -1)
    cv2.rectangle(img1, (70, 310), (80, 325), (30, 30, 220), -1)   # red taillight
    cv2.imwrite(str(OUTPUT_DIR / "pos_pothole_with_car.jpg"), img1)

    # 2. POSITIVE: Multiple potholes on road
    img2 = road_bg()
    draw_realistic_pothole(img2, center=(240, 410), r=(45, 22))
    draw_realistic_pothole(img2, center=(520, 430), r=(50, 25))
    cv2.imwrite(str(OUTPUT_DIR / "pos_multiple_potholes.jpg"), img2)

    # 3. NEGATIVE: Road + Soft Shadow only (must report UNKNOWN)
    img3 = road_bg()
    shadow_overlay = img3.copy()
    pts = np.array([[380, 340], [580, 320], [640, 460], [400, 480]], dtype=np.int32)
    cv2.fillPoly(shadow_overlay, [pts], (72, 70, 68))
    # Soft penumbra blend (diffuse edge)
    blurred_shadow = cv2.GaussianBlur(shadow_overlay, (21, 21), 0)
    img3 = cv2.addWeighted(blurred_shadow, 0.45, img3, 0.55, 0)
    cv2.imwrite(str(OUTPUT_DIR / "neg_shadow_only.jpg"), img3)

    # 4. NEGATIVE: Road markings (painted white/yellow lane lines)
    img4 = road_bg()
    # White dashed lane markings
    cv2.rectangle(img4, (335, 300), (355, 360), (230, 230, 232), -1)
    cv2.rectangle(img4, (330, 400), (355, 480), (235, 235, 238), -1)
    # Yellow curb line
    cv2.rectangle(img4, (20, 470), (280, 490), (30, 200, 230), -1)
    cv2.imwrite(str(OUTPUT_DIR / "neg_road_marking.jpg"), img4)

    # 5. NEGATIVE: Clean normal road
    img5 = road_bg()
    cv2.imwrite(str(OUTPUT_DIR / "neg_normal_road.jpg"), img5)

    # 6. POSITIVE: Wall + actual irregular crack
    img6 = wall_bg()
    np.random.seed(42)
    pts = [(130, 70)]
    x, y = 130, 70
    for _ in range(42):
        x += np.random.randint(4, 10)
        y += np.random.randint(3, 9)
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(img6, pts[i], pts[i + 1], (55, 52, 50), 2)
    # Branching segment
    bx, by = pts[20]
    for _ in range(12):
        bx += np.random.randint(3, 8)
        by -= np.random.randint(2, 6)
        cv2.line(img6, pts[20], (bx, by), (55, 52, 50), 1)
    cv2.imwrite(str(OUTPUT_DIR / "pos_wall_crack.jpg"), img6)

    # 7. NEGATIVE: Wall + Window and Pipe (must NOT detect crack)
    img7 = wall_bg()
    # Window with glass pane and dark frame
    cv2.rectangle(img7, (280, 140), (440, 320), (145, 165, 180), -1)
    cv2.rectangle(img7, (280, 140), (440, 320), (85, 85, 88), 5)
    cv2.line(img7, (360, 140), (360, 320), (85, 85, 88), 4)
    cv2.line(img7, (280, 230), (440, 230), (85, 85, 88), 4)
    # Straight vertical pipe
    cv2.rectangle(img7, (560, 40), (575, 460), (110, 110, 112), -1)
    cv2.line(img7, (560, 40), (560, 460), (70, 70, 72), 2)
    cv2.line(img7, (575, 40), (575, 460), (70, 70, 72), 2)
    cv2.imwrite(str(OUTPUT_DIR / "neg_wall_window.jpg"), img7)

    # 8. NEGATIVE: Clean normal wall
    img8 = wall_bg()
    cv2.imwrite(str(OUTPUT_DIR / "neg_normal_wall.jpg"), img8)

    # 9. Realistic AI-generated photo test cases
    import shutil
    ai_img1 = Path(r"C:\Users\Anuj\.gemini\antigravity\brain\aaa7a3d0-1329-471b-9e64-058f2d1453db\road_pothole_test_1790334163460.jpg")
    if ai_img1.exists():
        shutil.copy(ai_img1, OUTPUT_DIR / "road_pothole_city.jpg")

    ai_img2 = Path(r"C:\Users\Anuj\.gemini\antigravity\brain\aaa7a3d0-1329-471b-9e64-058f2d1453db\multiple_potholes_road_1790334281012.jpg")
    if ai_img2.exists():
        shutil.copy(ai_img2, OUTPUT_DIR / "road_potholes_rural.jpg")

    print(f"Generated test images in: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all()
