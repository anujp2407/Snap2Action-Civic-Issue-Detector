# Snap2Action_CV — Civic Issue Localization & Damage Detection

A local desktop application built with **CustomTkinter** and **OpenCV** to reliably detect and localize civic infrastructure damage (potholes on roads, cracks on walls) while rejecting non-civic objects, vehicles, and optical false positives.

---

## The Core Problem Solved

Traditional contour-based and naive thresholding methods treat:
$$\text{"strong contrast / high edge density"} \longrightarrow \text{"civic issue"}$$

This causes severe false detections:
- **Cars, bikes, and tyres** detected as potholes.
- **Tree and cloud shadows** detected as craters.
- **Painted road markings** detected as defects.
- **Windows, pipes, and brick mortar grids** detected as wall cracks.

Snap2Action_CV replaces this with a structured, multi-stage physical verification pipeline:
$$\text{INFRASTRUCTURE SURFACE} \longrightarrow \text{DAMAGE LOCALIZATION} \longrightarrow \text{PHYSICAL VERIFICATION} \longrightarrow \text{SEVERITY ESTIMATION}$$

---

## Detection Architecture

```text
               INPUT IMAGE
                   │
                   ▼
     Preprocessing & Scale Normalization
   (Gaussian filter, texture & gradient maps)
                   │
                   ▼
       Adaptive Surface ROI Estimation
 (Extract asphalt/concrete, suppress cars, sky)
                   │
                   ▼
         Candidate Generation
  (Local surface depression & thin fractures)
                   │
                   ▼
     Multi-Feature Physical Scoring
  (Contrast, boundary gradient, roughness, shape)
                   │
                   ▼
       False-Positive Rejection
 (Shadow penumbra, vehicle paint, tyres, markings)
                   │
                   ▼
  Confidence Calibration & NMS Suppression
  (Threshold >= 0.48, otherwise 'UNKNOWN')
                   │
                   ▼
         Severity Estimation
    (Area ratio, depth, crack tortuosity)
                   │
                   ▼
        Final Structured Result
    (Overlay, SQLite Record, PDF Report)
```

---

## Key Algorithmic Innovations

1. **Adaptive Surface ROI with Vehicle & Sky Suppression (`image_processor.py`)**:
   - Asphalt surfaces are achromatic ($S < 50$). Vehicle paint, taillights, and colorful debris ($S \ge 50$) are segmented and excluded from the road surface prior.
   - Specular reflections on vehicle hoods and glass are detected and masked out.
   - Upper low-texture sky and distant vertical tree/building lines are eliminated.

2. **Shadow vs. Pothole Physical Discriminator (`detector.py`)**:
   - **Potholes**: Have sharp physical fracture rims with high boundary gradient magnitude ($\nabla I \ge 22$) and increased internal broken asphalt roughness ($\sigma_{inner} > \sigma_{outer}$).
   - **Shadows**: Have diffuse, gradual penumbrae with low boundary gradient magnitude ($\nabla I \le 12$) and retain the underlying smooth road texture. Shadows are rejected via explicit penalty.

3. **Crack Tortuosity vs. Architectural Edges (`detector.py`)**:
   - Real structural cracks are tortuous and meandering ($\text{tortuosity} > 1.25$).
   - Straight architectural lines (pipes, wires, window frames, regular brick mortar) have $\text{tortuosity} \approx 1.0$ and are penalized as non-damage.

4. **Calibrated Confidence**:
   - If no candidate clears the multi-feature confidence bar ($0.48$), the system honestly reports `UNKNOWN` with an explicit reason rather than blaming an arbitrary object.

---

## Project Structure

```text
├── main.py              # Application entrypoint
├── gui.py               # CustomTkinter GUI (Detect, Dashboard, History)
├── image_processor.py   # Adaptive ROI, preprocessing, obstacle suppression
├── detector.py          # Candidate generation, multi-feature scoring, rejection
├── severity.py          # Localized damage severity estimation
├── database.py          # SQLite database storage & statistics
├── report_generator.py  # PDF report generation (reportlab)
├── visualization.py     # Overlay drawing & 8-stage debug contact sheet
├── make_test_images.py  # Positive and negative test case generator
├── test_pipeline.py     # Automated test suite and scorecard
├── requirements.txt     # Dependencies
└── data/
    ├── db/              # SQLite database (snap2action.db)
    ├── reports/         # Exported PDF reports
    └── uploads/         # Cached original and detected images
```

---

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
python test_pipeline.py
```
This runs 8 challenging positive and negative cases:
- `pos_pothole_with_car.jpg`: Pothole localized, car completely ignored.
- `pos_multiple_potholes.jpg`: Multiple potholes localized.
- `neg_shadow_only.jpg`: Classified as UNKNOWN (shadow ignored).
- `neg_road_marking.jpg`: Classified as UNKNOWN (markings ignored).
- `neg_normal_road.jpg`: Classified as UNKNOWN.
- `pos_wall_crack.jpg`: Meandering crack localized.
- `neg_wall_window.jpg`: Classified as UNKNOWN (window/pipe ignored).
- `neg_normal_wall.jpg`: Classified as UNKNOWN (brick pattern ignored).

### 3. Launch the Desktop Application
```bash
python main.py
```
