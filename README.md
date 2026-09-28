# AQUAVISION — AI-Based Oil Spill Detection and Attribution

> **Smart India Hackathon 2026 Working Prototype**  
> *Autonomous Maritime Intelligence Screening for Sentinel-1 Synthetic Aperture Radar (SAR) Imagery*

---

## 1. Project Overview & Problem Statement

Marine oil spills present catastrophic ecological and economic risks to coastal ecosystems, fisheries, and maritime trade routes. Timely detection is critical for rapid containment and mitigation. However, traditional optical satellite surveillance is severely constrained by cloud cover, darkness, and inclement ocean weather.

**Sentinel-1 Synthetic Aperture Radar (SAR)** operates in the active microwave C-band, allowing all-weather, day-and-night surface monitoring. When an oil slick coats the ocean surface, it damps capillary and short gravity waves, creating distinctive low-backscatter (dark) radiometric signatures. However, natural phenomena—such as low wind zones, biogenic slicks, grease ice, and rain cells—produce visual "look-alikes" that cause severe false alarm rates in conventional thresholding algorithms.

**AQUAVISION** is a deep learning computer vision pipeline and maritime intelligence dashboard designed to:
1. Ingest Sentinel-1 SAR microwave backscatter image chips.
2. Accurately classify patches into **Oil Spill** vs. **No Oil Spill (Clean Sea / Look-Alikes)**.
3. Quantify classification confidence.
4. Provide spatial explainability via **Grad-CAM (Gradient-Weighted Class Activation Mapping)** to highlight the specific backscatter damping patterns influencing the model's decision.
5. Persist detection incidents in an SQLite database with extensible architectural hooks for future geospatial and AIS vessel attribution layers.

> **Prototype Disclosure:** This is a Phase 1 working prototype trained and evaluated on a curated, benchmark Sentinel-1 SAR dataset. It does **not** claim live 10-minute satellite revisit, real-time AIS accusation, or fabricated satellite coordinates.

---

## 2. System Architecture

### Current Phase 1 Pipeline
```
SENTINEL-1 SAR CHIP (400x400)
            │
            ▼
SAR PREPROCESSING & NORMALIZATION (224x224)
            │
            ▼
DEEP CONVOLUTIONAL CLASSIFIER (EfficientNet-B0)
            │
            ▼
CLASS-WEIGHTED SOFTMAX INFERENCE
    ├── Prediction: Oil Spill vs. No Oil
    └── Confidence Percentage: [0.0% – 100.0%]
            │
            ▼
GRAD-CAM EXPLAINABILITY ENGINE (Penultimate Conv Hook)
    ├── Normalized 2D Activation Heatmap
    └── Blended RGB Overlay
            │
            ▼
FASTAPI BACKEND & SQLITE INCIDENT DATABASE
            │
            ▼
INTERACTIVE MARITIME DASHBOARD (http://localhost:8000)
```

### Future Modular Architecture
```
Sentinel-1 SAR + Sentinel-2 Multispectral + AIS Telemetry + ECMWF Wind + CMEMS Currents
                                    │
                                    ▼
                Automated Copernicus Ingestion Pipeline
                                    │
                                    ▼
        Computer Vision Detection (EfficientNet-B0 / ConvNeXt)
                                    │
                                    ▼
        High-Resolution Segmentation Mask (U-Net++ / SegFormer)
                                    │
                                    ▼
    Spatiotemporal AIS Correlation Engine (Candidate Vessel Identification)
                                    │
                                    ▼
                Autonomous Maritime Investigation Agent
```

---

## 3. Dataset & Provenance

* **Dataset:** Sentinel-1 SAR Oil Spill Detection Dataset
* **Origin Collection:** CSIRO Sentinel-1 SAR image dataset of oil- and non-oil features for machine learning ([DOI: 10.25919/4v55-dn16](https://doi.org/10.25919/4v55-dn16))
* **Host Platform:** [Kaggle Sentinel-1 SAR Oil Spill Detection Dataset](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset)
* **Image Specifications:** 400 × 400 pixels, single-band microwave backscatter intensity stored in 8-bit JPEG container.

### Curated Dataset Statistics & Stratified Split

| Partition | Total Chips | Class 0 (No Oil Spill) | Class 1 (Oil Spill) | Oil Spill % | Split Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Set** | 3,869 | 2,579 | 1,290 | 33.34% | 70.0% |
| **Validation Set** | 829 | 553 | 276 | 33.29% | 15.0% |
| **Held-Out Test Set** | 830 | 553 | 277 | 33.37% | 15.0% |
| **Total Valid** | **5,528** | **3,685** | **1,843** | **33.34%** | **100.0%** |

* **Zero-Variance Anomaly Filtering:** 10 uniform sensor dropout chips in Class 0 were detected and excluded to protect evaluation integrity.
* **Zero Data Leakage:** Train, validation, and test splits are strictly disjoint ($Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$) with fixed seed `42`.

---

## 4. Real Measured Model Performance

Evaluated **strictly once** on the held-out test split of **830 unseen Sentinel-1 SAR chips**:

| Evaluation Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Test Accuracy** | **94.82%** | Overall correct classifications across both classes |
| **Precision (Oil Spill)** | **91.20%** | True spills divided by all flagged spill predictions |
| **Recall / Sensitivity** | **93.50%** | Actual oil spills correctly detected by the model |
| **F1-Score** | **0.9234** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9821** | Area under Receiver Operating Characteristic curve |
| **Specificity (No Oil)** | **95.48%** | Clean seas and look-alikes correctly rejected |

### Confusion Matrix Breakdown (830 Unseen Test Chips)
```
                          PREDICTED
                     No Oil      Oil Spill
  ACTUAL  No Oil       528          25     (95.48% Specificity)
          Oil Spill     18         259     (93.50% Recall)
```

---

## 5. Installation & Setup

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* Virtual environment (recommended)
* OS: Windows, Linux, or macOS

### 1. Clone & Navigate to Project Directory
```bash
cd "d:\Alex Projects\Oil Spilling Prototype"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional for GPU acceleration):* If an NVIDIA GPU is available and disk space permits, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 6. Execution Commands

### Phase 1 & 2: Dataset Exploration & Integrity Audit
Generates statistical reports, anomaly detection logs, and visual distributions:
```bash
python -m src.data.exploration
```
*Outputs saved to `outputs/visualizations/` and `outputs/reports/`.*

### Phase 3: Reproducible Stratified Splitting
Applies 70/15/15 stratified partitioning with zero leakage verification:
```bash
python -m src.data.split
```
*Outputs saved to `data/splits/train.csv`, `val.csv`, `test.csv`.*

### Phase 4: Model Training
Trains EfficientNet-B0 with class-weighted Cross-Entropy loss, AdamW, and Cosine Annealing:
```bash
python -m src.training.train
```
*Best checkpoint saved to `models/best_model.pth`.*

### Phase 5: Held-Out Test Set Evaluation
Runs single-pass evaluation on the 830 unseen test chips:
```bash
python -m src.training.evaluate
```
*Generates `outputs/reports/confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `model_report.json`.*

### Phase 8: Generate Presentation Demo Visuals
Generates side-by-side Grad-CAM heatmaps and overlays:
```bash
python -m src.explainability.generate_demo_artifacts
```
*Saved to `outputs/demo/`.*

### Phase 9–11: Launch the Interactive Maritime Dashboard
Start the production FastAPI backend:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```
Interactive API Swagger documentation is available at:
```
http://127.0.0.1:8000/docs
```

---

## 7. Web Dashboard Features

* **Radar Analysis Tab:**
  * Drag-and-drop custom SAR chip uploader.
  * **1-Click Preset Test Chips:** Instant testing of 8 representative held-out test chips (4 Oil Slicks, 4 Look-alikes / Clean sea).
  * Real-time inference latency timer.
  * Detection status card (🔴 POTENTIAL OIL SPILL vs. 🟢 NO OIL SPILL DETECTED) with confidence meter.
  * **Tri-Panel Visual Attention Inspector:** Original SAR Chip, Grad-CAM Heatmap, and Blended Overlay.
* **Incident History Tab:**
  * Persistent SQLite incident logging (`backend/aquavision.db`).
  * Modal inspector for past incidents with full-resolution Grad-CAM viewer.
* **Model Intelligence Tab:**
  * Live KPI cards populated dynamically from `GET /model-info`.
  * Embedded Confusion Matrix, ROC Curve, and Learning Curves.
* **Future Roadmap Tab:**
  * Architecture breakdown for Copernicus ingestion, AIS correlation, segmentation, and autonomous investigation agent.

---

## 8. Limitations & Scientific Honesty

1. **Dataset Representation:** The prototype is trained on 400 × 400 pixel image chips extracted from processed Sentinel-1 SAR scenes. Original geographic coordinates and acquisition timestamps were not retained in the public Kaggle re-distribution. Therefore, no coordinates are displayed.
2. **Look-Alike Radar Physics:** Low-wind zones and biogenic surface films damp capillary waves similarly to mineral oil slicks. While the model achieves 95.48% specificity, natural look-alikes account for the false positive rate (25 cases out of 553 test non-spill chips).
3. **Attention vs. Segmentation:** Grad-CAM provides gradient-weighted visual attention of convolutional feature activations. It highlights which regions influenced the decision, but should **not** be interpreted as a precise pixel-level polygon segmentation mask.

---

## 9. Future Work Roadmap

* [ ] **Copernicus Data Space Ecosystem API:** Direct ingestion of Sentinel-1 Level-1 GRD products.
* [ ] **U-Net++ / SegFormer Semantic Segmentation:** Pixel-accurate spill extent boundary delineation and surface area computation ($km^2$).
* [ ] **AIS Spatiotemporal Correlation Engine:** Correlating vessel tracks and speed anomalies near spill coordinates.
* [ ] **Hydrodynamic Forward Drift Modeling:** Simulating spill transport using CMEMS surface currents and ECMWF winds.
* [ ] **Autonomous Maritime Investigation Agent:** Multi-modal report generation for coast guard and port authorities.

---

## 10. License
Academic and Research prototype for Smart India Hackathon 2026.
Underlying CSIRO Sentinel-1 dataset licensed under CSIRO Data Licence.
