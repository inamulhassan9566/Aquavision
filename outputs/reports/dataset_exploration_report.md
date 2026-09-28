# AQUAVISION — Dataset Exploration & Audit Report
**Project:** AI-Based Oil Spill Detection and Vessel Attribution (Smart India Hackathon 2026)  
**Dataset Source:** [Kaggle Sentinel-1 SAR Oil Spill Detection Dataset](https://www.kaggle.com/datasets/harikrishnacs/sentinel-1-sar-oil-spill-detection-dataset)  
**Underlying Collection:** CSIRO Sentinel-1 SAR Deep Learning Collection (DOI: 10.25919/4v55-dn16)  

---

## 1. Executive Summary
- **Total Satellite SAR Chips:** `5,538`
- **Resolution / Dimensions:** `400 x 400` pixels (100% uniform across all chips)
- **File Format:** JPEG (`JPEG`)
- **Color Mode:** `RGB` (1-band microwave SAR backscatter replicated across R, G, B channels)
- **Data Integrity:** **0 corrupted files** detected (100% readable)
- **Exact Duplicates:** **0 identical files** (SHA-256 hash unique across all 5,538 files)
- **Class Balance:** Moderate imbalance (~2.01 : 1 ratio)

---

## 2. Class Distribution Breakdown

| Class Label | Category Description | Chip Count | Percentage | Sequence ID Range |
| :--- | :--- | :--- | :--- | :--- |
| **Class 0** | No Oil Spill (Clean Sea, Look-alikes, Low Wind) | **3,695** | **66.72%** | `class_0_00001` to `class_0_03695` |
| **Class 1** | Oil Spill (Confirmed Slick Backscatter Damping) | **1,843** | **33.28%** | `class_1_00001` to `class_1_01843` |
| **Total** | Full Curated Dataset | **5,538** | **100.0%** | Consecutive, No Gaps |

### Class Imbalance Handling Note
- Oil Spill (Class 1) comprises **33.28%** of the dataset, while Non-Oil (Class 0) comprises **66.72%**.
- This is a natural reflection of marine surveillance where clean sea / look-alikes dominate.
- Recommended strategy: Use class-weighted cross-entropy loss `weights = [1.0, 2.005]` or balanced sampling to prevent bias towards majority non-spill predictions.

---

## 3. Physical & Radiometric SAR Characteristics
- **Dimensions:** Exactly 400 × 400 pixels for all 5,538 images.
- **Radiometric Channels:** Single-band SAR backscatter intensity stored in 8-bit RGB format (`R == G == B` identically across all pixels).
- **Pixel Intensity Stats (Normalized [0, 1]):**
  - **Class 0 Mean:** `0.5385` (Patch Std: `0.1078`)
  - **Class 1 Mean:** `0.4823` (Patch Std: `0.0558`)
  - **Global Dataset Mean:** `0.5198`
  - **Global Dataset Std:** `0.0905`
- **Storage Profile:** Mean file size is `12.1 KB` (Min: `3.1 KB`, Max: `43.0 KB`).

---

## 4. Anomalies & Data Leakage Prevention Considerations
1. **Zero-Variance Artifact Chips:**
   - Detected **10** images in `Class_0` that consist of flat, single-color pixels (e.g. constant value 142, 215, 255 with `std = 0.0`).
   - These represent sensor dropout or masked areas during chip extraction.
   - They will be documented transparently and can either be pruned or safely handled.
2. **Scene / Group Metadata Availability:**
   - In this Kaggle redistribution, file names are sequentially indexed (`class_0_00001.jpg` to `class_0_03695.jpg`).
   - Original CSIRO acquisition scene IDs, satellite acquisition timestamps, and bounding coordinates were not retained in image EXIF tags.
   - **Limitation Documented:** Group-aware splitting by raw scene ID is therefore unavailable; a **stratified split (70% Train / 15% Val / 15% Test) with a deterministic fixed random seed** will be applied to prevent data leakage and ensure 100% reproducibility.

---

## 5. Generated Visual Artifacts
The following publication-grade charts have been exported to `outputs/visualizations/`:
1. `class_distribution.png`: Bar and Donut chart depicting class counts and percentages.
2. `oil_spill_samples.png`: Grid of confirmed oil spill chips displaying characteristic backscatter dampening.
3. `no_oil_samples.png`: Grid of non-oil chips displaying sea clutter and ambient ocean texture.
4. `random_samples_grid.png`: Unbiased random sample batch across both classes.
5. `pixel_intensity_distribution.png`: Kernel density estimation of normalized pixel intensities.
6. `zero_variance_samples.png`: Inspection of the 10 uniform artifact chips found in Class 0.
