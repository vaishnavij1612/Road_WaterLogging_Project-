# Dataset Analysis Report

**Project Title:** AI-Based Road Waterlogging Detection and Severity Estimation Using Deep Learning  
**Date of Inspection:** 2026-10-04  
**Dataset:** FloodIMG — Flood Image DataBase System  
**Source:** Kaggle — `hhrclemson/flooding-image-dataset`  
**Archive Size:** ~11.14 GB  
**Inspection Status:** COMPLETE — Full dataset extracted and systematically inspected

---

## 1. Dataset Overview

| Attribute | Finding |
| :--- | :--- |
| **Total images** | **9,296** |
| **Total annotation files** | **25** (LabelMe JSON polygon format) |
| **Image folder** | `dataset/Flood Images/` |
| **Annotation folder** | `dataset/Annotation/` |
| **Additional metadata files** | **None** (no CSV, XML, TXT, YAML, or secondary JSON found) |
| **Official train/val/test split** | **None provided** |

---

## 2. Image Properties

| Attribute | Finding |
| :--- | :--- |
| **Image formats** | `.jpg` (6,403 files), `.png` (2,892 files), `.jpeg` (1 file) |
| **Color modes (sampled 500)** | RGB (498), RGBA (1), L/Grayscale (1) — effectively all RGB |
| **Width range** | 299 px – 4,954 px |
| **Height range** | 168 px – 3,049 px |
| **Median resolution** | 1,280 × 1,023 px (varies significantly — no fixed size) |
| **Corrupted images** | **0 detected** in 500-image sample |
| **Duplicate images** | At least 1 confirmed pair (`flood_2572.jpg` = `flood_5309.jpg`); likely more exist at scale |

---

## 3. Image Labels / Class Distribution

**Critical finding:**

> **ALL 9,296 images have filenames of the form `flood_NNNN.jpg` or `flood_NNNN.png`.**
>
> **There are NO non-flood / normal / dry road images in the dataset.**
>
> **There is NO binary class label (flood vs. non-flood) provided.**

This dataset is a **single-class flood imagery collection**, not a binary classification dataset.

---

## 4. Annotation Content — Detailed Findings

### 4.1 Annotation Format

- **Format:** LabelMe JSON (polygon annotations)
- **Shape type:** 100% `polygon` — no bounding boxes, no pixel masks
- **Only 25 of the 9,296 images have annotations**
- **All 25 annotated images exist in the Flood Images folder** (confirmed by cross-check)

### 4.2 Labels Found in Annotations

| Label | Instance Count (across 25 images) |
| :--- | :--- |
| Person | 19 |
| Tree | 14 |
| House | 13 |
| Builbing *(sic — typo for "Building")* | 13 |
| Traffic Sign | 12 |
| Bridge | 10 |
| Forest | 8 |
| Car | 6 |
| Truck | 4 |
| **Road** | **4** |

**Total polygon instances:** 103 across 25 images.

### 4.3 Critical Assessment of Annotation Labels

> **None of the 10 label categories are water, flood, inundation, puddle, depth, or severity.**

The polygon annotations describe **scene objects** (people, vehicles, buildings, bridges, trees, roads) that appear *in* flood photographs — they do **not** delineate the flood/water region itself.

- `Road` label appears in only **4 out of 25** annotated images with 1 polygon each.
- There are **no water-specific polygon annotations** in any file.
- There are **no severity or depth labels** anywhere in the dataset.

### 4.4 What the Annotated Polygons Actually Represent

Each JSON file corresponds to a scene-level annotation of objects *visible in* a flood photograph. For example:
- `flood_10.json` → `['Truck', 'Road']` (a truck and a road segment in a flood scene)
- `flood_11.json` → `['House', 'Person', 'Tree']` (houses, people, trees in a flood scene)
- `flood_109.json` → `['Bridge', 'Person', 'Traffic Sign']`

These are general scene-understanding annotations, **not flood/water delineation annotations.**

---

## 5. What This Dataset Does and Does NOT Support

| Capability | Supported? | Reason |
| :--- | :--- | :--- |
| **Binary flood/non-flood classification** | ❌ **NO** | No negative (non-flood) images exist in the dataset |
| **Flood water pixel segmentation (U-Net)** | ❌ **NO** | Annotations label scene objects, not water regions |
| **Flood water bounding box detection (YOLO)** | ❌ **NO** | No water/flood bounding boxes in annotations |
| **Affected road area calculation** | ❌ **NO** | No water mask or road-water intersection available |
| **Severity or depth label prediction** | ❌ **NO** | No severity/depth labels anywhere in dataset |
| **Scene object detection** | ✅ YES | 25 images have polygon annotations for 10 object classes |
| **Scene object segmentation** | ✅ YES | Polygon annotations can be converted to masks for 25 images |
| **Transfer learning feature extraction** | ✅ YES | 9,296 flood-scene images available for feature learning |
| **Self-supervised / unsupervised flood learning** | ✅ POSSIBLE | Could explore anomaly detection or clustering approaches |

---

## 6. Official Train/Test Split

**None provided.** No `train/`, `val/`, `test/` subdirectories exist. No metadata CSV or split file was found.

---

## 7. Technical Conclusion

### What the dataset actually is:
A collection of **9,296 web-scraped flood scene photographs** from real flooding events (primarily USA), with sparse polygon annotations on **25 images** describing visible scene objects (persons, vehicles, buildings, infrastructure). It is a **detection/segmentation dataset for general scene objects in flood environments** — it is NOT a water segmentation or flood severity dataset.

### What this means for the project:

1. **Binary classification (flood vs. non-flood) requires sourcing negative images.** The dataset must be supplemented with normal dry-road images to train any meaningful waterlogging detector.
2. **Segmentation of water regions is not supported** — the annotation polygons do not cover water/flood pixels.
3. **Severity estimation from labels is impossible** — no depth, severity, or level data exists.
4. **The 25 annotated images** are sufficient for training a small object detection or segmentation model for the labeled categories (Person, Bridge, Car, etc.) — but this is tangential to waterlogging detection.

---

## 8. What CAN Be Built — Honest Alternatives

### Option A (Recommended): Augmented Binary Flood Classifier
- **Supplement** the dataset with dry road / non-flood imagery from a public source (e.g., ImageNet road images, KITTI dataset, Google Open Images — all permissive license).
- Train **MobileNetV2 binary classifier** (flood vs. normal).
- Use **Grad-CAM strictly as qualitative visual attribution**, not affected-area measurement.
- Severity categorized as project-defined approximation only.

### Option B: Scene Object Detection on 25 Annotated Images
- Convert 25 LabelMe polygon JSONs to COCO or YOLO format.
- Train object detection model on available classes (Bridge, Person, Car, Road, etc.).
- Use Road polygon presence as a proxy for road-adjacent flood context.
- Very small training set — transfer learning is essential.

### Option C: Unsupervised / Self-Supervised Anomaly Detection
- Use all 9,296 flood images to learn flood scene features via self-supervised pretraining (SimCLR, DINO, or similar).
- No labels needed; flood images form one distribution.
- More advanced but feasible and academically honest.

---

## 9. Integrity Statement

> All findings in this report are based solely on verified inspection of the actual dataset files.
> No annotations, labels, masks, splits, or severity values have been invented or assumed.
> No model has been trained.
