# AI-Based Road Waterlogging Detection and Severity Estimation Using Deep Learning

A technically rigorous Deep Learning project for automated detection, spatial localization, affected-area estimation, and severity categorization of road waterlogging events using the **FloodNet** aerial-imagery dataset and a **U-Net** semantic segmentation model.

---

## 1. Project Title
**AI-Based Road Waterlogging Detection and Severity Estimation Using Deep Learning**

---

## 2. Problem Statement
Urban road waterlogging caused by heavy precipitation and inadequate stormwater drainage leads to traffic paralysis, vehicular damage, accidents, and life-threatening urban flooding. Traditional monitoring relies on manual visual inspections or costly in-situ physical depth sensors installed at fixed locations, which cannot provide widespread, real-time spatial coverage across entire city road networks.

---

## 3. Motivation
Cameras (traffic CCTV, dashcams, smartphones, aerial drones) are already omnipresent in modern smart cities. Developing automated deep-learning algorithms capable of analyzing optical imagery enables early flood alerting, dynamic route replanning, and automated hazard grading without expensive dedicated hardware.

---

## 4. Project Pipeline

```
Input road/aerial image
        │
        ▼
   Preprocessing
  (resize 256×256,
   ImageNet normalize)
        │
        ▼
 U-Net Semantic Segmentation
  (10-class FloodNet taxonomy)
        │
        ▼
 Road-flooded region extraction
  (Class 3: Road-flooded,
   Class 4: Road-non-flooded,
   Class 5: Water)
        │
        ▼
 Flooded-road area percentage
  = flooded_road_pixels /
    total_road_pixels × 100
        │
        ▼
 Project-defined Severity Tier
  (NONE / LOW / MODERATE / SEVERE)
        │
        ▼
 Streamlit Web Application output
```

---

## 5. Model Architecture

### U-Net (Semantic Segmentation)
- **Architecture:** Classic encoder–decoder U-Net with skip connections.
- **Input:** 256 × 256 RGB image.
- **Output:** 10-class per-pixel segmentation mask (FloodNet taxonomy).
- **Parameters:** ~31,038,218 trainable parameters.
- **Loss:** Cross-entropy (with optional weighted/Dice variants in `train.py`).
- **Framework:** PyTorch.

### FloodNet Class Taxonomy (10 Classes)
| Class ID | Label | Relevance |
| :---: | :--- | :--- |
| 0 | Background | — |
| 1 | Building-flooded | Context |
| 2 | Building-non-flooded | Context |
| **3** | **Road-flooded** | ✅ Primary target |
| **4** | **Road-non-flooded** | ✅ Primary target |
| **5** | **Water** | ✅ Primary target |
| 6 | Tree | — |
| 7 | Vehicle | — |
| 8 | Pool | — |
| 9 | Grass | — |

---

## 6. Severity Estimation

Severity is categorized using transparent, documented surface-coverage thresholds (project-defined; no physical water-depth sensor is required):

| Severity Tier | Flooded-Road Area | Operational Impact |
| :--- | :--- | :--- |
| **NONE / DRY** | 0.0% | Road dry, normal traffic. |
| **LOW** | 0.1% – 15.0% | Minor puddles; passable with caution. |
| **MODERATE** | 15.1% – 40.0% | Significant accumulation; aquaplaning risk. |
| **SEVERE** | > 40.0% | Extensive submergence; road impassable. |

---

## 7. Dataset — FloodNet (NOT included in this repository)

> ⚠️ **The FloodNet dataset is NOT included in this GitHub repository because of its large size (~20+ GB).**

### Obtaining the Dataset
The **FloodNet-Supervised v1.0** dataset is publicly available:

- **Official source:** [https://github.com/BinaLab/FloodNet-Supervised_v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0)
- **IEEE DataPort / Zenodo:** Search for "FloodNet Supervised v1.0".

### Expected Local Structure After Download
Place the extracted dataset at the following path relative to the project root:

```
Road_Waterlogging_Project/
└── floodnet_dataset/
    ├── FloodNet-Supervised_v1.0/
    │   ├── train/
    │   │   ├── train-org-img/
    │   │   └── train-label-img/
    │   ├── val/
    │   │   ├── val-org-img/
    │   │   └── val-label-img/
    │   └── test/
    │       └── test-org-img/
    └── ColorMasks-FloodNetv1.0/
```

This path is pre-configured in [`src/config.py`](src/config.py).

---

## 8. Project Structure

```
Road_Waterlogging_Project/
├── src/
│   ├── config.py            # Central configuration (paths, hyperparameters, class map)
│   ├── data_inspection.py   # Dataset analyzer & report generator
│   ├── extract_dataset.py   # Dataset extraction helper
│   ├── preprocessing.py     # Resizing, normalization, joint augmentations
│   ├── dataset_loader.py    # PyTorch DataLoader (train/val/test splits)
│   ├── model.py             # U-Net architecture definition
│   ├── train.py             # Training loop with checkpointing & early stopping
│   ├── evaluate.py          # Evaluation (mIoU, pixel accuracy, per-class IoU)
│   ├── predict.py           # End-to-end inference & visualization pipeline
│   ├── segmentation.py      # Segmentation utilities, area calculation
│   ├── severity.py          # Severity estimation logic
│   └── utils.py             # Shared utilities
├── notebooks/
│   ├── FloodNet_UNet_Training.ipynb        # Colab-compatible training notebook
│   └── exploration_and_experiments.ipynb   # Exploratory analysis
├── tests/
│   ├── test_pipeline.py       # End-to-end pipeline tests
│   ├── test_floodnet.py       # FloodNet dataset loader tests
│   └── test_train_resume.py   # Checkpoint resume tests
├── outputs/
│   ├── graphs/          # Training curves, confusion matrices (generated)
│   ├── masks/           # Segmentation mask outputs (generated)
│   ├── predictions/     # Prediction overlay images (generated)
│   └── visualizations/  # Additional visualizations (generated)
├── reports/
│   ├── dataset_analysis.md    # Dataset inspection findings
│   ├── dataset_summary.json   # Machine-readable dataset summary
│   └── training_history.json  # Partial training history (3 epochs)
├── models/              # Model checkpoints — excluded from Git (see .gitignore)
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
├── .gitignore           # Git exclusion rules
└── README.md            # This file
```

---

## 9. Current Training Status

> ⚠️ **Training is currently incomplete / in progress.**

A partial training run of **3 epochs** was completed on the FloodNet dataset:

| Epoch | Train Loss | Val Loss | mIoU | Pixel Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 1.3076 | 1.2867 | 0.1789 | 53.2% |
| 2 | 1.1347 | 1.1596 | 0.2072 | 56.7% |
| 3 | 1.0841 | 1.0947 | 0.1933 | 56.7% |

- Best mIoU after 3 epochs: **0.2072** (epoch 2).
- `road_flooded_iou` = **0.0** at this stage — the model has not yet learned the flooded-road class (expected; this class is rare and requires more training epochs + class weighting).
- Full training requires a GPU and the complete FloodNet dataset. Recommended: ≥15 epochs on a GPU (Google Colab A100 or local CUDA GPU).
- Model checkpoint files (`models/*.pth`) are **not included** in this repository due to size (~89 MB each). They must be generated by running training locally.

---

## 10. Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU recommended for training (CPU inference is supported but slow)

### Install Dependencies
```bash
pip install -r requirements.txt
```

For GPU support with PyTorch, visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and install the appropriate CUDA version.

---

## 11. Usage

### Dataset Inspection
Inspect dataset structure, verify splits, and generate analysis reports:
```bash
python -m src.data_inspection --dataset-dir floodnet_dataset/FloodNet-Supervised_v1.0 --output-dir reports
```

### Training
Train the U-Net model on FloodNet (requires dataset and GPU):
```bash
python -m src.train
```
Checkpoints are saved to `models/` automatically. Training can be resumed from the last checkpoint.

### Evaluation
Evaluate a saved model checkpoint on the validation set:
```bash
python -m src.evaluate
```
Metrics reported: mIoU, per-class IoU (Road-flooded, Road-non-flooded, Water), pixel accuracy.

### Prediction / Inference
Run inference on a single image:
```bash
python -m src.predict --image path/to/image.jpg
```

### Web Application (Streamlit)
Launch the interactive Streamlit web interface:
```bash
streamlit run app.py
```
The app accepts image uploads, runs segmentation, and displays the predicted mask, flooded-road area percentage, and severity tier.

### Tests
```bash
python -m pytest tests/
```

---

## 12. What Teammates Need To Do After Cloning

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vaishnavij1612/Road_WaterLogging_Project-.git
   cd Road_WaterLogging_Project-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the FloodNet dataset** from the official source (see Section 7 above) and place it at `floodnet_dataset/FloodNet-Supervised_v1.0/`.

4. **Run training** to generate model checkpoints:
   ```bash
   python -m src.train
   ```

5. **(Optional) Resume from a shared checkpoint** — if a teammate shares `models/floodnet_unet_best.pth`, place it in `models/` before running evaluation or the web app.

---

## 13. What Remains To Be Completed

- [ ] Continue training for ≥15 epochs (currently 3 epochs completed).
- [ ] Improve `Road-flooded` class IoU (currently 0.0 — requires class-weighted loss or focal loss).
- [ ] Run full evaluation and report final mIoU / per-class metrics.
- [ ] Generate sample prediction visualizations for the README.
- [ ] Publish trained model checkpoint via GitHub Releases or cloud storage.
- [ ] Integrate weather data API for real-world deployment.
- [ ] Optimize Streamlit app for production use.

---

## 14. Limitations & Future Work

### Limitations
- **Training incomplete:** Results above are from early-stage training (3 epochs). Final accuracy/mIoU pending.
- **Class imbalance:** `Road-flooded` is a rare class in FloodNet; class-weighted loss or oversampling is needed.
- **Depth Measurement:** This system estimates 2D surface coverage, not physical 3D water depth.
- **Domain Shift:** Aerial imagery (FloodNet) may not directly generalize to street-level camera perspectives.

### Future Work
- Integration with real-time municipal CCTV streams.
- Weather radar / rainfall precipitation API integration.
- Monocular depth estimation (e.g., MiDaS / Depth-Anything) for volumetric estimation.
- Multi-city generalization and domain adaptation.

---

## 15. Preprocessing & Augmentation

- **Images:** Resized to 256 × 256 using bilinear interpolation; normalized with ImageNet statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]).
- **Masks:** Resized using nearest-neighbor interpolation to preserve discrete class labels.
- **Augmentation:** Synchronized horizontal flips, rotations (±15°), and affine transforms applied identically to image and mask; photometric adjustments (brightness/contrast) applied only to the RGB image.

---

## 16. Acknowledgements

- **FloodNet Dataset:** Rahnemoonfar et al., "FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding," IEEE Access, 2021.
- **U-Net:** Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015.
