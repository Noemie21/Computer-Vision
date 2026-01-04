# 🎯 Computer Vision Projects

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![CLIP](https://img.shields.io/badge/CLIP-OpenAI-412991)
![Polytechnique Montréal](https://img.shields.io/badge/Polytechnique_Montréal-INF6804-red)

A collection of computer vision projects exploring classical and deep learning approaches for image description, segmentation, and object tracking.

---

## 📚 Projects Overview

| Project | Topic | Methods Compared | Key Findings |
|---------|-------|------------------|--------------|
| [**01_ROI_Description**](./01_ROI_Description/) | Image Feature Extraction | HOG vs CLIP | CLIP 100% Top-5 accuracy, HOG 2.7× faster |
| [**02_Video_Segmentation**](./02_Video_Segmentation/) | Semantic Segmentation | CLIPseg vs YOLOv8 | YOLO 26× better IoU, CLIPseg wins zero-shot |
| [**03_Object_Tracking**](./03_Object_Tracking/) | Multi-Object Tracking | DeepSORT + YOLOv8 | 82.4% LocA, 38.8% HOTA on MOT17 |

---

## 🏗️ Repository Structure

```
Computer-Vision/
│
├── 01_ROI_Description/          # HOG vs CLIP feature extraction
│   ├── HOG.ipynb
│   ├── CLIP.ipynb
│   └── README.md
│
├── 02_Video_Segmentation/       # CLIPseg vs YOLO segmentation
│   ├── CLIPseg.ipynb
│   ├── YOLO.ipynb
│   └── README.md
│
├── 03_Object_Tracking/          # DeepSORT multi-object tracking
│   ├── YOLO_Track.ipynb
│   ├── results.txt
│   └── README.md
│
└── README.md                    # This file
```

---

## 🔬 Methods & Technologies

### Classical Computer Vision
- **HOG (Histogram of Oriented Gradients)** — Handcrafted feature descriptor using gradient orientations

### Deep Learning / Vision-Language Models
- **CLIP** — OpenAI's contrastive vision-language model for zero-shot classification
- **CLIPseg** — CLIP-based semantic segmentation with text prompts
- **YOLOv8** — State-of-the-art real-time object detection and segmentation
- **DeepSORT** — Deep learning-enhanced multi-object tracking with appearance descriptors

### Frameworks & Libraries
```
torch, torchvision          # Deep learning
ultralytics                 # YOLOv8
transformers                # CLIP, CLIPseg
scikit-image                # HOG implementation
opencv-python               # Image processing
scipy                       # Distance metrics
TrackEval                   # MOT benchmark evaluation
```

---

## 📊 Key Results Summary

### 01 — Feature Extraction (HOG vs CLIP)
| Metric | HOG | CLIP |
|--------|-----|------|
| Execution Speed | **2.7× faster** | 1× |
| Face Recognition (avg distance) | 0.314 | **0.073** |
| Top-5 Accuracy | ~60% | **100%** |

### 02 — Segmentation (CLIPseg vs YOLO)
| Test Case | Winner |
|-----------|--------|
| Zero-shot (rare animals) | **CLIPseg** |
| Cluttered scenes | **CLIPseg** |
| Small objects | **CLIPseg** |
| IoU Precision | **YOLO** (26× higher) |
| Speed | **YOLO** (1.72× faster) |
| Occlusion handling | **YOLO** |

### 03 — Object Tracking (DeepSORT)
| Metric | Score | Description |
|--------|-------|-------------|
| LocA | 82.4% | Bounding box localization |
| AssA | 45.2% | ID association over time |
| DetA | 33.5% | Detection accuracy |
| HOTA | 38.8% | Overall tracking quality |

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Noemie21/Computer-Vision.git
cd Computer-Vision

# Install dependencies
pip install torch torchvision ultralytics transformers scikit-image opencv-python scipy

# Run any notebook
jupyter notebook 01_ROI_Description/CLIP.ipynb
```

---


## 📄 License

This project is part of academic coursework at Polytechnique Montréal.

---

## 🔗 References

- [CLIP Paper (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [CLIPseg Paper (Lüddecke & Ecker, 2022)](https://arxiv.org/abs/2112.10003)
- [Ultralytics YOLOv8](https://docs.ultralytics.com)
- [DeepSORT](https://github.com/nwojke/deep_sort)
- [HOTA Metric (Luiten et al., 2021)](https://arxiv.org/abs/2009.07736)
