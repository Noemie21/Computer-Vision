# 🎯 Multi-Object Tracking: DeepSORT + YOLOv8

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![DeepSORT](https://img.shields.io/badge/DeepSORT-Tracking-green)
![MOT17](https://img.shields.io/badge/MOT17-Benchmark-orange)
![Polytechnique Montréal](https://img.shields.io/badge/Polytechnique_Montréal-INF6804-red)

Implementation of multi-object tracking using DeepSORT with YOLOv8 detection, evaluated on the MOT17 benchmark.

---

## 📋 Overview

**Task:** Track multiple objects (cups/people) across video frames while maintaining consistent IDs.

**Solution:** DeepSORT — an evolution of SORT that adds deep appearance features to reduce identity switches.

---

## 🏗️ DeepSORT Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         DeepSORT Pipeline                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ┌─────────────┐      ┌─────────────────────────────────────────┐    │
│   │   YOLOv8    │      │              DeepSORT                   │    │
│   │  Detection  │─────→│  ┌─────────────┐  ┌──────────────────┐  │    │
│   └─────────────┘      │  │   Kalman    │  │    Appearance    │  │    │
│                        │  │   Filter    │  │    Descriptor    │  │    │
│   Video Frame          │  │  (Motion)   │  │    (Deep CNN)    │  │    │
│        │               │  └──────┬──────┘  └────────┬─────────┘  │    │
│        ▼               │         │                  │            │    │
│   ┌─────────────┐      │         ▼                  ▼            │    │
│   │  Bounding   │      │  ┌─────────────────────────────────┐    │    │
│   │    Boxes    │─────→│  │     Hungarian Algorithm         │    │──→ Tracked IDs
│   └─────────────┘      │  │  (Mahalanobis + Cosine dist)    │    │    │
│                        │  └─────────────────────────────────┘    │    │
│                        └─────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 SORT vs DeepSORT

| Component | SORT | DeepSORT |
|-----------|------|----------|
| Detection | External (YOLO) | External (YOLO) |
| Motion Model | Kalman Filter | Kalman Filter |
| Association | IoU only | **IoU + Appearance** |
| Distance Metric | IoU | **Mahalanobis + Cosine** |
| Re-identification | ❌ Poor | ✅ Robust |
| Occlusion Handling | ❌ Weak | ✅ Strong |

---

## 🎬 Video Challenges Addressed

| Challenge | DeepSORT Solution |
|-----------|-------------------|
| **Occlusion** | Kalman prediction + appearance memory |
| **Disappearance/Reappearance** | Track buffer with appearance matching |
| **Zoom/Scale changes** | Kalman state includes scale |
| **Motion blur** | Robust appearance descriptors |
| **Similar objects** | Deep appearance features distinguish |
| **Transparent objects** | Falls back to motion prediction |

---

## 📊 MOT17 Benchmark Results

Evaluated using **TrackEval** with **HOTA** metric.

### Component Metrics

| Video | AssA | LocA | DetA |
|-------|------|------|------|
| MOT17-02 | 41.9 | 83.3 | 18.6 |
| MOT17-04 | 47.4 | 83.3 | 34.8 |
| MOT17-05 | 47.1 | 79.2 | 44.3 |
| MOT17-09 | 37.7 | 82.9 | 51.3 |
| MOT17-10 | 39.9 | 79.1 | 32.6 |
| MOT17-11 | 49.2 | 85.2 | 50.4 |
| MOT17-13 | 38.6 | 79.1 | 24.0 |
| **Average** | **45.2** | **82.4** | **33.5** |

### HOTA Scores

| Video | HOTA | HOTA(0) |
|-------|------|---------|
| MOT17-02 | 27.8 | 33.9 |
| MOT17-04 | 40.5 | 51.3 |
| MOT17-05 | 45.5 | 63.4 |
| MOT17-09 | 43.8 | 56.8 |
| MOT17-10 | 35.9 | 48.0 |
| MOT17-11 | **49.7** | 61.6 |
| MOT17-13 | 30.2 | 39.2 |
| **Average** | **38.8** | **49.5** |

### Comparison with State-of-the-Art

| Tracker | HOTA | AssA | DetA |
|---------|------|------|------|
| **DeepSORT (Ours)** | 38.8 | 45.2 | 33.5 |
| MPNTrack17 | 46.6 | 47.3 | 46.2 |
| eTC17 | 45.1 | 46.4 | 44.1 |
| Tracktor++v2 | 45.1 | 45.0 | 45.3 |

**Analysis:** DeepSORT achieves competitive **AssA** (association accuracy) but lower **DetA** due to YOLOv8's struggles with distant/small objects.

---

## 📈 Metrics Explained

| Metric | Description | Our Score |
|--------|-------------|-----------|
| **LocA** | Bounding box alignment (IoU) | 82.4% ✅ |
| **DetA** | Detection accuracy (TP / TP+FN+FP) | 33.5% |
| **AssA** | ID consistency over time | 45.2% |
| **HOTA** | Combined metric √(DetA × AssA) | 38.8% |

---

## 💻 Implementation

### Cup Tracking Pipeline
```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def track_cups(image_paths, target_class=41):  # 41 = cup in COCO
    results = []
    
    for path in image_paths:
        # Run tracking (DeepSORT integrated in Ultralytics)
        tracking_results = model.track(
            source=path,
            persist=True,
            classes=[target_class],
            iou=0.7
        )
        
        for box in tracking_results[0].boxes:
            if box.cls == target_class and box.id is not None:
                frame_id = extract_frame_id(path)
                cup_id = int(box.id)
                x, y, w, h = box.xywh[0].tolist()
                results.append((frame_id, cup_id, x, y, w, h))
    
    return results

def save_results(results, output_path="results.txt"):
    with open(output_path, "w") as f:
        for r in results:
            f.write(f"{r[0]} {r[1]} {r[2]:.1f} {r[3]:.1f} {r[4]:.1f} {r[5]:.1f}\n")
```

### MOT17 Evaluation
```python
# Run tracking on MOT17 sequences
model = YOLO("yolov8n.pt")

for seq in ["MOT17-02", "MOT17-04", ...]:
    results = model.track(
        source=f"MOT17/{seq}/img1",
        persist=True,
        classes=[0],  # 0 = person
        iou=0.7
    )
    save_mot_format(results, f"results/{seq}.txt")

# Evaluate with TrackEval
# python TrackEval/scripts/run_mot_challenge.py \
#     --TRACKERS_TO_EVAL DeepSort \
#     --METRICS HOTA \
#     --USE_PARALLEL False
```

### Output Format
```
# MOT Challenge format:
# <frame_id> <object_id> <x> <y> <width> <height> <conf> <-1> <-1> <-1>

1 1 912.0 484.0 97.0 109.0 1 -1 -1 -1
1 2 1067.0 436.0 73.0 62.0 1 -1 -1 -1
2 1 912.0 484.0 97.0 109.0 1 -1 -1 -1
```

---

## 📁 Project Structure

```
03_Object_Tracking/
├── YOLO_Track.ipynb    # Main tracking implementation
├── results.txt         # Cup tracking results
├── data/
│   ├── cups/           # Cup video frames
│   └── MOT17/          # MOT17 benchmark data
├── evaluation/
│   └── TrackEval/      # Evaluation toolkit
└── README.md           # This file
```

---

## 🔧 Requirements

```bash
pip install ultralytics opencv-python numpy

# For MOT17 evaluation
git clone https://github.com/JonathonLuiten/TrackEval.git
pip install -r TrackEval/requirements.txt
```

---

## 🎯 Key Findings

### Strengths of DeepSORT
- ✅ Excellent **localization** (LocA: 82.4%)
- ✅ Competitive **association** despite occlusions
- ✅ Real-time performance
- ✅ Robust to temporary disappearances

### Limitations
- ❌ **Detection accuracy** limited by YOLOv8's performance on small/distant objects
- ❌ Dependent on external detector quality
- ❌ Lower HOTA than transformer-based trackers

### Recommendations
| Scenario | Best Choice |
|----------|-------------|
| Real-time surveillance | **DeepSORT** |
| Academic benchmarks | Transformer trackers |
| Embedded systems | **DeepSORT** (lightweight) |
| Crowded scenes | Consider ByteTrack |

---

## 📚 References

1. Wojke, N., et al. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric*. ICIP.
2. Bewley, A., et al. (2016). *Simple Online and Realtime Tracking*. ICIP.
3. Luiten, J., et al. (2021). *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking*. IJCV.
4. Ultralytics. (2024). *YOLOv8 Documentation*. https://docs.ultralytics.com
5. Milan, A., et al. (2016). *MOT16: A Benchmark for Multi-Object Tracking*. arXiv.

---

## 👥 Authors

- **Noémie Kpatenon** & **Slimane Boussafeur**
- Polytechnique Montréal — INF6804 (2025)
