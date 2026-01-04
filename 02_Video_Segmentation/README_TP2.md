# 🎬 Video Object Segmentation: CLIPseg vs YOLO

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![CLIPseg](https://img.shields.io/badge/CLIPseg-HuggingFace-yellow)
![Polytechnique Montréal](https://img.shields.io/badge/Polytechnique_Montréal-INF6804-red)

Comparative study of zero-shot (CLIPseg) vs specialized (YOLOv8) approaches for video object segmentation.

---

## 📋 Overview

| Aspect | CLIPseg | YOLOv8 |
|--------|---------|--------|
| **Architecture** | CLIP + Decoder | CNN + FPN |
| **Approach** | Zero-shot (text prompts) | Trained on 80 classes |
| **Strengths** | Flexible, any class | Fast, precise |
| **Input** | Image + Text description | Image only |

---

## 🏗️ Architecture Comparison

```
┌────────────────────────────────────────────────────────────────┐
│                      CLIPseg Pipeline                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│  │  Image  │───→│ CLIP Visual      │───→│                 │   │
│  └─────────┘    │ Transformer      │    │   CLIPseg       │   │
│                 └──────────────────┘    │   Decoder       │──→│ Mask
│  ┌─────────┐    ┌──────────────────┐    │   (FiLM)        │   │
│  │  Text   │───→│ CLIP Text        │───→│                 │   │
│  │ "a car" │    │ Transformer      │    └─────────────────┘   │
│  └─────────┘    └──────────────────┘                          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                       YOLOv8 Pipeline                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Image  │───→│ Backbone │───→│   Neck   │───→│   Head   │  │
│  └─────────┘    │ CSPNet   │    │   FPN    │    │ Segment  │──→│ Mask
│                 └──────────┘    └──────────┘    └──────────┘  │
│                        ↓                                       │
│               S×S Grid + Class Probabilities                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Experiments & Results

All tests conducted with **confidence threshold = 0.5** for fair comparison.

### 1️⃣ Zero-Shot Segmentation (Rare Animals)

| Animal | YOLO | CLIPseg |
|--------|------|---------|
| Platypus | ❌ "Bird" | ✅ Correct |
| Seahorse | ❌ "Bird" | ⚠️ Partial |
| Octopus | ❌ No detection | ✅ Correct |
| Aye-Aye | ❌ "Cat" | ⚠️ Weak |

**Winner: CLIPseg** — Text embeddings enable zero-shot recognition of unseen classes.

---

### 2️⃣ Cluttered Scene Segmentation

Testing on COCO images with multiple objects.

| Scene | YOLO Detections | CLIPseg Detections |
|-------|-----------------|-------------------|
| Baseball stadium | 4 people, 1 bat | **~30 people**, seats, bats, cap |
| Living room | Person, table, glasses, bowls, chair, sofa | **More variety**: fridge, window, painting, radiator, guitar |

**Winner: CLIPseg** — Detects more object classes including background elements.

---

### 3️⃣ Small Object Detection

Progressive zoom test on table objects (fork, glass, remote, plate).

| Zoom Level | YOLO | CLIPseg |
|------------|------|---------|
| Original | 0/4 | 0/4 |
| Zoom 1 | 1/4 (glass) | 1/4 (glass) |
| Zoom 2 | 1/4 (remote) | 2/4 |
| Zoom 3 | 1/4 (glass) | 3/4 |
| Zoom 4 | **0/4** | **4/4** ✅ |

**Winner: CLIPseg** — Better at detecting small objects with zoom.

---

### 4️⃣ Precision (IoU on Highway Video)

Using CDNET 2012 Highway sequence (frames 1100-1200).

| Metric | CLIPseg | YOLOv8 |
|--------|---------|--------|
| Mean IoU | 0.051 | **0.953** |
| Median IoU | 0.038 | **0.979** |
| IoU Range | 0.01 - 0.22 | 0.4 - 1.0 |

**Winner: YOLO** — **17-26× better** precision on known classes (cars).

---

### 5️⃣ Execution Speed

Processing time for 1, 10, 25, 50, 100 images.

| Images | YOLO | CLIPseg | Speedup |
|--------|------|---------|---------|
| 1 | 3.6s | 4.6s | 1.3× |
| 25 | 28.4s | 46.9s | 1.7× |
| 100 | 112.3s | 193.9s | **1.72×** |

**Winner: YOLO** — Consistently **1.72× faster** due to simpler architecture.

---

### 6️⃣ Occlusion Robustness

Testing on civil images (crowded streets, film crews).

| Scene | CLIPseg | YOLO |
|-------|---------|------|
| Film crew | Few detections | ✅ People behind cameras |
| Street | Partial detections | ✅ Occluded cars & people |

**Winner: YOLO** — Better trained on occluded objects in COCO dataset.

---

## 📊 Summary Table

| Test | Winner | Margin |
|------|--------|--------|
| Zero-shot | **CLIPseg** | Unlimited classes |
| Cluttered scenes | **CLIPseg** | More variety |
| Small objects | **CLIPseg** | Progressive detection |
| Precision (IoU) | **YOLO** | 26× higher |
| Speed | **YOLO** | 1.72× faster |
| Occlusion | **YOLO** | Better robustness |

---

## 💻 Implementation

### YOLOv8 Segmentation
```python
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

def segment_yolo(image_path, conf=0.5):
    results = model(image_path, conf=conf)
    return results[0].masks, results[0].boxes
```

### CLIPseg Segmentation
```python
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

def segment_clipseg(image, text_prompt, threshold=0.5):
    inputs = processor(
        text=[text_prompt], 
        images=[image], 
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    
    mask = torch.sigmoid(outputs.logits)
    binary_mask = (mask > threshold).float()
    return binary_mask
```

### IoU Calculation
```python
def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0
```

---

## 📁 Project Structure

```
02_Video_Segmentation/
├── CLIPseg.ipynb       # CLIPseg implementation
├── YOLO.ipynb          # YOLOv8 implementation
├── data/
│   ├── highway/        # CDNET 2012 frames
│   ├── coco/           # COCO test images
│   └── animals/        # Zero-shot test images
├── results/            # Output visualizations
└── README.md           # This file
```

---

## 🔧 Requirements

```bash
pip install ultralytics transformers torch torchvision opencv-python matplotlib numpy
```

---

## 🎯 When to Use Each Method

| Scenario | Recommended |
|----------|-------------|
| Real-time surveillance | **YOLO** |
| Novel/rare objects | **CLIPseg** |
| Production deployment | **YOLO** |
| Research/exploration | **CLIPseg** |
| Medical imaging (rare classes) | **CLIPseg** |
| Autonomous vehicles | **YOLO** |

---

## 📚 References

1. Lüddecke, T., & Ecker, A. (2022). *Image Segmentation Using Text and Image Prompts*. CVPR.
2. Ultralytics. (2024). *YOLOv8 Documentation*. https://docs.ultralytics.com
3. Lin, T.Y., et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV.
4. Goyette, N., et al. (2012). *CDNET: A New Dataset for Change Detection*. CVPR Workshops.

---

## 👥 Author

- **Noémie Kpatenon**
