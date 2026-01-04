# 🔍 ROI Description: HOG vs CLIP

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![CLIP](https://img.shields.io/badge/CLIP-OpenAI-412991)
![scikit-image](https://img.shields.io/badge/scikit--image-HOG-orange)
![Polytechnique Montréal](https://img.shields.io/badge/Polytechnique_Montréal-INF6804-red)

Comparative study of classical (HOG) vs deep learning (CLIP) approaches for region of interest description and image retrieval.

---

## 📋 Overview

This project compares two fundamentally different approaches to image feature extraction:

| Aspect | HOG | CLIP |
|--------|-----|------|
| **Type** | Handcrafted features | Vision-Language Model |
| **Year** | 2005 | 2021 |
| **Input** | Image gradients | Image + Text |
| **Output** | Fixed-size histogram | 512/768-dim embedding |
| **Training** | None (algorithm) | 400M image-text pairs |

---

## 🏗️ Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                         HOG Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  Image → Grayscale → Gradient → Cell Histograms → Block Norm   │
│                                        ↓                        │
│                              Feature Vector (3780-dim)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        CLIP Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  Image ──→ ViT Encoder ──→ Image Embedding ─┐                  │
│                                              ├→ Cosine Similarity│
│  Text ───→ Transformer ──→ Text Embedding ──┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Experiments

### 1️⃣ Execution Speed
Measuring processing time for feature extraction.

| Method | Time per Image | Relative Speed |
|--------|---------------|----------------|
| HOG | ~0.015s | **2.7× faster** |
| CLIP | ~0.040s | 1× (baseline) |

### 2️⃣ Recognition Accuracy
Image retrieval using cosine distance (lower = better match).

| Test Case | HOG (avg dist) | CLIP (avg dist) | Winner |
|-----------|---------------|-----------------|--------|
| Face recognition | 0.314 | **0.073** | CLIP |
| Object matching | Higher variance | **More consistent** | CLIP |

### 3️⃣ Top-K Accuracy
Percentage of correct matches in top-5 retrieved images.

| Method | Top-5 Accuracy |
|--------|---------------|
| HOG | ~60% |
| CLIP | **100%** |

### 4️⃣ Texture vs Object Recognition
Testing whether models prioritize texture or semantic content.

**Finding:** Both models show bias toward texture features over object type, but CLIP maintains better semantic understanding.

### 5️⃣ Rotation Robustness
Testing invariance to image rotation.

| Method | Rotation Robustness |
|--------|-------------------|
| HOG | Sensitive to rotation |
| CLIP | More robust |

### 6️⃣ Bounding Box Impact
Effect of tight vs loose bounding boxes on feature quality.

**Finding:** CLIP is less sensitive to bounding box variations due to its global context understanding.

---

## 📊 CLIP Patch Size Comparison

| Model Variant | Patch Size | Speed | Detail Level |
|--------------|------------|-------|--------------|
| ViT-B/32 | 32×32 | **Fastest** | Lower |
| ViT-B/16 | 16×16 | **Best balance** | Medium |
| ViT-L/14 | 14×14 | Slowest | **Highest** |

---

## 💻 Implementation

### HOG Feature Extraction
```python
from skimage.feature import hog
from skimage import io, color

def extract_hog(image_path):
    image = io.imread(image_path)
    gray = color.rgb2gray(image)
    
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    return features
```

### CLIP Feature Extraction
```python
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def extract_clip(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.numpy().flatten()
```

### Similarity Computation
```python
from scipy.spatial.distance import cosine

def compute_similarity(feat1, feat2):
    return 1 - cosine(feat1, feat2)  # Higher = more similar
```

---

## 📁 Project Structure

```
01_ROI_Description/
├── HOG.ipynb           # HOG implementation and experiments
├── CLIP.ipynb          # CLIP implementation and experiments
├── images/             # Test images
├── results/            # Output figures and metrics
└── README.md           # This file
```

---

## 🔧 Requirements

```bash
pip install torch torchvision transformers scikit-image scipy matplotlib numpy
```

---

## 📈 Key Takeaways

| Use Case | Recommended Method |
|----------|-------------------|
| Real-time processing | **HOG** (2.7× faster) |
| High accuracy needed | **CLIP** (100% Top-5) |
| Limited compute | **HOG** (no GPU needed) |
| Zero-shot classification | **CLIP** (text prompts) |
| Edge deployment | **HOG** (lightweight) |

---

## 📚 References

1. Dalal, N., & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR.
2. Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML.
3. Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. ICLR.

---

## 👥 Authors

- **Noémie Kpatenon** & **Slimane Boussafeur**
- Polytechnique Montréal — INF6804 (2025)
