# GVLiD: Grape Leaf Disease Image Dataset

A curated dataset of grapevine leaf images for automated disease detection using Deep Learning and Computer Vision.
##  Overview

**GVLiD (Grape Vine Leaf Disease Dataset)** is a structured image dataset designed for machine learning applications in plant pathology. It contains high-resolution grape leaf images categorized into multiple disease classes, supporting tasks such as:
* Image classification
* Disease detection
* Transfer learning
## Dataset Details

* **Total Images:** 3477
* **Image Resolution:** 1080 × 1080 pixels
* **Format:** JPEG (.jpg)
* **Classes:**

  * Black_rot
  * Esca
  * Leaf_blight
  * Healthy

## Directory Structure

GVLiD/
├── Black_rot/
├── Esca/
├── Leaf_blight/
└── Healthy/
```

Each folder contains images corresponding to a specific disease class.
## Metadata

The dataset includes a comprehensive metadata file:
metadata.json
### Key Fields:

* `Image_ID` – Unique identifier (e.g., IMG_0001)
* `image_filename` – Name of the image file
* `Class_Label` – Disease category
* `Location` – Geographical information
* `Environmental_Conditions` – Temperature, humidity, etc.
* `Plant_Health_Status` – Healthy / Diseased
* `Disease_Type` – Specific disease
* `Device_Specifications` – Capture device details
* `time_of_capture` – ISO timestamp

## Scientific Relevance

This dataset supports research in:

* Precision Agriculture
* Plant Disease Diagnosis
* Computer Vision for Agriculture
* Deep Learning model benchmarking

It is suitable for models such as:

* CNN architectures (ResNet, DenseNet, VGG)
* Transfer Learning frameworks
* Explainable AI methods

## Usage

### Load Images (Python Example)

```python
import os
from PIL import Image

dataset_path = "GVLiD/Black_rot"

for img_name in os.listdir(dataset_path):
    img = Image.open(os.path.join(dataset_path, img_name))
    img.show()
```

---

### Load Metadata

```python
import json

with open("metadata_final.json", "r") as f:
    metadata = json.load(f)

print(metadata[0])
```

---

## Applications

* Automated disease classification
* Crop health monitoring systems
* Mobile-based plant diagnosis tools
* Smart agriculture platforms

---

## Citation

If you use this dataset, please cite:

```
GVLiD: Grape Vine Leaf Disease Dataset, Mendeley Data, Version 5
```

---

##  Disclaimer

This dataset is intended for research and educational purposes only. It should not be used as a sole diagnostic tool in real-world agricultural decision-making without expert validation.

---

## Author
Shikalgar, Anisa; Savalkar, Ayush; Bhasme, Avishkar; Chavan, Snehal; Nikam, Vaishnavi (2026)

---

## 🔗 Dataset Link

Mendeley Data (Version 5):
https://data.mendeley.com/datasets/wkymf8bhcg/5

---

## Acknowledgements

We acknowledge open-source tools, datasets, and the research community contributing to advancements in AI-based agriculture.
