# 🏠 Defect Detection in Manufacturing

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Grad--CAM-FF6F61?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/EfficientNet--B0-EE4C2C?style=for-the-badge" />
</p>
🏠 Defect Detection in Manufacturing

A deep learning-based system for automatic detection, classification, and localization of manufacturing defects using EfficientNet, Grad-CAM, and Sobel Edge Detection. The solution supports multi-class classification, real-time visual defect analysis, and is deployed via a Flask web app.

🔗 GitHub Repository

🔗 GitHub Repository

🃀 Bonus Features Implemented

✅ Grad-CAM for model interpretability✅ Flask Web App for deployment✅ Multi-Class Defect Classification (16 unified classes)✅ Transfer Learning using EfficientNet-B0✅ Edge Detection with Sobel filters for defect localization

📌 Project Highlights

Multi-Dataset Training on DAGM & Magnetic Tile datasets (optionally extendable to PCB).

Unified Classification across 16 defect types.

Grad-CAM Visualization for defect interpretability.

Edge Detection using Sobel filters for localization.

Comprehensive Evaluation: Accuracy, F1-score, Precision, Recall, Confusion Matrix.

Flask App with real-time analysis and downloadable PDF reports.

📊 Evaluation Highlights

Metric

Value

Accuracy

92.45%

Precision

0.913

Recall

0.901

F1-Score

0.907

✅ Per-Dataset Accuracy

Dataset

Accuracy

DAGM

93.1%

MT

91.7%

🔢 Reproducibility

All experiments are fully reproducible via the provided requirements.txt, training.py, and dataset structure instructions.

📊 Visualizations

Accuracy Curve

Loss Curve





Class Distribution

Confusion Matrix





Per-Dataset Accuracy

Validation Metrics





🔧 Setup Instructions

Clone Repository

git clone https://github.com/your-username/defect-detection.git
cd defect-detection

Create Environment

conda create -n defect-env python=3.8
conda activate defect-env

Install Requirements

pip install -r requirements.txt

📁 Prepare Dataset

Dataset structure:

  Dataset/
  ├── DAGM/
  │   ├── Class1/
  │   ├── Class2/
  │   ...
  └── Magnetic-Tile-Defect/
      ├── MT_Blowhole/
      ├── MT_Crack/
      ...

Download and unzip:

Download Dataset

🚀 Running the Model

1. Training

python training.py

Outputs:

best_model_unified.pth

charts/ (for plots)

results_unified.json

gradcam_visualization_unified.png

2. Launch Flask App

python app.py

Go to http://localhost:5000

Upload an image or choose a sample

View Grad-CAM, bounding boxes, edge overlay, and download the defect report PDF.

🧐 Model Architecture

Backbone: EfficientNet-B0 pretrained on ImageNet.

Classifier Head: Single Linear layer for 16 classes.

Augmentation: Rescale, Flip, ColorJitter, Gaussian Blur, Random Erasing.

📊 Methodology Summary

Dataset Harmonization:

Unified class mapping.

MT dataset upsampling to balance internal class distribution.

Training Strategy:

Balanced sampling across all classes.

Learning rate scheduling with ReduceLROnPlateau.

Evaluation:

Balanced and stratified validation.

Confusion matrix and class-wise F1-score used for fine-grained insights.

Interpretability:

Grad-CAM for heatmap visualization.

Sobel edge detection over CAM-highlighted regions.

Deployment:

Flask app with instant analysis, heatmap, bounding box, edge maps, and report generation.

🧪 Sample Analysis Output

Original Image

Heatmap Overlay

Bounding Box

Sobel Edge









📂 Directory Structure

.
├── training.py
├── app.py
├── charts/
├── dataset/
├── best_model_unified.pth
├── results_unified.json
├── gradcam_visualization_unified.png
├── templates/
├── static/
├── sample/
├── requirements.txt
└── README_FOR_SUBMISSION.md

💡 Recommendations

Consider including PCB dataset for broader generalization.

Further augment DAGM defects to improve recall.

Optuna-based hyperparameter tuning for optimal performance.

## 📌 References

* [DAGM 2007 Dataset](https://hci.iwr.uni-heidelberg.de/node/3616)
* [Magnetic Tile Dataset](https://github.com/zhiyongfu/Magnetic-Tile-Defect)
* [EfficientNet Paper](https://arxiv.org/abs/1905.16946)

---

## 📣 Credits

Developed as part of the **DevifyX ML Job Assignment**

---
