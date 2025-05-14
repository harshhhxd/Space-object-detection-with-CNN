# Space-object-detection-with-CNN
## 🌌 Overview
This project detects and classifies **streaks (satellites/space debris)** and **stars** in astronomical images using a **U-Net-like CNN**. It processes 16-bit grayscale TIFF images (4418x4418 pixels), outputs centroid coordinates, and classifies objects based on shape. Designed for ground-based telescope imagery analysis.

---
Dataset Link: https://drive.google.com/drive/folders/1AqrixSQ7VLfR5aMmq4HIg0OLqKEbFaUh?usp=sharing

---
## 🛠️ Core Algorithms
1. **U-Net CNN**: Encoder-decoder architecture for precise binary segmentation.
2. **Binary Cross-Entropy Loss**: Optimizes pixel-wise classification.
3. **Connected Components Analysis**: Extracts object centroids and classifies streaks (`eccentricity > 0.9`) vs stars.
4. **Data Augmentation**: Rotations/flips to combat limited training data (35 images).

---

## 📂 File Structure
Datasets/
├── Raw_Images/ # 16-bit TIFFs (4418x4418)
└── Reference_Images/ # JPEG masks (ground truth)
src/
├── train.py # Model training script
├── predict.py # Inference on new images
└── utils.py # Helper functions
models/
└── streak_star_detector.h5 #trained model

---

## ⚙️ Requirements
- Python 3.8+
- GPU (Recommended) with CUDA 11.0+

Install dependencies
pip install -r requirements.txt
tensorflow==2.12.0
scikit-image==0.20.0
matplotlib==3.7.1
opencv-python==4.7.0
scikit-learn==1.2.2

🚀 How to Run
1. Training
python src/train.py \
  --data_dir Datasets/ \
  --batch_size 8 \
  --epochs 30 \
  --image_size 4418  # Supports 256/512/4418 (patch-based)
2. Inference
   python src/predict.py \
  --model_path models/streak_star_detector.h5 \
  --image_path sample.tiff \
  --output_dir results/
📊 Performance Metrics
Metric	Score
Test Accuracy	75.3%
IoU (Streaks)	0.91
IoU (Stars)	0.88
Inference Speed	3s/img (512x512 on RTX 3090)

📝 Notes
Reference Images: Must be binary masks (white=object, black=background).

GPU Memory: >24GB needed for full-resolution training.

Troubleshooting: Set TF_FORCE_GPU_ALLOW_GROWTH=true if facing CUDA errors.

