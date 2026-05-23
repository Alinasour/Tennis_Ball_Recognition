# Tennis_Ball_Recognition
# 🎾 Smart Tennis Training Robot – Computer Vision Module  

This repository contains the code, data, and documentation of my internship project at the **Engineering Science Laboratory (ESLAB), University of Tehran**.  
The goal of the project was to design a computer vision pipeline for **real-time tennis ball detection and impact point estimation** to support a smart tennis training robot.  

## ✨ Features  
- **YOLOv8** for real-time tennis ball detection with high accuracy.  
- Perspective correction using **ArUco markers** to convert pixel coordinates into real-world (cm) units.  
- Three complementary methods for impact point estimation:  
  1. **Side view (line–box intersection)**.  
  2. **Stereo vision (two cameras with disparity estimation)**.  
  3. **Hybrid (impact timing from side, coordinates from front)**.  
- Results saved in **CSV format** including coordinates (X, Y) and depth (Z).  
- Direct integration with the robot controller via **JSON/UDP messages** for targeted ball delivery.  

## 📂 Repository Structure  
- `data/` → Sample images and dataset.  
- `training/` → YOLOv8 training scripts and results (loss curves, PR curves, confusion matrix).  
- `scripts/` → Core code for detection, perspective correction, and coordinate estimation.  
- `results/` → Example outputs, CSV records, and saved impact images.  

## 🛠 Requirements  
- Python 3.9+  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- OpenCV  
- NumPy  


📊 Results
Detection accuracy (mAP@0.5): ~0.99

Mean localization error: < 2 cm in controlled experiments

📌 About
This project was conducted as part of my undergraduate internship under the supervision of Dr. Ehsan Maani at ESLAB.
