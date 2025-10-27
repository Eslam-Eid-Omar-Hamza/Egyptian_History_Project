#  Egyptian Landmarks Detection App

This project detects and identifies famous Egyptian landmarks (like the Great Pyramids, the Sphinx, and statues of ancient Pharaohs) using a trained YOLO model.  
The app includes a bilingual (Arabic/English) graphical interface built with PyQt5 and displays historical information about each detected landmark.

---

##  Overview
- **Type:** Computer Vision Project (Final AI Project)
- **Developer:** Eslam Eid Omar Hamza
- **Frameworks:** YOLOv8, PyQt5, OpenCV, Torch
- **Goal:** Identify Egyptian landmarks from images or videos and show their historical background.

---

##  How to Run

### 1 Install dependencies
Make sure Python 3.8+ is installed, then run:
```bash
pip install -r requirements.txt
```
---


## 2 Run the app 

python main.py
---


##  Libraries Used
| Library | Purpose |
|----------|----------|
| **ultralytics** | YOLO model for image detection |
| **opencv-python** | Reading images, videos, and using the camera |
| **PyQt5** | Graphical User Interface (GUI) for the app |
| **torch & torchvision** | Deep learning framework used by YOLO |
| **numpy** | Math and array operations |
| **pandas** | Handling data files (CSV, JSON, etc.) |

---


## Model Details
Model Used: YOLOv8

Trained Model File: best_giza_landmarks.pt

Dataset: Egyptian landmarks images (pyramids, sphinx, statues, temples, etc.)

Data Augmentation: Rotations, flips, zooming, and brightness changes were used to increase the dataset size.
---



## Project Structure
egyptian_landmarks_project/
├── Assets/
│   ├── Pharaoh_Small1.png             # Welcome screen image
│   └── Pharaoh_Small2.png
│
├── Data/
│   ├── StoryAR.json                   # Arabic historical data
│   └── StoryEN.json                   # English historical data
│
├── Images.zip/                            # Test images for detection (uploaded separately as ZIP)
│
├── Model/
│   └── best_giza_landmarks.pt         # Trained YOLO model
│
├── augment_dataset.ipynb              # Data augmentation & balancing notebook
├── Egyptian_Landmarks_Project.ipynb   # Main notebook (testing & visualization)
├── Python_Main.py                     # Main GUI application
├── requirements.txt                   # Dependencies list
├── ReadME.md                          # Project documentation
├── .gitignore                         # Ignored files for Git
---


## Dataset Preparation

The dataset was prepared and balanced using a separate notebook:  
[`augment_dataset.ipynb`](augment_dataset.ipynb)

This notebook performs:
- Automatic class balancing (up to 5000 images per class)
- Advanced augmentation using **Albumentations** (flips, rotations, brightness, scaling, etc.)
- Label validation to ensure correct class IDs and consistent YOLO formatting

Additionally:
- Test images used for model evaluation are provided in a separate ZIP file  
  (`Images.zip`) inside the **Images/** file.

---


## Features
Detects landmarks in images and videos
Displays historical details about each landmark
Supports both Arabic and English
Simple and elegant graphical interface
Can use the camera for live detection
---



## Author
Created by: Eslam Eid Omar  Hamza  
AI Final Project – October 2025
---


##  Acknowledgements
Special thanks to Eng. George for his valuable guidance and support during the project.
