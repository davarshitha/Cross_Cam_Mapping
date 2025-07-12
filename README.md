# Cross-Camera Player Re-Identification

This project addresses the problem of identifying and matching players across two different camera views (broadcast and tacticam) in sports footage.

## Problem Statement

Given two unsynchronized videos of the same match from different camera angles, identify which players appear in both views. This is a challenging re-identification (Re-ID) problem involving visual similarity under different lighting, perspective, and resolution.

## Methodology

1. **Detection**  
   Players are detected and cropped from each video using a fine-tuned YOLOv11 model.

2. **Feature Extraction**  
   Each player crop is passed through a pretrained ResNet50 model (without classification head) to extract visual embeddings.

3. **Matching**  
   Cosine similarity is computed between every pair of embeddings across views. The most similar pairings are considered as matched players.

4. **Visualization**  
   Top matches are visualized side-by-side and saved as `match_*.jpg` in the `matches/` folder.

## Folder Structure
Cross_Cam_Mapping/
├── best.pt                  # YOLOv11 detection model
├── detections/              # Cropped player images
│   ├── broadcast/
│   └── tacticam/
├── features/                # ResNet50 feature embeddings
├── matches/                 # Visualized match results
├── utils/                   # Detection and matching scripts
├── videos/                  # Input video files
├── run_detection.py         # Runs YOLO on both videos
├── run_matching.py          # Extracts features & matches players
├── run_visualize.py         # Generates output visualizations
├── requirements.txt         # Python dependencies
├── README.md                # This file


## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run player detection:
    python run_detection.py

3. Extract features and players:
    python run_matching.py

4. Visualize the top 10 matches:
    python run_visualize.py
