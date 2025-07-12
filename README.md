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

## Download best.pt

This repository requires a pre-trained model file best.pt, which is not included in the repo due to GitHub’s 100 MB file size limit.

Download Manually:
https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view
After downloading, Place the file in the root of the Project(Cross_Cam_Mapping/best.pt)

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run player detection:
    python run_detection.py

3. Extract features and players:
    python run_matching.py

4. Visualize the top 10 matches:
    python run_visualize.py
