import os
import cv2
import numpy as np
from utils.feature_matching import match_players

broadcast_dir = "detections/broadcast"
tacticam_dir = "detections/tacticam"

broadcast_images = sorted([f for f in os.listdir(broadcast_dir) if f.endswith(".jpg")])
tacticam_images = sorted([f for f in os.listdir(tacticam_dir) if f.endswith(".jpg")])

broadcast_features = np.load("features/broadcast_features.npy")
tacticam_features = np.load("features/tacticam_features.npy")

matches = match_players(broadcast_features, tacticam_features)

os.makedirs("matches", exist_ok=True)

for idx, (t_idx, b_idx, sim) in enumerate(matches[:10]):
    tac_path = os.path.join(tacticam_dir, tacticam_images[t_idx])
    bro_path = os.path.join(broadcast_dir, broadcast_images[b_idx])

    tac_img = cv2.imread(tac_path)
    bro_img = cv2.imread(bro_path)

    h = 224
    tac_img = cv2.resize(tac_img, (h, h))
    bro_img = cv2.resize(bro_img, (h, h))

    combined = cv2.hconcat([tac_img, bro_img])
    cv2.putText(combined, f"Sim: {sim:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = f"matches/match_{idx + 1}.jpg"
    cv2.imwrite(out_path, combined)
    print(f"Saved {out_path}")
