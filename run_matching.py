import numpy as np
from utils.feature_matching import extract_features_from_folder, match_players

# Extract features
broadcast_features, broadcast_names = extract_features_from_folder("detections/broadcast", max_images=500)
tacticam_features, tacticam_names = extract_features_from_folder("detections/tacticam", max_images=500)

np.save("features/broadcast_features.npy", broadcast_features)
np.save("features/tacticam_features.npy", tacticam_features)

# Match players across views
matches = match_players(broadcast_features, tacticam_features)

# Print results
print("\n Player Matching Results:\n")
for t_idx, b_idx, sim in matches:
    print(f"Tacticam: {tacticam_names[t_idx]} ↔ Broadcast: {broadcast_names[b_idx]}  |  Similarity: {sim:.2f}")
