import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load pretrained ResNet50 and remove classification head
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Extract feature embeddings
def extract_features_from_folder(folder_path, max_images=50): 
    features = []
    image_names = []
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])[:max_images] 
    for idx, file in enumerate(all_files):
        print(f"[{folder_path}] Processing {idx + 1}/{len(all_files)}: {file}")
        image = Image.open(os.path.join(folder_path, file)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = resnet(tensor).squeeze().numpy()
        features.append(embedding)
        image_names.append(file)
    return np.array(features), image_names

# Matching features
def match_players(features_1, features_2):
    matches = []
    for i, feat in enumerate(features_2):
        similarities = cosine_similarity([feat], features_1)[0]
        best_match_idx = np.argmax(similarities)
        matches.append((i, best_match_idx, similarities[best_match_idx]))
    return matches
