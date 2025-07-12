from ultralytics import YOLO
from utils.extract_detections import extract_detections

model = YOLO("best.pt")

extract_detections("videos/broadcast.mp4", "detections/broadcast", model, prefix="b", max_frames=500)
extract_detections("videos/tacticam.mp4", "detections/tacticam", model, prefix="t", max_frames=500)
