import cv2
import os

def extract_detections(video_path, save_dir, model, prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while True:
        # Read each frame
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the current frame to get detections
        results = model(frame)
        boxes = results[0].boxes

        if boxes is not None:
            for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box[:4])
                cropped = frame[y1:y2, x1:x2]
                filename = f"{prefix}_f{frame_num}_p{i}.jpg"
                cv2.imwrite(os.path.join(save_dir, filename), cropped)

        frame_num += 1

    cap.release()
    print(f"Finished processing: {video_path}")
