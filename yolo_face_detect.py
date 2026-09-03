import cv2
import numpy as np
from ultralytics import YOLO
import time
import facenet_pytorch
import os
import json

output_path = 'image_frames'
os.makedirs(output_path, exist_ok=True)

face_model = YOLO("model.pt")
box_info = []
cap = cv2.VideoCapture("sample.mp4")
frame_count = 0
human_id = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % 3 != 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_model.predict(rgb_frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0:
            human_id = boxes.shape[0]
            for i, box in enumerate(boxes):
                boxes_json = {} # fresh map for each human face detected

                x1, y1, x2, y2 = map(int, box)
                boxes_json["human_num"] = i
                boxes_json["bbox"] = (x1, y1, x2, y2)
                boxes_json["frame_id"] = frame_count
                boxes_json['path'] = f'image_frames/frame{frame_count}.jpg'
                box_info.append(boxes_json)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.imwrite(f"./image_frames/frame{frame_count}.jpg", frame)
                print(f"Frame {frame_count} saved")
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

with open('boxes.json', 'w') as f:
    json.dump(box_info, f, indent=2)
print("Done")
