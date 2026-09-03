import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from sklearn.cluster import DBSCAN

# =========================================================
# DEVICE
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# FACE DETECTOR
# =========================================================

face_model = YOLO("model.pt")

# =========================================================
# DINOv2
# =========================================================

dinov2 = torch.hub.load(
    'facebookresearch/dinov2',
    'dinov2_vits14'
)

dinov2.eval().to(device)

# =========================================================
# PREPROCESS
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# STORAGE
# =========================================================

all_embeddings = []
all_face_crops = []
all_metadata = []

# =========================================================
# VIDEO
# =========================================================

cap = cv2.VideoCapture("sample.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

print("FPS:", fps)

frame_id = 0

# =========================================================
# PROCESS VIDEO
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1

    # =====================================================
    # FRAME SKIPPING
    #
    # Video = 30 FPS
    # Process every 10th frame
    #
    # Effective embedding FPS = 3
    # =====================================================

    if frame_id % 10 != 0:
        continue

    # =====================================================
    # FACE DETECTION
    # =====================================================

    results = face_model.predict(
        frame,
        verbose=False
    )

    for r in results:

        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            # =================================================
            # EXPANDED SQUARE FACE CROP
            # =================================================

            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # =================================================
            # RGB CONVERSION
            # =================================================

            face_rgb = cv2.cvtColor(
                face_crop,
                cv2.COLOR_BGR2RGB
            )

            pil_img = Image.fromarray(face_rgb)

            # =================================================
            # PREPROCESS
            # =================================================

            input_tensor = transform(pil_img)

            input_tensor = input_tensor.unsqueeze(0).to(device)

            # =================================================
            # DINOv2 EMBEDDING
            # =================================================

            with torch.no_grad():

                embedding = dinov2(input_tensor)

            embedding = embedding.squeeze().cpu().numpy()

            # =================================================
            # L2 NORMALIZATION
            # =================================================

            embedding = embedding / np.linalg.norm(embedding)

            # =================================================
            # STORE
            # =================================================

            all_embeddings.append(embedding)

            all_face_crops.append(face_crop)

            all_metadata.append({
                "frame_id": frame_id,
                "bbox": (x1, y1, x2, y2)
            })

cap.release()

print(f"Collected embeddings: {len(all_embeddings)}")

# =========================================================
# DBSCAN CLUSTERING
# =========================================================

embeddings_np = np.array(all_embeddings)

clustering = DBSCAN(
    eps=0.25,
    min_samples=3,
    metric='cosine'
)

labels = clustering.fit_predict(embeddings_np)

# =========================================================
# UNIQUE IDENTITIES
# =========================================================

unique_labels = set(labels)

# remove noise cluster
if -1 in unique_labels:
    unique_labels.remove(-1)

print("Unique persons detected:", len(unique_labels))

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

import os

os.makedirs("unique_faces", exist_ok=True)

# =========================================================
# SAVE REPRESENTATIVE FACE OF EACH CLUSTER
# =========================================================

for cluster_id in unique_labels:

    # indices belonging to cluster
    cluster_indices = np.where(labels == cluster_id)[0]

    # pick first representative
    representative_idx = cluster_indices[0]

    representative_face = all_face_crops[
        representative_idx
    ]

    cv2.imwrite(
        f"unique_faces/person_{cluster_id}.jpg",
        representative_face
    )

print("Unique face images saved.")

# =========================================================
# OPTIONAL:
# DISPLAY UNIQUE IDENTITIES
# =========================================================

for cluster_id in unique_labels:

    cluster_indices = np.where(labels == cluster_id)[0]

    representative_idx = cluster_indices[0]

    representative_face = all_face_crops[
        representative_idx
    ]

    cv2.imshow(
        f"Person {cluster_id}",
        representative_face
    )

cv2.waitKey(0)
cv2.destroyAllWindows()