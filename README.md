This project provides a robust pipeline for detecting faces in video streams and clustering them to identify unique individuals. It leverages state-of-the-art deep learning models for detection and feature extraction, followed by unsupervised clustering to isolate distinct identities.

---

### 🚀 Technologies Used

*   **Detection:**
    *   **YOLO (You Only Look Once):** Used for high-speed, accurate face detection (`model.pt`).
    *   **MediaPipe:** Alternative lightweight face detection for real-time applications.
    *   **Haar Cascades (OpenCV):** Baseline classical computer vision detection.
*   **Feature Embedding:**
    *   **DINOv2 (Facebook Research):** A self-supervised Vision Transformer (ViT) used to extract high-dimensional semantic features from face crops.
    *   **FaceNet (InceptionResnetV1):** Pre-trained on VGGFace2, used for generating discriminative face embeddings.
*   **Clustering & Analysis:**
    *   **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Used to group face embeddings into clusters representing unique individuals without needing to pre-define the number of people.
    *   **Cosine Similarity:** Metric used to compare the distance between face embeddings.
*   **Core Libraries:**
    *   `OpenCV`, `PyTorch`, `NumPy`, `Scikit-learn`, `Ultralytics`, `MediaPipe`.

---

### 📂 Project Structure

*   `yolo_detect.py`: The primary pipeline. Detects faces via YOLO, extracts embeddings using DINOv2, clusters them using DBSCAN, and saves a representative image for each unique person detected.
*   `yolo_face_detect.py`: Detects faces and logs metadata (bounding boxes, frame IDs) to `boxes.json`.
*   `face_embedding.py`: Computes face embeddings and similarity scores using FaceNet.
*   `face_mediapipe.py`: Real-time face detection using the MediaPipe framework.
*   `face_identify.py`: Basic face detection using OpenCV Haar Cascades.
*   `boxes.json`: Data store for detected face coordinates and frame references.
*   `unique_faces/`: Output directory where representative images of each unique person are stored.

---

### 🛠️ How to Use

#### 1. Installation
Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install torch torchvision ultralytics opencv-python mediapipe scikit-learn facenet-pytorch pillow
```

*Note: If you have a specific `.whl` file for `insightface`, install it manually using `pip install [filename].whl`.*

#### 2. Running the Main Pipeline
To process a video (`sample.mp4`) and extract unique faces:

```bash
python yolo_detect.py
```
This script will:
1.  Read the video and detect faces every 10 frames.
2.  Crop and normalize each detected face.
3.  Generate embeddings using DINOv2.
4.  Cluster the embeddings to find unique identities.
5.  Save the results in the `unique_faces/` folder.

#### 3. Face Metadata Extraction
To just detect faces and save their positions to a JSON file:

```bash
python yolo_face_detect.py
```

#### 4. Individual Comparison
To check the similarity between two specific faces saved in `image_frames/`:

```bash
python face_embedding.py
```

---

### 📊 Key Features
*   **Automated Identity Extraction:** Automatically detects how many unique people are in a video.
*   **Noise Robustness:** DBSCAN identifies and ignores "noise" (false detections or blurry faces).
*   **Multi-Model Support:** Choose between YOLO, MediaPipe, or Haar Cascades depending on your performance needs.