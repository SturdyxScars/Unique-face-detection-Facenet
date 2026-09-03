import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# OpenCV video capture
cap = cv2.VideoCapture('sample.mp4')  # 0 = default webcam

with mp_face_detection.FaceDetection(
        model_selection=0,  # 0 = short-range (2m), 1 = full-range (5m)
        min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Convert the BGR frame (OpenCV) to RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = face_detection.process(frame_rgb)

        # Draw detections
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Show the frame
        cv2.imshow('MediaPipe Face Detection', frame)

        if cv2.waitKey(27) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
