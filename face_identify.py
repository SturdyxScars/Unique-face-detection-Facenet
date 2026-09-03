import cv2
import numpy as np
import os

cap = cv2.VideoCapture("sample.mp4")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
output_dir = "frames"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("Directory ", output_dir,  " Created ")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

i=0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("frame", frame)
    if i%5 == 0:
        file_path = os.path.join(os.getcwd(), output_dir, f"frames_{i}.png")
        success = cv2.imwrite(file_path, frame)
        print(success, file_path)
    if cv2.waitKey(27) & 0xFF == ord('q'):
       break
    i+=1
print(os.getcwd())
cap.release()
cv2.destroyAllWindows()
print(frame_count)
