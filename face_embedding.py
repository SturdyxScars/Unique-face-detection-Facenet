import json
import os
import cv2
import numpy as np
from facenet_pytorch import  InceptionResnetV1
import torch
from sklearn.metrics.pairwise import cosine_similarity
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Face Embedding Model Loaded")

with open('boxes.json', 'r') as f:
    box_dict = json.load(f)

frame_id_1 = 14
frame_id_2 = 19

image_1 = cv2.imread(f'./image_frames/frame{frame_id_1}.jpg')
cv2.imshow("cropped_1", image_1)
cv2.waitKey(0)
image_2 = cv2.imread(f'./image_frames/frame{frame_id_2}.jpg')
cv2.imshow("cropped_1", image_2)
cv2.waitKey(0)
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

bbox_1 = [map['bbox'] for map in box_dict if map['frame_id'] == frame_id_1]
bbox_2 = [map['bbox'] for map in box_dict if map['frame_id'] == frame_id_2]

cropped_1 = image_1[bbox_1[0][1] : bbox_1[0][3], bbox_1[0][0] : bbox_1[0][2]]
cropped_1 = cv2.resize(cropped_1, (160, 160))
cropped_1 = np.expand_dims(cropped_1, axis=0)
print(cropped_1.shape)
cropped_1_tensor = torch.tensor(cropped_1, dtype=torch.float32).permute(0, 3, 1, 2)
cropped_1_tensor = (cropped_1_tensor - 127.5) / 128.0

cropped_2 = image_2[bbox_2[0][1] : bbox_2[0][3], bbox_2[0][0] : bbox_2[0][2]]
cropped_2 = cv2.resize(cropped_2, (160, 160))
cropped_2 = np.expand_dims(cropped_2, axis=0)
print(cropped_2.shape)
cropped_2_tensor = torch.tensor(cropped_2, dtype=torch.float32).permute(0, 3, 1, 2)
cropped_2_tensor = (cropped_2_tensor - 127.5) / 128.0

with torch.no_grad():
    emb_1 = resnet(cropped_1_tensor.to(device))
    emb_2 = resnet(cropped_2_tensor.to(device))
    print(emb_1.shape, emb_2.shape)

def get_similarity(emb_1, emb_2):
    emb_1 = emb_1.cpu().numpy().reshape(1, -1)
    emb_2 = emb_2.cpu().numpy().reshape(1, -1)
    return cosine_similarity(emb_1, emb_2)[0][0]
print(get_similarity(emb_1, emb_2))
print(emb_1.shape, emb_2.shape)