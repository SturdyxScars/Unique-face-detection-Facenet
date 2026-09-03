import json
with open('boxes.json', 'r') as f:
    box_dict = json.load(f)

frame_id_1 = 14
frame_id_2 = 19


bbox_1 = [map['bbox'] for map in box_dict if map['frame_id'] == frame_id_1]
bbox_2 = [map['bbox'] for map in box_dict if map['frame_id'] == frame_id_2]

print(bbox_1[1])
print(bbox_2[3])