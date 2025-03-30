# Import libraries
from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
from collections import defaultdict

# Function to apply NMS manually to avoid duplicate detections
def apply_nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

# Function to check if a point is inside a polygon
def point_in_polygon(point, polygon):
    x, y = point
    num = len(polygon)
    j = num - 1
    inside = False
    for i in range(num):
        if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
           (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            inside = not inside
        j = i
    return inside

# Load the trained YOLOv8 model
model = YOLO('car_detection.pt')

# Input and output video paths
input_video = 'input.mp4'
output_video = 'output_video.mp4'

# Initialize DeepSort tracker with tuned parameters
tracker = DeepSort(
    max_age=50,
    nn_budget=200,
    n_init=2,
    max_iou_distance=0.7,
    max_cosine_distance=0.4
)

# Open the input video
cap = cv2.VideoCapture(input_video)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Load zones from zones.json
with open("zones.json", "r") as file:
    zones = json.load(file)

# Verify that the loaded zones contain all eight required zones
required_zones = ['north_in', 'north_out', 'south_in', 'south_out', 'east_in', 'east_out', 'west_in', 'west_out']
if not all(zone in zones for zone in required_zones):
    exit()

# Dictionary to store turn history for each car ID
car_turns = {}
count_carId = defaultdict(int)

# Process video frame by frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    
    # Perform inference with YOLOv8
    results = model.predict(source=frame, conf=0.50, verbose=False)
    
    # Prepare detections for NMS
    boxes = []
    scores = []
    class_ids = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = float(box.conf)
            class_id = int(box.cls)
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
            class_ids.append(class_id)
    
    # Apply NMS to remove duplicate detections
    keep_indices = apply_nms(boxes, scores, iou_threshold=0.3)
    
    # Filter detections based on NMS
    filtered_detections = []
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes[idx]
        confidence = scores[idx]
        class_id = class_ids[idx]
        detection = ([x1, y1, x2 - x1, y2 - y1], confidence, class_id)
        filtered_detections.append(detection)
    
    # Update DeepSort tracker with filtered detections
    tracks = tracker.update_tracks(filtered_detections, frame=frame)
    
    # Process tracked objects
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center = (center_x, center_y)
        
        count_carId[track_id] += 1
        
        
        if track_id not in car_turns:
            car_turns[track_id] = {'right': False, 'left': False, 'u_turn': False, 'entry': None, 'exit': None}
        
        if car_turns[track_id]['entry'] is None:
            for zone_name in ['north_in', 'south_in', 'east_in', 'west_in']:
                if point_in_polygon(center, zones[zone_name]):
                    car_turns[track_id]['entry'] = zone_name.split('_')[0]
                    break
        
        if car_turns[track_id]['entry'] is not None and car_turns[track_id]['exit'] is None:
            for zone_name in ['north_out', 'south_out', 'east_out', 'west_out']:
                if point_in_polygon(center, zones[zone_name]):
                    car_turns[track_id]['exit'] = zone_name.split('_')[0]
                    break
        
        if car_turns[track_id]['entry'] and car_turns[track_id]['exit']:
            entry = car_turns[track_id]['entry']
            exit = car_turns[track_id]['exit']
            
            if entry == 'north':
                if exit == 'north':
                    car_turns[track_id]['u_turn'] = True
                elif exit == 'east':
                    car_turns[track_id]['left'] = True
                elif exit == 'west':
                    car_turns[track_id]['right'] = True
            elif entry == 'south':
                if exit == 'south':
                    car_turns[track_id]['u_turn'] = True
                elif exit == 'west':
                    car_turns[track_id]['left'] = True
                elif exit == 'east':
                    car_turns[track_id]['right'] = True
            elif entry == 'east':
                if exit == 'east':
                    car_turns[track_id]['u_turn'] = True
                elif exit == 'south':
                    car_turns[track_id]['left'] = True
                elif exit == 'north':
                    car_turns[track_id]['right'] = True
            elif entry == 'west':
                if exit == 'west':
                    car_turns[track_id]['u_turn'] = True
                elif exit == 'north':
                    car_turns[track_id]['left'] = True
                elif exit == 'south':
                    car_turns[track_id]['right'] = True
        
        if count_carId[track_id] >= 50:
            if car_turns[track_id]['u_turn']:
                color = (0, 0, 0)
            elif car_turns[track_id]['right']:
                color = (0, 0, 255)
            elif car_turns[track_id]['left']:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f'ID: {track_id}'
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    out.write(frame)
    
    frame_count += 1

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

# Final analysis
total_cars = len(car_turns)
right_turn_cars = sum(1 for turns in car_turns.values() if turns['right'])
left_turn_cars = sum(1 for turns in car_turns.values() if turns['left'])
u_turn_cars = sum(1 for turns in car_turns.values() if turns['u_turn'])

print(f'\nVideo processing complete. Output saved as {output_video}')
print('\n--- Final Analysis ---')
print(f'Total number of cars: {total_cars}')
print(f'Cars that made a right turn: {right_turn_cars}')
print(f'Cars that made a left turn: {left_turn_cars}')
print(f'Cars that made a U-turn: {u_turn_cars}')
print('\nDetailed Turn History:')
for track_id, turns in car_turns.items():
    print(f'Car ID {track_id}: Right={turns["right"]}, Left={turns["left"]}, U-turn={turns["u_turn"]}, Entry={turns["entry"]}, Exit={turns["exit"]}, Frames Appeared={count_carId[track_id]}')