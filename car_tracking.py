# Import libraries
from ultralytics import YOLO  # YOLOv8 for object detection, chosen for its speed and accuracy
import cv2  # OpenCV for video processing and drawing bounding boxes
import numpy as np  # NumPy for numerical operations, e.g., NMS and polygon checks
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSort for robust multi-object tracking
import json  # JSON for loading zone definitions
import os  # OS for file existence checks
from collections import defaultdict  # Defaultdict to count frames per car ID
import sys  # Sys for graceful exits on errors
import matplotlib.pyplot as plt  # Matplotlib for visualizing turn analysis
from tqdm import tqdm  # TQDM for progress bar during video processing

# Function to apply NMS manually to avoid duplicate detections
def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression (NMS) removes overlapping bounding boxes to prevent duplicate detections.
    Why: YOLOv8 may detect the same car multiple times; NMS ensures only the best detection is kept.
    """
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
    """
    Ray-casting algorithm to determine if a point (car center) is inside a zone polygon.
    Why: Essential for classifying turns by checking which zones a car enters/exits.
    """
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

# Configuration: Define file paths as constants for easy modification
MODEL_PATH = 'car_detection.pt'  # Pre-trained YOLOv8 model for car detection
INPUT_VIDEO = 'input.mp4'  # Input video containing cars
OUTPUT_VIDEO = 'output_video.mp4'  # Output video with annotations
ZONES_FILE = 'zones.json'  # JSON file defining turn zones

# Verify required files exist to ensure the script runs smoothly
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found. Please provide a valid YOLOv8 model.")
    sys.exit(1)
if not os.path.exists(INPUT_VIDEO):
    print(f"Error: Input video '{INPUT_VIDEO}' not found. Please provide a valid video file.")
    sys.exit(1)
if not os.path.exists(ZONES_FILE):
    print(f"Error: Zones file '{ZONES_FILE}' not found. Run 'flask_app.py' to create it.")
    sys.exit(1)

# Load the trained YOLOv8 model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    sys.exit(1)

# Initialize DeepSort tracker with tuned parameters
tracker = DeepSort(
    max_age=50,  # Max frames a track can persist without detection (handles occlusions)
    nn_budget=200,  # Memory for appearance features (improves tracking robustness)
    n_init=2,  # Frames needed to confirm a track (reduces false positives)
    max_iou_distance=0.7,  # Max IoU distance for track association (motion-based)
    max_cosine_distance=0.4  # Max cosine distance for appearance (feature-based)
)
# Why: DeepSort combines motion and appearance for reliable tracking in crowded scenes.

# Open the input video
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Error: Could not open video '{INPUT_VIDEO}'. Check file format or path.")
    sys.exit(1)

# Get video properties for output configuration and progress bar
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames for tqdm
# Why: These properties ensure the output video matches the input and enable progress tracking.

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec for compatibility
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))
# Why: VideoWriter saves the processed frames as a video file.

# Load zones from zones.json
try:
    with open(ZONES_FILE, "r") as file:
        zones = json.load(file)
except json.JSONDecodeError:
    print(f"Error: '{ZONES_FILE}' is not a valid JSON file.")
    sys.exit(1)

# Verify that the loaded zones contain all eight required zones
required_zones = ['north_in', 'north_out', 'south_in', 'south_out', 'east_in', 'east_out', 'west_in', 'west_out']
if not all(zone in zones for zone in required_zones):
    print(f"Error: '{ZONES_FILE}' must contain all required zones: {required_zones}")
    sys.exit(1)
# Why: All zones are needed to accurately classify turns in all directions.

# Dictionary to store turn history and frame counts for each car ID
car_turns = {}  # Tracks turn types and entry/exit zones
count_carId = defaultdict(int)  # Counts frames each car appears in

# Process video frame by frame with progress bar
frame_count = 0
with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform inference with YOLOv8
        results = model.predict(source=frame, conf=0.50, verbose=False)
        # Why: Confidence threshold of 0.5 balances precision and recall for car detection.
        
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
        keep_indices = apply_nms(boxes, scores, iou_threshold=0.5)
        
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
        # Why: Tracks cars across frames, assigning consistent IDs.
        
        # Process tracked objects
        for track in tracks:
            if not track.is_confirmed():
                continue  # Skip unconfirmed tracks to avoid noise
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center = (center_x, center_y)
            
            count_carId[track_id] += 1  # Increment frame count for this car
            
            if track_id not in car_turns:
                car_turns[track_id] = {'right': False, 'left': False, 'u_turn': False, 'entry': None, 'exit': None}
            
            # Detect entry zone
            if car_turns[track_id]['entry'] is None:
                for zone_name in ['north_in', 'south_in', 'east_in', 'west_in']:
                    if point_in_polygon(center, zones[zone_name]):
                        car_turns[track_id]['entry'] = zone_name.split('_')[0]
                        break
            
            # Detect exit zone
            if car_turns[track_id]['entry'] is not None and car_turns[track_id]['exit'] is None:
                for zone_name in ['north_out', 'south_out', 'east_out', 'west_out']:
                    if point_in_polygon(center, zones[zone_name]):
                        car_turns[track_id]['exit'] = zone_name.split('_')[0]
                        break
            
            # Classify turns based on entry and exit zones
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
            # Why: Turn classification logic maps entry/exit combinations to specific turns.
            
            # Draw bounding boxes after 50 frames to ensure stable tracking
            if count_carId[track_id] >= 50:
                if car_turns[track_id]['u_turn']:
                    color = (0, 0, 0)  # Black for U-turn
                elif car_turns[track_id]['right']:
                    color = (0, 0, 255)  # Red for right turn
                elif car_turns[track_id]['left']:
                    color = (0, 255, 0)  # Green for left turn
                else:
                    color = (255, 0, 0)  # Blue for no turn (BGR format)
                
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                label = f'ID: {track_id}'
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        out.write(frame)
        frame_count += 1
        pbar.update(1)  # Update progress bar

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

# Why: Properly closes video resources to avoid memory leaks.

# Final analysis: Filter cars appearing >= 100 seconds
MIN_SECONDS = 1  # Minimum time threshold in seconds
min_frames = MIN_SECONDS * fps  # Convert seconds to frames based on FPS

# Filter car_turns for cars meeting the time threshold
filtered_car_turns = {track_id: turns for track_id, turns in car_turns.items() 
                      if count_carId[track_id] >= min_frames}

total_cars = len(filtered_car_turns)
right_turn_cars = sum(1 for turns in filtered_car_turns.values() if turns['right'])
left_turn_cars = sum(1 for turns in filtered_car_turns.values() if turns['left'])
u_turn_cars = sum(1 for turns in filtered_car_turns.values() if turns['u_turn'])
no_turn_cars = sum(1 for turns in filtered_car_turns.values() 
                   if not (turns['right'] or turns['left'] or turns['u_turn']))
# Why: Added no_turn_cars to count cars with no detected turns, enhancing analysis completeness.

print(f'\nVideo processing complete. Output saved as {OUTPUT_VIDEO}')
print(f'\n--- Final Analysis ---\n')
print(f'Total number of cars: {total_cars}')
print(f'Cars that made a right turn: {right_turn_cars}')
print(f'Cars that made a left turn: {left_turn_cars}')
print(f'Cars that made a U-turn: {u_turn_cars}')
print(f'Cars with no turns: {no_turn_cars}')
print('\nDetailed Turn History (Filtered):')

# Visualization of turn analysis including "No Turn"
turn_types = ['Right Turn', 'Left Turn', 'U-Turn', 'No Turn']
turn_counts = [right_turn_cars, left_turn_cars, u_turn_cars, no_turn_cars]
colors = ['red', 'green', 'black', 'blue']

plt.figure(figsize=(10, 6))
plt.bar(turn_types, turn_counts, color=colors)
plt.title('Turn Analysis')
plt.xlabel('Turn Type')
plt.ylabel('Number of Cars')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(turn_counts):
    plt.text(i, v + 0.5, str(v), ha='center', va='bottom')
plt.savefig('turn_analysis.png')
plt.show()
# Why: Bar chart now includes "No Turn" for a complete visual summary of all car behaviors.