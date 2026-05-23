import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

def tennis_hit_detection(model_path, frame, line_start=(0, 270), line_end=(640, 300), 
                        save_folder="tennis_hit_records", conf_threshold=0.5, cooldown=1.0):
    """
    Detects tennis ball hits crossing a specified line in a single frame.
    
    Parameters:
    - model_path: Path to the YOLO model (.pt file)
    - frame: Input frame (numpy array)
    - line_start: Starting point of the detection line (x1, y1)
    - line_end: Ending point of the detection line (x2, y2)
    - save_folder: Folder to save hit frames
    - conf_threshold: Confidence threshold for detection
    - cooldown: Minimum seconds between saves to prevent duplicate saves
    
    Returns:
    - bool: True if hit detected, False otherwise
    - frame: Annotated frame (optional)
    """
    
    def check_line_box_intersection(line_start, line_end, box):
        """Check if line intersects with bounding box"""
        x1, y1, x2, y2 = box
        box_lines = [
            ((x1, y1), (x2, y1)),  # Top line
            ((x2, y1), (x2, y2)),  # Right line
            ((x1, y2), (x2, y2)),  # Bottom line
            ((x1, y1), (x1, y2))   # Left line
        ]
        
        for box_line in box_lines:
            if line_intersection(line_start, line_end, box_line[0], box_line[1]):
                return True
        return False

    def line_intersection(a1, a2, b1, b2):
        """Calculate intersection between two lines"""
        s1 = (a2[0] - a1[0], a2[1] - a1[1])
        s2 = (b2[0] - b1[0], b2[1] - b1[1])
        
        denom = (-s2[0] * s1[1] + s1[0] * s2[1])
        if denom == 0:  # Parallel lines
            return False
        
        s = (-s1[1] * (a1[0] - b1[0]) + s1[0] * (a1[1] - b1[1])) / denom
        t = ( s2[0] * (a1[1] - b1[1]) - s2[1] * (a1[0] - b1[0])) / denom
        
        return (0 <= s <= 1 and 0 <= t <= 1)

    # Load YOLO model
    model = YOLO(model_path)
    
    # Process the frame
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]
    hit_detected = False
    annotated_frame = frame.copy()

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # Tennis ball class
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Check line intersection
            if check_line_box_intersection(line_start, line_end, (x1, y1, x2, y2)):
                cv2.putText(annotated_frame, "Hit Detected!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                print(f"⚠️ Hit detected! Bounding Box: ({x1},{y1})-({x2},{y2})")
                hit_detected = True

    # Draw detection line
    cv2.line(annotated_frame, line_start, line_end, (255, 0, 0), 2)

    # Save frame if hit detected (with cooldown)
    if hit_detected:
        os.makedirs(save_folder, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{save_folder}/hit_frame_{timestamp}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"💾 Saved frame as {filename}")

    return hit_detected, annotated_frame
