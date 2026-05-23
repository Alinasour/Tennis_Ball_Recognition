import cv2
import numpy as np
from ultralytics import YOLO
from correct_prespective import correct_perspective
import time
import os
import threading
import queue
from collections import deque

# --- Global variables for line selection ---
line_points = []
temp_frame = None
line_set = False


def mouse_callback(event, x, y, flags, param):
    """Mouse callback for selecting line points on hit detection camera"""
    global line_points, temp_frame, line_set

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
            cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)

            if len(line_points) == 2:
                cv2.line(temp_frame, line_points[0], line_points[1], (255, 0, 0), 2)
                line_set = True
                print(f"Line set: Start {line_points[0]}, End {line_points[1]}")
        cv2.imshow("Set Detection Line", temp_frame)


# --- Modified hit detection function ---
def tennis_hit_detection(model, frame, line_start, line_end, conf_threshold=0.5):
    """
    Detects tennis ball hits crossing a specified line in a single frame.
    Returns hit status and annotated frame.
    """
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
                cv2.putText(annotated_frame, "HIT DETECTED!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                hit_detected = True

    # Draw detection line
    cv2.line(annotated_frame, line_start, line_end, (255, 0, 0), 2)

    return hit_detected, annotated_frame


# --- Helper functions for hit detection ---
def line_intersection(a1, a2, b1, b2):
    """Calculate intersection between two line segments"""
    s1 = (a2[0] - a1[0], a2[1] - a1[1])
    s2 = (b2[0] - b1[0], b2[1] - b1[1])

    denom = -s2[0] * s1[1] + s1[0] * s2[1]
    if abs(denom) < 1e-5:  # Parallel lines
        return False

    s = (-s1[1] * (a1[0] - b1[0]) + s1[0] * (a1[1] - b1[1])) / denom
    t = (s2[0] * (a1[1] - b1[1]) - s2[1] * (a1[0] - b1[0])) / denom

    return (0 <= s <= 1 and 0 <= t <= 1)


def check_line_box_intersection(line_start, line_end, box):
    """Check if line intersects with bounding box"""
    x1, y1, x2, y2 = box
    box_lines = [
        ((x1, y1), (x2, y1)),  # Top
        ((x2, y1), (x2, y2)),  # Right
        ((x1, y2), (x2, y2)),  # Bottom
        ((x1, y1), (x1, y2))  # Left
    ]

    for box_line in box_lines:
        if line_intersection(line_start, line_end, box_line[0], box_line[1]):
            return True
    return False


# --- Camera capture threads ---
class CameraThread(threading.Thread):
    def __init__(self, camera_index, name):
        threading.Thread.__init__(self)
        self.camera_index = camera_index
        self.name = name
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        print(f"Started {self.name} camera (index: {self.camera_index})")

        while self.running:
            ret, frame = cap.read()
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                print(f"⚠️ Error reading frame from {self.name} camera")
                time.sleep(0.1)

        cap.release()
        print(f"Stopped {self.name} camera")

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False


# --- Main function ---
def main():
    global line_points, temp_frame, line_set

    # --- Settings ---
    HIT_CAMERA_INDEX = 2  # Camera for hit detection (side view)
    POSITION_CAMERA_INDEX = 3  # Camera for position tracking (top view)
    MARKER_IDS_TO_FIND = [0, 1, 2, 3]  # TL, TR, BR, BL
    WALL_DIMENSIONS_CM = (174, 67)
    SAVE_FOLDER = "tennis_hit_records"
    COOLDOWN = 1.0  # seconds between hit saves
    HIT_CONF_THRESHOLD = 0.3  # Confidence for hit camera
    POS_CONF_THRESHOLD = 0.1  # Lower confidence for position camera
    MODEL_PATH = "/home/alinasour/Documents/best.pt"
    POSITION_BUFFER_SIZE = 5  # Number of frames to buffer for position

    # --- Initialize ---
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    last_hit_time = 0
    position_buffer = deque(maxlen=POSITION_BUFFER_SIZE)  # Buffer for position frames

    # Load YOLO model (shared for both cameras)
    model = YOLO(MODEL_PATH)

    # Start camera threads
    hit_camera = CameraThread(HIT_CAMERA_INDEX, "Hit Detection")
    position_camera = CameraThread(POSITION_CAMERA_INDEX, "Position Tracking")
    hit_camera.start()
    position_camera.start()

    # Allow cameras to initialize
    time.sleep(2.0)

    # --- Line Setup Phase (using hit camera) ---
    print("Click two points to set detection line...")
    while not line_set:
        frame = hit_camera.get_frame()
        if frame is None:
            continue

        temp_frame = frame.copy()
        cv2.imshow("Set Detection Line", temp_frame)
        cv2.setMouseCallback("Set Detection Line", mouse_callback)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if line_set:
        cv2.destroyWindow("Set Detection Line")
        print(f"Detection line set: {line_points[0]} -> {line_points[1]}")
    else:
        print("Line not set, using default")
        line_points = [(0, 270), (640, 300)]

    # --- Main Loop ---
    while True:
        # Get frames from both cameras
        hit_frame = hit_camera.get_frame()
        position_frame = position_camera.get_frame()

        if hit_frame is None or position_frame is None:
            continue

        # --- Process Hit Camera ---
        hit_detected, hit_annotated = tennis_hit_detection(
            model, hit_frame, line_points[0], line_points[1], HIT_CONF_THRESHOLD
        )

        # --- Process Position Camera ---
        corrected, matrix = correct_perspective(position_frame, MARKER_IDS_TO_FIND, WALL_DIMENSIONS_CM)
        ball_position = None
        best_confidence = 0
        annotated_pos = position_frame.copy()

        if corrected is not None:
            # Run detection on corrected perspective with lower confidence threshold
            results = model.predict(corrected, conf=POS_CONF_THRESHOLD, verbose=False)
            annotated_pos = corrected.copy()

            # Calculate pixel-to-cm ratio
            pixel_cm_ratio_x = corrected.shape[1] / WALL_DIMENSIONS_CM[0]
            pixel_cm_ratio_y = corrected.shape[0] / WALL_DIMENSIONS_CM[1]
            pixel_cm_ratio = (pixel_cm_ratio_x + pixel_cm_ratio_y) / 2

            # Find the best ball detection (highest confidence)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    if result.names[cls_id] != "tennis_ball":
                        continue

                    # Get confidence score
                    conf = float(box.conf)

                    # Only consider this detection if it's better than previous
                    if conf > best_confidence:
                        best_confidence = conf

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                        # Calculate center position
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        ball_position = (cx / pixel_cm_ratio, cy / pixel_cm_ratio)

            # Draw the best detection if found
            if ball_position is not None:
                # Draw bounding box and position
                cv2.rectangle(annotated_pos, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated_pos, (cx, cy), 5, (0, 255, 255), -1)

                # Display position on the image
                position_text = f"Position: {ball_position[0]:.1f}, {ball_position[1]:.1f} cm"
                cv2.putText(
                    annotated_pos,
                    position_text,
                    (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
                )
            else:
                cv2.putText(annotated_pos, "Ball not detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(annotated_pos, "Markers not found", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Add position to buffer with timestamp
        position_buffer.append({
            "time": time.time(),
            "position": ball_position,
            "frame": annotated_pos.copy()
        })

        # --- Handle Hit Detection ---
        current_time = time.time()
        if hit_detected and (current_time - last_hit_time) > COOLDOWN:
            last_hit_time = current_time
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Save hit frame
            hit_filename = f"{SAVE_FOLDER}/hit_{timestamp}.jpg"
            cv2.imwrite(hit_filename, hit_annotated)

            # Find best position frame from buffer (closest in time)
            best_position = None
            best_frame = None
            min_time_diff = float('inf')

            for pos_data in position_buffer:
                if pos_data["position"] is None:
                    continue

                time_diff = abs(current_time - pos_data["time"])
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_position = pos_data["position"]
                    best_frame = pos_data["frame"]

            # Save position frame with ball info
            if best_frame is not None:
                pos_filename = f"{SAVE_FOLDER}/position_{timestamp}.jpg"
                cv2.imwrite(pos_filename, best_frame)

                # Save position data
                with open(f"{SAVE_FOLDER}/positions.csv", "a") as f:
                    f.write(f"{timestamp},{best_position[0]},{best_position[1]}\n")

                print(f"💥 Hit detected! Position: ({best_position[0]:.1f}, {best_position[1]:.1f}) cm")
            else:
                # Save current position frame even without detection
                pos_filename = f"{SAVE_FOLDER}/position_{timestamp}.jpg"
                cv2.imwrite(pos_filename, annotated_pos)
                print(f"💥 Hit detected! Position: Unknown")

        # Display results
        cv2.imshow("Hit Detection (Side View)", hit_annotated)
        cv2.imshow("Ball Position (Top View)", annotated_pos)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    hit_camera.stop()
    position_camera.stop()
    cv2.destroyAllWindows()
    hit_camera.join()
    position_camera.join()


if __name__ == "__main__":
    main()
