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
        ((x1, y1), (x1, y2))   # Left
    ]
    for box_line in box_lines:
        if line_intersection(line_start, line_end, box_line[0], box_line[1]):
            return True
    return False


# --- Modified hit detection function ---
def tennis_hit_detection(model, frame, line_start, line_end, conf_threshold=0.5):
    """
    Detects tennis ball hits crossing a specified line in a single frame.
    Returns hit status and annotated frame.
    """
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]
    hit_detected = False
    annotated_frame = frame.copy()

    for box in results.boxes:
        cls = int(box.cls[0])
        # If your ball class isn't 0, you can also check using the class name
        if cls == 0:
            # Change: keep coordinates as float
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # float
            # Drawing needs int
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if check_line_box_intersection(line_start, line_end, (x1, y1, x2, y2)):
                cv2.putText(annotated_frame, "HIT DETECTED!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                hit_detected = True

    cv2.line(annotated_frame, line_start, line_end, (255, 0, 0), 2)
    return hit_detected, annotated_frame


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

    # Final output similar to the second code
    OUTPUT_W, OUTPUT_H = 800, 600

    # --- Initialize ---
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    last_hit_time = 0
    position_buffer = deque(maxlen=POSITION_BUFFER_SIZE)

    model = YOLO(MODEL_PATH)

    hit_camera = CameraThread(HIT_CAMERA_INDEX, "Hit Detection")
    position_camera = CameraThread(POSITION_CAMERA_INDEX, "Position Tracking")
    hit_camera.start()
    position_camera.start()
    time.sleep(2.0)

    # --- Line Setup Phase ---
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
        annotated_pos = position_frame.copy()

        ball_position = None
        best_confidence = -1.0
        best_bbox = None  # for later drawing

        if corrected is not None:
            # Detect on the corrected top-view
            results = model.predict(corrected, conf=POS_CONF_THRESHOLD, verbose=False)
            # Pixel-to-cm ratio as float
            pixel_cm_ratio_x = corrected.shape[1] / float(WALL_DIMENSIONS_CM[0])
            pixel_cm_ratio_y = corrected.shape[0] / float(WALL_DIMENSIONS_CM[1])
            pixel_cm_ratio = (pixel_cm_ratio_x + pixel_cm_ratio_y) / 2.0

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    if conf <= best_confidence:
                        continue
                    # If needed, check by class name: result.names[cls_id] == "tennis_ball"
                    # Float coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Change: compute center as float (no integer division)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    best_confidence = conf
                    best_bbox = (x1, y1, x2, y2, cx, cy)
                    # Position in centimeters with float precision
                    ball_position = (cx / pixel_cm_ratio, cy / pixel_cm_ratio)

            if ball_position is not None:
                # Match the second code's output: resize to 800x600 and show only Position text
                display_img = cv2.resize(corrected, (OUTPUT_W, OUTPUT_H))
                # Scale factors for drawing
                scale_x = OUTPUT_W / float(WALL_DIMENSIONS_CM[0])
                scale_y = OUTPUT_H / float(WALL_DIMENSIONS_CM[1])

                # Optionally draw the center point
                _, _, _, _, cx, cy = best_bbox
                x_scaled = ball_position[0] * scale_x  # x_cm * scale_x
                y_scaled = ball_position[1] * scale_y  # y_cm * scale_y

                cv2.circle(display_img, (int(x_scaled), int(y_scaled)), 5, (0, 0, 255), -1)

                position_text = f"Position: ({ball_position[0]:.1f}cm, {ball_position[1]:.1f}cm)"
                cv2.putText(display_img, position_text, (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                annotated_pos = display_img  # replace display output
            else:
                annotated_pos = cv2.resize(corrected, (OUTPUT_W, OUTPUT_H))
                cv2.putText(annotated_pos, "Ball not detected", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            annotated_pos = cv2.resize(annotated_pos, (OUTPUT_W, OUTPUT_H))
            cv2.putText(annotated_pos, "Markers not found", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Add position to buffer with timestamp
        position_buffer.append({
            "time": time.time(),
            "position": ball_position,     # <-- tuple of floats or None
            "frame": annotated_pos.copy()  # already 800x600 if corrected
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
                    best_position = pos_data["position"]  # floats
                    best_frame = pos_data["frame"]        # 800x600 with Position text

            # Save position frame with ball info
            pos_filename = f"{SAVE_FOLDER}/position_{timestamp}.jpg"
            if best_frame is not None:
                cv2.imwrite(pos_filename, best_frame)
                # Save position data (floats)
                with open(f"{SAVE_FOLDER}/positions.csv", "a") as f:
                    f.write(f"{timestamp},{best_position[0]},{best_position[1]}\n")
                print(f"💥 Hit detected! Position: ({best_position[0]:.1f}, {best_position[1]:.1f}) cm")
            else:
                # Save current position frame even without detection
                cv2.imwrite(pos_filename, annotated_pos)
                print("💥 Hit detected! Position: Unknown")

        # Display results
        cv2.imshow("Hit Detection (Side View)", hit_annotated)
        cv2.imshow("Ball Position (Top View)", annotated_pos)

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
