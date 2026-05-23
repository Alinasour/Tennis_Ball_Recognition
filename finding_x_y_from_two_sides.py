import cv2
import numpy as np
import os
import csv
from datetime import datetime
from ultralytics import YOLO
from Hit_detection_2 import tennis_hit_detection

#--- Configuration ---

BOARD_WIDTH = 78.0 # cm
BOARD_HEIGHT = 47.0 # cm
REAL_BOARD_POINTS = np.array([ [0, 0],
                               [BOARD_WIDTH, 0],
                               [BOARD_WIDTH, BOARD_HEIGHT],
                               [0, BOARD_HEIGHT]
                               ], dtype=np.float32)

CAMERA_TOP_ID = 2
CAMERA_SIDE_ID = 3
MODEL_PATH = "best.pt"
CSV_OUTPUT = "tennis_hit_records/hits.csv"

#--- Init ---

os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
model = YOLO(MODEL_PATH)

#--- Utilities ---

def select_board_corners(image, view_name):
    points = []
    clone = image.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(view_name, clone)

    print(f"📌 Select 4 board corners in {view_name} view (Top-Left → Clockwise)")
    cv2.imshow(view_name, clone)
    cv2.setMouseCallback(view_name, click_event)

    while len(points) < 4:
        cv2.waitKey(1)

    cv2.destroyWindow(view_name)
    return np.array(points, dtype=np.float32)

def detect_ball_center(frame, model, view_type="top"):
    results = model(frame, verbose=False)[0]
    if len(results.boxes) > 0:
        box = results.boxes[0].xyxy[0]
        x1, y1, x2, y2 = box
        if view_type == "top":
            # Use BOTTOM-CENTER for top view (better for X-coordinate)
            cx = int((x1 + x2) / 2)
            cy = int(y2)  # Bottom edge
            return (cx, cy)
        elif view_type == "side":
            # Bottom-center for side view (Y-coordinate)
            return (int((x1 + x2) / 2), int(y2))  # Now using bottom-center here too
        else:
            # Use regular center for side view (Y-coordinate)
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    return None

def pixel_to_real(point, H):
    pt = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
    real = cv2.perspectiveTransform(pt, H)
    return real[0][0] # (x, y)

def adjust_line(frame, default_start=(100, 200), default_end=(500, 200)):
    points = []
    clone = frame.copy()

    def click_line(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (255, 0, 255), -1)
            if len(points) == 2:
                cv2.line(clone, points[0], points[1], (255, 0, 255), 2)
            cv2.imshow("Line Adjust", clone)

    print("📌 Click two points to adjust hit detection line, or press ENTER for default.")
    cv2.imshow("Line Adjust", clone)
    cv2.setMouseCallback("Line Adjust", click_line)

    while len(points) < 2:
        key = cv2.waitKey(1)
        if key == 13: # ENTER
            points = [default_start, default_end]
            break

    cv2.destroyWindow("Line Adjust")
    return points[0], points[1]

def save_hit_info(x, y, frame_top, frame_side):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    info = f"X = {x:.1f} cm, Y = {y:.1f} cm"
    print("🎾 Hit Detected:", info)

    with open(CSV_OUTPUT, mode='a', newline='') as f:
        csv.writer(f).writerow([timestamp, round(x, 2), round(y, 2)])

    cv2.imwrite(f"tennis_hit_records/hit_top_{timestamp}.jpg", frame_top)
    cv2.imwrite(f"tennis_hit_records/hit_side_{timestamp}.jpg", frame_side)

#--- Main ---

cap_top = cv2.VideoCapture(CAMERA_TOP_ID)
cap_side = cv2.VideoCapture(CAMERA_SIDE_ID)
camera_matrix = np.array([
    [554.2563, 0,       320],  # Assuming image width=640 (center at 320)
    [0,        554.2563, 180],  # Assuming image height=360 (center at 180)
    [0,        0,        1]
])

dist_coeffs = np.array([
    -0.10771770030260086,   # k1
    0.1213262677192688,     # k2
    0.00091733073350042105, # p1
    0.00010589254816295579, # p2
    0.04875476285815239     # k3
])

ret1, frame_top = cap_top.read()
ret2, frame_side = cap_side.read()


# Undistort frames
frame_top = cv2.undistort(frame_top, camera_matrix, dist_coeffs)
frame_side = cv2.undistort(frame_side, camera_matrix, dist_coeffs)

if not ret1 or not ret2:
    print("❌ Failed to read initial frames.")
    exit()

pts_top = select_board_corners(frame_top, "Top View")
pts_side = select_board_corners(frame_side, "Side View")

H_top = cv2.getPerspectiveTransform(pts_top, REAL_BOARD_POINTS)
H_side = cv2.getPerspectiveTransform(pts_side, REAL_BOARD_POINTS)


line_start, line_end = adjust_line(frame_side)

print("🚀 Starting detection loop. Press Q to quit.")
while True:
    ret1, frame_top = cap_top.read()
    ret2, frame_side = cap_side.read()

    # Undistort frames
    frame_top = cv2.undistort(frame_top, camera_matrix, dist_coeffs)
    frame_side = cv2.undistort(frame_side, camera_matrix, dist_coeffs)

    if not ret1 or not ret2:
        break

    center_top = detect_ball_center(frame_top, model, view_type="top")
    center_side = detect_ball_center(frame_side, model, view_type="side")

    x_real = pixel_to_real(center_top, H_top)[0] if center_top else None
    y_real = pixel_to_real(center_side, H_side)[1] if center_side else None

    if center_top:
        # Highlight bottom-center point
        cv2.circle(frame_top, center_top, 8, (0, 255, 0), -1)
        cv2.line(frame_top, (center_top[0], center_top[1] - 20),
                 (center_top[0], center_top[1] + 20), (0, 255, 0), 2)

        # Transform to real-world X (using H_top)
        x_real = pixel_to_real(center_top, H_top)[0]

    if center_side:
        # Draw bottom-center marker
        cv2.circle(frame_side, center_side, 8, (0, 0, 255), -1)

        # Draw vertical line through contact point
        cv2.line(frame_side,
                 (center_side[0] - 20, center_side[1]),
                 (center_side[0] + 20, center_side[1]),
                 (0, 0, 255), 2)

        # Label with Y-coordinate
        real_y = pixel_to_real(center_side, H_side)[1]
        cv2.putText(frame_side, f"Y: {real_y:.1f}cm",
                    (center_side[0] + 25, center_side[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    hit_detected, _ = tennis_hit_detection(
        model,
        frame=frame_side,
        line_start=line_start,
        line_end=line_end,
        conf_threshold=0.7
    )

    if hit_detected and x_real is not None and y_real is not None:
        save_hit_info(x_real, y_real, frame_top, frame_side)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.putText(frame_top, f"X: {x_real:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame_side, f"Y: {y_real:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        with open(CSV_OUTPUT, mode='a', newline='') as f:
            csv.writer(f).writerow([timestamp, round(x_real, 2), round(y_real, 2)])

        cv2.imwrite(f"tennis_hit_records/hit_top_{timestamp}.jpg", frame_top)
        cv2.imwrite(f"tennis_hit_records/hit_side_{timestamp}.jpg", frame_side)
        
    cv2.line(frame_side, line_start, line_end, (255, 0, 255), 2)
    cv2.imshow("Top View", frame_top)
    cv2.imshow("Side View", frame_side)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('l'):
        line_start, line_end = adjust_line(frame_side, line_start, line_end)

cap_top.release()
cap_side.release()
cv2.destroyAllWindows()
