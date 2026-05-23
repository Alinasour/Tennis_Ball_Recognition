import cv2
import numpy as np
from ultralytics import YOLO
from correct_prespective import correct_perspective

# --- Settings ---
CAMERA_INDEX = 2
MARKER_IDS_TO_FIND = [0, 1, 2, 3]  # TL, TR, BR, BL
WALL_DIMENSIONS_CM = (600, 400)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# --- Marker detection & perspective correction function ---


# --- Compute pixel-to-centimeter ratio ---
def compute_ratio():
    return 400 / 20  # pixels per centimeter

# --- Program start ---
cap = cv2.VideoCapture(CAMERA_INDEX)
yolo_model = YOLO("/home/alinasour/Documents/best.pt")  # or any desired model

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ error.")
        break

    corrected, _ = correct_perspective(frame, MARKER_IDS_TO_FIND, WALL_DIMENSIONS_CM)
    if corrected is not None:
        # Run YOLO model on the perspective-corrected image
        results = yolo_model.predict(corrected, conf=0.3, verbose=False)
        annotated = corrected.copy()
        pixel_cm_ratio = compute_ratio()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                label = result.names[int(box.cls)]
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label} ({cx/pixel_cm_ratio:.1f}, {cy/pixel_cm_ratio:.1f})cm",
                            (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Yolo", annotated)
    else:
        cv2.imshow("finding markers", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
