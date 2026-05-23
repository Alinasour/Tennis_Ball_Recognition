import cv2
import numpy as np

def correct_perspective(image, marker_ids, wall_dimensions_cm, aruco_dict_name=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) < 4:
        return None, None

    marker_centers = {}
    for i, marker_id in enumerate(ids):
        if marker_id[0] in marker_ids:
            M = cv2.moments(corners[i][0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                marker_centers[marker_id[0]] = (cX, cY)
                cv2.circle(image, (cX, cY), 8, (0, 255, 0), -1)
                cv2.putText(image, f"ID:{marker_id[0]}", (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if len(marker_centers) != 4:
        print(f"Error: Found {len(marker_centers)} markers, but require 4. Found IDs: {list(marker_centers.keys())}")
        return None, None

    sorted_centers = [marker_centers[id] for id in sorted(marker_ids)]
    src_pts = np.array(sorted_centers, dtype="float32")

    wall_width, wall_height = wall_dimensions_cm
    dst_pts = np.array([
        [0, 0],
        [wall_width, 0],
        [wall_width, wall_height],
        [0, wall_height]
    ], dtype="float32")

    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_image = cv2.warpPerspective(image, transform_matrix, (wall_width, wall_height))

    return warped_image, transform_matrix



if __name__ == "__main__":
    MARKER_IDS_TO_FIND = [0,1,2,3]
    WALL_WIDTH_CM = 300
    WALL_HEIGHT_CM = 200

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Camera opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        warped_view, matrix = correct_perspective(
            frame,
            MARKER_IDS_TO_FIND,
            (WALL_WIDTH_CM, WALL_HEIGHT_CM)
        )

        cv2.imshow("Original Camera Feed", frame)

        if warped_view is not None:
            cv2.imshow("Corrected View", warped_view)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
