import cv2
import numpy as np
from ultralytics import YOLO
import time
from correct_prespective import correct_perspective  # Using your existing module


def stereo_ball_tracker(model_path, left_cam_index, right_cam_index,
                        focal_length_px, baseline_mm, conf_threshold=0.5,
                        target_depth=2.1, depth_threshold=0.10):
    """
    Tracks tennis ball depth and captures position at target distance using your perspective correction
    """
    # Configuration
    WALL_DIMENSIONS_CM = (177, 67)  # Court dimensions (width, height)
    MARKER_IDS = [0, 1, 2, 3]  # TL, TR, BR, BL
    baseline_m = baseline_mm / 1000  # Convert to meters
    captured = False  # Capture flag to prevent duplicates

    # Load YOLO model
    model = YOLO(model_path)

    # Initialize cameras
    cap_left = cv2.VideoCapture(left_cam_index)
    cap_right = cv2.VideoCapture(right_cam_index)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("❌ Error opening cameras")
        return

    # Main processing loop
    while True:
        retL, frameL = cap_left.read()
        retR, frameR = cap_right.read()

        if not retL or not retR:
            print("❌ Frame read error")
            break

        # Detect balls in both frames
        resultsL = model(frameL, verbose=False)[0]
        resultsR = model(frameR, verbose=False)[0]

        # Find ball centers
        def get_ball_center(results):
            for box in results.boxes:
                if box.conf.item() > conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    return ((x1 + x2) / 2, (y1 + y2) / 2)
            return None

        centerL = get_ball_center(resultsL)
        centerR = get_ball_center(resultsR)

        # Process depth calculation
        if centerL and centerR:
            disparity = abs(centerL[0] - centerR[0])
            if disparity > 0:
                Z = (focal_length_px * baseline_m) / disparity
                depth_text = f"Depth: {Z:.2f}m"

                # Check if ball is at target depth
                if abs(Z - target_depth) < depth_threshold:
                    # Apply your perspective correction
                    corrected, homography = correct_perspective(frameL, MARKER_IDS, WALL_DIMENSIONS_CM)

                    if corrected is not None and homography is not None:
                        # Transform ball position to court coordinates
                        point = np.array([centerL[0], centerL[1], 1])
                        point_c = homography @ point
                        point_c /= point_c[2]
                        x_cm, y_cm = point_c[0], point_c[1]

                        # Define output dimensions (scale up if needed)
                        output_width = 800  # or your desired width
                        output_height = 600  # or your desired height

                        # Resize the corrected image
                        corrected = cv2.resize(corrected, (output_width, output_height))

                        # Scale the coordinates accordingly
                        scale_x = output_width / WALL_DIMENSIONS_CM[0]
                        scale_y = output_height / WALL_DIMENSIONS_CM[1]
                        x_scaled = x_cm * scale_x
                        y_scaled = y_cm * scale_y

                        # Draw ball position
                        cv2.circle(corrected, (int(x_scaled), int(y_scaled)),
                                   5, (0, 0, 255), -1)

                        # Add position text
                        position_text = f"Position: ({x_cm:.1f}cm, {y_cm:.1f}cm)"
                        cv2.putText(corrected, position_text, (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        # Add depth text
                        depth_text = f"Depth: {Z:.2f}m"
                        cv2.putText(corrected, depth_text, (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        # Save the image
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"capture_{timestamp}.png"
                        cv2.imwrite(filename, corrected)

                        print(f"✅ Capture saved: {filename}")
                        print(f"Ball position: ({x_cm:.1f}cm, {y_cm:.1f}cm)")
                        captured = True
                    else:
                        print("⚠️ Could not apply perspective correction")
            else:
                depth_text = "Disparity=0"
        else:
            depth_text = "Ball not detected"
            captured = False  # Reset capture flag

        # Display information
        cv2.putText(frameL, depth_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if captured:
            cv2.putText(frameL, "CAPTURED!", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Left Camera", frameL)
        cv2.imshow("Right Camera", frameR)

        # Reset capture if ball moves away
        if centerL and centerR and captured and abs(Z - target_depth) > depth_threshold * 2:
            captured = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


# ================== Execution ==================
if __name__ == "__main__":
    stereo_ball_tracker(
        model_path="/home/alinasour/Documents/best.pt",
        left_cam_index=2,
        right_cam_index=3,
        focal_length_px=554.2563,
        baseline_mm=82.84,
        target_depth=1.7,  # 1 meter target depth
        depth_threshold=0.1  # ±5cm tolerance
    )
