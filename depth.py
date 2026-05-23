import cv2
import numpy as np
from ultralytics import YOLO

def stereo_ball_distance_tracker(model_path="/home/alinasour/Documents/best.pt",
                               left_cam_index=2,
                               right_cam_index=3,
                               focal_length_px=554.2563,
                               baseline_mm=82.84,
                               conf_threshold=0.5,
                               calib_file=None):
    """
    Main function for ball tracking and stereo distance calculation
    
    Parameters:
        model_path: Path to YOLO model
        left_cam_index: Left camera index
        right_cam_index: Right camera index  
        focal_length_px: Focal length in pixels
        baseline_mm: Distance between cameras in mm
        conf_threshold: Detection confidence threshold
        calib_file: Path to stereo calibration file (optional)
    """

    # Convert units
    baseline_m = baseline_mm / 1000  # Convert mm to meters

    # Load YOLO model
    model = YOLO(model_path)

    # Load stereo calibration data if available
    rectify_maps = None
    if calib_file:
        try:
            calib_data = np.load(calib_file)
            left_map1 = calib_data['left_map1']
            left_map2 = calib_data['left_map2']
            right_map1 = calib_data['right_map1'] 
            right_map2 = calib_data['right_map2']
            
            # Use calibration parameters if available
            if 'focal_length' in calib_data:
                focal_length_px = calib_data['focal_length']
            if 'baseline' in calib_data:
                baseline_m = calib_data['baseline']
                
            rectify_maps = (left_map1, left_map2, right_map1, right_map2)
            print("✅ Calibration data loaded successfully")
        except Exception as e:
            print(f"❌ Error loading calibration file: {str(e)}")
            rectify_maps = None

    # Initialize cameras
    cap_left = cv2.VideoCapture(left_cam_index)
    cap_right = cv2.VideoCapture(right_cam_index)

    # Set camera resolution (should match calibration)
    if rectify_maps:
        h, w = left_map1.shape[:2]
        cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("❌ Error opening cameras")
        return

    print("Tracking running. Press 'q' to quit...")
    print("Note: For best accuracy, use stereo calibration file")

    while True:
        # Capture frames
        retL, frameL = cap_left.read()
        retR, frameR = cap_right.read()

        if not retL or not retR:
            print("❌ Frame capture failed")
            break

        # Rectify frames if calibration available
        if rectify_maps:
            frameL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)
            frameR = cv2.remap(frameR, right_map1, right_map2, cv2.INTER_LINEAR)

        # Run detection
        resultsL = model(frameL, verbose=False)
        resultsR = model(frameR, verbose=False)

        def get_ball_center(results, frame=None):
            """Helper function to get ball center coordinates"""
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    if conf > conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        
                        # Draw bounding box (optional)
                        if frame is not None:
                            cv2.rectangle(frame, (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        (0, 255, 0), 2)
                            cv2.circle(frame, (int(cx), int(cy)), 
                                    5, (0, 0, 255), -1)
                        return (cx, cy)
            return None

        # Get ball centers with visualization
        centerL = get_ball_center(resultsL, frameL)
        centerR = get_ball_center(resultsR, frameR)

        # Distance calculation
        if centerL and centerR:
            # Calculate signed disparity
            disparity = centerL[0] - centerR[0]
            
            # Only calculate distance for valid disparity
            if abs(disparity) > 1:  # Minimum disparity threshold
                Z = (focal_length_px * baseline_m) / abs(disparity)
                text = f"Distance: {Z:.2f}m | Disparity: {disparity:.1f}px"
                color = (0, 255, 0)  # Green for valid
            else:
                text = "Disparity too small (object too far)"
                color = (0, 0, 255)  # Red for warning
        else:
            # Detection failed cases
            if not centerL and not centerR:
                text = "Ball not detected in either camera"
            elif not centerL:
                text = "Ball not detected in left camera" 
            else:
                text = "Ball not detected in right camera"
            color = (0, 0, 255)  # Red for error

        # Display status
        cv2.putText(frameL, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2)

        # Show frames
        cv2.imshow("Left Camera", frameL)
        cv2.imshow("Right Camera", frameR)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage with calibration file
    stereo_ball_distance_tracker(
        calib_file="stereo_calib.npz"  # Optional: path to calibration file
    )
