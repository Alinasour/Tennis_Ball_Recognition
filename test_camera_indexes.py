import cv2

'''for i in range(6):
            test_cap = cv2.VideoCapture(i)
            if test_cap.isOpened():
                print(f"Found available camera at index {i}")'''
                
                

# Open the two cameras (usually 0 and 1 for built-in and external cameras)
cap1 = cv2.VideoCapture(2)  # First camera
cap2 = cv2.VideoCapture(3)  # Second camera

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both cameras")
    exit()

while True:
    # Read frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Could not read frame from one or both cameras")
        break

    # Display the frames
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and close windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
