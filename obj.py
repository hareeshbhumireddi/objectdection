import cv2
import numpy as np

# Specify the path to the local video file
video_path = r"C:\Users\HARI\OneDrive\Desktop\python script\objectdection\carvid.mp4"

# Load the video with OpenCV
cap = cv2.VideoCapture(video_path)

# Create background subtractor
background_object = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
kernel = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction and threshold
    fgmask = background_object.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Erode and dilate the mask to remove noise
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Copy the frame for drawing
    frame_copy = frame.copy()

    for cnt in contours:
        # Adjust contour area threshold for better detection
        if cv2.contourArea(cnt) > 500:  # Adjust this value as needed
            x, y, width, height = cv2.boundingRect(cnt)
            # Draw rectangle around detected objects
            cv2.rectangle(frame_copy, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(frame_copy, "Car Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Apply mask to get the foreground
    foreground = cv2.bitwise_and(frame, frame, mask=fgmask)

    # Stack the original frame and the foreground side by side
    stacked = np.hstack((frame, foreground))

    # Display the results
    cv2.imshow("Stacked", cv2.resize(stacked, None, fx=0.4, fy=0.4))
    cv2.imshow("Foreground", foreground)
    cv2.imshow("Frame Copy", frame_copy)
    cv2.imshow("FG Mask", fgmask)

    # Adjust the delay to slow down the video playback
    if cv2.waitKey(30) == ord('q'):  # Change 1 to 30 or higher for slower playback
        break

cap.release()
cv2.destroyAllWindows()
