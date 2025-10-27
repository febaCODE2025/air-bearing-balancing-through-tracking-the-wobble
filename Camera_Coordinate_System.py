import cv2
import numpy as np
import time

# --- Camera Configuration ---
CAM_INDEX = 0  # Set the camera index (usually 0 for the default camera).

# Requested resolution for a 2K camera.
FRAME_WIDTH = 2560
FRAME_HEIGHT = 1440

# A standard display resolution to resize the frame for optimal viewing on most laptops.
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# --- Setup ---
# Open the camera device.
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW) # Use cv2.CAP_DSHOW for better compatibility on Windows

# Check if the camera opened successfully.
if not cap.isOpened():
    raise SystemExit('Error: Cannot open camera. Please check the camera index or connection.')

# Attempt to set the desired resolution.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Check the actual resolution that was set by the camera.
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Requested resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
print(f"Actual camera resolution: {int(actual_width)}x{int(actual_height)}")

# --- Add a fallback mechanism if the requested resolution is not supported ---
if int(actual_width) != FRAME_WIDTH or int(actual_height) != FRAME_HEIGHT:
    print(f"Warning: Requested resolution {FRAME_WIDTH}x{FRAME_HEIGHT} not supported.")
    print(f"Falling back to a supported resolution: {int(actual_width)}x{int(actual_height)}")
    # The camera is already set to the fallback resolution.

# A simple check to ensure a consistent frame rate.
start_time = time.time()
frame_count = 0

# --- Main Loop ---
while True:
    # Read a frame from the camera.
    ok, frame = cap.read()
    
    # If a frame was not read, break the loop. This is the part that was failing.
    if not ok:
        print("Failed to grab frame. Exiting...")
        break
        
    # --- Resize the frame for display ---
    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    # Get the height and width of the resized frame to find the center.
    h, w, _ = frame.shape
    center_x = w // 2
    center_y = h // 2

    # --- Draw the Coordinate System ---
    cv2.line(frame, (0, center_y), (w, center_y), (0, 0, 255), 2)
    cv2.line(frame, (center_x, 0), (center_x, h), (255, 0, 0), 2)
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # --- Display Info ---
    frame_count += 1
    fps = frame_count / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the drawn axes.
    cv2.imshow('Camera with Coordinate System', frame)

    # Exit the loop if the 'q' or 'ESC' key is pressed.
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()