import cv2
import numpy as np
import time

# --- Config ---
# Ensure this dictionary matches the markers you've printed.
# The markers you provided previously are from DICT_4X4_50.
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, PARAMS)

# Set the camera index (0 is usually the default)
CAM_INDEX = 0

# --- Main Script ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit('Cannot open camera')

print("Press 'q' to quit the program.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Detect markers
    corners, ids, _ = DETECTOR.detectMarkers(frame)

    if ids is not None:
        # Draw a red dot at the center of each detected marker
        ids = ids.flatten()
        for i, c in zip(ids, corners):
            c = c.reshape(-1, 2)
            center = c.mean(axis=0)
            cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (int(center[0]) + 10, int(center[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Basic Marker Detection', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()