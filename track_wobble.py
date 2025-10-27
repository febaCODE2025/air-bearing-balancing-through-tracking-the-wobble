import cv2
import numpy as np
import time

# --- Config ---
# The dictionary is now correctly set to match the markers you are using.
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, PARAMS)

# The radius is set to the value you are using.
R_REF = 150.0  # mm radius of ref markers IDs 1..4

# The other settings remain the same.
CAM_INDEX = 0
RUN_SECONDS = 60 
SAVE_CSV = 'data/wobble_track.csv'
MIN_POINTS = 20

# Known platform coordinates (mm) for ref markers.
REF_COORDS = {
    1: np.array([+R_REF, 0.0]),
    2: np.array([0.0, +R_REF]),
    3: np.array([-R_REF, 0.0]),
    4: np.array([0.0, -R_REF]),
}
CENTER_ID = 0

# --- Helpers ---
def fit_circle_2d(points):
    """
    Algebraic circle fit with least squares
    points: Nx2 array
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = c
    r = np.sqrt(cx ** 2 + cy ** 2 + c0)
    return cx, cy, r

# --- Main Script ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit('Cannot open camera')

# Set camera resolution.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

pts_pix = []
start_time = time.time()
frame_count = 0
detected_frame_count = 0

while (time.time() - start_time) < RUN_SECONDS:
    ok, frame = cap.read()
    if not ok:
        break

    frame_count += 1
    
    # Detect markers and provide visual feedback.
    corners, ids, _ = DETECTOR.detectMarkers(frame)

    if ids is not None:
        cv2.putText(frame, f'Markers: {len(ids)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        ids = ids.flatten()
        
        ref_img = []
        ref_obj = []
        center_pix = None
        
        for i, c in zip(ids, corners):
            c = c.reshape(-1, 2)
            center = c.mean(axis=0)
            
            # Draw a green circle around the marker.
            cv2.circle(frame, (int(center[0]), int(center[1])), 20, (0, 255, 0), 2)
            # Draw a red dot at the center of each detected marker.
            cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

            if i in REF_COORDS:
                ref_img.append(center)
                ref_obj.append(REF_COORDS[i])
            if i == CENTER_ID:
                center_pix = center
        
        # This change prevents the code from crashing when it doesn't see all markers.
        if len(ref_img) >= 4 and center_pix is not None:
            detected_frame_count += 1
            ref_img = np.array(ref_img, dtype=np.float32)
            ref_obj = np.array(ref_obj, dtype=np.float32)
            H, _ = cv2.findHomography(ref_img, ref_obj, method=0)
            
            if H is not None:
                p = np.array([[center_pix[0], center_pix[1], 1.0]], dtype=np.float32).T
                q = H @ p
                q = (q[:2] / q[2]).flatten()
                pts_pix.append(q)
    else:
        # If no markers are detected, display a count of 0.
        cv2.putText(frame, 'Markers: 0', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    fps = frame_count / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('track', frame)
    
    if cv2.waitKey(1) == 27: # Press ESC to exit.
        break

cap.release()
cv2.destroyAllWindows()

# --- FIX IS HERE ---
# Check if enough points were collected
if len(pts_pix) < 20:
    raise SystemExit(f'Not enough points tracked. Only detected markers in {detected_frame_count} frames. Increase RUN_SECONDS or improve detection conditions.')

# Convert the list to a NumPy array before passing it to the function
pts_pix = np.array(pts_pix)

cx, cy, r = fit_circle_2d(pts_pix)
print(f'Circle center (mm): ({cx:.2f}, {cy:.2f}) radius ρ={r:.2f}mm')
print(f'Recommended COM correction vector d_COM ≈ (-{cx:.2f}, -{cy:.2f}) mm (proportional).')