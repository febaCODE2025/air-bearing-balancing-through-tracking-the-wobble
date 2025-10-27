import cv2
import cv2.aruco as aruco
import numpy as np
import csv
import time

# --- Initialize ArUco detector (OpenCV >= 4.7) ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# --- Video capture ---
cap = cv2.VideoCapture(0)

# Storage
marker_paths = {}
origin = None  # first position of marker 0
colors = {}

# CSV setup
csv_file = open("marker_paths.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["time", "marker_id", "px_x", "px_y", "rel_x", "rel_y"])

# Color generator
def get_marker_color(marker_id):
    if marker_id not in colors:
        rng = np.random.default_rng(marker_id)
        colors[marker_id] = tuple(int(c) for c in rng.integers(0, 255, 3))
    return colors[marker_id]

background = None

print("Press 'q' to quit...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if background is None:
        background = np.zeros_like(frame)

    if ids is not None:
        ids = ids.flatten()
        for i, corner in zip(ids, corners):
            pts = corner[0].astype(int)
            center = np.mean(pts, axis=0).astype(int)

            # --- Fix origin to first seen marker 0 ---
            if i == 0 and origin is None:
                origin = center.copy()
                print(f"Origin fixed at {origin}")

            # Compute relative position
            if origin is not None:
                rel_x = center[0] - origin[0]
                rel_y = center[1] - origin[1]
            else:
                rel_x, rel_y = 0, 0

            # Save path
            if i not in marker_paths:
                marker_paths[i] = []
            marker_paths[i].append((rel_x, rel_y))

            # Draw trajectory relative to origin
            color = get_marker_color(i)
            cv2.circle(background, (center[0], center[1]), 2, color, -1)

            # Draw current marker
            cv2.polylines(frame, [pts], True, color, 2)
            cv2.circle(frame, tuple(center), 5, color, -1)
            cv2.putText(frame, f"ID {i}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save CSV (absolute + relative)
            csv_writer.writerow([time.time(), i, center[0], center[1], rel_x, rel_y])

    # Combine camera + background drawing
    combined = cv2.addWeighted(frame, 0.7, background, 0.3, 0)
    cv2.imshow("Markers and Paths", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
