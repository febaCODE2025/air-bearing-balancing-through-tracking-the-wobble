import cv2
import numpy as np
import time
import os

# --- Config ---
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, PARAMS)

R_REF = 120.0  # mm

CAM_INDEX = 0
RUN_SECONDS = 90
CSV_FILE = 'data/markers_behavior_log.csv'

REF_COORDS = {
    1: np.array([+R_REF, 0.0]),
    2: np.array([0.0, +R_REF]),
    3: np.array([-R_REF, 0.0]),
    4: np.array([0.0, -R_REF]),
}
CENTER_ID = 0

# --- The list of colors for each marker path (BGR format).
PATH_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
]
marker_colors = {}

# --- Main Script ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit('Cannot open camera')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

marker_paths = {}
all_marker_data = []

start_time = time.time()
frame_count = 0
detected_frame_count = 0

_, background = cap.read()
if background is None:
    raise SystemExit('Cannot read a frame from the camera. Check camera connection.')
background = np.zeros_like(background)


while (time.time() - start_time) < RUN_SECONDS:
    ok, frame = cap.read()
    if not ok:
        break

    frame_count += 1

    corners, ids, _ = DETECTOR.detectMarkers(frame)

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    current_frame_data = {'time': time.time() - start_time}

    # Draw the fixed COR origin axes at the center of the frame.
    h, w, _ = frame.shape
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)  # Blue Y-axis.
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 2)  # Blue X-axis.

    if ids is not None:
        ids_flat = ids.flatten()

        for i, c in zip(ids_flat, corners):
            c = c.reshape(-1, 2)
            center = c.mean(axis=0)

            # Draw a green circle around the marker.
            cv2.circle(frame, (int(center[0]), int(center[1])), 20, (0, 255, 0), 2)
            # Draw a red dot at the center of each detected marker.
            cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (int(center[0]) + 10, int(center[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            current_frame_data[f'marker_{i}_px_x'] = center[0]
            current_frame_data[f'marker_{i}_px_y'] = center[1]

            if i not in marker_paths:
                marker_paths[i] = []

            if i not in marker_colors:
                marker_colors[i] = PATH_COLORS[len(marker_colors) % len(PATH_COLORS)]

            if len(marker_paths[i]) > 0:
                pt1 = tuple(marker_paths[i][-1].astype(int))
                pt2 = tuple(center.astype(int))
                path_color = marker_colors[i]
                if i == CENTER_ID:
                    path_color = (0, 255, 255) # Yellow for Marker 0.
                cv2.line(background, pt1, pt2, path_color, 2)

            marker_paths[i].append(center)

    final_frame = cv2.addWeighted(frame, 1, background, 1, 0)

    fps = frame_count / (time.time() - start_time)
    cv2.putText(final_frame, f'Markers: {len(ids) if ids is not None else 0}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(final_frame, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Markers Behavior', final_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

final_image_path = os.path.join('data', 'final_path_plot.png')
cv2.imwrite(final_image_path, background)
print(f"Final path plot saved to {final_image_path}")

if all_marker_data:
    os.makedirs('data', exist_ok=True)
    with open(CSV_FILE, 'w') as f:
        header = ['time']
        unique_markers = sorted(list(set(key.split('_')[1] for d in all_marker_data for key in d if key.startswith('marker'))))
        for marker_id in unique_markers:
            header.extend([f'{marker_id}_px_x', f'{marker_id}_px_y'])
        f.write(','.join(header) + '\n')

        for row_data in all_marker_data:
            row_str = [f'{row_data.get("time", "")}']
            for marker_id in unique_markers:
                row_str.extend([
                    f'{row_data.get(f"marker_{marker_id}_px_x", "")}',
                    f'{row_data.get(f"marker_{marker_id}_px_y", "")}'
                ])
            f.write(','.join(row_str) + '\n')
    print(f"Marker data saved to {CSV_FILE}")

print(f'Detected markers in {detected_frame_count} frames.')