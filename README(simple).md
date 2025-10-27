# CubeSat Air Bearing Balancing System

## What Does This Do?

This system balances a CubeSat on an air bearing table by:
1. Using a camera to track wobble motion
2. Calculating how far off-balance the system is
3. Telling you which direction to move weights
4. Repeating until balanced (3-5 times, ~20 minutes total)

**Result:** Balance improves from ~18mm offset to <3mm offset

---

## Quick File Summary

| File | What It Does |
|------|--------------|
| `take_calibration_photo.py` | Take photos of checkerboard (do once) |
| `calibrate_camera.py` | Calculate camera settings (do once) |
| `basic_detection.py` | Test if markers work (optional check) |
| `track_wobble.py` | **Main script - measures wobble** |
| `Camera_Coordinate_System.py` | Helper to align camera (optional) |

### Important File Created:
- **`calib/camera_intrinsics.npz`** - Camera calibration data (required by track_wobble.py)
  - Contains camera lens properties and distortion corrections
  - Without this file, nothing works!

---

## Installation

```bash
pip install opencv-python opencv-contrib-python numpy scipy
```

---

## How to Use (Step by Step)

### **ONE-TIME SETUP (30 minutes)**

#### Step 1: Take Calibration Photos
```bash
python take_calibration_photo.py
```
1. Print a 9×6 checkerboard pattern
2. Hold it in front of camera at different angles
3. Press **`s`** when green border appears (save photo)
4. Take 15-20 photos
5. Press **`q`** to quit

#### Step 2: Calibrate Camera
```bash
python calibrate_camera.py
```
- Automatically processes your photos
- Creates `calib/camera_intrinsics.npz` file
- Look for: "RMS reprojection error" < 1.0 (lower is better)

#### Step 3: Setup Markers
1. Print 5 ArUco markers from DICT_4X4_100 (IDs: 0, 1, 2, 3, 4)
   - Use this generator: https://chev.me/arucogen/
2. Place them on your table:
   ```
          Marker 2
              |
   Marker 3 --+-- Marker 1    (Markers 1-4 on table, 120mm from center)
              |
          Marker 4
          
        Marker 0               (Marker 0 on CubeSat)
   ```

#### Step 4: Measure Heights
Measure from camera lens downward:
- `Z_CAM` = Camera height above table (e.g., 500mm)
- `Z_TABLE` = Height to table surface (e.g., 360mm)  
- `Z_MARKER0` = Height to marker 0 surface (e.g., 240mm)

Edit `track_wobble.py` lines 24-26 with your measurements

---

### **BALANCING SESSION (20 minutes)**

#### Step 5: Measure Wobble
```bash
python track_wobble.py
```

**Before running:**
- Turn off AC/fans
- Don't speak during measurement
- Wait 5 seconds (it runs automatically)

**After it finishes, look at the console:**
```
Fitted Circle Center: (-15.52 mm, 9.36 mm)
Correction Vector:
  Magnitude: 18.12 mm
  Direction: 148.91 degrees
```

**What this means:**
- Your system is off-balance by 18.12mm
- Move weights in the 148.91° direction (upper-left)

#### Step 6: Calculate Weight Movement
Use this formula:
```
Δx' = -(x_center × 8870) / 476
Δy' = -(y_center × 8870) / 476
```

**Example:**
```
x_center = -15.52, y_center = 9.36

Δx' = -(-15.52 × 8870) / 476 = +289 mm
Δy' = -(9.36 × 8870) / 476 = -175 mm

→ Move X-weight +289mm to the right
→ Move Y-weight -175mm downward
```

#### Step 7: Move Weights & Repeat
1. Adjust weights by calculated amounts
2. Wait 60 seconds for system to stabilize
3. Run `track_wobble.py` again
4. Repeat until center is near (0, 0)

**You're balanced when:**
- Center is within ±3mm: `(-2.5, 1.0)` ✅
- Radius is < 0.5mm

---

## Understanding the Output

### Console Results
```
Fitted Circle Center: (x, y) mm  ← How far off-center you are
Fitted Circle Radius: r mm       ← How much wobble (want ~0)
Correction Vector:
  Magnitude: d mm                ← Total distance to fix
  Direction: θ degrees           ← Which way to move weights
```

### Output Images
- `data/final_path_plot_mm.png` - Visual showing:
  - Yellow line = wobble path
  - Blue dot = current balance point
  - White arrow = which way to move weights
  - Green dot at center = perfect balance goal

---

## Troubleshooting

### "Camera calibration file not found"
→ Run steps 1-2 first to create `calib/camera_intrinsics.npz`

### "Not enough data points"
→ Marker 0 not detected. Check lighting and marker visibility

### Markers not detected
→ Make sure you printed markers from DICT_4X4_100 (not DICT_4X4_50)

### Results jump around wildly
→ Turn off AC/fans, don't talk during measurement, wait 60s after moving weights

### Balance gets worse instead of better
→ Double-check your Z_CAM, Z_TABLE, Z_MARKER0 measurements

---

## Quick Reference

### Typical Session
```bash
# First time only
python take_calibration_photo.py
python calibrate_camera.py

# Every balancing session
python track_wobble.py    # Measure
# (move weights)
python track_wobble.py    # Measure again
# (move weights)
python track_wobble.py    # Measure again
# Done! (usually 3-5 iterations)
```

### Key Numbers
- System mass: 8870g
- Trim weight: 476g each
- Target: Center within ±3mm, radius < 0.5mm
- Typical iterations: 3-5
- Time per iteration: ~5 minutes

---

## That's It!

If something doesn't work, check the Troubleshooting section above or verify your setup matches the requirements.
