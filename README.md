# ZoneGuard-AI

Real-time **human fall detection** with **user-defined monitoring zones**, built
as a **C++ / Qt desktop application**. It ports the deep-learning pipeline from
the original `Human-Falling-Detect-Tracks` project (Tiny-YOLOv3 person detection
→ AlphaPose skeleton → ST-GCN action recognition) to ONNX Runtime, and adds
ROI-based zone monitoring plus real-time notifications.


## Pipeline

```
Camera / Video / RTSP
        │
        ▼
  Person detector  (Tiny-YOLOv3 one-class)
        │
        ▼
  Pose estimator   (AlphaPose SPPE FastPose)  → 13 keypoints + neck
        │
        ▼
  Tracker          (Kalman + IoU, 30-frame buffer per person)
        │
        ▼
  Action recognizer (Two-Stream ST-GCN)        → 7 actions incl. "Fall Down"
        │
        ▼
  Zone check + Alerts + Qt overlay
```

## Project layout

```
ZoneGuard-AI/
├── CMakeLists.txt          # Qt6/Qt5 + OpenCV + ONNX Runtime build
├── src/
│   ├── main.cpp
│   ├── core/
│   │   ├── detector.*      # ONNX Runtime Tiny-YOLOv3 detector
│   │   ├── pose_estimator.* # ONNX Runtime AlphaPose SPPE FastPose stage
│   │   ├── kalman_filter.* # constant-velocity Kalman filter (xyah state)
│   │   ├── tracker.*       # Kalman + IoU matching-cascade tracker
│   │   ├── action_recognizer.* # Two-Stream ST-GCN action recognizer
│   │   ├── zone.*          # monitoring zones (ROI) + containment checks
│   │   ├── notification.*  # debounced alert dispatch (pluggable sinks)
│   │   ├── pipeline.*      # threaded processing pipeline
│   │   └── types.h         # shared detection types
│   ├── vision/
│   │   └── frame_source.*  # threaded camera/file/RTSP capture
│   └── ui/
│       ├── video_widget.*  # QPainter frame + overlay rendering
│       └── main_window.*   # controls + status bar
├── models/onnx/            # exported ONNX models + manifests
```

## Build & run

Prerequisites:

- CMake >= 3.16
- C++17 compiler
- **Qt6 or Qt5** (Widgets)
- **OpenCV 4**

Notes:

- ONNX Runtime is fetched automatically by CMake unless `ONNXRUNTIME_ROOT` is set.
- Run commands from the repo root (`ZoneGuard-AI/`).

### 1) Configure

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

### 2) Build

```bash
cmake --build build -j
```

### 3) Run

```bash
./build/zoneguard
```

Enter a **source** in the toolbar:

- camera index: `0`
- local video file path
- stream URL: `rtsp://...` or `http://...`

Then press **Start**.

### Optional: override models directory at runtime

```bash
ZONEGUARD_MODELS_DIR=/absolute/path/to/models/onnx ./build/zoneguard
```

### Clean rebuild

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Troubleshooting

- If `cmake --build` fails, include the build directory explicitly:

```bash
cmake --build build -j
```

- Print recent build errors:

```bash
cmake --build build -j 2>&1 | tail -100
```
