# ZoneGuard-AI

Real-time **human fall detection** with **user-defined monitoring zones**, built
as a **C++ / Qt desktop application**. It ports the deep-learning pipeline from
the original `Human-Falling-Detect-Tracks` project (Tiny-YOLOv3 person detection
→ AlphaPose skeleton → ST-GCN action recognition) to ONNX Runtime, and adds
ROI-based zone monitoring plus real-time notifications.

> The original PyTorch repo is used **read-only** as a reference for model
> weights and pre/post-processing. Nothing in this project modifies it.

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
├── CMakeLists.txt          # Qt6/Qt5 + OpenCV build
├── src/
│   ├── main.cpp
│   ├── vision/             # capture / preprocessing
│   │   └── FrameSource.*   # threaded camera/file/RTSP capture
│   └── ui/
│       ├── VideoWidget.*   # QPainter frame + overlay rendering
│       └── MainWindow.*    # controls + status bar
├── models/onnx/            # exported ONNX models + manifests
└── tools/export/           # Python scripts that export the .pth -> .onnx
```

## Build & run

Prerequisites: CMake ≥ 3.16, a C++17 compiler, **Qt6 or Qt5** (Widgets), and
**OpenCV 4**.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/zoneguard
```

Enter a **source** in the toolbar — a camera index (`0`), a video file path, or
an `rtsp://` / `http://` URL — and press **Start**.

## Models

The three ONNX models in `models/onnx/` are produced by the export tooling. To
(re)generate them — e.g. after swapping a model — see
[tools/export/README.md](tools/export/README.md):

```bash
export ZONEGUARD_SOURCE_REPO=/path/to/Human-Falling-Detect-Tracks
cd tools/export
python export_all.py
```

Each model ships with a `*.manifest.json` describing its input/output tensors
and the exact pre/post-processing the C++ side reproduces — so swapping a model
is a matter of re-exporting, not changing app code.

## Status

- [x] **Part 1** — Model export to ONNX (detector, pose, action) + parity checks
- [x] **Milestone 1** — Qt app: threaded capture + live video widget
- [ ] Part 3 — C++ person detector (ONNX Runtime)
- [ ] Part 4 — Pose estimator
- [ ] Part 5 — Tracker (Kalman + IoU)
- [ ] Part 6 — Action recognizer
- [ ] Part 7 — Monitoring zones (ROI)
- [ ] Part 8 — Notification client
- [ ] Part 9 — UI polish (zone editor, alerts panel, settings)
