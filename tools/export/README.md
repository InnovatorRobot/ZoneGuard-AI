# Part 1 — Model export to ONNX

Converts the three PyTorch models of the original pipeline into device-agnostic
ONNX graphs that the C++/Qt application loads with ONNX Runtime. Each export
also writes a `*.manifest.json` describing input/output tensors and the exact
pre/post-processing the C++ side must reproduce — so the desktop app depends
only on the ONNX + manifest, never on this Python code.

## Models

| Script | Source model | ONNX output | Inputs | Output |
|---|---|---|---|---|
| `export_detector.py` | Tiny-YOLOv3 one-class (`Detection/Models.Darknet`) | `detector.onnx` | `image (1,3,384,384)` | `detections (1,N,6)` = cx,cy,w,h,obj,cls |
| `export_pose.py` | AlphaPose SPPE FastPose (`SPPE/.../FastPose`) | `pose.onnx` | `crops (N,3,224,160)` | `heatmaps (N,17,56,40)` |
| `export_action.py` | Two-Stream ST-GCN (`Actionsrecognition/.../TwoStream...`) | `action.onnx` | `points (B,3,30,14)`, `motion (B,2,29,14)` | `scores (B,7)` |

## Prerequisites

1. A checkout of the original `Human-Falling-Detect-Tracks` repo (read-only
   reference) with its pre-trained weights under `Models/`:
   - `Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg`
   - `Models/yolo-tiny-onecls/best-model.pth`
   - `Models/sppe/fast_res50_256x192.pth`
   - `Models/TSSTG/tsstg-model.pth`

   By default it is expected as a sibling folder of this project. Override with:
   ```bash
   export ZONEGUARD_SOURCE_REPO=/path/to/Human-Falling-Detect-Tracks
   ```
2. Install dependencies (a venv is recommended):
   ```bash
   pip install -r tools/export/requirements.txt
   ```

## Run

From this project root:

```bash
cd tools/export
python export_all.py          # exports all three + parity checks
# or individually:
python export_detector.py
python export_pose.py
python export_action.py
# smoke-test the produced graphs:
python verify_onnx.py
```

Artifacts are written to `models/onnx/` at the project root:
```
detector.onnx   detector.manifest.json
pose.onnx       pose.manifest.json
action.onnx     action.manifest.json
```

## Swapping a model later

This is designed for it:

1. Edit `config.py` — point `weight_file` (and `config_file`, sizes, class
   names, etc.) at the new model.
2. If the new architecture differs, adjust the `build_model()` in the relevant
   `export_*.py` (that is the only model-specific code).
3. Re-run the export. The regenerated `*.manifest.json` captures the new
   shapes/preprocessing, so the C++/Qt app adapts by reading the manifest
   rather than hard-coding sizes.

## Notes / gotchas baked into the scripts

- **Pose**: the runtime `InferenNet_fastRes50` hard-codes `.cuda()`, so the
  export builds the underlying `FastPose` backbone directly (CPU-friendly). The
  wrapper's extra `narrow(1,0,17)` is a no-op and intentionally omitted.
- **Action motion stream is 29 frames, not 30** — it is the per-frame delta of
  `(x, y)`. The manifest records this; the C++ side must build the two input
  tensors with `T=30` (points) and `T-1=29` (motion).
- **Pose post-processing** keeps heatmap channels `[0] + [5:]` (drops eyes/ears)
  → 13 joints, then the app appends a neck node (shoulder midpoint) → 14 graph
  nodes for the action model.
- Exports run on CPU with `opset 12`; the produced graph is device-agnostic.
