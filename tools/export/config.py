"""Central configuration for ONNX export.

Change values here when you swap a model. Each export script reads from this
file, so the C++/Qt side only ever depends on the produced `.onnx` files and
their generated `*.manifest.json` descriptors -- never on these Python paths.

This project (ZoneGuard-AI) only *reads* the original PyTorch repo. Set
`SOURCE_REPO` to where the `Human-Falling-Detect-Tracks` checkout lives (its
Python packages and `.pth` weights are imported from there). Override at runtime
with the `ZONEGUARD_SOURCE_REPO` environment variable.
"""

import os

# --- Source PyTorch repo (read-only reference) -------------------------------
# Default: sibling folder next to this ZoneGuard-AI project.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_REPO = os.environ.get(
    "ZONEGUARD_SOURCE_REPO",
    os.path.join(os.path.dirname(PROJECT_ROOT), "Human-Falling-Detect-Tracks"),
)
MODELS_DIR = os.path.join(SOURCE_REPO, "Models")

# Where exported artifacts are written (inside this project).
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "onnx")

# --- Export behaviour --------------------------------------------------------
# Export on CPU for portability/determinism. The produced graph is
# device-agnostic; the C++ runtime picks CPU/CUDA/DirectML at load time.
DEVICE = "cpu"
OPSET = 12
# Tolerances for the PyTorch-vs-ONNXRuntime numerical parity check.
RTOL = 1e-3
ATOL = 1e-4


# =============================================================================
# 1) PERSON DETECTOR  (Tiny-YOLOv3 one-class)
# =============================================================================
DETECTOR = {
    "name": "detector",
    "config_file": os.path.join(MODELS_DIR, "yolo-tiny-onecls", "yolov3-tiny-onecls.cfg"),
    "weight_file": os.path.join(MODELS_DIR, "yolo-tiny-onecls", "best-model.pth"),
    "input_size": 384,  # square; must be divisible by 32
    "conf_thres": 0.45,  # used by C++ post-processing (NMS)
    "nms_thres": 0.20,
    # Output is [cx, cy, w, h, obj_conf, class_conf] per candidate box, in
    # pixel units of the input image. Post-processing (xywh->xyxy, NMS,
    # rescale to original frame) happens in C++.
    "output_layout": "cxcywh_conf_cls",  # Box coords span 0..input_size, so a looser absolute tolerance is fine for
    # the parity smoke-check.
    "rtol": 1e-2,
    "atol": 1e-2,
}


# =============================================================================
# 2) POSE ESTIMATOR  (AlphaPose SPPE FastPose)
# =============================================================================
POSE = {
    "name": "pose",
    "backbone": "resnet50",  # 'resnet50' or 'resnet101'
    "weight_file": os.path.join(MODELS_DIR, "sppe", "fast_res50_256x192.pth"),
    "num_joints": 17,  # raw heatmap channels produced by the model
    # Crop size fed to the model (height, width). Matches main.py default
    # '224x160'. Heatmap output is (num_joints, H/4, W/4).
    "input_height": 224,
    "input_width": 160,
    # The runtime keeps only joints [0] + [5:] (drops eyes & ears) -> 13 joints,
    # then appends a neck node (midpoint of shoulders) -> 14 graph nodes.
    "keep_channels": [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
}


# =============================================================================
# 3) ACTION RECOGNIZER  (Two-Stream ST-GCN)
# =============================================================================
ACTION = {
    "name": "action",
    "weight_file": os.path.join(MODELS_DIR, "TSSTG", "tsstg-model.pth"),
    "strategy": "spatial",
    "class_names": [
        "Standing",
        "Walking",
        "Sitting",
        "Lying Down",
        "Stand up",
        "Sit down",
        "Fall Down",
    ],
    "num_nodes": 14,  # 13 body joints + neck
    "time_steps": 30,  # frames buffered per track before prediction
    # NOTE: the motion stream is the per-frame delta of (x, y), so its time
    # dimension is time_steps - 1 (29), NOT 30. The C++ side must build the two
    # input tensors accordingly.
    "pts_channels": 3,  # (x, y, score)
    "mot_channels": 2,  # (dx, dy)
}


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
