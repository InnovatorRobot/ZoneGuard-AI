"""Export the AlphaPose SPPE FastPose estimator to ONNX.

PoseEstimateLoader.SPPE_FastPose wraps InferenNet_fastRes50, which forces
`.cuda()`. To export on CPU we build the underlying `FastPose` backbone
directly (the InferenNet wrapper only adds a redundant `narrow(1, 0, 17)`).

The exported graph takes a batch of cropped person images (NCHW, [0, 1]) and
returns joint heatmaps (N, num_joints, H/4, W/4). The C++ side then:
  * keeps channels [0] + [5:] (drops eyes/ears) -> 13 joints,
  * argmax each heatmap -> (x, y) + score (getPrediction),
  * maps back to image coordinates using the crop boxes.
"""

import os

import torch

from _common import add_repo_root_to_path, export_onnx, parity_check, write_manifest
import config

add_repo_root_to_path()
from SPPE.src.models.FastPose import FastPose  # noqa: E402


def build_model(cfg, device):
    model = FastPose(cfg["backbone"], cfg["num_joints"])
    state = torch.load(cfg["weight_file"], map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    cfg = config.POSE
    out_dir = config.ensure_output_dir()
    device = config.DEVICE

    print(f"[pose] loading {cfg['weight_file']} ({cfg['backbone']})")
    model = build_model(cfg, device)

    h, w = cfg["input_height"], cfg["input_width"]
    # Batch dimension = number of detected persons (varies at runtime).
    dummy = torch.rand(2, 3, h, w, device=device)

    onnx_path = os.path.join(out_dir, f"{cfg['name']}.onnx")
    input_names = ["crops"]
    output_names = ["heatmaps"]

    print("[pose] exporting to ONNX")
    export_onnx(
        model,
        (dummy,),
        onnx_path,
        input_names,
        output_names,
        opset=config.OPSET,
        dynamic_axes={
            "crops": {0: "num_persons"},
            "heatmaps": {0: "num_persons"},
        },
    )

    print("[pose] verifying parity")
    parity_check(model, (dummy,), onnx_path, input_names, rtol=config.RTOL, atol=config.ATOL)

    write_manifest(
        out_dir,
        cfg["name"],
        {
            "name": cfg["name"],
            "task": "pose_estimation",
            "onnx_file": f"{cfg['name']}.onnx",
            "backbone": cfg["backbone"],
            "inputs": [
                {
                    "name": "crops",
                    "shape": ["num_persons", 3, h, w],
                    "layout": "NCHW",
                    "dtype": "float32",
                    "preprocessing": {
                        "crop": "crop_dets: expand each detection box, crop, resize",
                        "input_height": h,
                        "input_width": w,
                        "scale": "divide by 255 (ToTensor)",
                    },
                }
            ],
            "outputs": [
                {
                    "name": "heatmaps",
                    "shape": ["num_persons", cfg["num_joints"], h // 4, w // 4],
                    "layout": "NCHW",
                    "description": "per-joint confidence heatmaps",
                }
            ],
            "postprocessing": {
                "keep_channels": cfg["keep_channels"],
                "keep_note": "drop eyes/ears -> 13 joints; runtime later appends a "
                "neck node (midpoint of shoulders) -> 14 graph nodes",
                "decode": "getPrediction: argmax heatmap -> (x, y) + score, "
                "map to image coords via crop boxes (pt1, pt2)",
            },
            "opset": config.OPSET,
        },
    )
    print("[pose] done.\n")


if __name__ == "__main__":
    main()
