"""Export the Tiny-YOLOv3 one-class person detector to ONNX.

Reproduces DetectorLoader.TinyYOLOv3_onecls's model load. The exported graph
takes a single NCHW float image in [0, 1] and returns raw candidate boxes
`[cx, cy, w, h, obj_conf, class_conf]` in input-pixel units. All post-processing
(xywh->xyxy, confidence threshold, NMS, rescale to original frame, box expand)
is reproduced on the C++ side -- see the manifest for the constants.
"""

import os

import torch

from _common import add_repo_root_to_path, export_onnx, parity_check, write_manifest
import config

add_repo_root_to_path()
from Detection.Models import Darknet  # noqa: E402


def build_model(cfg, device):
    model = Darknet(cfg["config_file"])
    state = torch.load(cfg["weight_file"], map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    cfg = config.DETECTOR
    out_dir = config.ensure_output_dir()
    device = config.DEVICE

    print(f"[detector] loading {cfg['weight_file']}")
    model = build_model(cfg, device)

    size = cfg["input_size"]
    dummy = torch.rand(1, 3, size, size, device=device)

    onnx_path = os.path.join(out_dir, f"{cfg['name']}.onnx")
    input_names = ["image"]
    output_names = ["detections"]

    print("[detector] exporting to ONNX")
    export_onnx(
        model,
        (dummy,),
        onnx_path,
        input_names,
        output_names,
        opset=config.OPSET,
        # Batch is fixed at 1 (one frame at a time); spatial size is fixed.
        dynamic_axes={"detections": {1: "num_boxes"}},
    )

    print("[detector] verifying parity")
    parity_check(
        model,
        (dummy,),
        onnx_path,
        input_names,
        rtol=cfg.get("rtol", config.RTOL),
        atol=cfg.get("atol", config.ATOL),
    )

    write_manifest(
        out_dir,
        cfg["name"],
        {
            "name": cfg["name"],
            "task": "person_detection",
            "onnx_file": f"{cfg['name']}.onnx",
            "inputs": [
                {
                    "name": "image",
                    "shape": [1, 3, size, size],
                    "layout": "NCHW",
                    "dtype": "float32",
                    "preprocessing": {
                        "resize": "ResizePadding to square (keep aspect, pad)",
                        "color": "BGR->RGB",
                        "scale": "divide by 255 (ToTensor)",
                        "input_size": size,
                    },
                }
            ],
            "outputs": [
                {
                    "name": "detections",
                    "shape": [1, "num_boxes", 6],
                    "layout": cfg["output_layout"],
                    "fields": ["cx", "cy", "w", "h", "obj_conf", "class_conf"],
                    "units": "pixels of the (square) input image",
                }
            ],
            "postprocessing": {
                "box_convert": "cxcywh -> xyxy",
                "conf_thres": cfg["conf_thres"],
                "nms_thres": cfg["nms_thres"],
                "rescale": "undo ResizePadding to original frame coords",
                "expand_bb": 10,
            },
            "opset": config.OPSET,
        },
    )
    print("[detector] done.\n")


if __name__ == "__main__":
    main()
