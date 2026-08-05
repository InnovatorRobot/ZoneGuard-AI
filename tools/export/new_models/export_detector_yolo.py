"""Export a fine-tuned YOLO person detector to ONNX in ZoneGuard's legacy
[1, N, 6] = cx,cy,w,h,obj_conf,class_conf contract, so detector.cpp is unchanged."""

import os
import torch
from ultralytics import YOLO
import tools.export.config as config
from tools.export._common import add_repo_root_to_path, export_onnx, parity_check, write_manifest


class YoloToLegacy(torch.nn.Module):
    """Wrap Ultralytics output (1, 4+1, N) -> (1, N, 6) = cx,cy,w,h,obj,cls."""

    def __init__(self, det_model):
        super().__init__()
        self.model = det_model

    def forward(self, image):  # image: (1,3,S,S), RGB, /255
        out = self.model(image)[0]  # (1, 5, N): cx,cy,w,h,score in input px
        out = out.permute(0, 2, 1)  # (1, N, 5)
        box, score = out[..., :4], out[..., 4:5]
        return torch.cat([box, score, score], dim=-1)  # obj == cls == score


def main():
    cfg = config.DETECTOR
    out_dir = config.ensure_output_dir()
    size = cfg["input_size"]  # 384

    weights = os.environ.get(
        "ZONEGUARD_YOLO_WEIGHTS",
        "runs/detect/zoneguard_person/weights/best.pt",
    )
    inner = YOLO(weights).model.fuse().eval()
    wrapper = YoloToLegacy(inner).eval()

    dummy = torch.rand(1, 3, size, size)
    onnx_path = os.path.join(out_dir, f"{cfg['name']}.onnx")

    export_onnx(
        wrapper,
        (dummy,),
        onnx_path,
        input_names=["image"],
        output_names=["detections"],
        opset=config.OPSET,
        dynamic_axes={"detections": {1: "num_boxes"}},
    )
    parity_check(
        wrapper,
        (dummy,),
        onnx_path,
        ["image"],
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
    print("[detector-yolo] done.")


if __name__ == "__main__":
    main()
