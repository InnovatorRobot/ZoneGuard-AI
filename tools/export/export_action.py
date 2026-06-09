"""Export the Two-Stream ST-GCN action recognizer to ONNX.

TSSTG wraps `TwoStreamSpatialTemporalGraph`, whose `forward(inputs)` takes a
single tuple `(points, motion)`. ONNX prefers explicit tensor inputs, so we wrap
it to accept two separate tensors.

Important shape detail (from ActionsEstLoader.predict):
  * points stream:  (N, 3, T,   V)  -- (x, y, score), T = time_steps (30)
  * motion stream:  (N, 2, T-1, V)  -- per-frame delta of (x, y), so 29 frames

V = 14 graph nodes (13 body joints + neck). Output is (N, num_classes) after a
sigmoid; argmax gives the predicted action.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import add_repo_root_to_path, export_onnx, parity_check, write_manifest
import config

add_repo_root_to_path()
from Actionsrecognition.Models import (  # noqa: E402
    TwoStreamSpatialTemporalGraph,
    StreamSpatialTemporalGraph,
)


def _static_stream_forward(self, x):
    """Drop-in replacement for `StreamSpatialTemporalGraph.forward` that forces
    the reshape dimensions to be Python ints (compile-time constants).

    The original uses `x.view(N, V * C, T)` and `F.avg_pool2d(x, x.size()[2:])`
    with sizes pulled from the live tensor. The ONNX tracer turns those into
    dynamic `Gather`/`Mul` nodes it then refuses to export. Casting to `int`
    bakes them in as constants. Behaviour is identical for fixed-size input.
    """
    N, C, T, V = (int(d) for d in x.size())
    x = x.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
    x = x.view(N, V * C, T)
    x = self.data_bn(x)
    x = x.view(N, V, C, T)
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(N, C, T, V)

    for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
        x = gcn(x, self.A * importance)

    x = F.avg_pool2d(x, (int(x.size(2)), int(x.size(3))))
    x = self.cls(x)
    x = x.view(int(x.size(0)), -1)
    return x


class TwoStreamWrapper(nn.Module):
    """Adapts the tuple-input model to two explicit tensor inputs for ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, points, motion):
        return self.model((points, motion))


def build_model(cfg, device):
    # Apply the ONNX-friendly forward to both streams (export-time shim only;
    # the original source is untouched).
    StreamSpatialTemporalGraph.forward = _static_stream_forward

    graph_args = {"strategy": cfg["strategy"]}
    num_class = len(cfg["class_names"])
    model = TwoStreamSpatialTemporalGraph(graph_args, num_class)
    state = torch.load(cfg["weight_file"], map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return TwoStreamWrapper(model).to(device).eval()


def main():
    cfg = config.ACTION
    out_dir = config.ensure_output_dir()
    device = config.DEVICE

    print(f"[action] loading {cfg['weight_file']}")
    model = build_model(cfg, device)

    t = cfg["time_steps"]
    v = cfg["num_nodes"]
    pts = torch.rand(1, cfg["pts_channels"], t, v, device=device)
    mot = torch.rand(1, cfg["mot_channels"], t - 1, v, device=device)

    onnx_path = os.path.join(out_dir, f"{cfg['name']}.onnx")
    input_names = ["points", "motion"]
    output_names = ["scores"]

    print("[action] exporting to ONNX")
    export_onnx(
        model,
        (pts, mot),
        onnx_path,
        input_names,
        output_names,
        opset=config.OPSET,
        # Batch is fixed at 1: the runtime predicts one track at a time. The
        # ST-GCN reshapes use tensor-derived sizes, which cannot be traced to
        # ONNX with a symbolic (dynamic) batch dim, so we keep it static.
        dynamic_axes=None,
    )

    print("[action] verifying parity")
    parity_check(model, (pts, mot), onnx_path, input_names, rtol=config.RTOL, atol=config.ATOL)

    write_manifest(
        out_dir,
        cfg["name"],
        {
            "name": cfg["name"],
            "task": "action_recognition",
            "onnx_file": f"{cfg['name']}.onnx",
            "class_names": cfg["class_names"],
            "fall_classes": ["Fall Down", "Lying Down"],
            "inputs": [
                {
                    "name": "points",
                    "shape": [1, cfg["pts_channels"], t, v],
                    "layout": "N,C,T,V",
                    "dtype": "float32",
                    "fields": ["x", "y", "score"],
                },
                {
                    "name": "motion",
                    "shape": [1, cfg["mot_channels"], t - 1, v],
                    "layout": "N,C,T,V",
                    "dtype": "float32",
                    "fields": ["dx", "dy"],
                    "note": "consecutive-frame delta of (x, y); T-1 frames",
                },
            ],
            "outputs": [
                {
                    "name": "scores",
                    "shape": [1, len(cfg["class_names"])],
                    "activation": "sigmoid",
                    "decode": "argmax over classes",
                }
            ],
            "preprocessing": {
                "num_nodes": v,
                "time_steps": t,
                "steps": [
                    "normalize_points_with_size(pts, frame_w, frame_h)",
                    "scale_pose -> range [-1, 1] per pose",
                    "append neck node = midpoint of shoulders (joints 1 & 2)",
                    "points = (x, y, score); motion = delta of (x, y) between frames",
                ],
            },
            "opset": config.OPSET,
        },
    )
    print("[action] done.\n")


if __name__ == "__main__":
    main()
