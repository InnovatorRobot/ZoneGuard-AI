"""Shared helpers for exporting PyTorch models to ONNX.

Provides:
  * `add_repo_root_to_path()` so the SPPE/Detection/Actionsrecognition packages
    import correctly regardless of the current working directory.
  * `export_onnx()` to trace + save a model.
  * `parity_check()` to compare PyTorch vs ONNXRuntime outputs.
  * `write_manifest()` to emit a `*.manifest.json` describing the graph and the
    exact pre/post-processing the C++/Qt app must reproduce.
"""

import os
import sys
import json

import numpy as np
import torch


def add_repo_root_to_path():
    """Make the source PyTorch repo importable (Detection, SPPE, Actionsrecognition)."""
    import config

    repo_root = config.SOURCE_REPO
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _to_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def export_onnx(
    model, dummy_inputs, output_path, input_names, output_names, opset=12, dynamic_axes=None
):
    """Trace `model` with `dummy_inputs` and write an ONNX graph.

    `dummy_inputs` must be a tuple matching the model's forward signature.
    """
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamic_axes=dynamic_axes or {},
            do_constant_folding=True,
        )
    print(f"  [ok] wrote {output_path}")
    return output_path


def parity_check(model, dummy_inputs, output_path, input_names, rtol=1e-3, atol=1e-4):
    """Compare PyTorch and ONNXRuntime outputs for the same inputs.

    Returns the max absolute difference per output, raises on mismatch.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not installed; skipping parity check.")
        return None

    model.eval()
    with torch.no_grad():
        torch_out = model(*dummy_inputs) if isinstance(dummy_inputs, tuple) else model(dummy_inputs)
    if isinstance(torch_out, (list, tuple)):
        torch_out = [_to_numpy(o) for o in torch_out]
    else:
        torch_out = [_to_numpy(torch_out)]

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    # Map each declared input name to its numpy tensor.
    flat_inputs = dummy_inputs if isinstance(dummy_inputs, tuple) else (dummy_inputs,)
    # The model may receive a single tuple arg (e.g. action two-stream); flatten.
    feed = {}
    np_inputs = []
    for item in flat_inputs:
        if isinstance(item, (list, tuple)):
            np_inputs.extend(_to_numpy(x) for x in item)
        else:
            np_inputs.append(_to_numpy(item))
    for name, arr in zip(input_names, np_inputs):
        feed[name] = arr.astype(np.float32)

    ort_out = sess.run(None, feed)

    max_diffs = []
    for i, (a, b) in enumerate(zip(torch_out, ort_out)):
        diff = float(np.max(np.abs(a - b)))
        max_diffs.append(diff)
        ok = np.allclose(a, b, rtol=rtol, atol=atol)
        status = "ok" if ok else "MISMATCH"
        print(f"  [{status}] output[{i}] max|diff|={diff:.3e} shape={b.shape}")
        if not ok:
            raise AssertionError(
                f"Parity check failed for {output_path} output[{i}]: "
                f"max|diff|={diff:.3e} exceeds rtol={rtol} atol={atol}"
            )
    return max_diffs


def write_manifest(output_dir, name, manifest):
    """Write a `<name>.manifest.json` next to the ONNX file."""
    path = os.path.join(output_dir, f"{name}.manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [ok] wrote {path}")
    return path
