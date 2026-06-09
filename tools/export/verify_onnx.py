"""Standalone sanity check for already-exported ONNX models.

Loads each `*.onnx` from the output directory, prints input/output signatures,
and runs a single forward pass with random data using ONNXRuntime. Use this to
confirm an exported (or swapped) model loads and runs before wiring it into the
C++/Qt app.

Usage:
    python verify_onnx.py            # check all models present
    python verify_onnx.py detector   # check one
"""

import os
import sys

import numpy as np

import config


def _rand_for(inp):
    # Replace symbolic / None dims with concrete sizes for a smoke test.
    shape = []
    for d in inp.shape:
        shape.append(d if isinstance(d, int) and d > 0 else 1)
    return np.random.rand(*shape).astype(np.float32)


def check_model(name):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; run: pip install onnxruntime")
        return False

    path = os.path.join(config.OUTPUT_DIR, f"{name}.onnx")
    if not os.path.isfile(path):
        print(f"[{name}] MISSING: {path}")
        return False

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"[{name}] {path}")
    feed = {}
    for inp in sess.get_inputs():
        print(f"    input  {inp.name:10s} {inp.shape} {inp.type}")
        feed[inp.name] = _rand_for(inp)
    for out in sess.get_outputs():
        print(f"    output {out.name:10s} {out.shape} {out.type}")

    outs = sess.run(None, feed)
    for o, arr in zip(sess.get_outputs(), outs):
        print(f"    ran -> {o.name} shape={arr.shape}")
    print(f"[{name}] ok\n")
    return True


def main():
    names = sys.argv[1:] or [config.DETECTOR["name"], config.POSE["name"], config.ACTION["name"]]
    results = [check_model(n) for n in names]
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
