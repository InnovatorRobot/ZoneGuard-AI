"""Run all three ONNX exports in sequence.

    python export_all.py

Each step loads its PyTorch model, writes `<name>.onnx` and
`<name>.manifest.json` into export/onnx/, and runs a PyTorch-vs-ONNXRuntime
parity check. A failure in one step does not stop the others; a summary is
printed at the end.
"""

import traceback

import export_detector
import export_pose
import export_action


STEPS = [
    ("detector", export_detector.main),
    ("pose", export_pose.main),
    ("action", export_action.main),
]


def main():
    results = {}
    for name, fn in STEPS:
        try:
            fn()
            results[name] = "ok"
        except Exception as exc:  # keep going so one bad model doesn't block the rest
            results[name] = f"FAILED: {exc}"
            traceback.print_exc()
            print()

    print("=" * 50)
    print("Export summary")
    print("=" * 50)
    for name, status in results.items():
        print(f"  {name:10s} {status}")


if __name__ == "__main__":
    main()
