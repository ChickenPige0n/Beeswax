"""Validate local model contracts and recurrent streaming using installed Python ORT.

This does not validate the Rust ort binding, microphone routing, or speaker quality.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "models.json").read_text(encoding="utf-8"))
model_revision = manifest["revision"]
expected_hashes = {
    f"models/{model['name']}": (model["sha256"], int(model["bytes"]))
    for model in manifest["models"]
}
for relative_path, (expected_hash, expected_bytes) in expected_hashes.items():
    model_path = root / relative_path
    payload = model_path.read_bytes()
    assert len(payload) == expected_bytes, f"Unexpected model size: {relative_path}"
    actual_hash = hashlib.sha256(payload).hexdigest()
    assert actual_hash == expected_hash, f"Unexpected model revision: {relative_path}"

options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
options.add_session_config_entry("session.intra_op.allow_spinning", "0")
enroll = ort.InferenceSession(str(root / "models/enroll.onnx"), options, providers=["CPUExecutionProvider"])
chunk = ort.InferenceSession(str(root / "models/chunk_int8.onnx"), options, providers=["CPUExecutionProvider"])
assert [x.name for x in chunk.get_inputs()] == ["wav", "enroll_mag", "context", "h", "c"]
sample = (0.1 * np.sin(np.arange(48000, dtype=np.float32) * 0.086))[None]
mag, context = enroll.run(None, {"enroll": sample})
h = np.zeros((6, 32, 256), dtype=np.float32)
c = np.zeros_like(h)
times = []
for i in range(75):
    wav = (0.1 * np.sin((np.arange(1024, dtype=np.float32) + i * 640) * 0.086))[None]
    start = time.perf_counter()
    frames, h, c = chunk.run(None, {"wav": wav, "enroll_mag": mag, "context": context, "h": h, "c": c})
    times.append((time.perf_counter() - start) * 1000)
    assert frames.shape == (1, 5, 512), frames.shape
    assert h.shape == c.shape == (6, 32, 256)
    assert all(np.isfinite(x).all() for x in [frames, h, c])
report = {
    "runtime": ort.__version__, "reference": "VoxLock 1.1.1",
    "model_repository": manifest["repository"],
    "model_revision": model_revision, "interface": "VoxLock 1.1 ONNX",
    "runs": len(times),
    "chunk_ms": 40, "mean_inference_ms": round(float(np.mean(times)), 2),
    "p95_inference_ms": round(float(np.percentile(times, 95)), 2),
    "finite_output": True, "speech_quality_tested": False,
}
print(json.dumps(report, indent=2))
