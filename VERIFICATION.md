# Verification — 2026-11-03

## Build

- `cargo build --release --locked` succeeds. The default feature set (`cuda`) registers the CUDA execution provider and still runs on CPU-only ONNX Runtime installs.
- `cargo check --no-default-features --offline` succeeds (enrollment/UI build without ort).
- `cargo test --offline`: seven tests pass, including a 70 Hz high-pass response test.
- `cargo clippy --offline --all-targets -- -D warnings` and `cargo fmt --all -- --check`: clean.

## What changed

1. **Select popups.** Device pickers are deferred, absolutely positioned overlays anchored to the trigger, so opening a list never moves the page. The microphone picker opens downwards; the lower speaker/CABLE pickers open upwards. A transparent scrim dismisses them on an outside click; Escape also closes.
2. **Layout.** Four numbered cards — Microphone, Your voice, Where the voice goes, Processing — with Start/Stop, Refresh and the status line pinned in the footer.
3. **Backend choice.** CPU/CUDA are selectable in Processing and remembered in `data/settings.json`. CUDA is enabled only when the runtime advertises `CUDAExecutionProvider`; a failed CUDA session falls back to CPU with the reason shown in the card.
4. **Continuous filtering.** Capture overflow is no longer fatal, stale capture is trimmed to bound latency, model errors are tolerated for three blocks, and CPU inference uses up to eight threads (`BEESWAX_THREADS` overrides).
5. **Latency.** WASAPI streams ask for the shortest period the driver publishes (falling back to the host default if refused). The playback reserve starts at 60 ms instead of the previous 83 ms, grows by 20 ms only after a real underrun (cap 160 ms), and the Processing card shows the estimated added delay.
6. **Output routing.** Any playback endpoint can be the filtering output, not only VB-CABLE: CABLE Input for calls, headphones/speakers for direct monitoring. CABLE is preselected when installed; speaker monitoring warns about feedback.
7. **Model precision.** The loader prefers `chunk_fp32.onnx` / `chunk.onnx` > `chunk_fp16.onnx` > `chunk_int8.onnx` and reports the loaded precision in the Processing card. `scripts/fetch-model.ps1` downloads fp32 or fp16.
8. **Input conditioning.** A 70 Hz second-order Butterworth high-pass runs on the 16 kHz stream before the model; the Microphone card reports the capture sample rate and flags rates below 32 kHz.
9. **GPU diagnostics.** `--providers` prints the runtime DLL and the execution providers it actually exposes; the Processing card shows the same list. CUDA/cuDNN DLLs are preloaded from `BEESWAX_CUDA_DLLS`, `runtime/cuda-path.txt` (written by `scripts/setup-cuda-runtime.ps1`) or `runtime/cuda` before ONNX Runtime starts.

## Why CUDA was unavailable

- Hardware: Intel Iris Xe (iGPU) and NVIDIA GeForce RTX 3050 Laptop (dGPU, 4 GB), driver 610.47, CUDA UMD 13.3. **CUDA cannot run on the Intel iGPU at all**; that part needs DirectML or OpenVINO.
- `runtime/onnxruntime.dll` was copied from the **CPU-only** `onnxruntime` Python wheel: `--providers` prints `providers: CPU`, and the Python package reports `['AzureExecutionProvider','CPUExecutionProvider']`. It ships no `onnxruntime_providers_cuda.dll`, so `CUDAExecutionProvider` cannot be registered and the UI correctly disables CUDA. This is the blocker, not the GPU.
- `onnxruntime-gpu` 1.30.0 is available from the configured Tsinghua mirror and from the Aliyun mirror (161 MB wheel); direct PyPI access is blocked in this session.
- CUDA 12.6 and cuDNN 9 DLLs already exist in PyTorch's `torch\lib` (`cudart64_12`, `cublas64_12`, `cublasLt64_12`, `cufft64_11`, `cudnn64_9` plus the cuDNN 9 component DLLs), but none are on `PATH`, so a GPU ONNX Runtime needs them preloaded.
- VRAM is tight: a running game was holding 2.9–3.6 of 4.0 GB, which can make CUDA session creation fail even with the right runtime.

## CUDA verification (resolved)

1. Downloaded `onnxruntime-gpu==1.30.0` (cp314, 161 MB) from the Aliyun PyPI mirror and copied `onnxruntime.dll`, `onnxruntime_providers_cuda.dll`, `onnxruntime_providers_tensorrt.dll` and `onnxruntime_providers_shared.dll` into `runtime/`, keeping the CPU build as `runtime/onnxruntime-cpu.dll`.
2. Wrote `runtime/cuda-path.txt` pointing at PyTorch's `torch\lib` (CUDA 12.6 + cuDNN 9).
3. First CUDA attempt failed inside the Conv node: `LoadLibrary failed for cudnn64_9.dll with error 2`. Preloading alone did not satisfy ONNX Runtime's by-name load. Fix: put the CUDA directory first on the process `PATH` before `ort::init_from`, then preload. `scripts/setup-cuda-runtime.ps1` + the app's startup path now do both.
4. `--providers` now prints `providers: CPU, CUDA, TensorRT`, `cuda available: true`, `cuda dll dirs: ...torch\lib`; `--self-test --cuda` completes with engine CUDA and finite output.
5. Measured (game using the GPU concurrently):

| engine + model | 3 s of audio |
|---|---|
| CPU + int8 | 2.85 s |
| CPU + fp32 | 4.40 s |
| CUDA + int8 | 5.35 s |
| CUDA + fp32 | 2.74 s |

CUDA with int8 is slower than CPU because the quantized LSTM nodes fall back to the CPU, splitting the graph. With fp32 the whole graph stays on the GPU and CUDA becomes the fastest option, so the engine now picks int8 on CPU and the highest installed precision on CUDA.

The Intel Iris Xe iGPU still cannot use CUDA. Supporting it would need a DirectML (or OpenVINO) ONNX Runtime build and a matching backend; the app detects and reports those providers but does not expose them yet.

## Inference performance

Release `--self-test` on the development laptop (i7-12700H, CPU-only ONNX Runtime 1.30), processing 47,616 samples (2.98 s of audio):

| intra-op threads | time for 3 s of audio |
|---|---|
| 2 (previous default) | 4.13 s |
| 4 | 3.14 s |
| 6 | 2.87 s |
| 8 (new default) | 2.70–2.84 s |
| 12 | 2.95 s |

The old build needed 4.13 s per 3 s, so the worker fell behind real-time capture; the 0.5 s capture ring filled and the callback signalled a device fault that stopped the stream. With eight threads the build is slightly faster than real time, and overload now drops audio instead of ending the session.

## Audio-quality findings

Measured on this machine, not by listening:

- `data/voice.json` (the saved enrollment) has only 7.7 % of its energy between 1–3 kHz and 0.7 % between 4–8 kHz, with **14.5 % below 40 Hz**. That is a dark, rumble-heavy capture, so the input — not the Rust chain — limits fidelity. The 70 Hz high-pass removes the inaudible sub-bass.
- The same clip through the int8 model comes out at +0.7 dB relative to the input, with no samples over full scale, so the model is not attenuating or clipping the target.
- The 48 kHz → 16 kHz resampler is flat to 6 kHz, −1.8 dB at 7 kHz, −5.8 dB at 7.5 kHz and −18.6 dB at 8.2 kHz; 16 kHz → 48 kHz is flat to 7 kHz with images below −70 dB. Band-limiting to 8 kHz is inherent to the 16 kHz model.
- `models/chunk_int8.onnx` contains `com.microsoft.DynamicQuantizeLSTM` nodes: the LSTMs are int8 while the matrix multiplies stay float, matching the reference's fast "int8" precision. No fp32/fp16 model is installed here.

## UI checks

The release app was launched and captured with the Windows screen API. The window opens at the top with the header, READY pill and the first two cards, the footer pinned at the bottom, and the new output card text. Device pickers were verified opened (microphone list downwards, speakers list upwards) with a temporary startup hook that was removed before the final build.

## Limits

- Microphone capture still returns Windows `0x80070005` (Access denied) in this session, so real filtering, enrollment capture, speaker monitoring, VB-CABLE loopback and sustained-call audio were not exercised.
- CUDA could not be executed: the local runtime is CPU-only. CUDA is disabled in the UI here.
- No listening test was possible; the quality notes above are spectral/level measurements, not perceptual judgements.
- CABLE stays on the machine; the receiving app's own processing (noise suppression, AGC, compression) is outside Beeswax's control.

## Run

Use `Start-Beeswax.ps1` or `target/release/beeswax.exe`. Select a microphone, test it, record six seconds, preview the clip, choose CABLE Input or headphones, pick CPU or CUDA, and press Start filtering in the footer.

## Local model SHA-256

```
enroll.onnx      86645CBA23F207DEF0B31A6E74A615D3B7A813B4FDF957C53B3AF2905BA2FFD3
chunk_int8.onnx  F0D2B0AAF95AFFE8894EA008D0105BE4F5F5354F7244835534C31B1A7DD4699A
```
