# Beeswax

A Windows target-speaker virtual microphone built with **gpui-fast**, Rust **ort**, the **VoxLock 1.1.1** streaming implementation and its pinned ONNX model revision, and **VB-CABLE**. A light native interface records enrollment, selects audio endpoints, and starts/stops voice filtering. Missing required models download from the public VoxLock Hugging Face repository on first launch.

After building, launch `target/release/beeswax.exe`.

## Build and run

Requires Windows x64, Rust/MSVC with the Windows SDK, and VB-CABLE installed. The first app launch downloads the required models into `models/` and verifies them against their recorded SHA-256 checksums:

```
models/enroll.onnx
models/chunk_int8.onnx
```

From the project directory:

```powershell
# Copies the DLL from an existing Python onnxruntime installation; no download.
powershell -ExecutionPolicy Bypass -File scripts/setup-runtime.ps1
cargo build --release --locked
.\target\release\beeswax.exe
```

Alternatively pass `-RuntimeDll C:\path\onnxruntime.dll` to the setup script, or set `ORT_DYLIB_PATH` to that DLL. Use an ONNX Runtime supporting API 22 or newer (the local installation is CPU-only 1.30). The Rust binding is pinned to `ort 2.0.0-rc.10`.

The default build registers the CUDA execution provider and lets **Processing** choose between **CPU** and **CUDA**; CUDA is preselected when the provider is actually available, and the choice is remembered. CUDA is selectable only when the chosen ONNX Runtime DLL advertises `CUDAExecutionProvider`. **CUDA is NVIDIA-only**: an Intel or AMD integrated GPU cannot run it, no matter which DLL is installed. If a CUDA session cannot start, Beeswax falls back to CPU and shows the reason in the Processing card. Build without CUDA registration with `cargo build --release --no-default-features --features inference`; the choice is remembered in `data/settings.json`.

Beeswax checks the latest GitHub release at startup and compares it with the running version using SemVer. CI builds include the GitHub repository automatically; local builds discover it from `remote.origin.url` or accept `BEESWAX_GITHUB_REPOSITORY=owner/repo`. Tag a version such as `v0.2.0` to publish a Windows x64 release archive.

### Enabling CUDA

A GPU and a display driver are not enough: the normal ONNX Runtime wheel is CPU-only, so `runtime/onnxruntime.dll` exposes only `CPUExecutionProvider`. Check what you have:

```powershell
.\target\release\beeswax.exe --providers
# runtime dll: ...\runtime\onnxruntime.dll
# providers: CPU
# cuda available: false
```

Install the GPU build and its CUDA 12 / cuDNN 9 dependencies:

```powershell
# 1. Grab the GPU wheel (the CPU wheel has no onnxruntime_providers_cuda.dll).
python -m pip download --no-deps --dest .\target\ort-gpu onnxruntime-gpu==1.30.0

# 2. Copy its runtime next to Beeswax and point at a CUDA 12 / cuDNN 9 DLL folder.
#    Without -CudaDllDir the script looks inside PyTorch, which bundles both.
powershell -ExecutionPolicy Bypass -File scripts/setup-cuda-runtime.ps1 `
  -OrtWheel .\target\ort-gpu\onnxruntime_gpu-1.30.0-*.whl [-CudaDllDir 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin']

# 3. Verify.
.\target\release\beeswax.exe --providers      # -> providers: CPU, CUDA
.\target\release\beeswax.exe --self-test --cuda
```

The script backs up the previous runtime as `runtime/onnxruntime-cpu.dll`, copies `onnxruntime.dll` and `onnxruntime_providers_cuda.dll` into `runtime/`, and writes `runtime/cuda-path.txt`. Beeswax puts that folder first on `PATH` and preloads the CUDA/cuDNN DLLs from it (or from `BEESWAX_CUDA_DLLS`, semicolon-separated) before ONNX Runtime starts. Do not copy 1.5 GB of cuDNN DLLs into the project — pointing at the folder is enough.

Measured on the development laptop (RTX 3050 Laptop 4 GB, i7-12700H) with a game holding ~2.9 GB of VRAM:

| engine + model | 3 s of audio |
|---|---|
| CPU + int8 | 2.85 s |
| CPU + fp32 | 4.40 s |
| CUDA + int8 | 5.35 s (int8 LSTM nodes stay on the CPU) |
| CUDA + fp32 | 2.74 s |

So CUDA only pays off with the **fp32 model**; with int8 the CUDA session is slower than plain CPU.

Laptop GPUs are commonly 4 GB. A game holding most of the VRAM makes the CUDA session fail to allocate; close GPU-heavy apps before starting filtration.

The Intel iGPU cannot use CUDA. It would need a DirectML (or OpenVINO) build of ONNX Runtime, which Beeswax can report in `--providers` but does not expose as a backend yet.

Cargo must have access to its registry for the initial build. An environment with `CARGO_NET_OFFLINE=true` needs all dependencies pre-cached; the full inference build cannot run with only the UI dependencies cached. The app uses Windows' bundled `curl.exe` to download models and check for updates.

## Use

The window is arranged as four numbered steps, with the primary Start/Stop action pinned to the footer so it is always reachable while the steps scroll.

1. **Microphone.** Select a physical microphone from the picker (the Windows default is selected initially; full device names distinguish similar microphones). Use **Test microphone** to check the live dBFS meter, then **Stop** before recording.
2. **Your voice.** Click **Record my voice**. Speak alone for six seconds at normal volume. Beeswax selects the loudest three-second enrollment segment and rejects silence or excessive clipping. **Play enrollment** auditions the clip through the headphones or speakers picked in the same card.
3. **Where the voice goes.** Pick **CABLE Input (VB-Audio Virtual Cable)** to feed a call app, or any headphones/speakers to hear the separated voice directly. CABLE Input is preselected when installed. Monitoring through speakers can feed back into the microphone, so headphones are safer.
4. **Processing.** Pick the **CPU** or **CUDA** engine. If CUDA is unavailable the card says why, and a note appears if a requested backend had to fall back.
5. Click **Start filtering**. With CABLE Input selected, choose the matching **CABLE Output** as the microphone in Discord, Teams, OBS, or another app. Keep Beeswax running.
6. **Stop** closes capture and playback. **Forget voice** deletes the saved profile and WAV.

Pickers open as an overlay above the page: the list never pushes the steps below it out of place, and lower pickers open upwards so the whole list stays inside the window. Escape closes a list; clicking anywhere else dismisses it.

Install VB-CABLE separately from [VB-Audio](https://vb-audio.com/Cable/) if it is absent; its installer requires administrator access and may require a reboot. Beeswax does not install a driver or alter Windows default audio devices. Do not enable Windows “Listen to this device” back into the capture path.

Enrollment is stored locally in `data/voice.json` and a playable `data/enrollment.wav`. Only the selected three seconds are saved. No microphone audio is uploaded. Paths are resolved from the executable’s project ancestors when the models folder is present, otherwise from the working directory; use `BEESWAX_MODELS` and `BEESWAX_DATA` for explicit locations. Recording and re-enrollment are disabled while filtering runs. An unsuccessful recording preserves the existing profile.

## Microphone troubleshooting

If the meter does not respond, click **Test microphone** and read the status message. Windows error `0x80070005` means capture was denied. The **Windows microphone settings** button opens the privacy page; check microphone access and desktop-app access. If both are already enabled, launch the executable directly from your desktop session rather than a restricted agent session. This development session received Access denied even though the consent values were Allow.

## Audio pipeline

Physical microphone → WASAPI/CPAL → mono → streaming sinc resampler → 16 kHz / 40 ms blocks → VoxLock via ort → weighted overlap-add → cable-rate resampling → VB-CABLE playback endpoint.

The downloaded graphs use `enroll_mag` and `context`, with recurrent `h/c` state `[6,32,256]`. VoxLock 1.1.1's reference implementation pins the ONNX files to Hugging Face revision `a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03`; those weights expose the **1.1 ONNX interface**, not the incompatible v2 graphs. The 1.1.1 release changes automatic `live()` enrollment, while Beeswax uses explicit enrollment. The implementation preserves recurrent state, reflects the initial 256 samples, uses 512-sample frames with a 128-sample hop, and applies squared-Hann overlap normalization and the upstream -0.5 output scale.

Audio callbacks use bounded lock-free rings; model work runs on a worker thread. Missing samples produce silence. Capture and playback overload drop samples, trim stale capture when the model falls behind so latency stays bounded, and report the gap while keeping the session alive; repeated ONNX/device errors stop the stream, with no raw-audio bypass. Device IDs are used for opening endpoints rather than unstable enumeration positions. A stopped stream can be refreshed and restarted after device reconnection.

CPU inference runs the separation graph on all but two logical cores (at most eight; override with `BEESWAX_THREADS`). On the development laptop (i7-12700H) the int8 model needs about 2.8 s of CPU time per 3 s of audio with eight threads, so a busy machine can still fall behind; the stream then drops audio instead of stopping. The engine picks its model automatically: **CPU runs the int8 model**, **CUDA runs the highest precision installed** (fp32 > fp16 > int8), because the int8 LSTM nodes stay on the CPU even inside a CUDA session.

### Audio quality

The shipped separator is `models/chunk_int8.onnx`: a 16 kHz model with int8-quantized LSTMs. Working at 16 kHz band-limits everything to 8 kHz, so the result cannot sound like a raw 48 kHz microphone, and int8 is the fastest, least clean precision. The stream is high-passed at 70 Hz before the model to remove the sub-bass rumble that laptop and headset microphones pick up. The Microphone card reports the capture rate; a device that opens at 16 kHz or below is already band-limited before Beeswax sees it.

Ways to get better sound, in order of impact:

1. **Use a higher-precision model with CUDA.** Put `chunk.onnx` (fp32, the Hugging Face `onnx/chunk.onnx`) in `models/`; CUDA then runs fp32 while CPU keeps using int8. `scripts/fetch-model.ps1 -Precision fp32` downloads it. fp32 alone on CPU is slower than real time, so keep this combo on the GPU.
2. **Use the CUDA backend.** CPU inference on this laptop already needs about 2.8 s per 3 s of audio at int8; fp32 or a busy machine will drop audio, and dropped audio sounds like crackle and gaps.
3. **Check the Processing card while filtering.** Growing `gaps` or `dropped` counts mean the audio you hear is being cut, not that the model is noisy.
4. **Listen directly through headphones** instead of a call app, which may apply its own noise suppression, AGC or compression.
5. **Disable Windows microphone enhancements** (Sound settings → the input device → Enhancements) and use a headset microphone rather than a distant array if the input sounds distant.

### Latency

End-to-end delay comes from four places: the capture period (Beeswax asks WASAPI for the shortest period the driver publishes, typically 10 ms and at most the driver's minimum), the 0–40 ms wait while a 16 kHz block fills, the model's ~24 ms lookahead, and the playback path (Resampler < 2 ms, the output reserve, and the playback device period). The strongest lever is the reserve: Beeswax starts it at 60 ms — enough to cover one 40 ms block plus scheduling jitter — and adds 20 ms (up to 160 ms) only when the cable actually ran dry. The Processing card shows `≈ N ms added delay` while filtering, computed as capture period + 20 ms block wait + 24 ms model + reserve, so the effect of a backend change is visible.

The previous build used an 83 ms pre-roll and host-default WASAPI buffers; the new one asks for the driver's minimum period and starts at 60 ms, so the added delay drops by roughly 40–50 ms. VB-CABLE and the receiving call app add their own buffering, which Beeswax cannot observe or control.

There is no guarantee of completely removing other speakers. A clean enrollment and the same microphone/position improve the chance of retaining the intended voice. Independent hardware clocks can cause occasional gaps or an overflow on long sessions; adaptive clock-drift correction is not implemented.

## Verification

```powershell
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
cargo run --release -- --devices
cargo run --release -- --test-mic
cargo run --release -- --self-test
cargo run --release -- --self-test --cuda   # prints the effective engine and any fallback note
python scripts/verify-model.py
```

The Rust smoke test uses synthetic audio, exercises enrollment and streaming, and checks finite output and sample count. The Python check validates the actual local model contract and measures inference time independently; it does not test the Rust binding. Neither measures speaker separation quality.

For UI/audio development in an offline environment without the ort crates:

```powershell
cargo test --no-default-features --offline
cargo run --no-default-features --offline
```

This explicitly disables inference. Enrollment and device controls remain available; **Start filtering reports that inference is unavailable**. It is not a functional filtering build.

Hardware acceptance: enroll a real voice, record CABLE Output in a separate app while the enrolled person and a second speaker alternate and overlap, check retained-voice quality and background suppression, then exercise Stop, unplug/reconnect, re-enrollment, and a sustained call. This must be done on actual microphones and VB-CABLE; synthetic tests cannot establish it.

## Sources and licensing

Beeswax source code is licensed under MIT; see `LICENSE`. VoxLock's reference implementation and downloaded ONNX model files remain Apache-2.0 and are not relicensed by Beeswax. The model files are downloaded from the pinned upstream revision above. GPUI Fast and other dependencies retain their own licenses; VB-CABLE is a separate VB-Audio product.

- [gpui-fast](https://github.com/longbridge/gpui-fast) — native UI, Apache-2.0.
- [ort](https://github.com/pykeio/ort) — Rust ONNX Runtime binding.
- [VoxLock 1.1.1 streaming reference](https://github.com/arbazsiddiqui/voxlock/blob/v1.1.1/voxlock/core.py) and [Apache-2.0 model repository](https://huggingface.co/arbazsiddiqui/voxlock).
- [VB-CABLE routing](https://vb-audio.com/Cable/) — separately installed virtual driver.
