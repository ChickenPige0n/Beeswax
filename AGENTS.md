# Beeswax — Technical & Contributor Guide

This document collects the architecture, audio pipeline internals, build/vendoring workflow, CUDA setup, environment variables, and verification steps for contributors and coding agents.

---

## Repository Layout

- `src/main.rs` — Windows GUI entry point (`windows_subsystem = "windows"`), CLI diagnostic flags (`--devices`, `--test-mic`, `--providers`, `--self-test`, `--lang`), and GPUI window setup.
- `src/i18n.rs` — Lightweight runtime localization (`Language::En`, `Language::ZhCn`), Windows UI language detection (`GetUserDefaultUILanguage`), and atomic language state shared with worker threads.
- `src/ui/` — `gpui-fast` user interface (`mod.rs`, `sections.rs`, `select.rs`, `theme.rs`).
- `src/audio.rs` — WASAPI/CPAL device enumeration, low-latency stream setup, lock-free `ringbuf` capture/playback queues, worker loop, adaptive output reserve, preview playback, and `ModelCache`.
- `src/dsp.rs` — Precomputed polyphase windowed-sinc `Resampler` (64 taps, 4096 phases), 70 Hz Butterworth `HighPass`, squared-Hann `OverlapAdd`, silence `Gate`, and 3-second peak-energy enrollment extraction.
- `src/model.rs` / `src/model_disabled.rs` — ONNX Runtime (`ort 2.0.0-rc.10`) enrollment and streaming separator, protobuf graph specialization (`specialize_chunk_onnx`), preallocated output tensors, and automatic CUDA-to-CPU fallback.
- `src/runtime.rs` — `onnxruntime.dll` discovery and CUDA/cuDNN DLL path resolution (`BEESWAX_CUDA_DLLS`, `runtime/cuda-path.txt`, `runtime/cuda/`).
- `src/cpu.rs` — Hybrid x86 CPU topology inspection and performance-core affinity pinning.
- `src/profile.rs` — Atomic local persistence for `data/voice.json`, `data/enrollment.wav`, and `data/settings.json` (`backend`, `language`).
- `src/model_download.rs` — First-run fallback downloader (using hidden `curl.exe` with SHA-256 verification against `models.json`) for source builds missing `models/`.
- `src/updates.rs` — Startup release check via `https://github.com/<owner>/<repo>/releases/latest` HTTP redirect + SemVer comparison (avoids GitHub REST API rate limits).
- `src/bin/bench.rs` — Deterministic SI-SDR and chunk-latency benchmark binary (`beeswax-bench`).
- `assets/` — Embedded multi-size Windows icon (`icon.ico`), UI header icon (`icon.png`), Win32 resource script (`beeswax.rc`), manifest (`beeswax.manifest.xml`), and documentation screenshots.
- `build.rs` — Compiles `assets/beeswax.rc` via `embed-resource` so `beeswax.exe` carries icon resource ID `1` and PerMonitorV2 DPI manifest.
- `scripts/vendor.py` — Downloads, SHA-256 verifies, and packages ONNX Runtime 1.30.0 and VoxLock models.

---

## Build, Vendor, and Package

Requires Windows x64, Rust/MSVC with the Windows SDK, and Python 3.9+ for `scripts/vendor.py`:

```powershell
python scripts/vendor.py
cargo build --release --locked
.\target\release\beeswax.exe
```

`scripts/vendor.py` installs the pinned ONNX Runtime 1.30.0 CPU DLL and the two VoxLock models verified against `models.json`:

```text
runtime/onnxruntime.dll
models/enroll.onnx
models/chunk_int8.onnx
```

Useful commands:
- `python scripts/vendor.py check` — Verify `runtime/` and `models/` checksums without downloading.
- `python scripts/vendor.py package` — Assemble `target/beeswax-windows-x64.zip` (`beeswax.exe`, `runtime/onnxruntime.dll`, `models/*.onnx`, `models.json`, `README.md`, `LICENSE`).
- `python scripts/vendor.py runtime --force` — Restore the pinned CPU `onnxruntime.dll` over a manually installed CUDA runtime.

Release CI (`.github/workflows/ci.yml`) caches `runtime/` and `models/` keyed by `models.json` and `scripts/vendor.py`, runs formatting/lint/tests/smoke-tests, verifies that `vX.Y.Z` tags match `Cargo.toml`, and publishes `beeswax-windows-x64.zip`.

For offline UI/audio work when `ort` crates are unavailable:

```powershell
cargo test --no-default-features --offline
cargo run --no-default-features --offline
```

This uses `src/model_disabled.rs` and never substitutes raw unfiltered audio.

---

## Audio & Inference Architecture

### Signal Chain

`Physical microphone → WASAPI/CPAL → mono downmix → polyphase sinc resampler (16 kHz) → 70 Hz high-pass → 40 ms (640-sample) chunks → silence gate → VoxLock ONNX via ort → squared-Hann overlap-add → output sinc resampler → VB-CABLE / headphones`

### Model Specialization & Streaming State

- **Upstream weights**: VoxLock 1.1.1 reference pinned in `models.json` (`arbazsiddiqui/voxlock` revision `a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03`, 1.1 ONNX interface).
- **Inputs/State**: `enroll.onnx` maps 3 s (48,000 samples @ 16 kHz) to `enroll_mag` and `context`. Streaming `chunk*.onnx` consumes `wav` (`[1, 1024]`, 5 frames × 128 hop + 512 FFT) and recurrent `h`/`c` (`[6, 32, 256]`), returning `[1, 5, 512]` frames.
- **Graph constant folding**: `specialize_chunk_onnx` rewrites `enroll_mag` and `context` from `GraphProto.input` into `GraphProto.initializer` in memory and overrides `S = 1024` before ORT `GraphOptimizationLevel::Level3`, allowing ORT to constant-fold all 32 cross-attention K/V projections once at session creation.
- **Zero-allocation steady state**: Output tensors (`frames`, `h_out`, `c_out`) are preallocated on `RunOptions`, and DSP buffers are reused across 40 ms blocks.
- **Session cache**: `ModelCache` retains healthy CPU `Extractor` instances across Stop/Start cycles (clearing only `h`/`c`, pending samples, and overlap-add state). CUDA sessions are dropped on Stop so idle VRAM is freed immediately.

### Silence Gate & Real-Time Safety

- **Silence gate**: After 5 consecutive 40 ms blocks below `-60 dBFS` RMS (200 ms hangover), inference is skipped and exact zeros are pushed through the output resampler. Speech resumes inference immediately. After >1 s (25 blocks) of silence, the first speech block clears only `h`/`c` (`GateAction::ResumeState`) so stale recurrent context is discarded without shifting sample cadence.
- **Bounded latency & overload handling**: Audio callbacks communicate via lock-free `ringbuf` queues. Capture backlogs exceeding 250 ms are trimmed down to 50 ms so temporary CPU spikes cannot permanently inflate delay. Repeated ONNX errors stop the stream cleanly rather than leaking unfiltered audio.
- **Latency budget**: Capture period (driver minimum requested from WASAPI, ~10 ms) + ~20 ms average block fill + 24 ms model lookahead (384 samples @ 16 kHz) + output reserve (starts at 60 ms, steps +20 ms up to 160 ms only on actual playback underruns).

### Hybrid CPU Pinning

On hybrid x86 CPUs with SMT performance cores (e.g., Intel 12th–14th Gen), `beeswax::cpu::pin_to_performance_cores()` restricts process affinity to P-cores before ONNX Runtime initializes and caps intra-op threads (`2..=5`), preventing Windows from migrating worker threads onto slower E-cores while leaving E-cores free for games and call apps. Homogeneous CPUs are untouched.

---

## Enabling CUDA (Optional)

The bundled runtime is CPU-only (`CPUExecutionProvider`) and runs `chunk_int8.onnx` in real time out of the box. CUDA is NVIDIA-only and requires both the GPU ONNX Runtime DLLs and full-precision `models/chunk.onnx` (`fp32`), because quantized `int8` LSTM nodes remain on CPU even inside a CUDA session.

1. Download `onnxruntime-win-x64-gpu_cuda12-1.30.0.zip` from [ONNX Runtime v1.30.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0).
2. Copy `lib/onnxruntime.dll`, `lib/onnxruntime_providers_cuda.dll`, and `lib/onnxruntime_providers_shared.dll` into `runtime/`.
3. Download `onnx/chunk.onnx` (fp32) from the pinned [VoxLock revision](https://huggingface.co/arbazsiddiqui/voxlock/tree/a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03/onnx) into `models/chunk.onnx`.
4. Provide CUDA 12 / cuDNN 9 DLLs via `runtime/cuda/`, `BEESWAX_CUDA_DLLS`, or one directory path per line in `runtime/cuda-path.txt` (e.g., an existing PyTorch `lib` directory).
5. Verify with:
   ```powershell
   .\target\release\beeswax.exe --providers
   .\target\release\beeswax.exe --self-test --cuda
   ```

Reference timings for 3 s of audio on an RTX 3050 Laptop (4 GB) + i7-12700H:
- `CPU + int8`: ~0.80 s
- `CUDA + fp32`: ~0.96 s (offloads CPU when gaming/encoding; requires free VRAM)
- `CPU + fp32`: ~2.7 s
- `CUDA + int8`: ~5.3 s (int8 LSTMs fall back to CPU)

---

## Environment Variables & CLI Flags

### Environment Variables

| Variable | Purpose |
|---|---|
| `ORT_DYLIB_PATH` | Explicit path to `onnxruntime.dll` (API 22+ / ORT 1.22+). |
| `BEESWAX_MODELS` | Override path to the `models/` directory. |
| `BEESWAX_DATA` | Override path to the `data/` directory (`voice.json`, `enrollment.wav`, `settings.json`). |
| `BEESWAX_THREADS` | Override ONNX Runtime CPU intra-op thread count. |
| `BEESWAX_CUDA_DLLS` | Semicolon-separated directories containing CUDA 12 / cuDNN 9 DLLs. |
| `BEESWAX_LANG` | Override initial UI language (`en` or `zh-CN`). |
| `BEESWAX_GITHUB_REPOSITORY` | `owner/repo` used for release update checks. |
| `BEESWAX_DEBUG` | Print diagnostic preload warnings during runtime startup. |

### CLI Flags

Because `beeswax.exe` uses the Windows GUI subsystem (no console window on launch), diagnostic flags attach to the parent console or write to redirected pipes:

```powershell
.\target\release\beeswax.exe --help
.\target\release\beeswax.exe --devices
.\target\release\beeswax.exe --test-mic
.\target\release\beeswax.exe --providers
.\target\release\beeswax.exe --self-test
.\target\release\beeswax.exe --self-test --cuda
.\target\release\beeswax.exe --lang zh
```

---

## Verification Checklist

```powershell
python scripts/vendor.py check
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
.\target\release\beeswax.exe --providers
.\target\release\beeswax.exe --self-test
```

Optional benchmark (`src/bin/bench.rs`):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bench-fixture.ps1
cargo run --release --locked --bin beeswax-bench
```
