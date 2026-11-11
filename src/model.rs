use crate::{
    Backend,
    dsp::{FFT, HOP, OverlapAdd},
};
use anyhow::{Context, Result, ensure};
use ort::{
    memory::Allocator,
    session::{
        Session,
        builder::GraphOptimizationLevel,
        run_options::{HasSelectedOutputs, OutputSelector, RunOptions},
    },
    value::{Tensor, TensorRef},
};
use std::path::Path;

const BLOCK_FRAMES: usize = 5;
const BLOCK_WAV_LEN: usize = (BLOCK_FRAMES - 1) * HOP + FFT;
const STATE_SHAPE: [usize; 3] = [6, 32, 256];
const STATE_LEN: usize = 6 * 32 * 256;

/// Execution providers the loaded ONNX Runtime build actually exposes. This is the
/// quickest way to tell a CPU-only runtime from a GPU one.
pub fn providers() -> Vec<&'static str> {
    #[cfg(feature = "inference")]
    let list = std::panic::catch_unwind(|| {
        use ort::execution_providers::{
            CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider,
            ExecutionProvider, OpenVINOExecutionProvider, TensorRTExecutionProvider,
        };
        let mut names = Vec::new();
        for (name, available) in [
            (
                "CPU",
                CPUExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false),
            ),
            (
                "CUDA",
                CUDAExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false),
            ),
            (
                "TensorRT",
                TensorRTExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false),
            ),
            (
                "DirectML",
                DirectMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false),
            ),
            (
                "OpenVINO",
                OpenVINOExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false),
            ),
        ] {
            if available {
                names.push(name);
            }
        }
        names
    })
    .unwrap_or_default();
    #[cfg(not(feature = "inference"))]
    let list = Vec::new();
    list
}

/// Load CUDA/cuDNN runtime DLLs from a directory before the CUDA provider starts.
/// The GPU build of ONNX Runtime does not bundle them, and they are often already
/// present in a PyTorch installation.
#[cfg(feature = "cuda")]
pub fn preload_cuda_dylibs(dirs: &[std::path::PathBuf]) -> Vec<String> {
    use ort::execution_providers::cuda;
    let mut errors = Vec::new();
    for dir in dirs {
        if dir.is_dir()
            && let Err(error) = cuda::preload_dylibs(Some(dir), Some(dir))
        {
            errors.push(format!("{}: {error}", dir.display()));
        }
    }
    errors
}

#[cfg(not(feature = "cuda"))]
pub fn preload_cuda_dylibs(_: &[std::path::PathBuf]) -> Vec<String> {
    Vec::new()
}

/// True when this build and the loaded ONNX Runtime can use the CUDA provider.
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    let available = std::panic::catch_unwind(|| {
        use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
        CUDAExecutionProvider::default()
            .is_available()
            .unwrap_or(false)
    })
    .unwrap_or(false);
    #[cfg(not(feature = "cuda"))]
    let available = false;
    available
}

/// Intra-op threads for CPU inference. Leave a couple of cores for audio and UI work.
fn cpu_threads() -> usize {
    std::env::var("BEESWAX_THREADS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or_else(|| {
            // On a hybrid CPU the process is pinned to the performance cores,
            // so the worker pool must not outnumber them.
            let cap = match crate::cpu::performance_cpus() {
                0 => 5,
                count => count.clamp(2, 5),
            };
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(4)
                // The int8 LSTMs plateau before saturating a larger host; leaving a
                // little more CPU headroom also keeps capture and playback responsive.
                .saturating_sub(2)
                .clamp(2, cap)
        })
}

fn apply_backend(
    #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
    mut builder: ort::session::builder::SessionBuilder,
    backend: Backend,
) -> Result<ort::session::builder::SessionBuilder> {
    match backend {
        Backend::Cpu => {}
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                use ort::execution_providers::cuda::CuDNNConvAlgorithmSearch;
                // Heuristic search avoids benchmarking every convolution shape at start-up,
                // and TF32 keeps fp32 matmuls on the tensor cores.
                builder = builder.with_execution_providers([CUDAExecutionProvider::default()
                    .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Heuristic)
                    .with_tf32(true)
                    .build()])?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA support is not included in this build.");
            }
        }
    }
    Ok(builder)
}

fn open_session(path: &Path, backend: Backend, intra_threads: usize) -> Result<Session> {
    let builder = Session::builder()?
        .with_intra_threads(intra_threads)?
        .with_inter_threads(1)?;
    let builder = apply_backend(builder, backend)?;
    builder
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(path)
        .with_context(|| model_load_context(path))
}

fn open_chunk_session(
    bytes: &[u8],
    path: &Path,
    backend: Backend,
    intra_threads: usize,
    specialized: bool,
    mag_frames: i64,
    ctx_frames: i64,
) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_intra_threads(intra_threads)?
        .with_inter_threads(1)?
        .with_dimension_override("S", BLOCK_WAV_LEN as i64)?;
    if !specialized {
        builder = builder
            .with_dimension_override("Fe", mag_frames)?
            .with_dimension_override("Fc", ctx_frames)?;
    }
    let builder = apply_backend(builder, backend)?;
    builder
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_memory(bytes)
        .with_context(|| model_load_context(path))
}

fn model_load_context(path: &Path) -> String {
    format!(
        "Could not load {}",
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string())
    )
}

fn first_line(error: &anyhow::Error) -> String {
    let text = format!("{error:#}");
    let line = text.lines().next().unwrap_or_default().trim().to_owned();
    if line.chars().count() > 150 {
        format!("{}…", line.chars().take(150).collect::<String>())
    } else {
        line
    }
}

/// Pick the separation model for the chosen backend. int8 keeps the CPU real time;
/// CUDA has the headroom for full precision, and only then do the LSTMs run on the GPU.
fn chunk_model(folder: &Path, backend: Backend) -> (std::path::PathBuf, &'static str) {
    let order: [(&str, &str); 4] = if backend == Backend::Cuda {
        [
            ("chunk_fp32.onnx", "fp32"),
            ("chunk.onnx", "fp32"),
            ("chunk_fp16.onnx", "fp16"),
            ("chunk_int8.onnx", "int8"),
        ]
    } else {
        [
            ("chunk_int8.onnx", "int8"),
            ("chunk_fp16.onnx", "fp16"),
            ("chunk_fp32.onnx", "fp32"),
            ("chunk.onnx", "fp32"),
        ]
    };
    for (file, label) in order {
        let path = folder.join(file);
        if path.is_file() {
            return (path, label);
        }
    }
    (folder.join("chunk_int8.onnx"), "int8")
}

/// Precision labels of the separation models present in this folder.
pub fn installed_models(folder: &Path) -> Vec<&'static str> {
    let mut list = Vec::new();
    for (file, label) in [
        ("chunk.onnx", "fp32"),
        ("chunk_fp32.onnx", "fp32"),
        ("chunk_fp16.onnx", "fp16"),
        ("chunk_int8.onnx", "int8"),
    ] {
        if folder.join(file).is_file() && !list.contains(&label) {
            list.push(label);
        }
    }
    list
}

/// Bake the enrollment embeddings (`enroll_mag` and `context`) directly into the
/// ONNX `GraphProto` as constant initializers and remove them from `graph.input`.
///
/// ONNX Runtime's `AddInitializer` C API only overrides *overridable* initializers
/// (inputs that already carry a default `TensorProto`); pure graph inputs remain
/// runtime inputs, which prevents Level-3 constant folding across the 32 speaker
/// cross-attention key/value projections in `/attenFuse`. Rewriting those two
/// `ValueInfoProto` entries into `GraphProto.initializer` lets ORT fold all
/// enrollment-only subgraphs once when the session opens.
fn specialize_chunk_onnx(
    model_bytes: &[u8],
    mag_shape: &[i64],
    mag_values: &[f32],
    ctx_shape: &[i64],
    ctx_values: &[f32],
) -> Option<Vec<u8>> {
    let mut top = ProtoIter::new(model_bytes, 0, model_bytes.len());
    while let Some(field) = top.next_field()? {
        // ModelProto.graph = field 7 (length-delimited GraphProto).
        if field.number == 7
            && let WireSpan::Bytes(vs, ve) = field.span
        {
            let mut remove_spans = Vec::with_capacity(2);
            let mut found_mag = false;
            let mut found_ctx = false;
            let mut graph = ProtoIter::new(model_bytes, vs, ve);
            while let Some(gfield) = graph.next_field()? {
                // GraphProto.input = field 11 (ValueInfoProto).
                if gfield.number == 11
                    && let WireSpan::Bytes(ivs, ive) = gfield.span
                    && let Some(name) = first_bytes_field(model_bytes, ivs, ive, 1)?
                {
                    if name == b"enroll_mag" {
                        found_mag = true;
                        remove_spans.push((gfield.start, gfield.end));
                    } else if name == b"context" {
                        found_ctx = true;
                        remove_spans.push((gfield.start, gfield.end));
                    }
                }
            }
            if !(found_mag && found_ctx && remove_spans.len() == 2) {
                return None;
            }
            let mut extra = Vec::with_capacity((mag_values.len() + ctx_values.len()) * 4 + 128);
            encode_initializer_f32(&mut extra, "enroll_mag", mag_shape, mag_values);
            encode_initializer_f32(&mut extra, "context", ctx_shape, ctx_values);

            let removed: usize = remove_spans.iter().map(|(a, b)| b - a).sum();
            let new_graph_len = (ve - vs) - removed + extra.len();
            let mut out = Vec::with_capacity(model_bytes.len() - removed + extra.len() + 16);
            out.extend_from_slice(&model_bytes[..field.start]);
            push_varint(&mut out, (7u64 << 3) | 2);
            push_varint(&mut out, new_graph_len as u64);
            let mut cursor = vs;
            for (a, b) in remove_spans {
                out.extend_from_slice(&model_bytes[cursor..a]);
                cursor = b;
            }
            out.extend_from_slice(&model_bytes[cursor..ve]);
            out.extend_from_slice(&extra);
            out.extend_from_slice(&model_bytes[ve..]);
            return Some(out);
        }
    }
    None
}

#[derive(Clone, Copy)]
enum WireSpan {
    Other,
    Bytes(usize, usize),
}

struct ProtoField {
    number: u64,
    start: usize,
    end: usize,
    span: WireSpan,
}

struct ProtoIter<'a> {
    buf: &'a [u8],
    pos: usize,
    end: usize,
}

impl<'a> ProtoIter<'a> {
    fn new(buf: &'a [u8], pos: usize, end: usize) -> Self {
        Self { buf, pos, end }
    }

    fn next_field(&mut self) -> Option<Option<ProtoField>> {
        if self.pos >= self.end {
            return Some(None);
        }
        let start = self.pos;
        let tag = read_varint(self.buf, &mut self.pos, self.end)?;
        let number = tag >> 3;
        if number == 0 {
            return None;
        }
        let span = match tag & 7 {
            0 => {
                read_varint(self.buf, &mut self.pos, self.end)?;
                WireSpan::Other
            }
            1 => {
                self.pos = self.pos.checked_add(8)?;
                if self.pos > self.end {
                    return None;
                }
                WireSpan::Other
            }
            2 => {
                let len = usize::try_from(read_varint(self.buf, &mut self.pos, self.end)?).ok()?;
                let vs = self.pos;
                let ve = vs.checked_add(len)?;
                if ve > self.end {
                    return None;
                }
                self.pos = ve;
                WireSpan::Bytes(vs, ve)
            }
            5 => {
                self.pos = self.pos.checked_add(4)?;
                if self.pos > self.end {
                    return None;
                }
                WireSpan::Other
            }
            _ => return None,
        };
        Some(Some(ProtoField {
            number,
            start,
            end: self.pos,
            span,
        }))
    }
}

fn first_bytes_field(buf: &[u8], start: usize, end: usize, target: u64) -> Option<Option<&[u8]>> {
    let mut iter = ProtoIter::new(buf, start, end);
    while let Some(field) = iter.next_field()? {
        if field.number == target
            && let WireSpan::Bytes(vs, ve) = field.span
        {
            return Some(Some(&buf[vs..ve]));
        }
    }
    Some(None)
}

fn read_varint(buf: &[u8], pos: &mut usize, end: usize) -> Option<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    while *pos < end && shift < 64 {
        let byte = buf[*pos];
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
        shift += 7;
    }
    None
}

fn push_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(((value as u8) & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn encode_initializer_f32(out: &mut Vec<u8>, name: &str, shape: &[i64], values: &[f32]) {
    let raw_len = std::mem::size_of_val(values);
    let mut tensor = Vec::with_capacity(raw_len + name.len() + 32);
    // TensorProto.dims = repeated int64 (field 1, varint)
    for &dim in shape {
        push_varint(&mut tensor, 1 << 3);
        push_varint(&mut tensor, dim as u64);
    }
    // TensorProto.data_type = FLOAT = 1 (field 2, varint)
    push_varint(&mut tensor, 2 << 3);
    push_varint(&mut tensor, 1);
    // TensorProto.name = string (field 8, bytes)
    push_varint(&mut tensor, (8 << 3) | 2);
    push_varint(&mut tensor, name.len() as u64);
    tensor.extend_from_slice(name.as_bytes());
    // TensorProto.raw_data = bytes (field 9, little-endian IEEE-754 f32)
    push_varint(&mut tensor, (9 << 3) | 2);
    push_varint(&mut tensor, raw_len as u64);
    for &sample in values {
        tensor.extend_from_slice(&sample.to_le_bytes());
    }
    // GraphProto.initializer = repeated TensorProto (field 5, bytes)
    push_varint(out, (5 << 3) | 2);
    push_varint(out, tensor.len() as u64);
    out.extend_from_slice(&tensor);
}

pub struct Extractor {
    session: Session,
    engine: Backend,
    model_name: &'static str,
    note: Option<String>,
    specialized: bool,
    mag: Option<Tensor<f32>>,
    context: Option<Tensor<f32>>,
    run_options: RunOptions<HasSelectedOutputs>,
    h: Vec<f32>,
    c: Vec<f32>,
    pending: Vec<f32>,
    tail: Vec<f32>,
    started: bool,
    skip: usize,
    ola: OverlapAdd,
}

impl Extractor {
    pub fn new(folder: &Path, sample: &[f32], backend: Backend) -> Result<Self> {
        Self::with_threads(folder, sample, backend, cpu_threads())
    }

    /// Like [`Extractor::new`] but with an explicit CPU worker count. The
    /// benchmark uses this to compare pool sizes without touching the process
    /// environment; normal callers keep the policy in `cpu_threads`.
    pub fn with_threads(
        folder: &Path,
        sample: &[f32],
        backend: Backend,
        threads: usize,
    ) -> Result<Self> {
        let threads = threads.max(1);
        let mut enroll = open_session(&folder.join("enroll.onnx"), Backend::Cpu, threads)?;
        let input = TensorRef::from_array_view(([1usize, sample.len()], sample))?;
        let outputs = enroll.run(ort::inputs!["enroll" => input])?;
        let (ms, mv) = outputs["enroll_mag"].try_extract_tensor::<f32>()?;
        let mag_shape: Vec<i64> = ms.to_vec();
        let mag_values: Vec<f32> = mv.to_vec();
        let (cs, cv) = outputs["context"].try_extract_tensor::<f32>()?;
        let ctx_shape: Vec<i64> = cs.to_vec();
        let ctx_values: Vec<f32> = cv.to_vec();
        drop(outputs);
        drop(enroll);

        let mag_frames = *mag_shape.get(2).unwrap_or(&376);
        let ctx_frames = *ctx_shape.get(2).unwrap_or(&298);

        // The GPU keeps fewer threads busy; the CPU uses most of the cores.
        let chunk_threads = if backend == Backend::Cuda { 2 } else { threads };
        let (model, model_name) = chunk_model(folder, backend);
        let raw_bytes = std::fs::read(&model).with_context(|| model_load_context(&model))?;
        let specialized_bytes =
            specialize_chunk_onnx(&raw_bytes, &mag_shape, &mag_values, &ctx_shape, &ctx_values);
        let specialized = specialized_bytes.is_some();
        let model_bytes = specialized_bytes.as_deref().unwrap_or(&raw_bytes);

        let (session, engine, note) = match open_chunk_session(
            model_bytes,
            &model,
            backend,
            chunk_threads,
            specialized,
            mag_frames,
            ctx_frames,
        ) {
            Ok(session) => (session, backend, None),
            Err(error) if backend == Backend::Cuda => {
                let (cpu_model, cpu_name) = chunk_model(folder, Backend::Cpu);
                let cpu_raw;
                let cpu_spec;
                let (cpu_bytes, cpu_specialized) = if cpu_model == model {
                    (model_bytes, specialized)
                } else {
                    cpu_raw = std::fs::read(&cpu_model)
                        .with_context(|| model_load_context(&cpu_model))?;
                    cpu_spec = specialize_chunk_onnx(
                        &cpu_raw,
                        &mag_shape,
                        &mag_values,
                        &ctx_shape,
                        &ctx_values,
                    );
                    let is_spec = cpu_spec.is_some();
                    (cpu_spec.as_deref().unwrap_or(&cpu_raw), is_spec)
                };
                let session = open_chunk_session(
                    cpu_bytes,
                    &cpu_model,
                    Backend::Cpu,
                    threads,
                    cpu_specialized,
                    mag_frames,
                    ctx_frames,
                )
                .context("The CPU fallback could not load the voice model")?;
                return Self::finish(
                    session,
                    Backend::Cpu,
                    cpu_name,
                    Some(format!(
                        "CUDA could not start ({}). Filtering with the CPU instead.",
                        first_line(&error)
                    )),
                    cpu_specialized,
                    mag_shape,
                    mag_values,
                    ctx_shape,
                    ctx_values,
                );
            }
            Err(error) => return Err(error),
        };
        Self::finish(
            session,
            engine,
            model_name,
            note,
            specialized,
            mag_shape,
            mag_values,
            ctx_shape,
            ctx_values,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        session: Session,
        engine: Backend,
        model_name: &'static str,
        note: Option<String>,
        specialized: bool,
        mag_shape: Vec<i64>,
        mag_values: Vec<f32>,
        ctx_shape: Vec<i64>,
        ctx_values: Vec<f32>,
    ) -> Result<Self> {
        ensure!(
            specialized || session.inputs.iter().any(|i| i.name == "context"),
            "Expected the VoxLock 1.1 ONNX interface used by the 1.1.1 reference (context input)."
        );
        let allocator = Allocator::default();
        let run_options = RunOptions::new()?.with_outputs(
            OutputSelector::default()
                .preallocate(
                    "frames",
                    Tensor::<f32>::new(&allocator, [1usize, BLOCK_FRAMES, FFT])?,
                )
                .preallocate("h_out", Tensor::<f32>::new(&allocator, STATE_SHAPE)?)
                .preallocate("c_out", Tensor::<f32>::new(&allocator, STATE_SHAPE)?),
        );
        let (mag, context) = if specialized {
            (None, None)
        } else {
            (
                Some(Tensor::from_array((mag_shape, mag_values))?),
                Some(Tensor::from_array((ctx_shape, ctx_values))?),
            )
        };
        Ok(Self {
            session,
            engine,
            model_name,
            note,
            specialized,
            mag,
            context,
            run_options,
            h: vec![0f32; STATE_LEN],
            c: vec![0f32; STATE_LEN],
            pending: Vec::with_capacity(BLOCK_WAV_LEN * 2),
            tail: vec![0f32; BLOCK_WAV_LEN],
            started: false,
            skip: 256,
            ola: OverlapAdd::new(),
        })
    }

    /// Clear only the recurrent state. A long silence makes the carried `h/c`
    /// stale, but the pending samples and overlap-add history must keep their
    /// place so the 40 ms output cadence does not move.
    pub fn reset_state(&mut self) -> Result<()> {
        self.h.fill(0.);
        self.c.fill(0.);
        Ok(())
    }

    /// Clear the streaming state so a cached extractor can start a new filtering
    /// session. The sessions and the enrollment embedding are reused.
    pub fn reset(&mut self) -> Result<()> {
        self.reset_state()?;
        self.pending.clear();
        self.started = false;
        self.skip = 256;
        self.ola = OverlapAdd::new();
        Ok(())
    }

    /// The backend that is actually running, which may differ from the request.
    pub fn engine(&self) -> Backend {
        self.engine
    }

    /// Human readable reason when the requested backend was not used.
    pub fn note(&self) -> Option<&str> {
        self.note.as_deref()
    }

    /// Precision label of the loaded model ("int8", "fp16" or "fp32").
    pub fn model_label(&self) -> &'static str {
        self.model_name
    }

    pub fn push(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        let mut result = Vec::new();
        self.push_into(audio, &mut result)?;
        Ok(result)
    }

    pub fn push_into(&mut self, audio: &[f32], result: &mut Vec<f32>) -> Result<()> {
        result.clear();
        self.pending.extend_from_slice(audio);
        if !self.started {
            if self.pending.len() <= 256 {
                return Ok(());
            }
            let mut padded = Vec::with_capacity(256 + self.pending.len() + BLOCK_WAV_LEN);
            padded.extend(self.pending[1..257].iter().rev().copied());
            padded.extend_from_slice(&self.pending);
            self.pending = padded;
            self.started = true;
        }
        while self.pending.len() >= FFT {
            let count = (1 + (self.pending.len() - FFT) / HOP).min(BLOCK_FRAMES);
            let len = (count - 1) * HOP + FFT;
            let wav_slice = if len == BLOCK_WAV_LEN {
                &self.pending[..BLOCK_WAV_LEN]
            } else {
                self.tail[..len].copy_from_slice(&self.pending[..len]);
                self.tail[len..].fill(0.);
                &self.tail[..]
            };
            let wav = TensorRef::from_array_view(([1usize, BLOCK_WAV_LEN], wav_slice))?;
            let h_in = TensorRef::from_array_view((STATE_SHAPE, self.h.as_slice()))?;
            let c_in = TensorRef::from_array_view((STATE_SHAPE, self.c.as_slice()))?;
            let out = if self.specialized {
                self.session.run_with_options(
                    ort::inputs!["wav" => wav, "h" => h_in, "c" => c_in],
                    &self.run_options,
                )?
            } else {
                let mag = self.mag.as_ref().expect("unspecialized mag");
                let ctx = self.context.as_ref().expect("unspecialized context");
                self.session.run_with_options(
                    ort::inputs![
                        "wav" => wav,
                        "enroll_mag" => mag,
                        "context" => ctx,
                        "h" => h_in,
                        "c" => c_in,
                    ],
                    &self.run_options,
                )?
            };
            let (_, all_frames) = out["frames"].try_extract_tensor::<f32>()?;
            ensure!(
                all_frames.len() == BLOCK_FRAMES * FFT
                    && all_frames[..count * FFT].iter().all(|x| x.is_finite()),
                "Invalid model output; microphone stopped."
            );
            self.ola
                .push_into(&all_frames[..count * FFT], &mut self.skip, result);
            if count == BLOCK_FRAMES {
                let (_, values) = out["h_out"].try_extract_tensor::<f32>()?;
                self.h.copy_from_slice(values);
                let (_, values) = out["c_out"].try_extract_tensor::<f32>()?;
                self.c.copy_from_slice(values);
            }
            let consumed = count * HOP;
            let remaining = self.pending.len() - consumed;
            self.pending.copy_within(consumed.., 0);
            self.pending.truncate(remaining);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_model(inputs: &[&str]) -> Vec<u8> {
        let mut graph = Vec::new();
        // GraphProto.name = "test" (field 2)
        push_varint(&mut graph, (2 << 3) | 2);
        push_varint(&mut graph, 4);
        graph.extend_from_slice(b"test");
        for name in inputs {
            let mut vi = Vec::new();
            push_varint(&mut vi, (1 << 3) | 2);
            push_varint(&mut vi, name.len() as u64);
            vi.extend_from_slice(name.as_bytes());
            push_varint(&mut graph, (11 << 3) | 2);
            push_varint(&mut graph, vi.len() as u64);
            graph.extend_from_slice(&vi);
        }
        let mut model = Vec::new();
        // ModelProto.ir_version = 8 (field 1)
        push_varint(&mut model, 1 << 3);
        push_varint(&mut model, 8);
        // ModelProto.graph (field 7)
        push_varint(&mut model, (7 << 3) | 2);
        push_varint(&mut model, graph.len() as u64);
        model.extend_from_slice(&graph);
        model
    }

    #[test]
    fn specialization_replaces_enrollment_inputs_with_initializers() {
        let raw = minimal_model(&["wav", "enroll_mag", "context", "h", "c"]);
        let spec = specialize_chunk_onnx(&raw, &[1, 2, 3], &[1.0; 6], &[1, 2, 2], &[2.0; 4])
            .expect("valid VoxLock graph should specialize");

        let mut top = ProtoIter::new(&spec, 0, spec.len());
        let mut inputs = Vec::new();
        let mut initializers = Vec::new();
        while let Some(field) = top.next_field().unwrap() {
            if field.number == 7
                && let WireSpan::Bytes(vs, ve) = field.span
            {
                let mut g = ProtoIter::new(&spec, vs, ve);
                while let Some(gf) = g.next_field().unwrap() {
                    if gf.number == 11
                        && let WireSpan::Bytes(s, e) = gf.span
                        && let Some(name) = first_bytes_field(&spec, s, e, 1).unwrap()
                    {
                        inputs.push(std::str::from_utf8(name).unwrap().to_owned());
                    } else if gf.number == 5
                        && let WireSpan::Bytes(s, e) = gf.span
                        && let Some(name) = first_bytes_field(&spec, s, e, 8).unwrap()
                    {
                        initializers.push(std::str::from_utf8(name).unwrap().to_owned());
                    }
                }
            }
        }
        assert_eq!(inputs, vec!["wav", "h", "c"]);
        assert_eq!(initializers, vec!["enroll_mag", "context"]);

        // Missing inputs or truncated bytes safely fall back to None.
        assert!(
            specialize_chunk_onnx(&minimal_model(&["wav", "h", "c"]), &[1], &[0.], &[1], &[0.])
                .is_none()
        );
        assert!(specialize_chunk_onnx(&raw[..8], &[1], &[0.], &[1], &[0.]).is_none());
    }
}
