use crate::{
    Backend,
    dsp::{FFT, HOP, OverlapAdd},
};
use anyhow::{Context, Result, ensure};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use std::path::Path;

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
                0 => 6,
                count => count.clamp(2, 6),
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
        .with_context(|| {
            format!(
                "Could not load {}",
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string())
            )
        })
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

pub struct Extractor {
    session: Session,
    engine: Backend,
    model_name: &'static str,
    note: Option<String>,
    mag: Tensor<f32>,
    context: Tensor<f32>,
    h: Tensor<f32>,
    c: Tensor<f32>,
    pending: Vec<f32>,
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
        // The GPU keeps fewer threads busy; the CPU uses most of the cores.
        let chunk_threads = if backend == Backend::Cuda { 2 } else { threads };
        let (model, model_name) = chunk_model(folder, backend);
        let (session, engine, note) = match open_session(&model, backend, chunk_threads) {
            Ok(session) => (session, backend, None),
            Err(error) if backend == Backend::Cuda => {
                let session = open_session(&model, Backend::Cpu, threads)
                    .context("The CPU fallback could not load the voice model")?;
                (
                    session,
                    Backend::Cpu,
                    Some(format!(
                        "CUDA could not start ({}). Filtering with the CPU instead.",
                        first_line(&error)
                    )),
                )
            }
            Err(error) => return Err(error),
        };
        ensure!(
            session.inputs.iter().any(|i| i.name == "context"),
            "Expected the VoxLock 1.1 ONNX interface used by the 1.1.1 reference (context input)."
        );
        let input = Tensor::from_array(([1usize, sample.len()], sample.to_vec()))?;
        let outputs = enroll.run(ort::inputs!["enroll" => input])?;
        let (ms, mv) = outputs["enroll_mag"].try_extract_tensor::<f32>()?;
        let mag = Tensor::from_array((ms.to_vec(), mv.to_vec()))?;
        let (cs, cv) = outputs["context"].try_extract_tensor::<f32>()?;
        let context = Tensor::from_array((cs.to_vec(), cv.to_vec()))?;
        Ok(Self {
            session,
            engine,
            model_name,
            note,
            mag,
            context,
            h: Tensor::from_array(([6usize, 32, 256], vec![0f32; 6 * 32 * 256]))?,
            c: Tensor::from_array(([6usize, 32, 256], vec![0f32; 6 * 32 * 256]))?,
            pending: Vec::new(),
            started: false,
            skip: 256,
            ola: OverlapAdd::new(),
        })
    }

    /// Clear the streaming state so a cached extractor can start a new filtering
    /// session. The sessions and the enrollment embedding are reused.
    pub fn reset(&mut self) -> Result<()> {
        self.h = Tensor::from_array(([6usize, 32, 256], vec![0f32; 6 * 32 * 256]))?;
        self.c = Tensor::from_array(([6usize, 32, 256], vec![0f32; 6 * 32 * 256]))?;
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
        self.pending.extend_from_slice(audio);
        if !self.started {
            if self.pending.len() <= 256 {
                return Ok(Vec::new());
            }
            let mut padded: Vec<_> = self.pending[1..257].iter().rev().copied().collect();
            padded.append(&mut self.pending);
            self.pending = padded;
            self.started = true;
        }
        let mut result = Vec::new();
        while self.pending.len() >= FFT {
            let count = (1 + (self.pending.len() - FFT) / HOP).min(5);
            let len = (count - 1) * HOP + FFT;
            let wav = Tensor::from_array(([1usize, len], self.pending[..len].to_vec()))?;
            let out = self.session.run(ort::inputs!["wav" => wav,
                "enroll_mag" => &self.mag, "context" => &self.context, "h" => &self.h, "c" => &self.c])?;
            let (_, frames) = out["frames"].try_extract_tensor::<f32>()?;
            ensure!(
                frames.len() == count * FFT && frames.iter().all(|x| x.is_finite()),
                "Invalid model output; microphone stopped."
            );
            let samples = self.ola.push(frames);
            let drop = self.skip.min(samples.len());
            self.skip -= drop;
            result.extend_from_slice(&samples[drop..]);
            let (shape, values) = out["h_out"].try_extract_tensor::<f32>()?;
            self.h = Tensor::from_array((shape.to_vec(), values.to_vec()))?;
            let (shape, values) = out["c_out"].try_extract_tensor::<f32>()?;
            self.c = Tensor::from_array((shape.to_vec(), values.to_vec()))?;
            self.pending.drain(..count * HOP);
        }
        Ok(result)
    }
}
