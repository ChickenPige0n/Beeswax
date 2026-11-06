pub mod audio;
pub mod dsp;
#[cfg(feature = "inference")]
pub mod model;
#[cfg(not(feature = "inference"))]
#[path = "model_disabled.rs"]
pub mod model;

/// Which ONNX Runtime execution provider runs the separation model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// Portable CPU inference.
    #[default]
    Cpu,
    /// NVIDIA GPU inference through ONNX Runtime's CUDA provider.
    Cuda,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
