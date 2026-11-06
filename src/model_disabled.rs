//! Explicit enrollment/UI development build; never substitutes unfiltered audio.
pub fn cuda_available() -> bool {
    false
}
pub fn providers() -> Vec<&'static str> {
    Vec::new()
}
pub fn installed_models(_: &std::path::Path) -> Vec<&'static str> {
    Vec::new()
}
pub fn preload_cuda_dylibs(_: &[std::path::PathBuf]) -> Vec<String> {
    Vec::new()
}
pub struct Extractor;
impl Extractor {
    pub fn new(_: &std::path::Path, _: &[f32], _: crate::Backend) -> anyhow::Result<Self> {
        anyhow::bail!(
            "Inference is not included in this development build. Build with default features."
        )
    }
    pub fn engine(&self) -> crate::Backend {
        crate::Backend::Cpu
    }
    pub fn note(&self) -> Option<&str> {
        None
    }
    pub fn model_label(&self) -> &'static str {
        "none"
    }
    pub fn push(&mut self, _: &[f32]) -> anyhow::Result<Vec<f32>> {
        anyhow::bail!("Inference unavailable")
    }
}
