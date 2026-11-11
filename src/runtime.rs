//! ONNX Runtime loading, including CUDA/cuDNN dependency resolution.
//!
//! The same initialization runs in the app and the benchmark so both load the
//! identical runtime and provider DLLs.

use std::path::{Path, PathBuf};

/// Path to `onnxruntime.dll`: an explicit override, the packaged runtime, or a
/// copy beside the executable.
pub fn runtime_dll(root: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH") {
        return PathBuf::from(path);
    }
    let packaged = root.join("runtime").join("onnxruntime.dll");
    if packaged.is_file() {
        return packaged;
    }
    // Also accept the DLL copied directly beside beeswax.exe.
    if let Some(beside) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("onnxruntime.dll")))
        && beside.is_file()
    {
        return beside;
    }
    packaged
}

/// Directories that may hold cuBLAS/cuDNN, in resolution order.
pub fn cuda_dll_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(value) = std::env::var("BEESWAX_CUDA_DLLS") {
        for entry in value.split(';') {
            let entry = entry.trim();
            if !entry.is_empty() {
                dirs.push(PathBuf::from(entry));
            }
        }
    }
    if let Ok(list) = std::fs::read_to_string(root.join("runtime").join("cuda-path.txt")) {
        for line in list.lines() {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with('#') {
                dirs.push(PathBuf::from(line));
            }
        }
    }
    let local = root.join("runtime").join("cuda");
    if local.is_dir() {
        dirs.push(local);
    }
    dirs
}

#[cfg(feature = "inference")]
pub fn initialize(root: &Path) -> anyhow::Result<()> {
    let path = runtime_dll(root);
    anyhow::ensure!(
        path.is_file(),
        "{}",
        crate::i18n::tr(
            "ONNX Runtime (onnxruntime.dll) is missing. Reinstall Beeswax from the Windows release package, or set ORT_DYLIB_PATH to an ONNX Runtime 1.22+ DLL.",
            "缺少 ONNX Runtime (onnxruntime.dll)。请从 Windows 发行包重新安装 Beeswax，或将 ORT_DYLIB_PATH 指向 ONNX Runtime 1.22+ DLL。"
        )
    );
    // The CUDA provider DLL is loaded by onnxruntime.dll itself; its cublas/cuDNN
    // dependencies must already be resolvable. Put the directory first on PATH so the
    // loader can resolve them by name, then preload them by absolute path.
    let dirs = cuda_dll_dirs(root);
    if !dirs.is_empty() {
        let existing = std::env::var_os("PATH").unwrap_or_default();
        let mut entries: Vec<PathBuf> = dirs.to_vec();
        entries.extend(std::env::split_paths(&existing));
        if let Ok(joined) = std::env::join_paths(entries) {
            // SAFETY: this runs during startup, before the UI and audio threads exist.
            unsafe { std::env::set_var("PATH", joined) };
        }
        for error in crate::model::preload_cuda_dylibs(&dirs) {
            if std::env::var_os("BEESWAX_DEBUG").is_some() {
                eprintln!("CUDA DLL preload: {error}");
            }
        }
    }
    ort::init_from(path.to_string_lossy())
        .with_name("Beeswax")
        .with_telemetry(false)
        .commit()?;
    Ok(())
}

#[cfg(not(feature = "inference"))]
pub fn initialize(_: &Path) -> anyhow::Result<()> {
    anyhow::bail!(
        "{}",
        crate::i18n::tr(
            "Enrollment/UI development build. Rebuild with default features to enable filtering.",
            "当前为声纹/界面开发构建，请使用默认特性重新编译以启用人声过滤。"
        )
    )
}
