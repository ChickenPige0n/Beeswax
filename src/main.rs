mod model_download;
mod profile;
mod ui;
mod updates;

use beeswax::{
    Backend,
    audio::{self, Job, Status},
    model::{self, Extractor},
};
use gpui::{App, AppContext as _, Bounds, TitlebarOptions, WindowBounds, WindowOptions, px};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

fn main() -> anyhow::Result<()> {
    // Hybrid CPUs: keep ONNX Runtime on the performance cores. This runs before
    // any session is created, so its threads inherit the affinity.
    beeswax::cpu::pin_to_performance_cores();
    // Double-clicking the executable must still find the project's models/runtime.
    let executable = std::env::current_exe()?;
    let root = executable
        .ancestors()
        .skip(1)
        .find(|p| {
            p.join("Cargo.toml").is_file() || p.join(".git").exists() || p.join("models").is_dir()
        })
        .map(PathBuf::from)
        .unwrap_or(std::env::current_dir()?);
    let update_repository = updates::repository(&root);
    let model_dir = std::env::var_os("BEESWAX_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("models"));
    let data_dir = std::env::var_os("BEESWAX_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("data"));
    if std::env::args().any(|a| a == "--devices") {
        for input in [true, false] {
            for device in audio::devices(input)? {
                println!(
                    "{}: {} [{}]",
                    if input { "Input" } else { "Output" },
                    device.name,
                    device.id
                );
            }
        }
        return Ok(());
    }
    if std::env::args().any(|a| a == "--test-mic") {
        let list = audio::devices(true)?;
        let device = list
            .iter()
            .find(|d| d.is_default)
            .ok_or_else(|| anyhow::anyhow!("No default microphone"))?;
        println!("Testing {} for three seconds (not saved)", device.name);
        let status = Arc::new(Mutex::new(Status::default()));
        let cache = Arc::new(audio::ModelCache::new());
        let job = Job::start(
            device.id.clone(),
            None,
            model_dir,
            None,
            status.clone(),
            true,
            Backend::Cpu,
            cache,
        );
        for _ in 0..6 {
            std::thread::sleep(Duration::from_millis(500));
            let s = status.lock().unwrap();
            println!("{:.1} dBFS: {}", 20. * s.level.max(1e-5).log10(), s.message);
            if job.finished() {
                anyhow::bail!("Microphone test stopped: {}", s.message);
            }
        }
        job.cancel();
        return Ok(());
    }
    let runtime_error = std::panic::catch_unwind(|| beeswax::runtime::initialize(&root))
        .unwrap_or_else(|_| {
            Err(anyhow::anyhow!(
                "ONNX Runtime could not initialize. Reinstall Beeswax, or set ORT_DYLIB_PATH to an ONNX Runtime 1.22+ DLL for this architecture."
            ))
        })
        .err()
        .map(|e| format!("{e:#}"));
    if std::env::args().any(|a| a == "--providers") {
        println!(
            "runtime dll: {}",
            beeswax::runtime::runtime_dll(&root).display()
        );
        if let Some(error) = &runtime_error {
            println!("runtime error: {error}");
            println!("providers: unavailable");
            return Ok(());
        }
        let providers = model::providers();
        println!(
            "providers: {}",
            if providers.is_empty() {
                "none".to_owned()
            } else {
                providers.join(", ")
            }
        );
        println!("cuda available: {}", model::cuda_available());
        #[cfg(feature = "inference")]
        {
            let dirs = beeswax::runtime::cuda_dll_dirs(&root);
            println!(
                "cuda dll dirs: {}",
                if dirs.is_empty() {
                    "none".to_owned()
                } else {
                    dirs.iter()
                        .map(|dir| dir.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            );
            for error in model::preload_cuda_dylibs(&dirs) {
                println!("cuda preload error: {error}");
            }
        }
        return Ok(());
    }
    if std::env::args().any(|a| a == "--self-test") {
        if let Some(e) = runtime_error {
            anyhow::bail!(e);
        }
        let backend = if std::env::args().any(|a| a == "--cuda") {
            Backend::Cuda
        } else {
            Backend::Cpu
        };
        let sample: Vec<_> = (0..48000).map(|i| 0.1 * (i as f32 * 0.086).sin()).collect();
        let mut model = Extractor::new(&model_dir, &sample, backend)?;
        // Warm up first: GPU backends build cuDNN plans and allocate memory on the first runs.
        let mut count = 0;
        for part in sample.chunks(640).take(5) {
            count += model.push(part)?.len();
        }
        let start = std::time::Instant::now();
        for part in sample.chunks(640).skip(5) {
            count += model.push(part)?.len();
        }
        anyhow::ensure!(count >= 47000, "Unexpected output sample count: {count}");
        println!(
            "Model smoke test passed: {count} finite samples, {:.2}s processing for 3s input with {} + {} model (CUDA available: {}). This is not a speech-quality test.",
            start.elapsed().as_secs_f32(),
            model.engine().label(),
            model.model_label(),
            model::cuda_available()
        );
        if let Some(note) = model.note() {
            println!("Backend note: {note}");
        }
        return Ok(());
    }
    gpui_platform::application().run(move |cx: &mut App| {
        cx.on_window_closed(|cx, _| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();
        let bounds = Bounds::centered(None, gpui::size(px(740.), px(780.)), cx);
        cx.open_window(
            WindowOptions {
                titlebar: Some(TitlebarOptions {
                    title: Some("Beeswax".into()),
                    ..Default::default()
                }),
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(gpui::size(px(580.), px(640.))),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|cx| {
                    ui::Beeswax::new(model_dir, data_dir, runtime_error, update_repository, cx)
                })
            },
        )
        .expect("Unable to open Beeswax window");
        cx.activate(true);
    });
    Ok(())
}
