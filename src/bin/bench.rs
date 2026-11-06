//! Deterministic benchmark for the filtering pipeline.
//!
//! Workload: the saved enrollment clip (data/voice.json or data/enrollment.wav)
//! mixed with a fixed, locally synthesized competing speaker at 0 dB SNR. The
//! interference fixture is produced by scripts/bench-fixture.ps1 and reused
//! byte-for-byte, so runs compare the same audio.
//!
//! Quality is SI-SDR of the model output against the clean enrollment clip,
//! minus SI-SDR of the mixture (which is 0 dB by construction). Performance is
//! the steady-state wall-clock cost per 40 ms block through the same chain the
//! app runs: 48 kHz capture resampling, high-pass, streaming model, overlap-add
//! and 48 kHz output resampling.

use beeswax::{
    Backend,
    dsp::{self, CHUNK, HighPass, RATE, Resampler, rms},
    model::Extractor,
};
use std::{path::Path, time::Instant};

const RATE_U32: u32 = RATE as u32;
const AUDIO_SECONDS: usize = 3;
const FIXTURE_OFFSET_SECONDS: usize = 3;
const CAPTURE_RATE: u32 = 48_000;
const CAPTURE_BLOCK: usize = 480;
const WARMUP_PASSES: usize = 1;
const TIMED_PASSES: usize = 6;
const COMPARE_PASSES: usize = 4;
const MAX_LAG: isize = 1024;

fn main() -> anyhow::Result<()> {
    raise_priority();
    beeswax::cpu::pin_to_performance_cores();
    let root = std::env::current_dir()?;
    init_runtime(&root)?;
    let models = root.join("models");
    let target = load_target(&root)?;
    let interference = load_interference(&root, &target)?;
    let mixture = mix(&target, &interference);

    if let Ok(specification) = std::env::var("BENCH_COMPARE") {
        return compare_threads(&models, &target, &mixture, &specification);
    }

    let started = Instant::now();
    let mut extractor = Extractor::new(&models, &target, Backend::Cpu)?;
    let startup_ms = started.elapsed().as_secs_f64() * 1000.;
    let quality = measure_quality(&mut extractor, &target, &interference, &mixture)?;
    let perf = measure_pipeline(extractor, &mixture, startup_ms)?;

    // Second start in the same process: filtering stopped and the user pressed
    // Start again. Without a session cache this reloads both models.
    let restart = Instant::now();
    let reloaded = Extractor::new(&models, &target, Backend::Cpu)?;
    let warm_start_ms = restart.elapsed().as_secs_f64() * 1000.;
    drop(reloaded);

    println!("METRIC chunk_latency_ms={:.3}", perf.chunk_ms);
    println!("METRIC chunk_latency_median_ms={:.3}", perf.chunk_median_ms);
    println!("METRIC chunk_latency_max_ms={:.3}", perf.chunk_max_ms);
    println!("METRIC chunk_cpu_ms={:.3}", perf.chunk_cpu_ms);
    println!("METRIC model_latency_ms={:.3}", perf.model_ms);
    println!("METRIC dsp_latency_ms={:.3}", perf.dsp_ms);
    println!("METRIC rtf={:.4}", perf.rtf);
    println!("METRIC startup_ms={:.1}", perf.startup_ms);
    println!("METRIC warm_start_ms={:.1}", warm_start_ms);
    println!("METRIC si_sdr_improvement_db={:.3}", quality.improvement);
    println!("METRIC si_sdr_in_db={:.3}", quality.in_sdr);
    println!("METRIC si_sdr_out_db={:.3}", quality.out_sdr);
    println!("METRIC target_gain_db={:.3}", quality.target_gain);
    println!(
        "METRIC interference_suppression_db={:.3}",
        quality.interference_suppression
    );
    println!("METRIC alignment_lag={}", quality.lag);
    Ok(())
}

fn init_runtime(root: &Path) -> anyhow::Result<()> {
    let path = root.join("runtime").join("onnxruntime.dll");
    anyhow::ensure!(
        path.is_file(),
        "{} is missing; run `python scripts/vendor.py runtime` first",
        path.display()
    );
    ort::init_from(path.to_string_lossy())
        .with_name("BeeswaxBench")
        .with_telemetry(false)
        .commit()?;
    Ok(())
}

/// Benchmark the pipeline's capability rather than the background load: the
/// measurement process runs high-priority, while the app itself stays normal.
fn raise_priority() {
    win::raise_priority();
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;

    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn SetPriorityClass(process: *mut c_void, class: u32) -> i32;
        fn GetProcessTimes(
            process: *mut c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }

    pub fn raise_priority() {
        const HIGH_PRIORITY_CLASS: u32 = 0x0000_0080;
        // SAFETY: the pseudo-handle from GetCurrentProcess is always valid.
        unsafe {
            SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        }
    }

    /// Total processor time (kernel + user) consumed by every thread so far.
    pub fn cpu_seconds() -> f64 {
        let mut creation = FileTime::default();
        let mut exit = FileTime::default();
        let mut kernel = FileTime::default();
        let mut user = FileTime::default();
        // SAFETY: all pointers refer to live, correctly sized FileTime values.
        let ok = unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut creation,
                &mut exit,
                &mut kernel,
                &mut user,
            )
        };
        if ok == 0 {
            return 0.;
        }
        fn seconds(time: &FileTime) -> f64 {
            (((time.high as u64) << 32) | time.low as u64) as f64 * 1e-7
        }
        seconds(&kernel) + seconds(&user)
    }
}

#[cfg(not(windows))]
mod win {
    pub fn raise_priority() {}
    pub fn cpu_seconds() -> f64 {
        0.
    }
}

/// The app's streaming chain, minus the audio devices and ring buffers.
struct Pipeline {
    input: Resampler,
    high_pass: HighPass,
    extractor: Extractor,
    output: Resampler,
    pending: Vec<f32>,
}

impl Pipeline {
    fn new(extractor: Extractor) -> Self {
        Self {
            input: Resampler::new(CAPTURE_RATE, RATE_U32),
            high_pass: HighPass::new(RATE as f32, 70.),
            extractor,
            output: Resampler::new(RATE_U32, CAPTURE_RATE),
            pending: Vec::with_capacity(CHUNK * 2),
        }
    }

    /// Feed one capture-rate block. Returns how many 40 ms blocks the model ran
    /// and how long those model calls took.
    fn push(&mut self, block: &[f32]) -> anyhow::Result<(usize, f64)> {
        let mut mono = self.input.push(block);
        self.high_pass.process(&mut mono);
        self.pending.extend_from_slice(&mono);
        let mut chunks = 0;
        let mut model_ms = 0.;
        while self.pending.len() >= CHUNK {
            let started = Instant::now();
            let filtered = self.extractor.push(&self.pending[..CHUNK])?;
            model_ms += started.elapsed().as_secs_f64() * 1000.;
            self.pending.drain(..CHUNK);
            let _ = self.output.push(&filtered);
            chunks += 1;
        }
        Ok((chunks, model_ms))
    }
}

struct Quality {
    improvement: f64,
    in_sdr: f64,
    out_sdr: f64,
    target_gain: f64,
    interference_suppression: f64,
    lag: isize,
}

fn measure_quality(
    extractor: &mut Extractor,
    target: &[f32],
    interference: &[f32],
    mixture: &[f32],
) -> anyhow::Result<Quality> {
    let mut output = Vec::with_capacity(mixture.len());
    for chunk in mixture.chunks(CHUNK) {
        output.extend(extractor.push(chunk)?);
    }
    let lag = best_lag(&output, target, MAX_LAG);
    let (estimate, start) = match lag {
        lag if lag >= 0 => (&output[lag as usize..], 0),
        lag => (&output[..], (-lag) as usize),
    };
    let length = estimate.len().min(target.len() - start);
    let estimate = &estimate[..length];
    let reference = &target[start..start + length];
    let noise = &interference[start..start + length];
    let mixture = &mixture[start..start + length];

    let in_sdr = si_sdr(mixture, reference);
    let out_sdr = si_sdr(estimate, reference);
    let (target_gain, interference_gain) = projection(estimate, reference, noise);
    Ok(Quality {
        improvement: out_sdr - in_sdr,
        in_sdr,
        out_sdr,
        target_gain: 20. * target_gain.abs().log10(),
        interference_suppression: -20. * interference_gain.abs().max(1e-12).log10(),
        lag,
    })
}

struct Performance {
    chunk_ms: f64,
    chunk_median_ms: f64,
    chunk_max_ms: f64,
    chunk_cpu_ms: f64,
    model_ms: f64,
    dsp_ms: f64,
    rtf: f64,
    startup_ms: f64,
}

fn measure_pipeline(
    extractor: Extractor,
    mixture: &[f32],
    startup_ms: f64,
) -> anyhow::Result<Performance> {
    // Each pass is three seconds of audio at the capture rate. The first pass is
    // warm-up; the rest are timed and the best one is reported. Operating system
    // scheduling only ever adds time, so the minimum is the most stable estimate
    // of what the pipeline can do on this machine.
    let passes = WARMUP_PASSES + TIMED_PASSES;
    let capture = build_capture(mixture, passes);
    let pass_len = capture.len() / passes;
    anyhow::ensure!(pass_len > 0, "The benchmark capture is empty");

    let mut pipeline = Pipeline::new(extractor);
    let mut measured = Vec::with_capacity(TIMED_PASSES);
    for (index, pass) in capture[..pass_len * passes].chunks(pass_len).enumerate() {
        let cpu_started = win::cpu_seconds();
        let started = Instant::now();
        let mut chunks = 0usize;
        let mut model_ms = 0.;
        for block in pass.chunks(CAPTURE_BLOCK) {
            let (ran, spent) = pipeline.push(block)?;
            chunks += ran;
            model_ms += spent;
        }
        let elapsed = started.elapsed().as_secs_f64() * 1000.;
        let cpu_ms = (win::cpu_seconds() - cpu_started) * 1000.;
        if index >= WARMUP_PASSES {
            anyhow::ensure!(chunks > 0, "A timed pass produced no 40 ms blocks");
            measured.push((
                elapsed / chunks as f64,
                cpu_ms / chunks as f64,
                model_ms / chunks as f64,
                (elapsed - model_ms).max(0.) / chunks as f64,
            ));
        }
    }
    anyhow::ensure!(
        measured.len() == TIMED_PASSES,
        "Expected {TIMED_PASSES} timed passes, got {}",
        measured.len()
    );
    measured.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let best = measured[0];
    Ok(Performance {
        chunk_ms: best.0,
        chunk_median_ms: (measured[TIMED_PASSES / 2 - 1].0 + measured[TIMED_PASSES / 2].0) / 2.,
        chunk_max_ms: measured[TIMED_PASSES - 1].0,
        chunk_cpu_ms: best.1,
        model_ms: best.2,
        dsp_ms: best.3,
        rtf: best.0 / (CHUNK as f64 / RATE as f64 * 1000.),
        startup_ms,
    })
}

/// Build `passes` back-to-back captures of the mixture at the device rate.
fn build_capture(mixture: &[f32], passes: usize) -> Vec<f32> {
    let mut upsampler = Resampler::new(RATE_U32, CAPTURE_RATE);
    let mut capture = Vec::with_capacity(passes * AUDIO_SECONDS * CAPTURE_RATE as usize);
    for _ in 0..passes {
        capture.extend(upsampler.push(mixture));
    }
    capture
}

/// Paired thread-pool comparison: run both extractors on the same audio in the
/// same process, alternating which one goes first, and report the best pass.
/// This removes the cross-run machine drift that makes separate runs useless
/// for one- or two-millisecond differences.
fn compare_threads(
    models: &Path,
    target: &[f32],
    mixture: &[f32],
    specification: &str,
) -> anyhow::Result<()> {
    let threads: Vec<usize> = specification
        .split(',')
        .map(|value| value.trim().parse::<usize>())
        .collect::<Result<_, _>>()
        .map_err(|_| anyhow::anyhow!("BENCH_COMPARE expects two comma-separated thread counts"))?;
    anyhow::ensure!(
        threads.len() == 2 && threads.iter().all(|count| *count > 0),
        "BENCH_COMPARE expects two positive thread counts"
    );
    let (first_threads, second_threads) = (threads[0], threads[1]);
    let first = Extractor::with_threads(models, target, Backend::Cpu, first_threads)?;
    let second = Extractor::with_threads(models, target, Backend::Cpu, second_threads)?;

    let passes = COMPARE_PASSES + 1;
    let capture = build_capture(mixture, passes);
    let pass_len = capture.len() / passes;
    anyhow::ensure!(pass_len > 0, "The comparison capture is empty");

    let mut first_pipeline = Pipeline::new(first);
    let mut second_pipeline = Pipeline::new(second);
    let mut first_ms = Vec::with_capacity(COMPARE_PASSES);
    let mut second_ms = Vec::with_capacity(COMPARE_PASSES);
    for (index, pass) in capture[..pass_len * passes].chunks(pass_len).enumerate() {
        // Alternate order so a warm cache and the scheduler bias do not always
        // favour the same side.
        let (a, b) = if index % 2 == 0 {
            (
                time_pipeline(&mut first_pipeline, pass)?,
                time_pipeline(&mut second_pipeline, pass)?,
            )
        } else {
            let b = time_pipeline(&mut second_pipeline, pass)?;
            let a = time_pipeline(&mut first_pipeline, pass)?;
            (a, b)
        };
        if index > 0 {
            first_ms.push(a);
            second_ms.push(b);
        }
    }
    anyhow::ensure!(
        first_ms.len() == COMPARE_PASSES,
        "Expected {COMPARE_PASSES} timed passes, got {}",
        first_ms.len()
    );
    let best = |values: &[f64]| values.iter().copied().fold(f64::INFINITY, f64::min);
    let first_ms = best(&first_ms);
    let second_ms = best(&second_ms);
    println!(
        "threads {first_threads} vs {second_threads}: {first_ms:.3} vs {second_ms:.3} ms per chunk"
    );
    println!("METRIC compare_a_ms={first_ms:.3}");
    println!("METRIC compare_b_ms={second_ms:.3}");
    println!("METRIC compare_ratio={:.4}", first_ms / second_ms);
    Ok(())
}

fn time_pipeline(pipeline: &mut Pipeline, pass: &[f32]) -> anyhow::Result<f64> {
    let started = Instant::now();
    let mut chunks = 0usize;
    for block in pass.chunks(CAPTURE_BLOCK) {
        chunks += pipeline.push(block)?.0;
    }
    anyhow::ensure!(chunks > 0, "The comparison pass produced no blocks");
    Ok(started.elapsed().as_secs_f64() * 1000. / chunks as f64)
}

/// Enrollment clip saved by the app: `data/voice.json` first, then the PCM16 WAV.
fn load_target(root: &Path) -> anyhow::Result<Vec<f32>> {
    let data = root.join("data");
    let json = data.join("voice.json");
    let samples = if json.is_file() {
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&json)?)?;
        let samples: Vec<f32> = serde_json::from_value(
            value
                .get("samples")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{} has no samples", json.display()))?,
        )?;
        samples
    } else {
        let wav = data.join("enrollment.wav");
        anyhow::ensure!(
            wav.is_file(),
            "No enrolled voice found. Record a voice in Beeswax first ({} or {}).",
            json.display(),
            wav.display()
        );
        read_wav(&wav)?
    };
    let sample = dsp::enrollment(&samples)?;
    anyhow::ensure!(
        sample.len() == AUDIO_SECONDS * RATE,
        "Enrollment clip has {} samples, expected {}",
        sample.len(),
        AUDIO_SECONDS * RATE
    );
    Ok(sample)
}

/// A 3 s competing-talker segment scaled so the mixture is 0 dB SNR.
fn load_interference(root: &Path, target: &[f32]) -> anyhow::Result<Vec<f32>> {
    let path = root
        .join("target")
        .join("autoresearch")
        .join("interference.wav");
    anyhow::ensure!(
        path.is_file(),
        "{} is missing; run scripts/bench-fixture.ps1",
        path.display()
    );
    let wav = read_wav(&path)?;
    let start = FIXTURE_OFFSET_SECONDS * RATE;
    anyhow::ensure!(
        wav.len() >= start + AUDIO_SECONDS * RATE,
        "{} is shorter than {} seconds",
        path.display(),
        FIXTURE_OFFSET_SECONDS + AUDIO_SECONDS
    );
    let mut segment = wav[start..start + AUDIO_SECONDS * RATE].to_vec();
    let gain = rms(target) / rms(&segment);
    for sample in &mut segment {
        *sample *= gain;
    }
    Ok(segment)
}

fn mix(target: &[f32], interference: &[f32]) -> Vec<f32> {
    target
        .iter()
        .zip(interference)
        .map(|(speech, noise)| speech + noise)
        .collect()
}

/// PCM16, mono, 16 kHz WAV reader. Ignores unknown chunks (e.g. LIST).
fn read_wav(path: &Path) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    anyhow::ensure!(
        bytes.len() >= 44 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "{} is not a RIFF/WAVE file",
        path.display()
    );
    let mut format = 0u16;
    let mut channels = 0u16;
    let mut rate = 0u32;
    let mut bits = 0u16;
    let mut data: Option<&[u8]> = None;
    let mut position = 12;
    while position + 8 <= bytes.len() {
        let id = &bytes[position..position + 4];
        let size =
            u32::from_le_bytes(bytes[position + 4..position + 8].try_into().unwrap()) as usize;
        let start = position + 8;
        let end = start.saturating_add(size).min(bytes.len());
        match id {
            b"fmt " if end >= start + 16 => {
                format = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
                channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap());
                rate = u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
                bits = u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap());
            }
            b"data" => data = Some(&bytes[start..end]),
            _ => {}
        }
        position = end + (size & 1);
    }
    anyhow::ensure!(
        format == 1 && channels == 1 && rate == RATE_U32 && bits == 16,
        "{} must be 16 kHz mono PCM16",
        path.display()
    );
    let data = data.ok_or_else(|| anyhow::anyhow!("{} has no data chunk", path.display()))?;
    Ok(data
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.)
        .collect())
}

/// Lag in `[-max, max]` that maximises normalised correlation with `reference`.
fn best_lag(estimate: &[f32], reference: &[f32], max: isize) -> isize {
    let mut best = (0, f64::NEG_INFINITY);
    for lag in -max..=max {
        let (estimate, start) = if lag >= 0 {
            (&estimate[lag as usize..], 0)
        } else {
            (estimate, (-lag) as usize)
        };
        let length = estimate.len().min(reference.len().saturating_sub(start));
        if length < RATE {
            continue;
        }
        let estimate = &estimate[..length];
        let reference = &reference[start..start + length];
        let mut dot = 0f64;
        let mut estimate_energy = 0f64;
        let mut reference_energy = 0f64;
        for (&a, &b) in estimate.iter().zip(reference) {
            dot += a as f64 * b as f64;
            estimate_energy += (a as f64).powi(2);
            reference_energy += (b as f64).powi(2);
        }
        let denominator = (estimate_energy * reference_energy).sqrt();
        let score = if denominator > 0. {
            dot / denominator
        } else {
            f64::NEG_INFINITY
        };
        if score > best.1 {
            best = (lag, score);
        }
    }
    best.0
}

fn si_sdr(estimate: &[f32], reference: &[f32]) -> f64 {
    let (mut estimate, mut reference) = (estimate.to_vec(), reference.to_vec());
    remove_mean(&mut estimate);
    remove_mean(&mut reference);
    let reference_energy: f64 = reference.iter().map(|x| (*x as f64).powi(2)).sum();
    let dot: f64 = estimate
        .iter()
        .zip(&reference)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let alpha = if reference_energy > 0. {
        dot / reference_energy
    } else {
        0.
    };
    let signal: f64 = reference.iter().map(|x| (alpha * *x as f64).powi(2)).sum();
    let error: f64 = estimate
        .iter()
        .zip(&reference)
        .map(|(a, b)| (*a as f64 - alpha * *b as f64).powi(2))
        .sum();
    10. * (signal / error.max(1e-18)).log10()
}

fn remove_mean(samples: &mut [f32]) {
    let mean = samples.iter().map(|x| *x as f64).sum::<f64>() / samples.len() as f64;
    for sample in samples {
        *sample -= mean as f32;
    }
}

/// Least-squares gains for [reference, interference]; returns (target, interference).
fn projection(estimate: &[f32], reference: &[f32], interference: &[f32]) -> (f64, f64) {
    let (mut estimate, mut reference, mut interference) =
        (estimate.to_vec(), reference.to_vec(), interference.to_vec());
    remove_mean(&mut estimate);
    remove_mean(&mut reference);
    remove_mean(&mut interference);
    let mut rr = 0.;
    let mut ri = 0.;
    let mut ii = 0.;
    let mut er = 0.;
    let mut ei = 0.;
    for ((&e, &r), &i) in estimate.iter().zip(&reference).zip(&interference) {
        let (e, r, i) = (e as f64, r as f64, i as f64);
        rr += r * r;
        ri += r * i;
        ii += i * i;
        er += e * r;
        ei += e * i;
    }
    let determinant = rr * ii - ri * ri;
    if determinant.abs() < 1e-12 {
        return (0., 0.);
    }
    (
        (er * ii - ei * ri) / determinant,
        (ei * rr - er * ri) / determinant,
    )
}
