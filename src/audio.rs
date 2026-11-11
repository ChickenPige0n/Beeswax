use crate::{
    dsp::{self, CHUNK, RATE, Resampler},
    model::Extractor,
};
use anyhow::{Context, Result, bail, ensure};
use cpal::{
    FromSample, Sample, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use ringbuf::{HeapCons, HeapProd, HeapRb, traits::*};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Debug)]
pub struct Device {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}
pub fn devices(input: bool) -> Result<Vec<Device>> {
    let host = cpal::default_host();
    let default = if input {
        host.default_input_device()
    } else {
        host.default_output_device()
    }
    .and_then(|d| d.id().ok())
    .map(|id| id.to_string());
    let list = if input {
        host.input_devices()?
    } else {
        host.output_devices()?
    };
    Ok(list
        .filter_map(|d| {
            let id = d.id().ok()?.to_string();
            let description = d.description().ok()?;
            let detail = description.extended().join(" · ");
            let name = if detail.is_empty() {
                description.name().to_owned()
            } else {
                detail
            };
            Some(Device {
                is_default: default.as_ref() == Some(&id),
                id,
                name,
            })
        })
        .collect())
}

#[derive(Clone, Default)]
pub struct Status {
    pub message: String,
    pub active: bool,
    pub recording: bool,
    pub monitoring: bool,
    pub previewing: bool,
    pub level: f32,
    pub output_level: f32,
    pub gated: bool,
    pub progress: f32,
    pub inference_ms: f32,
    /// Device sample rate of the selected microphone, in Hz.
    pub capture_rate: u32,
    /// Configured capture period in milliseconds.
    pub capture_ms: f32,
    /// Output reserve plus playback device period in milliseconds.
    pub buffer_ms: f32,
    pub underruns: usize,
    /// Samples discarded to keep the pipeline close to real time.
    pub dropped: usize,
    /// Backend that is actually running the model ("CPU" or "CUDA").
    pub engine: String,
    /// Precision of the loaded separation model ("int8", "fp16" or "fp32").
    pub model: String,
    /// Explanation when the requested backend could not be used.
    pub note: String,
    pub enrollment: Option<Vec<f32>>,
}
/// Keeps the loaded ONNX sessions between filtering jobs. A restart only has to
/// clear the streaming state instead of parsing the model files again.
pub struct ModelCache {
    slot: Mutex<Option<CachedModel>>,
}

struct CachedModel {
    folder: PathBuf,
    sample: Vec<f32>,
    backend: crate::Backend,
    extractor: Extractor,
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelCache {
    pub fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// Take a ready extractor for this enrollment, or load one on a miss.
    pub fn acquire(
        &self,
        folder: &Path,
        sample: &[f32],
        backend: crate::Backend,
    ) -> Result<Extractor> {
        let cached = {
            let mut slot = self.slot.lock().unwrap();
            let matches = slot.as_ref().is_some_and(|cached| {
                cached.folder.as_path() == folder
                    && cached.backend == backend
                    && cached.sample.as_slice() == sample
            });
            if matches {
                slot.take()
            } else {
                // A different enrollment, backend, or model folder makes the
                // cached sessions useless; drop them now rather than keep the
                // memory until the next release.
                *slot = None;
                None
            }
        };
        match cached {
            Some(mut cached) => {
                cached.extractor.reset()?;
                Ok(cached.extractor)
            }
            None => Extractor::new(folder, sample, backend),
        }
    }

    /// Put a healthy CPU extractor back for the next start. A CUDA session owns
    /// GPU memory while idle, so it is released instead of cached.
    pub fn release(
        &self,
        folder: &Path,
        sample: &[f32],
        backend: crate::Backend,
        extractor: Extractor,
    ) {
        if extractor.engine() != crate::Backend::Cpu {
            return;
        }
        *self.slot.lock().unwrap() = Some(CachedModel {
            folder: folder.to_path_buf(),
            sample: sample.to_vec(),
            backend,
            extractor,
        });
    }

    /// Drop the cached sessions (voice forgotten, backend changed).
    pub fn clear(&self) {
        *self.slot.lock().unwrap() = None;
    }
}

pub struct Job {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}
impl Job {
    pub fn cancel(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
    pub fn finished(&self) -> bool {
        self.handle.as_ref().is_none_or(|h| h.is_finished())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        input: String,
        output: Option<String>,
        model_dir: PathBuf,
        sample: Option<Vec<f32>>,
        state: Arc<Mutex<Status>>,
        monitor: bool,
        backend: crate::Backend,
        cache: Arc<ModelCache>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let handle = thread::spawn(move || {
            let recording = sample.is_none();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(
                    input, output, model_dir, sample, &flag, &state, monitor, backend, &cache,
                )
            }))
            .unwrap_or_else(|_| {
                Err(anyhow::anyhow!(
                    "Audio worker failed. Output stopped; check the runtime and restart."
                ))
            });
            let mut s = state.lock().unwrap();
            s.active = false;
            s.recording = false;
            s.monitoring = false;
            s.level = 0.;
            s.output_level = 0.;
            s.gated = false;
            if let Err(e) = result {
                s.message = audio_error(&format!("{e:#}"));
            } else if flag.load(Ordering::Relaxed) {
                s.message = "Stopped. Output is silent.".into();
            } else if recording {
                s.message = "Voice enrolled. Preview the clip before filtering.".into();
            }
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }
    pub fn preview(output: String, sample: Vec<f32>, state: Arc<Mutex<Status>>) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let handle = thread::spawn(move || {
            let result = play_preview(&output, &sample, &flag);
            let mut s = state.lock().unwrap();
            s.active = false;
            s.previewing = false;
            s.message = match result {
                Ok(()) if flag.load(Ordering::Relaxed) => "Preview stopped.".into(),
                Ok(()) => "Preview finished.".into(),
                Err(e) => format!("Could not play enrollment: {e:#}"),
            };
        });
        Self {
            stop,
            handle: Some(handle),
        }
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        self.cancel();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn input_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut producer: HeapProd<f32>,
    overruns: Arc<std::sync::atomic::AtomicUsize>,
    fault: Arc<AtomicBool>,
) -> Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels as usize;
    let err = fault.clone();
    Ok(device.build_input_stream(
        config,
        move |data: &[T], _| {
            let mut lost = 0usize;
            for frame in data.chunks_exact(channels) {
                let mono =
                    frame.iter().map(|x| f32::from_sample(*x)).sum::<f32>() / channels as f32;
                if !mono.is_finite() {
                    continue;
                }
                if producer.try_push(mono).is_err() {
                    // Never kill the stream for a full queue; count it and keep going.
                    lost += 1;
                }
            }
            if lost > 0 {
                overruns.fetch_add(lost, Ordering::Relaxed);
            }
        },
        move |_| {
            err.store(true, Ordering::Relaxed);
        },
        None,
    )?)
}
fn output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut consumer: HeapCons<f32>,
    stop: Arc<AtomicBool>,
    fault: Arc<AtomicBool>,
    misses: Arc<std::sync::atomic::AtomicUsize>,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    Ok(device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            let muted = stop.load(Ordering::Relaxed);
            let mut missing = false;
            for frame in data.chunks_exact_mut(channels) {
                let value = if muted {
                    0.
                } else {
                    consumer.try_pop().unwrap_or_else(|| {
                        missing = true;
                        0.
                    })
                };
                frame.fill(T::from_sample(value.clamp(-1., 1.)));
            }
            if missing {
                misses.fetch_add(1, Ordering::Relaxed);
            }
        },
        move |_| {
            fault.store(true, Ordering::Relaxed);
        },
        None,
    )?)
}

/// Ask WASAPI for the shortest period the device advertises. The host default is kept
/// when the driver does not publish a buffer range.
fn low_latency_config(supported: &cpal::SupportedStreamConfig) -> cpal::StreamConfig {
    let mut config = supported.config();
    if let cpal::SupportedBufferSize::Range { min, .. } = supported.buffer_size() {
        config.buffer_size = cpal::BufferSize::Fixed((*min).max(32));
    }
    config
}

fn buffer_ms(config: &cpal::StreamConfig, rate: u32) -> f32 {
    match config.buffer_size {
        cpal::BufferSize::Fixed(frames) => frames as f32 / rate as f32 * 1000.,
        cpal::BufferSize::Default => 10.,
    }
}

fn build_input(
    device: &cpal::Device,
    format: cpal::SampleFormat,
    config: &cpal::StreamConfig,
    capacity: usize,
    overruns: &Arc<std::sync::atomic::AtomicUsize>,
    fault: &Arc<AtomicBool>,
) -> Result<(cpal::Stream, HeapCons<f32>)> {
    let (producer, consumer) = HeapRb::<f32>::new(capacity).split();
    let stream = match format {
        cpal::SampleFormat::F32 => {
            input_stream::<f32>(device, config, producer, overruns.clone(), fault.clone())?
        }
        cpal::SampleFormat::I16 => {
            input_stream::<i16>(device, config, producer, overruns.clone(), fault.clone())?
        }
        cpal::SampleFormat::I32 => {
            input_stream::<i32>(device, config, producer, overruns.clone(), fault.clone())?
        }
        cpal::SampleFormat::U16 => {
            input_stream::<u16>(device, config, producer, overruns.clone(), fault.clone())?
        }
        other => bail!("Unsupported microphone format: {other:?}"),
    };
    Ok((stream, consumer))
}

fn build_output(
    device: &cpal::Device,
    format: cpal::SampleFormat,
    config: &cpal::StreamConfig,
    capacity: usize,
    stop: Arc<AtomicBool>,
    fault: Arc<AtomicBool>,
    misses: Arc<std::sync::atomic::AtomicUsize>,
) -> Result<(cpal::Stream, HeapProd<f32>)> {
    let (producer, consumer) = HeapRb::<f32>::new(capacity).split();
    let stream = match format {
        cpal::SampleFormat::F32 => {
            output_stream::<f32>(device, config, consumer, stop, fault, misses)?
        }
        cpal::SampleFormat::I16 => {
            output_stream::<i16>(device, config, consumer, stop, fault, misses)?
        }
        cpal::SampleFormat::I32 => {
            output_stream::<i32>(device, config, consumer, stop, fault, misses)?
        }
        cpal::SampleFormat::U16 => {
            output_stream::<u16>(device, config, consumer, stop, fault, misses)?
        }
        other => bail!("Unsupported output format: {other:?}"),
    };
    Ok((stream, producer))
}

#[allow(clippy::too_many_arguments)]
fn run(
    input: String,
    output: Option<String>,
    model_dir: PathBuf,
    sample: Option<Vec<f32>>,
    stop: &Arc<AtomicBool>,
    state: &Arc<Mutex<Status>>,
    monitor: bool,
    backend: crate::Backend,
    cache: &ModelCache,
) -> Result<()> {
    let recording = sample.is_none();
    let mut model = match sample.as_ref() {
        Some(sample) => Some(cache.acquire(&model_dir, sample, backend)?),
        None => None,
    };
    let result = run_stream(input, output, recording, &mut model, stop, state, monitor);
    if result.is_ok()
        && let (Some(sample), Some(extractor)) = (sample.as_ref(), model.take())
    {
        cache.release(&model_dir, sample, backend, extractor);
    }
    result
}

fn run_stream(
    input: String,
    output: Option<String>,
    recording: bool,
    model: &mut Option<Extractor>,
    stop: &Arc<AtomicBool>,
    state: &Arc<Mutex<Status>>,
    monitor: bool,
) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .device_by_id(&input.parse()?)
        .context("Microphone disconnected. Refresh devices.")?;
    if let Some(model) = model.as_ref() {
        let mut status = state.lock().unwrap();
        status.engine = model.engine().label().to_owned();
        status.model = model.model_label().to_owned();
        if let Some(note) = model.note() {
            status.note = note.to_owned();
        }
    }
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    let supported = device
        .default_input_config()
        .context("Cannot open the selected microphone")?;
    let rate = supported.sample_rate();
    let fault = Arc::new(AtomicBool::new(false));
    let overruns = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    // Ask for the shortest capture period the driver reports, and keep the host default
    // if the device refuses it.
    let mut input_config = low_latency_config(&supported);
    let input_format = supported.sample_format();
    let (input_stream, mut consumer) = match build_input(
        &device,
        input_format,
        &input_config,
        rate as usize * 3,
        &overruns,
        &fault,
    ) {
        Ok(pair) => pair,
        Err(_) => {
            input_config.buffer_size = cpal::BufferSize::Default;
            build_input(
                &device,
                input_format,
                &input_config,
                rate as usize * 3,
                &overruns,
                &fault,
            )?
        }
    };
    let capture_ms = buffer_ms(&input_config, rate);
    {
        let mut status = state.lock().unwrap();
        status.capture_ms = capture_ms;
        status.capture_rate = rate;
    }

    let mut output_stream_handle = None;
    let mut output_producer = None;
    let mut output_resampler = None;
    let mut output_rate = 0u32;
    let mut output_device_ms = 0f32;
    // Start with one 40 ms block of slack plus jitter room. Underruns raise it.
    let mut reserve_ms = 60u32;
    let misses = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    if !recording {
        let output_id = output.context("Choose an output device.")?;
        let device = host
            .device_by_id(&output_id.parse()?)
            .context("Output device disconnected. Refresh devices.")?;
        // Any playback endpoint works: CABLE Input feeds a call app, headphones or
        // speakers let the separated voice be monitored directly.
        let supported = device.default_output_config()?;
        let out_rate = supported.sample_rate();
        let out_format = supported.sample_format();
        let mut config = low_latency_config(&supported);
        let (stream, mut producer) = match build_output(
            &device,
            out_format,
            &config,
            out_rate as usize / 4,
            stop.clone(),
            fault.clone(),
            misses.clone(),
        ) {
            Ok(pair) => pair,
            Err(_) => {
                config.buffer_size = cpal::BufferSize::Default;
                build_output(
                    &device,
                    out_format,
                    &config,
                    out_rate as usize / 4,
                    stop.clone(),
                    fault.clone(),
                    misses.clone(),
                )?
            }
        };
        let pre_roll = out_rate as usize * reserve_ms as usize / 1000;
        producer.push_slice(&vec![0.; pre_roll]);
        output_device_ms = buffer_ms(&config, out_rate);
        output_rate = out_rate;
        output_stream_handle = Some(stream);
        output_producer = Some(producer);
        output_resampler = Some(Resampler::new(RATE as u32, out_rate));
    }
    let mut resampler = Resampler::new(rate, RATE as u32);
    let mut high_pass = dsp::HighPass::new(RATE as f32, 70.);
    let mut gate = dsp::Gate::new(true);
    let mut pending = Vec::with_capacity(CHUNK * 2);
    let mut mono = Vec::with_capacity(2048);
    let mut filtered = Vec::with_capacity(CHUNK);
    let mut output = Vec::with_capacity(output_rate as usize / 20);
    let mut captured = Vec::new();
    let mut buf = vec![0.; 4096];
    input_stream.play()?;
    if let Some(stream) = &output_stream_handle {
        stream.play()?;
    }
    let mut last_input = Instant::now();
    state.lock().unwrap().message = if monitor {
        "Microphone test is active. Speak and watch the input level; nothing is recorded or played."
    } else if recording {
        "Speak alone, naturally, for six seconds."
    } else {
        "Filtering your voice to the selected output."
    }
    .into();
    let mut dropped = 0usize;
    let mut failures = 0u32;
    let mut last_misses = 0usize;
    while !stop.load(Ordering::Relaxed) {
        ensure!(
            !fault.load(Ordering::Relaxed),
            "Audio device failed. Output stopped; refresh devices and retry."
        );
        // Stay close to real time: when inference or the system falls behind, discard the
        // oldest capture instead of letting an ever-growing backlog add latency.
        let backlog = consumer.occupied_len();
        if backlog > rate as usize / 4 {
            let mut left = backlog.saturating_sub(rate as usize / 20);
            while left > 0 {
                let take = left.min(buf.len());
                let n = consumer.pop_slice(&mut buf[..take]);
                if n == 0 {
                    break;
                }
                left -= n;
                dropped += n;
            }
        }
        let n = consumer.pop_slice(&mut buf);
        if n == 0 {
            ensure!(
                last_input.elapsed() < Duration::from_secs(5),
                "No microphone samples received for five seconds."
            );
            thread::sleep(Duration::from_millis(2));
            continue;
        }
        last_input = Instant::now();
        resampler.push_into(&buf[..n], &mut mono);
        high_pass.process(&mut mono);
        if !mono.is_empty() {
            let mut s = state.lock().unwrap();
            s.level = dsp::rms(&mono).max(s.level * 0.93);
        }
        if monitor {
            continue;
        }
        if recording {
            captured.extend_from_slice(&mono);
            state.lock().unwrap().progress = captured.len() as f32 / (6 * RATE) as f32;
            if captured.len() >= 6 * RATE {
                let sample = dsp::enrollment(&captured)?;
                state.lock().unwrap().enrollment = Some(sample);
                return Ok(());
            }
        } else {
            pending.extend_from_slice(&mono);
            while pending.len() >= CHUNK {
                if stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let start = Instant::now();
                let mut skipped = false;
                match gate.next(&pending[..CHUNK]) {
                    // The hangover already fed silence to the model; skipping keeps the
                    // stream in time while emitting exact silence through the resampler.
                    dsp::GateAction::Skip => {
                        skipped = true;
                        filtered.clear();
                        filtered.resize(CHUNK, 0.);
                    }
                    action => {
                        let extractor = model.as_mut().unwrap();
                        if action == dsp::GateAction::ResumeState
                            && let Err(error) = extractor.reset_state()
                        {
                            state.lock().unwrap().message = format!(
                                "Could not clear the stale voice state; continuing: {error:#}"
                            );
                        }
                        if let Err(error) = extractor.push_into(&pending[..CHUNK], &mut filtered) {
                            // A single bad block should not end the call: drop it, keep the stream
                            // alive, and only stop when the model keeps failing.
                            failures += 1;
                            if failures >= 3 {
                                return Err(
                                    error.context("The voice model stopped returning audio")
                                );
                            }
                            state.lock().unwrap().message =
                                format!("Voice model hiccup; recovering: {error:#}");
                            filtered.clear();
                            filtered.resize(CHUNK, 0.);
                        } else {
                            failures = 0;
                        }
                    }
                }
                let remaining = pending.len() - CHUNK;
                pending.copy_within(CHUNK.., 0);
                pending.truncate(remaining);
                let duration = start.elapsed().as_secs_f32() * 1000.;
                output_resampler
                    .as_mut()
                    .unwrap()
                    .push_into(&filtered, &mut output);
                let producer = output_producer.as_mut().unwrap();
                let pushed = producer.push_slice(&output);
                let short = pushed < output.len();
                if short {
                    dropped += output.len() - pushed;
                }
                let misses_now = misses.load(Ordering::Relaxed);
                if misses_now > last_misses {
                    last_misses = misses_now;
                    // Playback ran dry: add slack so the next blocks keep the cable fed.
                    if reserve_ms < 160 {
                        let extra = output_rate as usize * 20 / 1000;
                        if producer.vacant_len() >= extra {
                            producer.push_slice(&vec![0.; extra]);
                            reserve_ms += 20;
                        }
                    }
                }
                let out_rms = if skipped { 0. } else { dsp::rms(&filtered) };
                let mut s = state.lock().unwrap();
                if !skipped {
                    s.inference_ms = duration;
                }
                s.gated = skipped;
                s.output_level = out_rms.max(s.output_level * 0.88);
                s.buffer_ms = reserve_ms as f32 + output_device_ms;
                s.underruns = misses_now;
                s.dropped = dropped + overruns.load(Ordering::Relaxed);
                let engine = s.engine.clone();
                if duration > 40. {
                    s.message = format!(
                        "{engine} inference needs {duration:.0} ms per 40 ms of audio. Filtering continues, but audio is dropped to stay in time."
                    );
                } else if short {
                    s.message =
                        "The playback queue filled up; some audio was dropped. Filtering continues.".into();
                }
            }
        }
    }
    Ok(())
}

fn audio_error(error: &str) -> String {
    if error.to_lowercase().contains("0x80070005") {
        format!(
            "Windows denied microphone access. Check Settings > Privacy & security > Microphone: enable microphone access and desktop app access. If already enabled, launch Beeswax directly outside the restricted session. Details: {error}"
        )
    } else {
        error.to_owned()
    }
}

fn play_preview(output: &str, sample: &[f32], stop: &Arc<AtomicBool>) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .device_by_id(&output.parse()?)
        .context("Preview device disconnected")?;
    ensure!(
        !device
            .description()?
            .name()
            .to_lowercase()
            .contains("cable"),
        "Choose headphones or speakers for preview."
    );
    let cfg = device.default_output_config()?;
    let rate = cfg.sample_rate();
    let mut resampler = Resampler::new(RATE as u32, rate);
    let mut padded = sample.to_vec();
    padded.extend([0.; 64]);
    let audio = resampler.push(&padded);
    let duration = Duration::from_secs_f64(audio.len() as f64 / rate as f64);
    let (mut p, c) = HeapRb::new(audio.len() + 1).split();
    p.push_slice(&audio);
    let fault = Arc::new(AtomicBool::new(false));
    let misses = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let config = cfg.config();
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => {
            output_stream::<f32>(&device, &config, c, stop.clone(), fault.clone(), misses)?
        }
        cpal::SampleFormat::I16 => {
            output_stream::<i16>(&device, &config, c, stop.clone(), fault.clone(), misses)?
        }
        cpal::SampleFormat::I32 => {
            output_stream::<i32>(&device, &config, c, stop.clone(), fault.clone(), misses)?
        }
        cpal::SampleFormat::U16 => {
            output_stream::<u16>(&device, &config, c, stop.clone(), fault.clone(), misses)?
        }
        other => bail!("Unsupported preview format: {other:?}"),
    };
    stream.play()?;
    let start = Instant::now();
    while start.elapsed() < duration + Duration::from_millis(200) && !stop.load(Ordering::Relaxed) {
        ensure!(!fault.load(Ordering::Relaxed), "Preview device failed");
        thread::sleep(Duration::from_millis(10));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn localized_access_denied_has_actionable_guidance() {
        let message = audio_error("拒绝访问。 (0x80070005)");
        assert!(message.contains("desktop app access"));
        assert!(message.contains("0x80070005"));
        assert_eq!(audio_error("device unplugged"), "device unplugged");
    }
}
