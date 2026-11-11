mod sections;
mod select;
pub mod theme;

use crate::{model_download, profile, updates};
use beeswax::{
    Backend,
    audio::{self, Device, Job, Status},
    model,
};
use gpui::{Context, FocusHandle, MouseButton, Window, div, prelude::*, px, rgb};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use theme::{BG, INK, waveform_peaks};

/// The streaming model keeps about three 128-sample hops of lookahead (384 / 16000).
pub(super) const MODEL_LOOKAHEAD_MS: f32 = 24.;
/// Average wait while a 40 ms block fills.
pub(super) const CHUNK_WAIT_MS: f32 = 20.;

pub(super) const SELECT_MIC: u8 = 0;
pub(super) const SELECT_OUTPUT: u8 = 1;
pub(super) const SELECT_SPEAKER: u8 = 2;

pub struct Beeswax {
    pub(super) inputs: Vec<Device>,
    pub(super) outputs: Vec<Device>,
    pub(super) input: usize,
    pub(super) speakers: Vec<Device>,
    pub(super) speaker: usize,
    pub(super) output: usize,
    pub(super) open_select: Option<u8>,
    pub(super) select_focus: [FocusHandle; 3],
    pub(super) status: Arc<Mutex<Status>>,
    pub(super) job: Option<Job>,
    pub(super) cache: Arc<audio::ModelCache>,
    pub(super) voice: Option<Vec<f32>>,
    pub(super) waveform: Vec<f32>,
    pub(super) show_preview_picker: bool,
    pub(super) model_dir: PathBuf,
    pub(super) data_dir: PathBuf,
    pub(super) runtime_error: Option<String>,
    pub(super) backend: Backend,
    pub(super) cuda: bool,
    pub(super) providers: String,
    pub(super) models: String,
    pub(super) model_setup: Arc<Mutex<model_download::State>>,
    pub(super) update_state: Arc<Mutex<updates::State>>,
}

impl Beeswax {
    pub fn new(
        model_dir: PathBuf,
        data_dir: PathBuf,
        runtime_error: Option<String>,
        update_repository: Option<String>,
        cx: &mut Context<Self>,
    ) -> Self {
        let voice = profile::load(&data_dir);
        let message = match &voice {
            Ok(_) => "Voice profile loaded. Ready to filter.".to_owned(),
            Err(_) if !data_dir.join("voice.json").exists() => {
                "Record a 6-second voice sample to get started.".to_owned()
            }
            Err(e) => format!("Saved enrollment could not be loaded: {e}. Record again."),
        };
        let cuda = runtime_error.is_none() && model::cuda_available();
        let models = format_models(&model_dir);
        let providers = if runtime_error.is_none() {
            let names = model::providers();
            if names.is_empty() {
                "none".to_owned()
            } else {
                names.join(", ")
            }
        } else {
            "unavailable".to_owned()
        };
        let backend = match profile::load_backend(&data_dir) {
            Some(Backend::Cuda) if cuda => Backend::Cuda,
            _ => Backend::Cpu,
        };
        let voice_opt = voice.ok();
        let waveform = voice_opt.as_deref().map(waveform_peaks).unwrap_or_default();
        let model_setup = Arc::new(Mutex::new(model_download::State::for_directory(&model_dir)));
        let update_state = Arc::new(Mutex::new(updates::State::default()));
        let mut app = Self {
            inputs: vec![],
            outputs: vec![],
            input: 0,
            speakers: vec![],
            speaker: 0,
            output: 0,
            open_select: None,
            select_focus: std::array::from_fn(|_| cx.focus_handle()),
            status: Arc::new(Mutex::new(Status {
                message: runtime_error.clone().unwrap_or(message),
                ..Default::default()
            })),
            job: None,
            cache: Arc::new(audio::ModelCache::new()),
            voice: voice_opt,
            waveform,
            show_preview_picker: false,
            model_dir,
            data_dir,
            runtime_error,
            backend,
            cuda,
            providers,
            models,
            model_setup,
            update_state,
        };
        app.refresh();
        model_download::start(app.model_dir.clone(), app.model_setup.clone());
        updates::start(update_repository, app.update_state.clone());

        cx.spawn(async move |view, cx| {
            let mut last_snapshot = PollSnapshot::default();
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(60))
                    .await;
                if view
                    .update(cx, |this, cx| {
                        let mut changed = false;
                        if this.models == "none" && this.model_setup.lock().unwrap().ready {
                            let updated = format_models(&this.model_dir);
                            if updated != this.models {
                                this.models = updated;
                                changed = true;
                            }
                        }
                        if this.job.as_ref().is_some_and(Job::finished) {
                            this.job.take();
                            let sample = this.status.lock().unwrap().enrollment.take();
                            if let Some(sample) = sample {
                                if let Err(e) = profile::save(&this.data_dir, &sample) {
                                    this.status.lock().unwrap().message = format!(
                                        "Enrolled for this session, but could not save: {e}"
                                    );
                                }
                                this.waveform = waveform_peaks(&sample);
                                this.voice = Some(sample);
                            }
                            changed = true;
                        }
                        let current = PollSnapshot::capture(this);
                        if changed || current != last_snapshot {
                            last_snapshot = current;
                            cx.notify();
                        }
                    })
                    .is_err()
                {
                    break;
                }
            }
        })
        .detach();
        app
    }

    pub(super) fn refresh(&mut self) {
        if self.job.is_some() {
            return;
        }
        let old_input = self.inputs.get(self.input).map(|d| d.id.clone());
        let old_output = self.outputs.get(self.output).map(|d| d.id.clone());
        let old_speaker = self.speakers.get(self.speaker).map(|d| d.id.clone());
        match (audio::devices(true), audio::devices(false)) {
            (Ok(inputs), Ok(outputs)) => {
                self.inputs = inputs
                    .into_iter()
                    .filter(|d| !d.name.to_lowercase().contains("cable"))
                    .collect();
                self.speakers = outputs
                    .iter()
                    .filter(|d| !d.name.to_lowercase().contains("cable"))
                    .cloned()
                    .collect();
                self.speaker = self
                    .speakers
                    .iter()
                    .position(|d| Some(&d.id) == old_speaker.as_ref())
                    .unwrap_or_else(|| {
                        self.speakers.iter().position(|d| d.is_default).unwrap_or(0)
                    });
                self.outputs = outputs;
                self.input = self
                    .inputs
                    .iter()
                    .position(|d| Some(&d.id) == old_input.as_ref())
                    .unwrap_or_else(|| self.inputs.iter().position(|d| d.is_default).unwrap_or(0));
                self.output = self
                    .outputs
                    .iter()
                    .position(|d| Some(&d.id) == old_output.as_ref())
                    .or_else(|| {
                        self.outputs
                            .iter()
                            .position(|d| d.name.to_lowercase().contains("cable"))
                    })
                    .or_else(|| self.outputs.iter().position(|d| d.is_default))
                    .unwrap_or(0);
            }
            (Err(e), _) | (_, Err(e)) => {
                self.status.lock().unwrap().message = format!("Cannot enumerate audio devices: {e}")
            }
        }
    }

    pub(super) fn begin(&mut self, record: bool, monitor: bool) {
        if self.job.is_some() {
            return;
        }
        if !record && !self.model_setup.lock().unwrap().ready {
            self.status.lock().unwrap().message = "Required voice models are not ready yet.".into();
            return;
        }
        let Some(input) = self.inputs.get(self.input) else {
            self.status.lock().unwrap().message =
                "Connect a physical microphone, then refresh devices.".into();
            return;
        };
        if !record {
            if let Some(error) = &self.runtime_error {
                self.status.lock().unwrap().message = error.clone();
                return;
            }
            if self.voice.is_none() {
                self.status.lock().unwrap().message = "Record your voice first.".into();
                return;
            }
            if self.outputs.is_empty() {
                self.status.lock().unwrap().message =
                    "Connect headphones, speakers or VB-CABLE, then refresh devices.".into();
                return;
            }
        }
        *self.status.lock().unwrap() = Status {
            active: true,
            recording: record && !monitor,
            monitoring: monitor,
            message: if monitor {
                "Opening microphone for level check…"
            } else if record {
                "Opening microphone for enrollment…"
            } else {
                "Preparing voice separator…"
            }
            .into(),
            ..Default::default()
        };
        self.job = Some(Job::start(
            input.id.clone(),
            self.outputs.get(self.output).map(|d| d.id.clone()),
            self.model_dir.clone(),
            if record { None } else { self.voice.clone() },
            self.status.clone(),
            monitor,
            self.backend,
            self.cache.clone(),
        ));
        self.open_select = None;
    }

    pub(super) fn retry_model_download(&mut self) {
        model_download::start(self.model_dir.clone(), self.model_setup.clone());
    }

    pub(super) fn preview(&mut self) {
        if self.job.is_some() {
            return;
        }
        let Some(sample) = self.voice.clone() else {
            return;
        };
        let Some(device) = self.speakers.get(self.speaker) else {
            self.status.lock().unwrap().message =
                "Connect headphones or speakers, then refresh devices.".into();
            return;
        };
        *self.status.lock().unwrap() = Status {
            active: true,
            previewing: true,
            message: format!("Playing enrollment clip on {}", device.name),
            ..Default::default()
        };
        self.open_select = None;
        self.job = Some(Job::preview(device.id.clone(), sample, self.status.clone()));
    }

    pub(super) fn forget_voice(&mut self) {
        if self.job.is_some() {
            return;
        }
        match profile::forget(&self.data_dir) {
            Ok(()) => {
                self.voice = None;
                self.waveform.clear();
                self.show_preview_picker = false;
                self.cache.clear();
                self.status.lock().unwrap().message = "Voice profile removed.".into();
            }
            Err(e) => {
                self.status.lock().unwrap().message = format!("Could not delete profile: {e}");
            }
        }
    }

    pub(super) fn stop(&mut self) {
        if let Some(job) = &self.job {
            job.cancel();
            self.status.lock().unwrap().message = "Stopping…".into();
        }
    }

    pub(super) fn output_is_cable(&self) -> bool {
        self.outputs
            .get(self.output)
            .is_some_and(|device| device.name.to_lowercase().contains("cable"))
    }

    pub(super) fn choose_backend(&mut self, backend: Backend) {
        if self.job.is_some() || (backend == Backend::Cuda && !self.cuda) {
            return;
        }
        self.backend = backend;
        profile::save_backend(&self.data_dir, backend);
    }
}

impl Render for Beeswax {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.status.lock().unwrap().clone();
        let model_setup = self.model_setup.lock().unwrap().clone();
        let update_state = self.update_state.lock().unwrap().clone();
        let busy = self.job.is_some();
        let can_start = model_setup.ready && self.runtime_error.is_none();
        let filtering = busy && !state.recording && !state.monitoring && !state.previewing;

        let mic_select = self.device_select(SELECT_MIC, cx);
        let output_select = self.device_select(SELECT_OUTPUT, cx);
        let speaker_select = self.device_select(SELECT_SPEAKER, cx);

        div()
            .id("app")
            .size_full()
            .bg(rgb(BG))
            .text_color(rgb(INK))
            .font_family("Segoe UI")
            .flex()
            .flex_col()
            .child(self.render_header(&state, &update_state, busy, cx))
            .child(
                div().id("main-scroll").flex_1().overflow_y_scroll().child(
                    div()
                        .w_full()
                        .px_6()
                        .py_4()
                        .flex()
                        .flex_col()
                        .items_center()
                        .child(
                            div()
                                .w_full()
                                .max_w(px(680.))
                                .flex()
                                .flex_col()
                                .gap_3p5()
                                .child(self.render_telemetry_strip(&state, filtering))
                                .child(self.render_voice_section(&state, busy, speaker_select, cx))
                                .child(self.render_routing_section(
                                    &state,
                                    busy,
                                    filtering,
                                    mic_select,
                                    output_select,
                                    cx,
                                ))
                                .child(self.render_engine_section(&state, &model_setup, busy, cx)),
                        ),
                ),
            )
            .child(self.render_footer(&state, &model_setup, &update_state, busy, can_start, cx))
            .when(self.open_select.is_some(), |element| {
                element.child(
                    div()
                        .id("select-scrim")
                        .absolute()
                        .inset_0()
                        .occlude()
                        .on_mouse_down(
                            MouseButton::Left,
                            cx.listener(|this, _, _, cx| {
                                if this.open_select.take().is_some() {
                                    cx.notify();
                                }
                            }),
                        ),
                )
            })
    }
}

fn format_models(folder: &std::path::Path) -> String {
    let list = model::installed_models(folder);
    if list.is_empty() {
        "none".to_owned()
    } else {
        list.join(", ")
    }
}

#[derive(Default, PartialEq, Eq)]
struct PollSnapshot {
    active: bool,
    recording: bool,
    monitoring: bool,
    previewing: bool,
    gated: bool,
    level_q: i32,
    out_level_q: i32,
    progress_q: i32,
    inference_q: i32,
    buffer_q: i32,
    underruns: usize,
    dropped: usize,
    capture_rate: u32,
    message: String,
    note: String,
    model_ready: bool,
    model_active: bool,
    model_msg: Option<String>,
    model_err: Option<String>,
    update_msg: Option<String>,
    update_ver: Option<String>,
}

impl PollSnapshot {
    fn capture(app: &Beeswax) -> Self {
        let s = app.status.lock().unwrap();
        let m = app.model_setup.lock().unwrap();
        let u = app.update_state.lock().unwrap();
        Self {
            active: s.active,
            recording: s.recording,
            monitoring: s.monitoring,
            previewing: s.previewing,
            gated: s.gated,
            level_q: (s.level * 500.) as i32,
            out_level_q: (s.output_level * 500.) as i32,
            progress_q: (s.progress * 200.) as i32,
            inference_q: (s.inference_ms * 10.) as i32,
            buffer_q: s.buffer_ms as i32,
            underruns: s.underruns,
            dropped: s.dropped,
            capture_rate: s.capture_rate,
            message: s.message.clone(),
            note: s.note.clone(),
            model_ready: m.ready,
            model_active: m.active,
            model_msg: m.message.clone(),
            model_err: m.error.clone(),
            update_msg: u.message.clone(),
            update_ver: u.available.as_ref().map(|(v, _)| v.clone()),
        }
    }
}
