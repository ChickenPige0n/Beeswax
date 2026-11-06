use beeswax::{
    Backend,
    audio::{self, Device, Job, Status},
    model::{self, Extractor},
};
use gpui::{
    App, Bounds, Context, MouseButton, Window, WindowBounds, WindowOptions, deferred, div,
    prelude::*, px, rgb,
};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
mod model_download;
mod profile;
mod updates;

const BG: u32 = 0xf4f6f0;
const CARD: u32 = 0xffffff;
const BORDER: u32 = 0xe2e7db;
const INK: u32 = 0x202b25;
const MUTED: u32 = 0x68766d;
const GREEN: u32 = 0x245c46;
const GREEN_DARK: u32 = 0x194b37;
const GREEN_SOFT: u32 = 0xe9f1ea;
const AMBER: u32 = 0x9a6517;
const AMBER_SOFT: u32 = 0xf8ecd8;
const RED: u32 = 0x9b3524;

/// The streaming model keeps about three 128-sample hops of lookahead (384 / 16000).
const MODEL_LOOKAHEAD_MS: f32 = 24.;
/// Average wait while a 40 ms block fills.
const CHUNK_WAIT_MS: f32 = 20.;

const SELECT_MIC: u8 = 0;
const SELECT_OUTPUT: u8 = 1;
const SELECT_SPEAKER: u8 = 2;

#[derive(Clone, Copy)]
enum Tone {
    Secondary,
    Ghost,
}

struct Beeswax {
    inputs: Vec<Device>,
    outputs: Vec<Device>,
    input: usize,
    speakers: Vec<Device>,
    speaker: usize,
    output: usize,
    open_select: Option<u8>,
    select_focus: [gpui::FocusHandle; 3],
    status: Arc<Mutex<Status>>,
    job: Option<Job>,
    cache: Arc<audio::ModelCache>,
    voice: Option<Vec<f32>>,
    model_dir: PathBuf,
    data_dir: PathBuf,
    runtime_error: Option<String>,
    backend: Backend,
    cuda: bool,
    providers: String,
    models: String,
    model_setup: Arc<Mutex<model_download::State>>,
    update_state: Arc<Mutex<updates::State>>,
}

impl Beeswax {
    fn new(
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
                "Record your voice to get started.".to_owned()
            }
            Err(e) => format!("Saved enrollment could not be loaded: {e}. Record again."),
        };
        let cuda = runtime_error.is_none() && model::cuda_available();
        let models = {
            let list = model::installed_models(&model_dir);
            if list.is_empty() {
                "none".to_owned()
            } else {
                list.join(", ")
            }
        };
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
        // Default to CPU: on the development laptop it beats the CUDA fp32 path in
        // latency and separation quality, restarts from the session cache instead of
        // reloading, and leaves the GPU to games. A saved CUDA choice is honored.
        let backend = match profile::load_backend(&data_dir) {
            Some(Backend::Cuda) if cuda => Backend::Cuda,
            _ => Backend::Cpu,
        };
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
            voice: voice.ok(),
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
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(80))
                    .await;
                if view
                    .update(cx, |this, cx| {
                        if this.models == "none" && this.model_setup.lock().unwrap().ready {
                            let installed = model::installed_models(&this.model_dir);
                            if !installed.is_empty() {
                                this.models = installed.join(", ");
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
                                this.voice = Some(sample);
                            }
                        }
                        cx.notify();
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

    fn refresh(&mut self) {
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
                // Every playback endpoint is valid: CABLE Input for call apps, or
                // headphones/speakers for listening to the isolated voice directly.
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

    fn begin(&mut self, record: bool, monitor: bool) {
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
            message: if record {
                "Opening microphone…"
            } else {
                "Loading voice model…"
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

    fn retry_model_download(&mut self) {
        model_download::start(self.model_dir.clone(), self.model_setup.clone());
    }

    fn preview(&mut self) {
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
            message: format!("Playing enrollment through {}", device.name),
            ..Default::default()
        };
        self.open_select = None;
        self.job = Some(Job::preview(device.id.clone(), sample, self.status.clone()));
    }

    fn stop(&mut self) {
        if let Some(job) = &self.job {
            job.cancel();
            self.status.lock().unwrap().message = "Stopping…".into();
        }
    }

    fn output_is_cable(&self) -> bool {
        self.outputs
            .get(self.output)
            .is_some_and(|device| device.name.to_lowercase().contains("cable"))
    }

    fn choose_backend(&mut self, backend: Backend) {
        if self.job.is_some() || (backend == Backend::Cuda && !self.cuda) {
            return;
        }
        self.backend = backend;
        profile::save_backend(&self.data_dir, backend);
    }

    /// A device picker whose list is painted as an overlay, so opening it never moves
    /// the surrounding controls.
    fn device_select(&self, kind: u8, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let (devices, selected) = match kind {
            SELECT_MIC => (&self.inputs, self.input),
            SELECT_OUTPUT => (&self.outputs, self.output),
            _ => (&self.speakers, self.speaker),
        };
        let busy = self.job.is_some();
        let open = self.open_select == Some(kind) && !busy;
        let label = devices
            .get(selected)
            .map(|device| {
                format!(
                    "{}{}",
                    device.name,
                    if device.is_default {
                        " · Windows default"
                    } else {
                        ""
                    }
                )
            })
            .unwrap_or_else(|| "No device available".into());

        let options = devices
            .iter()
            .enumerate()
            .map(|(index, device)| {
                let text = format!(
                    "{}{}{}",
                    if index == selected { "✓  " } else { "" },
                    device.name,
                    if device.is_default {
                        " · Windows default"
                    } else {
                        ""
                    }
                );
                div()
                    .id(("device-option", kind as usize * 1000 + index))
                    .px_3()
                    .py_2()
                    .rounded_sm()
                    .text_size(px(13.))
                    .text_color(rgb(INK))
                    .cursor_pointer()
                    .bg(rgb(if index == selected { GREEN_SOFT } else { CARD }))
                    .hover(|style| style.bg(rgb(0xf0f4ec)))
                    .child(text)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        if this.job.is_none() {
                            match kind {
                                SELECT_MIC => this.input = index,
                                SELECT_OUTPUT => this.output = index,
                                _ => this.speaker = index,
                            }
                            this.open_select = None;
                            this.status.lock().unwrap().level = 0.;
                        }
                        cx.notify();
                    }))
            })
            .collect::<Vec<_>>();

        div()
            .relative()
            .w_full()
            .flex()
            .flex_col()
            .child(
                div()
                    .id(("device-select", kind as usize))
                    .track_focus(&self.select_focus[kind as usize])
                    .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, _, cx| {
                        if this.job.is_some() {
                            return;
                        }
                        let (count, selected) = match kind {
                            SELECT_MIC => (this.inputs.len(), &mut this.input),
                            SELECT_OUTPUT => (this.outputs.len(), &mut this.output),
                            _ => (this.speakers.len(), &mut this.speaker),
                        };
                        match event.keystroke.key.as_str() {
                            "up" if count > 0 => {
                                *selected = selected.saturating_sub(1);
                                this.open_select = Some(kind);
                            }
                            "down" if count > 0 => {
                                *selected = (*selected + 1).min(count - 1);
                                this.open_select = Some(kind);
                            }
                            "enter" | "space" => {
                                this.open_select = if this.open_select == Some(kind) {
                                    None
                                } else {
                                    Some(kind)
                                };
                            }
                            "escape" => {
                                this.open_select = None;
                            }
                            _ => return,
                        }
                        cx.stop_propagation();
                        cx.notify();
                    }))
                    .h(px(42.))
                    .px_3()
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(if open { GREEN } else { 0xd6ddd0 }))
                    .bg(rgb(CARD))
                    .opacity(if busy { 0.55 } else { 1. })
                    .cursor_pointer()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .child(
                        div()
                            .flex_1()
                            .overflow_hidden()
                            .text_size(px(13.))
                            .child(label),
                    )
                    .child(
                        div()
                            .text_size(px(11.))
                            .text_color(rgb(MUTED))
                            .child(if open { "▴" } else { "▾" }),
                    )
                    .on_click(cx.listener(move |this, _, window, cx| {
                        if this.job.is_none() {
                            this.select_focus[kind as usize].focus(window, cx);
                            this.open_select = if this.open_select == Some(kind) {
                                None
                            } else {
                                Some(kind)
                            };
                        }
                        cx.notify();
                    })),
            )
            .when(open, |element| {
                // Lower pickers open upwards so the list stays inside the window.
                let upwards = kind != SELECT_MIC;
                element.child(
                    // Deferred keeps the list out of layout and paints it above the cards.
                    deferred(
                        div()
                            .id(("device-options", kind as usize))
                            .absolute()
                            .when(upwards, |list| list.bottom_11())
                            .when(!upwards, |list| list.top_11())
                            .left_0()
                            .right_0()
                            .p_1()
                            .max_h(px(240.))
                            .overflow_y_scroll()
                            .rounded_md()
                            .border_1()
                            .border_color(rgb(BORDER))
                            .bg(rgb(CARD))
                            .shadow_lg()
                            .occlude()
                            .flex()
                            .flex_col()
                            .children(options),
                    )
                    .with_priority(20),
                )
            })
    }
}

fn button(
    id: &'static str,
    text: impl Into<gpui::SharedString>,
    tone: Tone,
) -> gpui::Stateful<gpui::Div> {
    let (background, foreground, hover) = match tone {
        Tone::Secondary => (0xedf0e8, INK, 0xe1e6da),
        Tone::Ghost => (0xf4f6ef, 0x4b5c51, 0xe9ede2),
    };
    div()
        .id(id)
        .px_4()
        .py_2()
        .rounded_md()
        .cursor_pointer()
        .text_size(px(13.))
        .bg(rgb(background))
        .text_color(rgb(foreground))
        .hover(move |style| style.bg(rgb(hover)))
        .child(text.into())
}

fn caption(text: impl Into<gpui::SharedString>) -> gpui::Div {
    div()
        .text_size(px(12.5))
        .text_color(rgb(MUTED))
        .child(text.into())
}

fn label(text: &'static str) -> gpui::Div {
    div()
        .text_size(px(11.))
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .text_color(rgb(MUTED))
        .child(text)
}

fn bar(fraction: f32, color: u32) -> gpui::Div {
    div()
        .h(px(6.))
        .w_full()
        .rounded_full()
        .bg(rgb(0xe6eae0))
        .overflow_hidden()
        .child(
            div()
                .h_full()
                .w(gpui::relative(fraction.clamp(0., 1.)))
                .rounded_full()
                .bg(rgb(color)),
        )
}

fn card(step: &'static str, title: &'static str, hint: &'static str) -> gpui::Div {
    div()
        .p_5()
        .rounded_lg()
        .border_1()
        .border_color(rgb(BORDER))
        .bg(rgb(CARD))
        .flex()
        .flex_col()
        .gap_4()
        .child(
            div()
                .flex()
                .items_start()
                .gap_3()
                .child(
                    div()
                        .size(px(24.))
                        .rounded_full()
                        .bg(rgb(GREEN_SOFT))
                        .text_color(rgb(GREEN))
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_size(px(12.))
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .child(step),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .text_size(px(15.))
                                .font_weight(gpui::FontWeight::MEDIUM)
                                .child(title),
                        )
                        .child(caption(hint)),
                ),
        )
}

fn choice(
    id: &'static str,
    title: &'static str,
    detail: &'static str,
    selected: bool,
    enabled: bool,
) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .flex_1()
        .p_3()
        .rounded_md()
        .border_1()
        .border_color(rgb(if selected { GREEN } else { BORDER }))
        .bg(rgb(if selected { GREEN_SOFT } else { CARD }))
        .opacity(if enabled { 1. } else { 0.55 })
        .when(enabled, |element| {
            element
                .cursor_pointer()
                .hover(|style| style.border_color(rgb(0xa9bdab)))
        })
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .text_size(px(13.))
                .font_weight(gpui::FontWeight::MEDIUM)
                .text_color(rgb(if selected { GREEN_DARK } else { INK }))
                .child(title),
        )
        .child(caption(detail))
}

impl Render for Beeswax {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.status.lock().unwrap().clone();
        let model_setup = self.model_setup.lock().unwrap().clone();
        let update_state = self.update_state.lock().unwrap().clone();
        let busy = self.job.is_some();
        let can_start = model_setup.ready && self.runtime_error.is_none();
        let filtering = busy && !state.recording && !state.monitoring && !state.previewing;
        let db = 20. * state.level.max(1e-5).log10();
        let level = ((db + 60.) / 60.).clamp(0., 1.);

        let (status_text, status_background, status_foreground) = if state.recording {
            ("RECORDING", AMBER_SOFT, AMBER)
        } else if state.monitoring {
            ("MICROPHONE TEST", AMBER_SOFT, AMBER)
        } else if state.previewing {
            ("PREVIEW", 0xe9eef7, 0x33507a)
        } else if busy {
            ("FILTERING", GREEN_SOFT, GREEN)
        } else {
            ("READY", 0xeef0e8, MUTED)
        };

        let mic_select = self.device_select(SELECT_MIC, cx);
        let output_select = self.device_select(SELECT_OUTPUT, cx);
        let speaker_select = self.device_select(SELECT_SPEAKER, cx);

        let hint = if busy {
            "Stop closes the microphone and ends playback."
        } else if model_setup.active {
            "Required voice models are downloading. Filtering starts when they are ready."
        } else if model_setup.error.is_some() {
            "The voice model download needs attention. Retry it in Processing."
        } else if self.runtime_error.is_some() {
            "Inference is unavailable in this build."
        } else if self.voice.is_none() {
            "Record your voice to enable filtering."
        } else if self.outputs.is_empty() {
            "Connect headphones or VB-CABLE to enable filtering."
        } else {
            "Audio stays on this computer."
        };

        div()
            .id("app")
            .size_full()
            .bg(rgb(BG))
            .text_color(rgb(INK))
            .font_family("Segoe UI")
            .flex()
            .flex_col()
            .child(
                div()
                    .id("main-scroll")
                    .flex_1()
                    .overflow_y_scroll()
                    .child(
                        div()
                            .w_full()
                            .p_8()
                            .flex()
                            .flex_col()
                            .items_center()
                            .child(
                                div()
                                    .w_full()
                                    .max_w(px(700.))
                                    .flex()
                                    .flex_col()
                                    .gap_4()
                                    // Header
                                    .child(
                                        div()
                                            .flex()
                                            .items_center()
                                            .justify_between()
                                            .gap_4()
                                            .child(
                                                div()
                                                    .flex()
                                                    .flex_col()
                                                    .gap_1()
                                                    .child(
                                                        div()
                                                            .text_size(px(26.))
                                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                                            .child("Beeswax"),
                                                    )
                                                    .child(caption(
                                                        "A virtual microphone that keeps only your voice.",
                                                    )),
                                            )
                                            .child(
                                                div()
                                                    .flex()
                                                    .items_center()
                                                    .gap_2()
                                                    .when(update_state.available.is_some(), |row| {
                                                        let (version, url) = update_state
                                                            .available
                                                            .clone()
                                                            .unwrap_or_default();
                                                        row.child(
                                                            button(
                                                                "view-update",
                                                                format!("Update {version}"),
                                                                Tone::Secondary,
                                                            )
                                                            .on_click(cx.listener(
                                                                move |_, _, _, cx| {
                                                                    updates::open_release(&url);
                                                                    cx.notify();
                                                                },
                                                            )),
                                                        )
                                                    })
                                                    .child(
                                                        div()
                                                            .px_3()
                                                            .py_1()
                                                            .rounded_full()
                                                            .bg(rgb(status_background))
                                                            .text_color(rgb(status_foreground))
                                                            .text_size(px(11.))
                                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                                            .child(status_text),
                                                    ),
                                            ),
                                    )
                                    // 1. Microphone
                                    .child(
                                        card(
                                            "1",
                                            "Microphone",
                                            "Beeswax listens only while you test, record, or filter.",
                                        )
                                        .child(mic_select)
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap_3()
                                                .child(
                                                    button(
                                                        "test-mic",
                                                        "Test microphone",
                                                        Tone::Secondary,
                                                    )
                                                    .on_click(cx.listener(|this, _, _, cx| {
                                                        this.begin(true, true);
                                                        cx.notify();
                                                    })),
                                                )
                                                .child(
                                                    div()
                                                        .flex_1()
                                                        .flex()
                                                        .flex_col()
                                                        .gap_1()
                                                        .child(bar(
                                                            if state.recording || state.monitoring {
                                                                level
                                                            } else {
                                                                0.
                                                            },
                                                            0x408763,
                                                        ))
                                                        .child(caption(
                                                            if state.recording || state.monitoring {
                                                                format!(
                                                                    "{db:.0} dBFS{}",
                                                                    if db < -50. {
                                                                        " · very quiet; check mute and Windows access"
                                                                    } else {
                                                                        ""
                                                                    }
                                                                )
                                                            } else {
                                                                "The level meter runs during testing and recording."
                                                                    .to_owned()
                                                            },
                                                        )),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .justify_between()
                                                .gap_3()
                                                .child(caption(format!(
                                                    "{} input device{} found{}",
                                                    self.inputs.len(),
                                                    if self.inputs.len() == 1 { "" } else { "s" },
                                                    if state.capture_rate > 0 {
                                                        format!(
                                                            " · captures at {} kHz{}",
                                                            state.capture_rate / 1000,
                                                            if state.capture_rate < 32_000 {
                                                                " (low detail)"
                                                            } else {
                                                                ""
                                                            }
                                                        )
                                                    } else {
                                                        String::new()
                                                    }
                                                )))
                                                .child(
                                                    button(
                                                        "microphone-settings",
                                                        "Windows microphone settings",
                                                        Tone::Ghost,
                                                    )
                                                    .on_click(cx.listener(|this, _, _, cx| {
                                                        if let Err(e) = std::process::Command::new(
                                                            "explorer.exe",
                                                        )
                                                        .arg("ms-settings:privacy-microphone")
                                                        .spawn()
                                                        {
                                                            this.status.lock().unwrap().message =
                                                                format!(
                                                                    "Could not open microphone settings: {e}"
                                                                );
                                                        }
                                                        cx.notify();
                                                    })),
                                                ),
                                        ),
                                    )
                                    // 2. Voice enrollment
                                    .child(
                                        card(
                                            "2",
                                            "Your voice",
                                            "Speak alone for six seconds; the clearest three seconds are kept.",
                                        )
                                        .child(
                                            div()
                                                .flex()
                                                .flex_col()
                                                .gap_1()
                                                .child(
                                                    div()
                                                        .text_size(px(14.))
                                                        .font_weight(gpui::FontWeight::MEDIUM)
                                                        .child(if self.voice.is_some() {
                                                            "Voice profile ready"
                                                        } else {
                                                            "No voice profile yet"
                                                        }),
                                                )
                                                .child(caption(if state.recording {
                                                    "Read aloud: This is my voice. I am speaking clearly at my normal volume."
                                                } else if self.voice.is_some() {
                                                    "Stored locally in data/voice.json."
                                                } else {
                                                    "Press Record and speak normally; silence and clipping are rejected."
                                                })),
                                        )
                                        .when(state.recording, |element| {
                                            element.child(
                                                div()
                                                    .flex()
                                                    .flex_col()
                                                    .gap_1()
                                                    .child(bar(state.progress, GREEN))
                                                    .child(caption(format!(
                                                        "{:.1} / 6 seconds",
                                                        (state.progress * 6.).min(6.)
                                                    ))),
                                            )
                                        })
                                        .child(
                                            div()
                                                .flex()
                                                .gap_2()
                                                .child(
                                                    button(
                                                        "record",
                                                        if self.voice.is_some() {
                                                            "Record again"
                                                        } else {
                                                            "Record my voice"
                                                        },
                                                        Tone::Secondary,
                                                    )
                                                    .on_click(cx.listener(|this, _, _, cx| {
                                                        this.begin(true, false);
                                                        cx.notify();
                                                    })),
                                                )
                                                .when(self.voice.is_some(), |element| {
                                                    element.child(
                                                        button(
                                                            "forget",
                                                            "Forget voice",
                                                            Tone::Ghost,
                                                        )
                                                        .on_click(cx.listener(
                                                            |this, _, _, cx| {
                                                                if this.job.is_none() {
                                                                    match profile::forget(
                                                                        &this.data_dir,
                                                                    ) {
                                                                        Ok(()) => {
                                                                            this.voice = None;
                                                                            this.cache.clear();
                                                                            this.status
                                                                                .lock()
                                                                                .unwrap()
                                                                                .message =
                                                                                "Voice profile deleted."
                                                                                    .into();
                                                                        }
                                                                        Err(e) => {
                                                                            this.status
                                                                                .lock()
                                                                                .unwrap()
                                                                                .message =
                                                                                format!(
                                                                                    "Could not delete profile: {e}"
                                                                                )
                                                                        }
                                                                    }
                                                                }
                                                                cx.notify();
                                                            },
                                                        )),
                                                    )
                                                }),
                                        )
                                        .when(self.voice.is_some(), |element| {
                                            element.child(
                                                div()
                                                    .flex()
                                                    .flex_col()
                                                    .gap_2()
                                                    .border_t_1()
                                                    .border_color(rgb(BORDER))
                                                    .pt_4()
                                                    .child(label("PREVIEW THROUGH"))
                                                    .child(
                                                        div()
                                                            .flex()
                                                            .items_center()
                                                            .gap_2()
                                                            .child(
                                                                div()
                                                                    .flex_1()
                                                                    .child(speaker_select),
                                                            )
                                                            .child(
                                                                button(
                                                                    "preview",
                                                                    "Play enrollment",
                                                                    Tone::Secondary,
                                                                )
                                                                .opacity(if busy { 0.5 } else { 1. })
                                                                .on_click(cx.listener(
                                                                    |this, _, _, cx| {
                                                                        this.preview();
                                                                        cx.notify();
                                                                    },
                                                                )),
                                                            ),
                                                    ),
                                            )
                                        }),
                                    )
                                    // 3. Output routing
                                    .child(
                                        card(
                                            "3",
                                            "Where the voice goes",
                                            "CABLE Input feeds a call app; headphones or speakers let you listen directly.",
                                        )
                                        .child(output_select)
                                        .child(caption(if self.outputs.is_empty() {
                                            "No playback device found. Install VB-CABLE or connect headphones, then press Refresh devices."
                                        } else if self.output_is_cable() {
                                            "In Discord, Teams, OBS or Zoom, choose the matching CABLE Output as the microphone."
                                        } else {
                                            "Monitoring through this device. Use headphones: speakers can feed back into the microphone."
                                        })),
                                    )
                                    // 4. Processing
                                    .child(
                                        card(
                                            "4",
                                            "Processing",
                                            "Choose where the voice model runs; a faster engine keeps audio seamless.",
                                        )
                                        .child(
                                            div()
                                                .flex()
                                                .gap_2()
                                                .child(
                                                    choice(
                                                        "backend-cpu",
                                                        "CPU",
                                                        "Runs on this processor",
                                                        self.backend == Backend::Cpu,
                                                        !busy,
                                                    )
                                                    .on_click(cx.listener(|this, _, _, cx| {
                                                        this.choose_backend(Backend::Cpu);
                                                        cx.notify();
                                                    })),
                                                )
                                                .child(
                                                    choice(
                                                        "backend-cuda",
                                                        "CUDA",
                                                        if self.cuda {
                                                            "Runs on the NVIDIA GPU"
                                                        } else if self.runtime_error.is_some() {
                                                            "Runtime unavailable"
                                                        } else {
                                                            "Requires an NVIDIA GPU"
                                                        },
                                                        self.backend == Backend::Cuda,
                                                        self.cuda && !busy,
                                                    )
                                                    .on_click(cx.listener(|this, _, _, cx| {
                                                        this.choose_backend(Backend::Cuda);
                                                        cx.notify();
                                                    })),
                                                ),
                                        )
                                        .child(caption(if self.cuda {
                                            format!(
                                                "Runtime providers: {}. CUDA is ready.",
                                                self.providers
                                            )
                                        } else if self.runtime_error.is_some() {
                                            "ONNX Runtime is unavailable, so filtering cannot start. See the error below.".to_owned()
                                        } else {
                                            format!(
                                                "Runtime providers: {}. CUDA is unavailable here; the CPU engine is used.",
                                                self.providers
                                            )
                                        }))
                                        .when(model_setup.active, |element| {
                                            element.child(
                                                div()
                                                    .py_2()
                                                    .text_size(px(12.5))
                                                    .text_color(rgb(GREEN))
                                                    .child(
                                                        model_setup
                                                            .message
                                                            .clone()
                                                            .unwrap_or_else(|| "Preparing voice models…".to_owned()),
                                                    ),
                                            )
                                        })
                                        .when(model_setup.error.is_some(), |element| {
                                            let error = model_setup.error.clone().unwrap_or_default();
                                            element.child(
                                                div()
                                                    .flex()
                                                    .items_center()
                                                    .gap_3()
                                                    .child(
                                                        div()
                                                            .flex_1()
                                                            .py_2()
                                                            .text_size(px(12.5))
                                                            .text_color(rgb(RED))
                                                            .child(error),
                                                    )
                                                    .child(
                                                        button(
                                                            "retry-model-download",
                                                            "Retry download",
                                                            Tone::Secondary,
                                                        )
                                                        .on_click(cx.listener(|this, _, _, cx| {
                                                            this.retry_model_download();
                                                            cx.notify();
                                                        })),
                                                    ),
                                            )
                                        })
                                        .child(caption(if self.models == "none" {
                                            if model_setup.active {
                                                "Installing the required voice models…".to_owned()
                                            } else if model_setup.error.is_some() {
                                                "Required voice models are missing. Retry the download to enable filtering.".to_owned()
                                            } else {
                                                "No voice model found in models/.".to_owned()
                                            }
                                        } else if self.cuda
                                            && !self.models.contains("fp32")
                                            && !self.models.contains("fp16")
                                        {
                                            format!(
                                                "Models installed: {}. Only int8 is available, and its LSTM nodes stay on the CPU inside a CUDA session — CUDA is slower than CPU until models/chunk.onnx (fp32) is added.",
                                                self.models
                                            )
                                        } else if self.cuda {
                                            format!(
                                                "Models installed: {}. CUDA runs the highest precision present; CPU runs int8 when available.",
                                                self.models
                                            )
                                        } else {
                                            format!(
                                                "Models installed: {}. The CPU engine runs int8 for real time.",
                                                self.models
                                            )
                                        }))
                                        .when(filtering, |element| {
                                            element
                                                .child(caption(format!(
                                                    "{} model · {} engine · {:.1} ms compute · {:.0} ms buffer · ≈{:.0} ms added delay · {} gaps",
                                                    if state.model.is_empty() {
                                                        "?"
                                                    } else {
                                                        state.model.as_str()
                                                    },
                                                    if state.engine.is_empty() {
                                                        "CPU"
                                                    } else {
                                                        state.engine.as_str()
                                                    },
                                                    state.inference_ms,
                                                    state.buffer_ms,
                                                    state.capture_ms
                                                        + CHUNK_WAIT_MS
                                                        + MODEL_LOOKAHEAD_MS
                                                        + state.buffer_ms,
                                                    state.underruns,
                                                )))
                                                .when(state.dropped > 0, |element| {
                                                    element.child(caption(format!(
                                                        "{} samples were dropped to stay in time.",
                                                        state.dropped
                                                    )))
                                                })
                                        })
                                        .when(!state.note.is_empty(), |element| {
                                            element.child(
                                                div()
                                                    .p_3()
                                                    .rounded_md()
                                                    .bg(rgb(0xfbf3e4))
                                                    .text_color(rgb(0x7a5314))
                                                    .text_size(px(12.5))
                                                    .child(state.note.clone()),
                                            )
                                        })
                                        .when(self.runtime_error.is_some(), |element| {
                                            element.child(
                                                div()
                                                    .p_3()
                                                    .rounded_md()
                                                    .bg(rgb(0xf9ece9))
                                                    .text_color(rgb(RED))
                                                    .text_size(px(12.5))
                                                    .child(
                                                        self.runtime_error.clone().unwrap_or_default(),
                                                    ),
                                            )
                                        }),
                                    )
                                    .child(caption(
                                        "Audio stays on this computer. Beeswax never uploads your voice.",
                                    )),
                            ),
                    ),
            )
            .child(
                div()
                    .border_t_1()
                    .border_color(rgb(BORDER))
                    .bg(rgb(0xfbfcf8))
                    .px_8()
                    .py_4()
                    .flex()
                    .items_center()
                    .gap_4()
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_size(px(13.))
                                    .text_color(rgb(INK))
                                    .child(state.message.clone()),
                            )
                            .child(caption(hint))
                            .when(update_state.message.is_some(), |element| {
                                element.child(caption(
                                    update_state.message.clone().unwrap_or_default(),
                                ))
                            }),
                    )
                    .child(                        button("refresh", "Refresh devices", Tone::Ghost).on_click(
                            cx.listener(|this, _, _, cx| {
                                this.refresh();
                                cx.notify();
                            }),
                        ),
                    )
                    .child(
                        div()
                            .id("toggle")
                            .px_6()
                            .py_3()
                            .rounded_md()
                            .when(busy || can_start, |element| element.cursor_pointer())
                            .text_size(px(14.))
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .bg(rgb(if busy {
                                0x8c3226
                            } else if can_start {
                                GREEN
                            } else {
                                0x87928b
                            }))
                            .text_color(rgb(0xffffff))
                            .hover(move |style| {
                                style.bg(rgb(if busy {
                                    0x74281e
                                } else if can_start {
                                    GREEN_DARK
                                } else {
                                    0x87928b
                                }))
                            })
                            .child(if busy {
                                "Stop"
                            } else if self.runtime_error.is_some() {
                                "Filtering unavailable"
                            } else if model_setup.active {
                                "Downloading models…"
                            } else if !model_setup.ready {
                                "Models unavailable"
                            } else {
                                "Start filtering"
                            })
                            .on_click(cx.listener(move |this, _, _, cx| {
                                if this.job.is_some() {
                                    this.stop();
                                } else if can_start {
                                    this.begin(false, false);
                                }
                                cx.notify();
                            })),
                    ),
            )
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
        let bounds = Bounds::centered(None, gpui::size(px(800.), px(900.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                window_min_size: Some(gpui::size(px(620.), px(720.))),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|cx| Beeswax::new(model_dir, data_dir, runtime_error, update_repository, cx))
            },
        )
        .expect("Unable to open Beeswax window");
        cx.activate(true);
    });
    Ok(())
}
