use super::theme::{
    AMBER, BORDER, BORDER_STRONG, CARD, GREEN, GREEN_DARK, GREEN_METER, GREEN_SOFT, INK, INK_SOFT,
    MUTED, PillTone, RED, RED_DARK, SUBTLE, SURFACE, Tone, banner, button, caption, level_meter,
    metric_cell, pill, progress_bar, section_label, waveform_strip,
};
use super::{Beeswax, CHUNK_WAIT_MS, MODEL_LOOKAHEAD_MS};
use crate::{model_download, updates};
use beeswax::{Backend, audio::Status};
use gpui::{Context, div, prelude::*, px, rgb};

impl Beeswax {
    pub(super) fn render_header(
        &self,
        state: &Status,
        update_state: &updates::State,
        busy: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let (badge_label, badge_tone) = if self.runtime_error.is_some() {
            ("RUNTIME ERROR", PillTone::Danger)
        } else if state.recording {
            ("ENROLLING", PillTone::Recording)
        } else if state.monitoring {
            ("MIC CHECK", PillTone::Recording)
        } else if state.previewing {
            ("PREVIEWING", PillTone::Info)
        } else if busy && state.gated {
            ("FILTERING · SILENCE GATE", PillTone::Active)
        } else if busy {
            ("FILTERING LIVE", PillTone::Active)
        } else if self.voice.is_some() {
            ("READY", PillTone::Neutral)
        } else {
            ("SETUP NEEDED", PillTone::Recording)
        };

        div()
            .px_6()
            .py_3p5()
            .bg(rgb(SURFACE))
            .border_b_1()
            .border_color(rgb(BORDER))
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .size(px(30.))
                            .rounded_md()
                            .bg(rgb(GREEN))
                            .text_color(rgb(0xffffff))
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_size(px(14.))
                            .font_weight(gpui::FontWeight::BOLD)
                            .child("B"),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .child(
                                div()
                                    .flex()
                                    .items_baseline()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_size(px(16.))
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .text_color(rgb(INK))
                                            .child("Beeswax"),
                                    )
                                    .child(
                                        div()
                                            .text_size(px(11.))
                                            .text_color(rgb(MUTED))
                                            .child(concat!("v", env!("CARGO_PKG_VERSION"))),
                                    ),
                            )
                            .child(caption(
                                "Target-speaker virtual microphone · local inference",
                            )),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .when(update_state.available.is_some(), |row| {
                        let (version, url) = update_state.available.clone().unwrap_or_default();
                        row.child(
                            button(
                                "view-update",
                                format!("Update {version}"),
                                Tone::Secondary,
                                true,
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
                        button("refresh", "Refresh", Tone::Ghost, !busy).on_click(cx.listener(
                            |this, _, _, cx| {
                                this.refresh();
                                cx.notify();
                            },
                        )),
                    )
                    .child(pill(badge_label, badge_tone)),
            )
    }

    pub(super) fn render_telemetry_strip(&self, state: &Status, filtering: bool) -> gpui::Div {
        let total_delay = state.capture_ms + CHUNK_WAIT_MS + MODEL_LOOKAHEAD_MS + state.buffer_ms;
        let engine_text = if state.engine.is_empty() {
            format!("{}", self.backend)
        } else if state.model.is_empty() {
            state.engine.clone()
        } else {
            format!("{} · {}", state.engine, state.model)
        };
        let compute_text = if !filtering {
            "—".to_owned()
        } else if state.gated {
            format!("0.0 ms (gated · {:.1} ms)", state.inference_ms)
        } else {
            format!("{:.1} ms / 40 ms", state.inference_ms)
        };
        let delay_text = if filtering {
            format!("≈{total_delay:.0} ms")
        } else {
            "—".to_owned()
        };
        let health_text = if !filtering {
            "Idle".to_owned()
        } else if state.underruns == 0 && state.dropped == 0 {
            "Clean stream".to_owned()
        } else {
            format!("{} gaps · {} dropped", state.underruns, state.dropped)
        };

        let out_db = 20. * state.output_level.max(1e-5).log10();
        let out_frac = ((out_db + 60.) / 60.).clamp(0., 1.);

        div()
            .p_3p5()
            .rounded_lg()
            .border_1()
            .border_color(rgb(if filtering { 0xbdd4c6 } else { BORDER }))
            .bg(rgb(if filtering { GREEN_SOFT } else { SURFACE }))
            .flex()
            .flex_col()
            .gap_2p5()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2p5()
                    .child(metric_cell("ENGINE", engine_text))
                    .child(metric_cell("COMPUTE", compute_text))
                    .child(metric_cell("ADDED DELAY", delay_text))
                    .child(metric_cell("STREAM HEALTH", health_text)),
            )
            .when(filtering, |el| {
                el.child(
                    div()
                        .flex()
                        .items_center()
                        .gap_3()
                        .child(
                            div()
                                .text_size(px(11.))
                                .font_weight(gpui::FontWeight::SEMIBOLD)
                                .text_color(rgb(GREEN_DARK))
                                .child("ISOLATED OUTPUT"),
                        )
                        .child(div().flex_1().child(level_meter(out_frac, true, GREEN)))
                        .child(
                            div()
                                .w(px(58.))
                                .text_size(px(11.5))
                                .text_color(rgb(INK_SOFT))
                                .child(if state.gated {
                                    "Muted".to_owned()
                                } else {
                                    format!("{out_db:.0} dBFS")
                                }),
                        ),
                )
            })
    }
    pub(super) fn render_voice_section(
        &self,
        state: &Status,
        busy: bool,
        speaker_select: impl IntoElement,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let has_voice = self.voice.is_some();
        let speaker_name = self
            .speakers
            .get(self.speaker)
            .map(|d| d.name.as_str())
            .unwrap_or("No speaker");

        div()
            .p_4()
            .rounded_lg()
            .border_1()
            .border_color(rgb(if !has_voice && !state.recording {
                0xdfc596
            } else {
                BORDER
            }))
            .bg(rgb(CARD))
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2p5()
                            .child(section_label("TARGET VOICE PROFILE"))
                            .child(if state.recording {
                                pill("Recording 6 s", PillTone::Recording)
                            } else if has_voice {
                                pill("Enrolled · 3.0 s", PillTone::Active)
                            } else {
                                pill("Not enrolled", PillTone::Recording)
                            }),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .when(has_voice, |row| {
                                row.child(
                                    button(
                                        "preview",
                                        if state.previewing {
                                            "Playing…"
                                        } else {
                                            "Play clip"
                                        },
                                        Tone::Secondary,
                                        !busy,
                                    )
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.preview();
                                        cx.notify();
                                    })),
                                )
                            })
                            .child(
                                button(
                                    "record",
                                    if has_voice {
                                        "Re-record"
                                    } else {
                                        "Record voice"
                                    },
                                    if has_voice {
                                        Tone::Secondary
                                    } else {
                                        Tone::Primary
                                    },
                                    !busy,
                                )
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.begin(true, false);
                                    cx.notify();
                                })),
                            )
                            .when(has_voice, |row| {
                                row.child(
                                    button("forget", "Forget", Tone::Ghost, !busy).on_click(
                                        cx.listener(|this, _, _, cx| {
                                            this.forget_voice();
                                            cx.notify();
                                        }),
                                    ),
                                )
                            }),
                    ),
            )
            .when(state.recording, |el| {
                el.child(
                    div()
                        .p_3()
                        .rounded_md()
                        .bg(rgb(SURFACE))
                        .border_1()
                        .border_color(rgb(BORDER))
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .text_size(px(13.))
                                .font_weight(gpui::FontWeight::MEDIUM)
                                .text_color(rgb(INK))
                                .child(
                                    "Read aloud: “This is my voice. I am speaking clearly at my normal volume.”",
                                ),
                        )
                        .child(progress_bar(state.progress, GREEN))
                        .child(caption(format!(
                            "{:.1} / 6.0 s captured — the clearest 3 seconds are kept automatically.",
                            (state.progress * 6.).min(6.)
                        ))),
                )
            })
            .when(!state.recording && has_voice, |el| {
                el.child(waveform_strip(&self.waveform, state.previewing))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .gap_3()
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(0.))
                                    .text_size(px(12.))
                                    .text_color(rgb(MUTED))
                                    .truncate()
                                    .child(format!(
                                        "Saved locally in data/voice.json · Preview speaker: {speaker_name}"
                                    )),
                            )
                            .child(
                                div()
                                    .id("toggle-preview-device")
                                    .text_size(px(12.))
                                    .font_weight(gpui::FontWeight::MEDIUM)
                                    .text_color(rgb(GREEN))
                                    .cursor_pointer()
                                    .hover(|s| s.text_color(rgb(GREEN_DARK)))
                                    .child(if self.show_preview_picker {
                                        "Hide speaker"
                                    } else {
                                        "Change speaker"
                                    })
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.show_preview_picker = !this.show_preview_picker;
                                        this.open_select = None;
                                        cx.notify();
                                    })),
                            ),
                    )
                    .when(self.show_preview_picker, |el| el.child(speaker_select))
            })
            .when(!state.recording && !has_voice, |el| {
                el.child(caption(
                    "Speak alone for six seconds in a quiet moment. Silence and clipping are rejected automatically.",
                ))
            })
    }
    pub(super) fn render_routing_section(
        &self,
        state: &Status,
        busy: bool,
        filtering: bool,
        mic_select: impl IntoElement,
        output_select: impl IntoElement,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let meter_active = state.recording || state.monitoring || filtering;
        let db = 20. * state.level.max(1e-5).log10();
        let level_frac = ((db + 60.) / 60.).clamp(0., 1.);
        let is_cable = self.output_is_cable();

        div()
            .p_4()
            .rounded_lg()
            .border_1()
            .border_color(rgb(BORDER))
            .bg(rgb(CARD))
            .flex()
            .flex_col()
            .gap_3p5()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(section_label("AUDIO ROUTING"))
                    .child(
                        div()
                            .id("microphone-settings")
                            .text_size(px(12.))
                            .text_color(rgb(MUTED))
                            .cursor_pointer()
                            .hover(|s| s.text_color(rgb(INK)))
                            .child("Windows mic privacy ↗")
                            .on_click(cx.listener(|this, _, _, cx| {
                                if let Err(e) = std::process::Command::new("explorer.exe")
                                    .arg("ms-settings:privacy-microphone")
                                    .spawn()
                                {
                                    this.status.lock().unwrap().message =
                                        format!("Could not open microphone settings: {e}");
                                }
                                cx.notify();
                            })),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_start()
                    .gap_3()
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.))
                            .flex()
                            .flex_col()
                            .gap_1p5()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .child(
                                        div()
                                            .text_size(px(12.5))
                                            .font_weight(gpui::FontWeight::MEDIUM)
                                            .text_color(rgb(INK))
                                            .child("Physical microphone"),
                                    )
                                    .child(caption(if state.capture_rate > 0 {
                                        format!("{} kHz", state.capture_rate / 1000)
                                    } else {
                                        format!("{} found", self.inputs.len())
                                    })),
                            )
                            .child(mic_select),
                    )
                    .child(
                        div()
                            .pt_7()
                            .text_size(px(14.))
                            .text_color(rgb(MUTED))
                            .child("→"),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.))
                            .flex()
                            .flex_col()
                            .gap_1p5()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .child(
                                        div()
                                            .text_size(px(12.5))
                                            .font_weight(gpui::FontWeight::MEDIUM)
                                            .text_color(rgb(INK))
                                            .child("Filtered output"),
                                    )
                                    .child(caption(if self.outputs.is_empty() {
                                        "None"
                                    } else if is_cable {
                                        "Virtual mic"
                                    } else {
                                        "Direct monitor"
                                    })),
                            )
                            .child(output_select),
                    ),
            )
            .child(
                div()
                    .pt_2p5()
                    .border_t_1()
                    .border_color(rgb(BORDER))
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_3()
                            .child(
                                button(
                                    "test-mic",
                                    if state.monitoring {
                                        "Stop test"
                                    } else {
                                        "Test mic"
                                    },
                                    if state.monitoring {
                                        Tone::Danger
                                    } else {
                                        Tone::Ghost
                                    },
                                    !busy || state.monitoring,
                                )
                                .on_click(cx.listener(|this, _, _, cx| {
                                    if this.job.is_some() {
                                        this.stop();
                                    } else {
                                        this.begin(true, true);
                                    }
                                    cx.notify();
                                })),
                            )
                            .child(
                                div()
                                    .flex_1()
                                    .child(level_meter(level_frac, meter_active, GREEN_METER)),
                            )
                            .child(
                                div()
                                    .w(px(64.))
                                    .text_size(px(11.5))
                                    .text_color(rgb(if meter_active && db < -50. {
                                        AMBER
                                    } else {
                                        MUTED
                                    }))
                                    .child(if meter_active {
                                        format!("{db:.0} dBFS")
                                    } else {
                                        "Mic idle".to_owned()
                                    }),
                            ),
                    )
                    .child(caption(if self.outputs.is_empty() {
                        "Install VB-CABLE or connect headphones, then press Refresh."
                    } else if meter_active && db < -50. {
                        "Input is below -50 dBFS — check hardware mute and Windows microphone permissions."
                    } else if is_cable {
                        "Select CABLE Output as your microphone in Discord, Zoom, Teams, or OBS."
                    } else {
                        "Direct listening active — wear headphones so speakers do not feed back into the mic."
                    })),
            )
    }

    pub(super) fn render_engine_section(
        &self,
        state: &Status,
        model_setup: &model_download::State,
        busy: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let cpu_selected = self.backend == Backend::Cpu;
        let cuda_selected = self.backend == Backend::Cuda;
        let cuda_enabled = self.cuda && !busy;

        div()
            .p_4()
            .rounded_lg()
            .border_1()
            .border_color(rgb(BORDER))
            .bg(rgb(CARD))
            .flex()
            .flex_col()
            .gap_2p5()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(section_label("INFERENCE ENGINE"))
                            .child(caption(format!(
                                "Providers: {} · Models: {}",
                                self.providers, self.models
                            ))),
                    )
                    .child(
                        div()
                            .p(px(3.))
                            .rounded_md()
                            .bg(rgb(SUBTLE))
                            .border_1()
                            .border_color(rgb(BORDER))
                            .flex()
                            .items_center()
                            .gap(px(2.))
                            .child(
                                segment_option("backend-cpu", "CPU · int8", cpu_selected, !busy)
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.choose_backend(Backend::Cpu);
                                        cx.notify();
                                    })),
                            )
                            .child(
                                segment_option(
                                    "backend-cuda",
                                    if self.cuda {
                                        "CUDA · GPU"
                                    } else {
                                        "CUDA (unavailable)"
                                    },
                                    cuda_selected,
                                    cuda_enabled,
                                )
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.choose_backend(Backend::Cuda);
                                    cx.notify();
                                })),
                            ),
                    ),
            )
            .when(
                self.cuda && !self.models.contains("fp32") && !self.models.contains("fp16"),
                |el| {
                    el.child(caption(
                        "Only int8 is installed; add models/chunk.onnx (fp32) so LSTM layers run on CUDA instead of CPU.",
                    ))
                },
            )
            .when(model_setup.active, |el| {
                el.child(caption(
                    model_setup
                        .message
                        .clone()
                        .unwrap_or_else(|| "Downloading voice models…".to_owned()),
                ))
            })
            .when(model_setup.error.is_some(), |el| {
                let err = model_setup.error.clone().unwrap_or_default();
                el.child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap_3()
                        .child(div().flex_1().child(banner(err, true)))
                        .child(
                            button("retry-model-download", "Retry", Tone::Secondary, true)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.retry_model_download();
                                    cx.notify();
                                })),
                        ),
                )
            })
            .when(!state.note.is_empty(), |el| {
                el.child(banner(state.note.clone(), false))
            })
            .when(self.runtime_error.is_some(), |el| {
                el.child(banner(
                    self.runtime_error.clone().unwrap_or_default(),
                    true,
                ))
            })
    }

    pub(super) fn render_footer(
        &self,
        state: &Status,
        model_setup: &model_download::State,
        update_state: &updates::State,
        busy: bool,
        can_start: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let hint = if busy {
            "Audio stays on this PC. Press Stop to release the microphone."
        } else if model_setup.active {
            "Finishing first-run model check…"
        } else if model_setup.error.is_some() {
            "Retry the model download above to enable filtering."
        } else if self.runtime_error.is_some() {
            "ONNX Runtime is unavailable in this build."
        } else if self.voice.is_none() {
            "Enroll your voice above to unlock Start filtering."
        } else if self.outputs.is_empty() {
            "Connect VB-CABLE or headphones, then refresh devices."
        } else {
            "100% local processing · zero cloud uploads."
        };

        let primary_label = if busy {
            "Stop"
        } else if self.runtime_error.is_some() {
            "Unavailable"
        } else if model_setup.active {
            "Preparing models…"
        } else if !model_setup.ready {
            "Models missing"
        } else if self.voice.is_none() {
            "Enroll voice first"
        } else {
            "Start filtering"
        };

        let actionable = busy || (can_start && self.voice.is_some() && !self.outputs.is_empty());

        div()
            .px_6()
            .py_3p5()
            .border_t_1()
            .border_color(rgb(BORDER))
            .bg(rgb(SURFACE))
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .child(
                div()
                    .flex_1()
                    .min_w(px(0.))
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(
                        div()
                            .text_size(px(13.))
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .text_color(rgb(INK))
                            .truncate()
                            .child(state.message.clone()),
                    )
                    .child(
                        div()
                            .text_size(px(11.5))
                            .text_color(rgb(MUTED))
                            .truncate()
                            .child(match &update_state.message {
                                Some(msg) if update_state.available.is_some() => {
                                    format!("{hint} · {msg}")
                                }
                                _ => hint.to_owned(),
                            }),
                    ),
            )
            .child(
                div()
                    .id("toggle")
                    .h(px(40.))
                    .px_6()
                    .rounded_md()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_size(px(13.5))
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(rgb(0xffffff))
                    .bg(rgb(if busy {
                        RED
                    } else if actionable {
                        GREEN
                    } else {
                        0x8c968f
                    }))
                    .when(busy || can_start, |el| {
                        el.cursor_pointer().hover(move |s| {
                            s.bg(rgb(if busy {
                                RED_DARK
                            } else if actionable {
                                GREEN_DARK
                            } else {
                                0x7d8780
                            }))
                        })
                    })
                    .child(primary_label)
                    .on_click(cx.listener(move |this, _, _, cx| {
                        if this.job.is_some() {
                            this.stop();
                        } else if can_start {
                            this.begin(false, false);
                        }
                        cx.notify();
                    })),
            )
    }
}

fn segment_option(
    id: &'static str,
    label: &'static str,
    selected: bool,
    enabled: bool,
) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .h(px(28.))
        .px_3()
        .rounded(px(5.))
        .border_1()
        .border_color(rgb(if selected { BORDER_STRONG } else { SUBTLE }))
        .bg(rgb(if selected { CARD } else { SUBTLE }))
        .text_color(rgb(if selected { INK } else { MUTED }))
        .text_size(px(12.))
        .font_weight(if selected {
            gpui::FontWeight::SEMIBOLD
        } else {
            gpui::FontWeight::MEDIUM
        })
        .opacity(if enabled || selected { 1. } else { 0.45 })
        .flex()
        .items_center()
        .justify_center()
        .when(enabled, |el| {
            el.cursor_pointer().hover(move |s| s.text_color(rgb(INK)))
        })
        .child(label)
}
