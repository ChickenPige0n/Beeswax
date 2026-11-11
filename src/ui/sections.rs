use super::theme::{
    AMBER, BORDER, BORDER_STRONG, CARD, GREEN, GREEN_DARK, GREEN_METER, GREEN_SOFT, INK, INK_SOFT,
    MUTED, PillTone, RED, RED_DARK, SUBTLE, SURFACE, Tone, app_icon, banner, button, caption,
    level_meter, metric_cell, pill, progress_bar, section_label, waveform_strip,
};
use super::{Beeswax, CHUNK_WAIT_MS, MODEL_LOOKAHEAD_MS};
use crate::{model_download, updates};
use beeswax::{
    Backend,
    audio::Status,
    i18n::{Language, pick},
};
use gpui::{Context, div, prelude::*, px, rgb};

impl Beeswax {
    pub(super) fn render_header(
        &self,
        state: &Status,
        update_state: &updates::State,
        busy: bool,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let lang = self.lang;
        let (badge_label, badge_tone) = if self.runtime_error.is_some() {
            (
                pick(lang, "RUNTIME ERROR", "运行环境异常"),
                PillTone::Danger,
            )
        } else if state.recording {
            (pick(lang, "ENROLLING", "正在录入"), PillTone::Recording)
        } else if state.monitoring {
            (pick(lang, "MIC CHECK", "麦克风测试"), PillTone::Recording)
        } else if state.previewing {
            (pick(lang, "PREVIEWING", "正在试听"), PillTone::Info)
        } else if busy && state.gated {
            (
                pick(lang, "FILTERING (SILENCE GATE)", "过滤中（静音门控）"),
                PillTone::Active,
            )
        } else if busy {
            (pick(lang, "FILTERING LIVE", "实时过滤中"), PillTone::Active)
        } else if self.voice.is_some() {
            (pick(lang, "READY", "已就绪"), PillTone::Neutral)
        } else {
            (
                pick(lang, "SETUP NEEDED", "待录入声纹"),
                PillTone::Recording,
            )
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
                    .child(app_icon().size(px(32.)).rounded_md())
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
                            .child(caption(pick(
                                lang,
                                "Target-speaker virtual microphone, local inference",
                                "目标说话人虚拟麦克风 | 本地离线推理",
                            ))),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .when(update_state.available.is_some(), |row| {
                        let (version, url) = update_state.available.clone().unwrap_or_default();
                        let update_label = match lang {
                            Language::En => format!("Update {version}"),
                            Language::ZhCn => format!("更新 {version}"),
                        };
                        row.child(
                            button("view-update", update_label, Tone::Secondary, true).on_click(
                                cx.listener(move |_, _, _, cx| {
                                    updates::open_release(&url);
                                    cx.notify();
                                }),
                            ),
                        )
                    })
                    .child(
                        div()
                            .p(px(2.))
                            .rounded_md()
                            .bg(rgb(SUBTLE))
                            .border_1()
                            .border_color(rgb(BORDER))
                            .flex()
                            .items_center()
                            .gap(px(2.))
                            .child(lang_option("lang-en", "EN", lang == Language::En).on_click(
                                cx.listener(|this, _, _, cx| {
                                    this.choose_language(Language::En);
                                    cx.notify();
                                }),
                            ))
                            .child(
                                lang_option("lang-zh", "中文", lang == Language::ZhCn).on_click(
                                    cx.listener(|this, _, _, cx| {
                                        this.choose_language(Language::ZhCn);
                                        cx.notify();
                                    }),
                                ),
                            ),
                    )
                    .child(
                        button("refresh", pick(lang, "Refresh", "刷新"), Tone::Ghost, !busy)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.refresh();
                                cx.notify();
                            })),
                    )
                    .child(pill(badge_label, badge_tone)),
            )
    }

    pub(super) fn render_telemetry_strip(&self, state: &Status, filtering: bool) -> gpui::Div {
        let lang = self.lang;
        let total_delay = state.capture_ms + CHUNK_WAIT_MS + MODEL_LOOKAHEAD_MS + state.buffer_ms;
        let engine_text = if state.engine.is_empty() {
            format!("{}", self.backend)
        } else if state.model.is_empty() {
            state.engine.clone()
        } else {
            format!("{} {}", state.engine, state.model)
        };
        let compute_text = if !filtering {
            "—".to_owned()
        } else if state.gated {
            match lang {
                Language::En => format!("0.0 ms (gated, {:.1} ms)", state.inference_ms),
                Language::ZhCn => format!("0.0 ms（静音门控，{:.1} ms）", state.inference_ms),
            }
        } else {
            format!("{:.1} ms / 40 ms", state.inference_ms)
        };
        let delay_text = if filtering {
            format!("≈{total_delay:.0} ms")
        } else {
            "—".to_owned()
        };
        let health_text = if !filtering {
            pick(lang, "Idle", "空闲").to_owned()
        } else if state.underruns == 0 && state.dropped == 0 {
            pick(lang, "Clean stream", "流状态正常").to_owned()
        } else {
            match lang {
                Language::En => format!("{} gaps, {} dropped", state.underruns, state.dropped),
                Language::ZhCn => format!("{} 次欠载，丢弃 {}", state.underruns, state.dropped),
            }
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
                    .child(metric_cell(pick(lang, "ENGINE", "推理引擎"), engine_text))
                    .child(metric_cell(pick(lang, "COMPUTE", "单块耗时"), compute_text))
                    .child(metric_cell(
                        pick(lang, "ADDED DELAY", "附加延迟"),
                        delay_text,
                    ))
                    .child(metric_cell(
                        pick(lang, "STREAM HEALTH", "音频流状态"),
                        health_text,
                    )),
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
                                .child(pick(lang, "ISOLATED OUTPUT", "分离后输出电平")),
                        )
                        .child(div().flex_1().child(level_meter(out_frac, true, GREEN)))
                        .child(
                            div()
                                .w(px(58.))
                                .text_size(px(11.5))
                                .text_color(rgb(INK_SOFT))
                                .child(if state.gated {
                                    pick(lang, "Muted", "静音").to_owned()
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
        let lang = self.lang;
        let has_voice = self.voice.is_some();
        let speaker_name = self
            .speakers
            .get(self.speaker)
            .map(|d| d.name.as_str())
            .unwrap_or_else(|| pick(lang, "No speaker", "无扬声器"));

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
                            .child(section_label(pick(
                                lang,
                                "TARGET VOICE PROFILE",
                                "目标声纹档案",
                            )))
                            .child(if state.recording {
                                pill(pick(lang, "Recording 6 s", "录制中 6 秒"), PillTone::Recording)
                            } else if has_voice {
                                pill(
                                    pick(lang, "Enrolled (3.0 s)", "已录入（3.0 秒）"),
                                    PillTone::Active,
                                )
                            } else {
                                pill(pick(lang, "Not enrolled", "未录入"), PillTone::Recording)
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
                                            pick(lang, "Playing…", "播放中…")
                                        } else {
                                            pick(lang, "Play clip", "试听片段")
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
                                        pick(lang, "Re-record", "重新录制")
                                    } else {
                                        pick(lang, "Record voice", "录制声纹")
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
                                    button(
                                        "forget",
                                        pick(lang, "Forget", "清除"),
                                        Tone::Ghost,
                                        !busy,
                                    )
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.forget_voice();
                                        cx.notify();
                                    })),
                                )
                            }),
                    ),
            )
            .when(state.recording, |el| {
                let captured = (state.progress * 6.).min(6.);
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
                                .child(pick(
                                    lang,
                                    "Read aloud: “This is my voice. I am speaking clearly at my normal volume.”",
                                    "请朗读：“这是我的声音，我正在以正常音量清晰地说话。”",
                                )),
                        )
                        .child(progress_bar(state.progress, GREEN))
                        .child(caption(match lang {
                            Language::En => format!(
                                "{captured:.1} / 6.0 s captured — the clearest 3 seconds are kept automatically."
                            ),
                            Language::ZhCn => format!(
                                "已采集 {captured:.1} / 6.0 秒 — 系统会自动截取最清晰的 3 秒片段。"
                            ),
                        })),
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
                                    .child(match lang {
                                        Language::En => format!(
                                            "Saved locally in data/voice.json | Preview speaker: {speaker_name}"
                                        ),
                                        Language::ZhCn => format!(
                                            "已保存在本地 data/voice.json | 试听扬声器：{speaker_name}"
                                        ),
                                    }),
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
                                        pick(lang, "Hide speaker", "收起扬声器")
                                    } else {
                                        pick(lang, "Change speaker", "更换扬声器")
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
                el.child(caption(pick(
                    lang,
                    "Speak alone for six seconds in a quiet moment. Silence and clipping are rejected automatically.",
                    "请在安静环境下独自说话 6 秒，系统会自动拦截静音或破音录音。",
                )))
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
        let lang = self.lang;
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
                    .child(section_label(pick(lang, "AUDIO ROUTING", "音频路由")))
                    .child(
                        div()
                            .id("microphone-settings")
                            .text_size(px(12.))
                            .text_color(rgb(MUTED))
                            .cursor_pointer()
                            .hover(|s| s.text_color(rgb(INK)))
                            .child(pick(
                                lang,
                                "Windows mic privacy ↗",
                                "Windows 麦克风隐私设置 ↗",
                            ))
                            .on_click(cx.listener(|this, _, _, cx| {
                                if let Err(e) = std::process::Command::new("explorer.exe")
                                    .arg("ms-settings:privacy-microphone")
                                    .spawn()
                                {
                                    this.status.lock().unwrap().message = match this.lang {
                                        Language::En => {
                                            format!("Could not open microphone settings: {e}")
                                        }
                                        Language::ZhCn => {
                                            format!("无法打开麦克风隐私设置：{e}")
                                        }
                                    };
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
                                            .child(pick(
                                                lang,
                                                "Physical microphone",
                                                "物理麦克风",
                                            )),
                                    )
                                    .child(caption(if state.capture_rate > 0 {
                                        format!("{} kHz", state.capture_rate / 1000)
                                    } else {
                                        match lang {
                                            Language::En => format!("{} found", self.inputs.len()),
                                            Language::ZhCn => format!("已发现 {} 个", self.inputs.len()),
                                        }
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
                                            .child(pick(lang, "Filtered output", "过滤后输出")),
                                    )
                                    .child(caption(if self.outputs.is_empty() {
                                        pick(lang, "None", "无")
                                    } else if is_cable {
                                        pick(lang, "Virtual mic", "虚拟麦克风")
                                    } else {
                                        pick(lang, "Direct monitor", "直接监听")
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
                                        pick(lang, "Stop test", "停止测试")
                                    } else {
                                        pick(lang, "Test mic", "测试麦克风")
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
                                        pick(lang, "Mic idle", "未采集").to_owned()
                                    }),
                            ),
                    )
                    .child(caption(if self.outputs.is_empty() {
                        pick(
                            lang,
                            "Install VB-CABLE or connect headphones, then press Refresh.",
                            "请安装 VB-CABLE 虚拟声卡或连接耳机，然后点击“刷新”。",
                        )
                    } else if meter_active && db < -50. {
                        pick(
                            lang,
                            "Input is below -50 dBFS — check hardware mute and Windows microphone permissions.",
                            "输入电平低于 -50 dBFS — 请检查硬件静音开关及 Windows 麦克风权限。",
                        )
                    } else if is_cable {
                        pick(
                            lang,
                            "Select CABLE Output as your microphone in Discord, Zoom, Teams, or OBS.",
                            "请在 Discord、腾讯会议、Zoom、Teams、OBS 或游戏语音中将麦克风设为 CABLE Output。",
                        )
                    } else {
                        pick(
                            lang,
                            "Direct listening active — wear headphones so speakers do not feed back into the mic.",
                            "当前为直接监听模式 — 请佩戴耳机，避免扬声器外放造成回声啸叫。",
                        )
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
        let lang = self.lang;
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
                            .child(section_label(pick(lang, "INFERENCE ENGINE", "推理引擎")))
                            .child(caption(match lang {
                                Language::En => {
                                    format!("Providers: {} | Models: {}", self.providers, self.models)
                                }
                                Language::ZhCn => {
                                    format!("可用后端：{} | 已装模型：{}", self.providers, self.models)
                                }
                            })),
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
                                segment_option(
                                    "backend-cpu",
                                    "CPU (int8)",
                                    cpu_selected,
                                    !busy,
                                )
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.choose_backend(Backend::Cpu);
                                    cx.notify();
                                })),
                            )
                            .child(
                                segment_option(
                                    "backend-cuda",
                                    if self.cuda {
                                        "CUDA (GPU)"
                                    } else {
                                        pick(lang, "CUDA (unavailable)", "CUDA（不可用）")
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
                    el.child(caption(pick(
                        lang,
                        "Only int8 is installed; add models/chunk.onnx (fp32) so LSTM layers run on CUDA instead of CPU.",
                        "当前仅安装了 int8 模型；放入 models/chunk.onnx (fp32) 后，LSTM 层才能在 CUDA GPU 上加速运行。",
                    )))
                },
            )
            .when(model_setup.active, |el| {
                el.child(caption(model_setup.message.clone().unwrap_or_else(|| {
                    pick(lang, "Downloading voice models…", "正在下载语音模型…").to_owned()
                })))
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
                            button(
                                "retry-model-download",
                                pick(lang, "Retry", "重试"),
                                Tone::Secondary,
                                true,
                            )
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
        let lang = self.lang;
        let hint = if busy {
            pick(
                lang,
                "Audio stays on this PC. Press Stop to release the microphone.",
                "所有音频仅在本机处理。点击“停止”即可释放麦克风。",
            )
        } else if model_setup.active {
            pick(
                lang,
                "Finishing first-run model check…",
                "正在完成首次模型检查…",
            )
        } else if model_setup.error.is_some() {
            pick(
                lang,
                "Retry the model download above to enable filtering.",
                "请点击上方的“重试”下载模型以启用人声过滤。",
            )
        } else if self.runtime_error.is_some() {
            pick(
                lang,
                "ONNX Runtime is unavailable in this build.",
                "当前运行环境中的 ONNX Runtime 不可用。",
            )
        } else if self.voice.is_none() {
            pick(
                lang,
                "Enroll your voice above to unlock Start filtering.",
                "请先在上方录制个人声纹以解锁开始过滤。",
            )
        } else if self.outputs.is_empty() {
            pick(
                lang,
                "Connect VB-CABLE or headphones, then refresh devices.",
                "请安装 VB-CABLE 或连接耳机后刷新设备列表。",
            )
        } else {
            pick(
                lang,
                "100% local processing, zero cloud uploads.",
                "100% 本地实时处理，零云端上传。",
            )
        };

        let primary_label = if busy {
            pick(lang, "Stop", "停止")
        } else if self.runtime_error.is_some() {
            pick(lang, "Unavailable", "不可用")
        } else if model_setup.active {
            pick(lang, "Preparing models…", "正在准备模型…")
        } else if !model_setup.ready {
            pick(lang, "Models missing", "缺少模型")
        } else if self.voice.is_none() {
            pick(lang, "Enroll voice first", "请先录制声纹")
        } else {
            pick(lang, "Start filtering", "开始过滤")
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
                                    format!("{hint} | {msg}")
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

fn lang_option(id: &'static str, label: &'static str, selected: bool) -> gpui::Stateful<gpui::Div> {
    div()
        .id(id)
        .h(px(28.))
        .px_2p5()
        .rounded(px(4.))
        .border_1()
        .border_color(rgb(if selected { BORDER_STRONG } else { SUBTLE }))
        .bg(rgb(if selected { CARD } else { SUBTLE }))
        .text_color(rgb(if selected { INK } else { MUTED }))
        .text_size(px(11.5))
        .font_weight(if selected {
            gpui::FontWeight::SEMIBOLD
        } else {
            gpui::FontWeight::MEDIUM
        })
        .cursor_pointer()
        .hover(move |s| s.text_color(rgb(INK)))
        .flex()
        .items_center()
        .justify_center()
        .child(label)
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
