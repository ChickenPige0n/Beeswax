use gpui::{SharedString, div, prelude::*, px, rgb};

pub const BG: u32 = 0xf4f5f0;
pub const SURFACE: u32 = 0xfafbf7;
pub const CARD: u32 = 0xffffff;
pub const SUBTLE: u32 = 0xedf0e8;
pub const SUBTLE_HOVER: u32 = 0xe2e7db;
pub const BORDER: u32 = 0xdde3d6;
pub const BORDER_STRONG: u32 = 0xc6d0bd;
pub const INK: u32 = 0x1a241e;
pub const INK_SOFT: u32 = 0x3b4940;
pub const MUTED: u32 = 0x627067;

pub const GREEN: u32 = 0x1f5a42;
pub const GREEN_DARK: u32 = 0x164532;
pub const GREEN_SOFT: u32 = 0xe6f0ea;
pub const GREEN_METER: u32 = 0x35825d;

pub const AMBER: u32 = 0x945f12;
pub const AMBER_SOFT: u32 = 0xf8ecd6;
pub const BLUE: u32 = 0x2b4d78;
pub const BLUE_SOFT: u32 = 0xe8eff8;
pub const RED: u32 = 0x993222;
pub const RED_DARK: u32 = 0x7d2619;
pub const RED_SOFT: u32 = 0xf9ebe8;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    Primary,
    Danger,
    Secondary,
    Ghost,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PillTone {
    Neutral,
    Active,
    Recording,
    Info,
    Danger,
}

pub fn caption(text: impl Into<SharedString>) -> gpui::Div {
    div()
        .text_size(px(12.))
        .line_height(px(17.))
        .text_color(rgb(MUTED))
        .child(text.into())
}

pub fn section_label(text: &'static str) -> gpui::Div {
    div()
        .text_size(px(11.))
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .text_color(rgb(MUTED))
        .child(text)
}

pub fn pill(text: impl Into<SharedString>, tone: PillTone) -> gpui::Div {
    let (bg, fg, dot) = match tone {
        PillTone::Neutral => (SUBTLE, MUTED, 0x8b988f),
        PillTone::Active => (GREEN_SOFT, GREEN, GREEN_METER),
        PillTone::Recording => (AMBER_SOFT, AMBER, 0xc7811b),
        PillTone::Info => (BLUE_SOFT, BLUE, 0x4570a8),
        PillTone::Danger => (RED_SOFT, RED, RED),
    };
    div()
        .px_2p5()
        .py_1()
        .rounded_full()
        .bg(rgb(bg))
        .text_color(rgb(fg))
        .flex()
        .items_center()
        .gap_1p5()
        .child(div().size(px(6.)).rounded_full().bg(rgb(dot)))
        .child(
            div()
                .text_size(px(11.))
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(text.into()),
        )
}

pub fn button(
    id: &'static str,
    text: impl Into<SharedString>,
    tone: Tone,
    enabled: bool,
) -> gpui::Stateful<gpui::Div> {
    let (bg, fg, hover_bg, border) = match tone {
        Tone::Primary => (GREEN, 0xffffff, GREEN_DARK, GREEN),
        Tone::Danger => (RED, 0xffffff, RED_DARK, RED),
        Tone::Secondary => (CARD, INK, SUBTLE, BORDER_STRONG),
        Tone::Ghost => (SUBTLE, INK_SOFT, SUBTLE_HOVER, SUBTLE),
    };
    div()
        .id(id)
        .h(px(34.))
        .px_3()
        .rounded_md()
        .border_1()
        .border_color(rgb(border))
        .bg(rgb(bg))
        .text_color(rgb(fg))
        .text_size(px(12.5))
        .font_weight(gpui::FontWeight::MEDIUM)
        .opacity(if enabled { 1. } else { 0.48 })
        .flex()
        .items_center()
        .justify_center()
        .when(enabled, move |el| {
            el.cursor_pointer().hover(move |s| s.bg(rgb(hover_bg)))
        })
        .child(text.into())
}

pub fn progress_bar(fraction: f32, color: u32) -> gpui::Div {
    div()
        .h(px(6.))
        .w_full()
        .rounded_full()
        .bg(rgb(0xe3e8dd))
        .overflow_hidden()
        .child(
            div()
                .h_full()
                .w(gpui::relative(fraction.clamp(0., 1.)))
                .rounded_full()
                .bg(rgb(color)),
        )
}

/// Segmented audio meter with smooth visual density and no layout jitter.
pub fn level_meter(fraction: f32, active: bool, accent: u32) -> gpui::Div {
    const SEGMENTS: usize = 24;
    let clamped = if active { fraction.clamp(0., 1.) } else { 0. };
    let lit = (clamped * SEGMENTS as f32).round() as usize;
    let mut row = div().h(px(8.)).w_full().flex().items_center().gap(px(2.));
    for idx in 0..SEGMENTS {
        let on = idx < lit;
        let seg_color = if !on {
            0xe3e8dd
        } else if idx >= 20 {
            AMBER
        } else {
            accent
        };
        row = row.child(div().flex_1().h_full().rounded(px(2.)).bg(rgb(seg_color)));
    }
    row
}

/// Compact 32-bar waveform summary computed from the 3-second enrollment clip.
pub fn waveform_peaks(samples: &[f32]) -> Vec<f32> {
    const BARS: usize = 36;
    if samples.is_empty() {
        return Vec::new();
    }
    let chunk_len = (samples.len() / BARS).max(1);
    let mut raw = Vec::with_capacity(BARS);
    let mut max_peak = 1e-4f32;
    for chunk in samples.chunks(chunk_len).take(BARS) {
        let rms = (chunk.iter().map(|v| v * v).sum::<f32>() / chunk.len() as f32).sqrt();
        max_peak = max_peak.max(rms);
        raw.push(rms);
    }
    raw.into_iter()
        .map(|v| (v / max_peak).clamp(0.12, 1.0))
        .collect()
}

pub fn waveform_strip(peaks: &[f32], playing: bool) -> gpui::Div {
    let color = if playing { BLUE } else { GREEN_METER };
    let mut row = div()
        .h(px(28.))
        .w_full()
        .px_2p5()
        .py_1()
        .rounded_md()
        .bg(rgb(SURFACE))
        .border_1()
        .border_color(rgb(BORDER))
        .flex()
        .items_center()
        .gap(px(3.));
    for &height in peaks {
        row = row.child(
            div()
                .flex_1()
                .h(gpui::relative(height))
                .rounded_full()
                .bg(rgb(color)),
        );
    }
    row
}

pub fn metric_cell(label: &'static str, value: impl Into<SharedString>) -> gpui::Div {
    div()
        .flex_1()
        .px_3()
        .py_2()
        .rounded_md()
        .bg(rgb(CARD))
        .border_1()
        .border_color(rgb(BORDER))
        .flex()
        .flex_col()
        .gap(px(2.))
        .child(
            div()
                .text_size(px(10.5))
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(rgb(MUTED))
                .child(label),
        )
        .child(
            div()
                .text_size(px(13.5))
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(rgb(INK))
                .truncate()
                .child(value.into()),
        )
}

pub fn banner(text: impl Into<SharedString>, is_error: bool) -> gpui::Div {
    let (bg, fg, border) = if is_error {
        (RED_SOFT, RED, 0xebcfc9)
    } else {
        (AMBER_SOFT, AMBER, 0xead5b0)
    };
    div()
        .px_3p5()
        .py_2p5()
        .rounded_md()
        .border_1()
        .border_color(rgb(border))
        .bg(rgb(bg))
        .text_color(rgb(fg))
        .text_size(px(12.))
        .line_height(px(17.))
        .child(text.into())
}
