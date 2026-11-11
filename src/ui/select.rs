use super::theme::{BORDER, BORDER_STRONG, CARD, GREEN, GREEN_SOFT, INK, MUTED};
use super::{Beeswax, SELECT_MIC, SELECT_OUTPUT, SELECT_SPEAKER};
use gpui::{Context, KeyDownEvent, deferred, div, prelude::*, px, rgb};

impl Beeswax {
    /// A device picker whose list is painted as an overlay, so opening it never moves
    /// the surrounding controls.
    pub(super) fn device_select(
        &self,
        kind: u8,
        cx: &mut Context<Self>,
    ) -> impl IntoElement + use<> {
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
                    if device.is_default { " · Default" } else { "" }
                )
            })
            .unwrap_or_else(|| "No device available".into());

        let options = devices
            .iter()
            .enumerate()
            .map(|(index, device)| {
                let is_selected = index == selected;
                let text = format!(
                    "{}{}",
                    device.name,
                    if device.is_default { " · Default" } else { "" }
                );
                div()
                    .id(("device-option", kind as usize * 1000 + index))
                    .px_2p5()
                    .py_1p5()
                    .rounded_sm()
                    .text_size(px(12.5))
                    .text_color(rgb(if is_selected { GREEN } else { INK }))
                    .font_weight(if is_selected {
                        gpui::FontWeight::MEDIUM
                    } else {
                        gpui::FontWeight::NORMAL
                    })
                    .cursor_pointer()
                    .bg(rgb(if is_selected { GREEN_SOFT } else { CARD }))
                    .hover(|style| style.bg(rgb(0xeff3ec)))
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        div()
                            .w(px(12.))
                            .text_size(px(11.))
                            .text_color(rgb(GREEN))
                            .child(if is_selected { "✓" } else { "" }),
                    )
                    .child(div().flex_1().min_w(px(0.)).truncate().child(text))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        if this.job.is_none() {
                            match kind {
                                SELECT_MIC => this.input = index,
                                SELECT_OUTPUT => this.output = index,
                                SELECT_SPEAKER => this.speaker = index,
                                _ => {}
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
            .min_w(px(0.))
            .flex()
            .flex_col()
            .child(
                div()
                    .id(("device-select", kind as usize))
                    .track_focus(&self.select_focus[kind as usize])
                    .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
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
                    .h(px(38.))
                    .px_3()
                    .rounded_md()
                    .border_1()
                    .border_color(rgb(if open { GREEN } else { BORDER_STRONG }))
                    .bg(rgb(CARD))
                    .opacity(if busy { 0.55 } else { 1. })
                    .when(!busy, |el| {
                        el.cursor_pointer()
                            .hover(|s| s.border_color(rgb(if open { GREEN } else { 0x9fb0a2 })))
                    })
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.))
                            .text_size(px(12.5))
                            .text_color(rgb(INK))
                            .truncate()
                            .child(label),
                    )
                    .child(
                        div()
                            .text_size(px(10.))
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
                let upwards = kind == SELECT_SPEAKER;
                element.child(
                    deferred(
                        div()
                            .id(("device-options", kind as usize))
                            .absolute()
                            .when(upwards, |list| list.bottom_10())
                            .when(!upwards, |list| list.top_10())
                            .left_0()
                            .right_0()
                            .p_1()
                            .max_h(px(220.))
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
