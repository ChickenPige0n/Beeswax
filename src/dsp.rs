use std::collections::VecDeque;

pub const RATE: usize = 16_000;
pub const HOP: usize = 128;
pub const FFT: usize = 512;
pub const CHUNK: usize = 640;
const RESAMPLER_TAPS: usize = 64;
const RESAMPLER_PHASES: usize = 4096;

/// Streaming windowed-sinc resampling. Retains phase and filter history across callbacks.
pub struct Resampler {
    samples: VecDeque<f32>,
    position: f64,
    step: f64,
    coefficients: Vec<f32>,
}
impl Resampler {
    pub fn new(input: u32, output: u32) -> Self {
        let cutoff = (output as f64 / input as f64).min(1.) * 0.94;
        let mut coefficients = Vec::with_capacity(RESAMPLER_PHASES * RESAMPLER_TAPS);
        for phase in 0..RESAMPLER_PHASES {
            let fraction = phase as f64 / RESAMPLER_PHASES as f64;
            let row_start = coefficients.len();
            let mut norm = 0.;
            for tap in 0..RESAMPLER_TAPS {
                let x = tap as f64 - 31. - fraction;
                let a = std::f64::consts::PI * x * cutoff;
                let sinc = if a.abs() < 1e-9 { 1. } else { a.sin() / a };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * x / 32.).cos();
                let weight = sinc * window * cutoff;
                coefficients.push(weight as f32);
                norm += weight;
            }
            for weight in &mut coefficients[row_start..] {
                *weight /= norm as f32;
            }
        }
        Self {
            samples: VecDeque::from(vec![0.; 32]),
            position: 32.,
            step: input as f64 / output as f64,
            coefficients,
        }
    }
    pub fn push(&mut self, input: &[f32]) -> Vec<f32> {
        self.samples.extend(input.iter().copied());
        let mut out = Vec::with_capacity((input.len() as f64 / self.step).ceil() as usize);
        while self.position + 32. < self.samples.len() as f64 {
            let center = self.position.floor() as isize;
            let phase = ((self.position - center as f64) * RESAMPLER_PHASES as f64) as usize;
            let weights = &self.coefficients[phase * RESAMPLER_TAPS..(phase + 1) * RESAMPLER_TAPS];
            let start = (center - 31) as usize;
            let (front, back) = self.samples.as_slices();
            let front_count = if start < front.len() {
                (front.len() - start).min(RESAMPLER_TAPS)
            } else {
                0
            };
            let front_start = start.min(front.len());
            let back_start = start.saturating_sub(front.len());
            let mut sum = 0.;
            for (&sample, &weight) in front[front_start..front_start + front_count]
                .iter()
                .zip(&weights[..front_count])
            {
                sum += sample * weight;
            }
            for (&sample, &weight) in back[back_start..back_start + RESAMPLER_TAPS - front_count]
                .iter()
                .zip(&weights[front_count..])
            {
                sum += sample * weight;
            }
            out.push(sum);
            self.position += self.step;
        }
        let remove = (self.position.floor() as usize).saturating_sub(32);
        self.samples.drain(..remove);
        self.position -= remove as f64;
        out
    }
}

pub struct OverlapAdd {
    sum: Vec<f32>,
    weight: Vec<f32>,
    window: Vec<f32>,
}
impl Default for OverlapAdd {
    fn default() -> Self {
        Self::new()
    }
}
impl OverlapAdd {
    pub fn new() -> Self {
        Self {
            sum: vec![0.; FFT],
            weight: vec![0.; FFT],
            window: (0..FFT)
                .map(|i| {
                    (0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / FFT as f32).cos()).powi(2)
                })
                .collect(),
        }
    }
    pub fn push(&mut self, frames: &[f32]) -> Vec<f32> {
        let (frames, remainder) = frames.as_chunks::<FFT>();
        assert!(remainder.is_empty());
        let mut result = Vec::with_capacity(frames.len() * HOP);
        for frame in frames {
            for (i, &sample) in frame.iter().enumerate() {
                self.sum[i] += sample;
                self.weight[i] += self.window[i];
            }
            for i in 0..HOP {
                result.push(
                    -0.5 * self.sum[i]
                        / if self.weight[i] > 1e-11 {
                            self.weight[i]
                        } else {
                            1.
                        },
                );
            }
            self.sum.copy_within(HOP.., 0);
            self.weight.copy_within(HOP.., 0);
            self.sum[FFT - HOP..].fill(0.);
            self.weight[FFT - HOP..].fill(0.);
        }
        result
    }
}

/// Second-order Butterworth high-pass. Removes the sub-80 Hz rumble that laptop and
/// headset microphones pick up, which otherwise eats headroom and distracts the model.
pub struct HighPass {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl HighPass {
    pub fn new(rate: f32, cutoff: f32) -> Self {
        let w0 = std::f32::consts::TAU * cutoff / rate;
        let cos = w0.cos();
        let alpha = w0.sin() / (2. * std::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1. + alpha;
        Self {
            b0: (1. + cos) / 2. / a0,
            b1: -(1. + cos) / a0,
            b2: (1. + cos) / 2. / a0,
            a1: -2. * cos / a0,
            a2: (1. - alpha) / a0,
            x1: 0.,
            x2: 0.,
            y1: 0.,
            y2: 0.,
        }
    }

    pub fn process(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            let x = *sample;
            let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2;
            self.x2 = self.x1;
            self.x1 = x;
            self.y2 = self.y1;
            self.y1 = y;
            *sample = y;
        }
    }
}

pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.;
    }
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

pub fn enrollment(x: &[f32]) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        x.len() >= 3 * RATE,
        "Record at least three seconds of speech."
    );
    anyhow::ensure!(
        x.iter().all(|v| v.is_finite()),
        "Invalid microphone samples."
    );
    let size = 3 * RATE;
    let mut energy: f64 = x[..size].iter().map(|v| (*v as f64).powi(2)).sum();
    let mut best = (energy, 0);
    for i in size..x.len().min(30 * RATE) {
        energy += (x[i] as f64).powi(2) - (x[i - size] as f64).powi(2);
        if energy > best.0 {
            best = (energy, i + 1 - size);
        }
    }
    let sample = x[best.1..best.1 + size].to_vec();
    anyhow::ensure!(
        rms(&sample) > 0.003,
        "Recording is too quiet. Move closer and record again."
    );
    anyhow::ensure!(
        sample.iter().filter(|v| v.abs() >= 0.99).count() < size / 100,
        "Recording is clipping. Lower the microphone level and record again."
    );
    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resampling_is_independent_of_callback_boundaries() {
        let input: Vec<_> = (0..48000).map(|i| (i as f32 * 0.07).sin()).collect();
        let whole = Resampler::new(48000, 16000).push(&input);
        let mut r = Resampler::new(48000, 16000);
        let split: Vec<_> = input.chunks(137).flat_map(|c| r.push(c)).collect();
        assert_eq!(whole.len(), split.len());
        assert!(whole.iter().zip(split).all(|(a, b)| (a - b).abs() < 1e-5));
        assert!((whole.len() as i32 - 16000).abs() < 32);
    }
    #[test]
    fn upsampling_is_stable_across_callback_boundaries() {
        let input: Vec<_> = (0..16000).map(|i| (i as f32 * 0.07).sin()).collect();
        let whole = Resampler::new(16000, 48000).push(&input);
        let mut r = Resampler::new(16000, 48000);
        let split: Vec<_> = input.chunks(137).flat_map(|c| r.push(c)).collect();
        assert_eq!(whole.len(), split.len());
        let max_error = whole
            .iter()
            .zip(split)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        // Re-basing the stream phase can select an adjacent polyphase table row.
        assert!(max_error < 0.0001, "max error was {max_error}");
        assert!((whole.len() as i32 - 48000).abs() < 100);
    }
    #[test]
    fn resampling_rejects_out_of_band_audio() {
        let input: Vec<_> = (0..48000)
            .map(|i| (std::f32::consts::TAU * 12000. * i as f32 / 48000.).sin())
            .collect();
        let out = Resampler::new(48000, 16000).push(&input);
        assert!(rms(&out[100..]) < 0.01);
    }
    #[test]
    fn polyphase_resampling_matches_windowed_sinc_reference() {
        let input: Vec<_> = (0..4800)
            .map(|i| {
                let t = i as f32 / 48000.;
                0.2 * (std::f32::consts::TAU * 7300. * t).sin()
                    + 0.1 * (std::f32::consts::TAU * 900. * t).sin()
            })
            .collect();
        let actual = Resampler::new(48000, 16000).push(&input);
        let mut samples = VecDeque::from(vec![0.; 32]);
        samples.extend(input);
        let mut position: f64 = 32.;
        let mut expected = Vec::new();
        let cutoff = 16000. / 48000. * 0.94;
        while position + 32. < samples.len() as f64 {
            let center = position.floor() as isize;
            let mut sum = 0.;
            let mut norm = 0.;
            for i in center - 31..=center + 32 {
                let x = i as f64 - position;
                let a = std::f64::consts::PI * x * cutoff;
                let sinc = if a.abs() < 1e-9 { 1. } else { a.sin() / a };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * x / 32.).cos();
                let weight = sinc * window * cutoff;
                sum += samples[i as usize] as f64 * weight;
                norm += weight;
            }
            expected.push((sum / norm) as f32);
            position += 3.;
        }
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0., f32::max);
        assert!(max_error < 0.0001, "max error was {max_error}");
    }
    #[test]
    fn overlap_add_restores_windowed_signal_and_gain() {
        let mut ola = OverlapAdd::new();
        let frame: Vec<_> = ola.window.iter().map(|w| -2. * w).collect();
        for _ in 0..8 {
            let out = ola.push(&frame);
            assert!(out[1..].iter().all(|v| (v - 1.).abs() < 1e-5));
        }
    }
    #[test]
    fn high_pass_removes_rumble_and_keeps_speech() {
        let rate = 16000.;
        let mut rumble = vec![0.; 16000];
        let mut voice = vec![0.; 16000];
        for (i, (r, v)) in rumble.iter_mut().zip(voice.iter_mut()).enumerate() {
            let t = i as f32 / rate;
            *r = (std::f32::consts::TAU * 30. * t).sin();
            *v = (std::f32::consts::TAU * 1000. * t).sin();
        }
        let mut hp = HighPass::new(rate, 70.);
        hp.process(&mut rumble);
        let mut hp = HighPass::new(rate, 70.);
        hp.process(&mut voice);
        assert!(rms(&rumble[2000..]) < 0.2, "30 Hz should be attenuated");
        assert!(rms(&voice[2000..]) > 0.6, "1 kHz should pass");
    }

    #[test]
    fn enrollment_rejects_silence_and_clipping() {
        assert!(enrollment(&vec![0.; 48000]).is_err());
        assert!(enrollment(&vec![1.; 48000]).is_err());
        let mut x = vec![0.; 48000];
        x.extend(vec![0.1; 48000]);
        assert_eq!(enrollment(&x).unwrap(), vec![0.1; 48000]);
    }
}
