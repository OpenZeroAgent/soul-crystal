use crate::crystal::FibonacciCrystal;
use num_complex::Complex;
use serde::Serialize;
use std::f32::consts::PI;

pub type C = Complex<f32>;

#[derive(Debug, Clone, Serialize)]
pub struct Emotions {
    pub energy: f32,
    pub energy_concentration: f32,
    pub phase_coherence: f32,
    pub depth: f32,
    pub ring_coherence: f32,
    pub spectral_richness: f32,
    pub arousal: f32,
    pub horizon_activity: f32,
}

impl Emotions {
    pub fn from_crystal(crystal: &FibonacciCrystal) -> Self {
        let state = &crystal.state;
        let n = crystal.total_nodes;
        let l = crystal.layers;
        let k = crystal.nodes_per_layer;

        // 1. Phase coherence
        let unit_sum: C = state
            .iter()
            .map(|c| {
                let norm = c.norm();
                if norm > 1e-9 { c / norm } else { C::new(0.0, 0.0) }
            })
            .sum();
        let phase_coherence = (unit_sum / C::new(n as f32, 0.0)).norm();

        // 2. Energy stats
        let mags: Vec<f32> = state.iter().map(|c| c.norm()).collect();
        let energy: f32 = mags.iter().sum::<f32>() / n as f32;
        let energy_std = {
            let var: f32 = mags.iter().map(|m| (m - energy).powi(2)).sum::<f32>() / n as f32;
            var.sqrt()
        };

        // 3. Layer energy gradient
        let mut layer_energies = vec![0.0f32; l];
        for layer in 0..l {
            let start = layer * k;
            let layer_e: f32 = (0..k).map(|j| mags[start + j]).sum::<f32>() / k as f32;
            layer_energies[layer] = layer_e;
        }
        let deep_energy: f32 = layer_energies[..l / 2].iter().sum::<f32>() / (l / 2) as f32;
        let surface_energy: f32 = layer_energies[l / 2..].iter().sum::<f32>() / (l - l / 2) as f32;
        let depth = (deep_energy - surface_energy) / (deep_energy + surface_energy + 1e-9);

        // 4. Ring coherence
        let mut ring_coh_sum = 0.0f32;
        for layer in 0..l {
            let start = layer * k;
            let ring_unit_sum: C = (0..k)
                .map(|j| {
                    let c = state[start + j];
                    let norm = c.norm();
                    if norm > 1e-9 { c / norm } else { C::new(0.0, 0.0) }
                })
                .sum();
            ring_coh_sum += (ring_unit_sum / C::new(k as f32, 0.0)).norm();
        }
        let ring_coherence = ring_coh_sum / l as f32;

        // 5. Spectral richness (entropy of FFT power spectrum)
        // Simple DFT for the state magnitudes
        let spectral_richness = spectral_entropy_norm(&mags);

        // 6. Arousal (from history ring buffer)
        let n_samples = crystal.history_idx.min(16);
        let arousal = if n_samples > 1 {
            let recent_phi: Vec<f32> = (0..n_samples)
                .map(|i| crystal.history_phi[(crystal.history_idx - 1 - i) % 64])
                .collect();
            let recent_energy: Vec<f32> = (0..n_samples)
                .map(|i| crystal.history_energy[(crystal.history_idx - 1 - i) % 64])
                .collect();
            (std_dev(&recent_phi) + std_dev(&recent_energy)) * 50.0
        } else {
            0.0
        };

        // 7. Horizon activity
        let horizon_energy: f32 = (0..k)
            .map(|j| mags[crystal.horizon_start + j])
            .sum::<f32>()
            / k as f32;
        let horizon_activity = horizon_energy / (energy + 1e-9);

        // 8. Gini coefficient (energy concentration)
        let mut sorted_mags = mags.clone();
        sorted_mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_f = n as f32;
        let gini_num: f32 = sorted_mags
            .iter()
            .enumerate()
            .map(|(i, m)| (i as f32 + 1.0) * m)
            .sum::<f32>();
        let gini_denom: f32 = sorted_mags.iter().sum::<f32>();
        let gini = if gini_denom > 1e-9 {
            2.0 * gini_num / (n_f * gini_denom) - (n_f + 1.0) / n_f
        } else {
            0.0
        };

        Emotions {
            energy,
            energy_concentration: gini,
            phase_coherence,
            depth,
            ring_coherence,
            spectral_richness,
            arousal,
            horizon_activity,
        }
    }

    pub fn vibe_string(&self) -> String {
        let mut moods: Vec<&str> = Vec::new();

        if self.phase_coherence > 0.10 {
            moods.push("Coherent");
        } else if self.phase_coherence > 0.06 {
            moods.push("Aligned");
        }
        if self.arousal > 1.0 {
            moods.push("Excited");
        } else if self.arousal > 0.3 {
            moods.push("Active");
        }
        if self.depth > 0.05 {
            moods.push("Deep");
        } else if self.depth < -0.08 {
            moods.push("Surface");
        }
        if self.energy_concentration > 0.35 {
            moods.push("Focused");
        } else if self.energy_concentration < 0.25 {
            moods.push("Diffuse");
        }
        if self.ring_coherence > 0.35 {
            moods.push("Harmonic");
        }
        if self.spectral_richness > 0.96 {
            moods.push("Complex");
        } else if self.spectral_richness < 0.88 {
            moods.push("Ordered");
        }
        if self.horizon_activity > 1.3 {
            moods.push("Scrambling");
        }

        let mood = match moods.len() {
            0 => "Resonant".to_string(),
            1 => moods[0].to_string(),
            _ => format!("{}-{}", moods[0], moods[1]),
        };

        format!(
            "{} (Φ={:.3}, C={:.3}, D={:+.3}, R={:.3})",
            mood, self.phase_coherence, self.energy_concentration, self.depth, self.spectral_richness
        )
    }
}

fn std_dev(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    let var: f32 = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    var.sqrt()
}

fn spectral_entropy_norm(signal: &[f32]) -> f32 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }

    // Compute DFT power spectrum
    let mut power = vec![0.0f32; n];
    for k in 0..n {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (j, &s) in signal.iter().enumerate() {
            let angle = -2.0 * PI * k as f32 * j as f32 / n as f32;
            re += s * angle.cos();
            im += s * angle.sin();
        }
        power[k] = re * re + im * im;
    }

    let total: f32 = power.iter().sum();
    if total < 1e-9 {
        return 0.0;
    }

    let entropy: f32 = power
        .iter()
        .map(|p| {
            let pn = p / total;
            if pn > 1e-9 {
                -pn * pn.ln()
            } else {
                0.0
            }
        })
        .sum();

    let max_entropy = (n as f32).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}
