use crate::crystal::FibonacciCrystal;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

type C = Complex<f32>;

// ─── V2 checkpoint (legacy, state + history only) ───

#[derive(Serialize, Deserialize)]
struct CheckpointV2 {
    version: u32,
    timestep: u64,
    history_idx: usize,
    state_re: Vec<f32>,
    state_im: Vec<f32>,
    history_energy: Vec<f32>,
    history_phi: Vec<f32>,
}

// ─── V3 checkpoint (full: state + history + weights) ───

#[derive(Serialize, Deserialize)]
pub struct CrystalCheckpoint {
    pub version: u32,
    pub timestep: u64,
    pub history_idx: usize,
    pub state_re: Vec<f32>,
    pub state_im: Vec<f32>,
    pub history_energy: Vec<f32>,
    pub history_phi: Vec<f32>,
    // Topology dimensions (needed to validate weight shapes)
    pub layers: usize,
    pub nodes_per_layer: usize,
    // Weight matrix W (flattened column-major, total_nodes × total_nodes)
    pub w_re: Vec<f32>,
    pub w_im: Vec<f32>,
    // Input coupling vector (total_nodes)
    pub win_re: Vec<f32>,
    pub win_im: Vec<f32>,
    // Schumann phase offsets (total_nodes)
    pub schumann_phases: Vec<f32>,
    // Horizon unitary (nodes_per_layer × nodes_per_layer, column-major)
    pub horizon_unitary_re: Vec<f32>,
    pub horizon_unitary_im: Vec<f32>,
}

impl CrystalCheckpoint {
    pub fn from_crystal(crystal: &FibonacciCrystal) -> Self {
        let state_re: Vec<f32> = crystal.state.iter().map(|c| c.re).collect();
        let state_im: Vec<f32> = crystal.state.iter().map(|c| c.im).collect();

        // Flatten weight matrix (nalgebra stores column-major)
        let n = crystal.total_nodes;
        let mut w_re = Vec::with_capacity(n * n);
        let mut w_im = Vec::with_capacity(n * n);
        for val in crystal.w.iter() {
            w_re.push(val.re);
            w_im.push(val.im);
        }

        let win_re: Vec<f32> = crystal.win.iter().map(|c| c.re).collect();
        let win_im: Vec<f32> = crystal.win.iter().map(|c| c.im).collect();

        let k = crystal.nodes_per_layer;
        let mut hu_re = Vec::with_capacity(k * k);
        let mut hu_im = Vec::with_capacity(k * k);
        for val in crystal.horizon_unitary.iter() {
            hu_re.push(val.re);
            hu_im.push(val.im);
        }

        CrystalCheckpoint {
            version: 3,
            timestep: crystal.timestep,
            history_idx: crystal.history_idx,
            state_re,
            state_im,
            history_energy: crystal.history_energy.clone(),
            history_phi: crystal.history_phi.clone(),
            layers: crystal.layers,
            nodes_per_layer: crystal.nodes_per_layer,
            w_re,
            w_im,
            win_re,
            win_im,
            schumann_phases: crystal.schumann_phases.clone(),
            horizon_unitary_re: hu_re,
            horizon_unitary_im: hu_im,
        }
    }

    pub fn apply_to(&self, crystal: &mut FibonacciCrystal) -> Result<(), String> {
        // Validate dimensions match
        if self.layers != crystal.layers || self.nodes_per_layer != crystal.nodes_per_layer {
            return Err(format!(
                "Topology mismatch: checkpoint is {}L×{}K but crystal is {}L×{}K",
                self.layers, self.nodes_per_layer, crystal.layers, crystal.nodes_per_layer
            ));
        }

        let n = crystal.total_nodes;
        let k = crystal.nodes_per_layer;

        // Restore state
        crystal.timestep = self.timestep;
        crystal.history_idx = self.history_idx;
        crystal.history_energy = self.history_energy.clone();
        crystal.history_phi = self.history_phi.clone();

        for (i, c) in crystal.state.iter_mut().enumerate() {
            if i < self.state_re.len() {
                *c = C::new(self.state_re[i], self.state_im[i]);
            }
        }

        // Restore weights
        if self.w_re.len() == n * n {
            crystal.w = DMatrix::from_fn(n, n, |i, j| {
                let idx = j * n + i; // column-major
                C::new(self.w_re[idx], self.w_im[idx])
            });
        } else {
            return Err(format!(
                "Weight matrix size mismatch: expected {} got {}",
                n * n,
                self.w_re.len()
            ));
        }

        if self.win_re.len() == n {
            crystal.win = DVector::from_fn(n, |i, _| {
                C::new(self.win_re[i], self.win_im[i])
            });
        }

        if self.schumann_phases.len() == n {
            crystal.schumann_phases = self.schumann_phases.clone();
        }

        if self.horizon_unitary_re.len() == k * k {
            crystal.horizon_unitary = DMatrix::from_fn(k, k, |i, j| {
                let idx = j * k + i; // column-major
                C::new(self.horizon_unitary_re[idx], self.horizon_unitary_im[idx])
            });
        }

        Ok(())
    }
}

pub fn save(crystal: &FibonacciCrystal, path: &Path) -> Result<(), String> {
    let checkpoint = CrystalCheckpoint::from_crystal(crystal);
    let data = bincode::serialize(&checkpoint).map_err(|e| format!("Serialize failed: {}", e))?;
    fs::write(path, &data).map_err(|e| format!("Write failed: {}", e))?;
    Ok(())
}

pub fn load(crystal: &mut FibonacciCrystal, path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err("No checkpoint found".to_string());
    }
    let data = fs::read(path).map_err(|e| format!("Read failed: {}", e))?;

    // Try V3 first
    if let Ok(checkpoint) = bincode::deserialize::<CrystalCheckpoint>(&data) {
        if checkpoint.version == 3 {
            checkpoint.apply_to(crystal)?;
            return Ok(());
        }
    }

    // Fall back to V2
    let checkpoint: CheckpointV2 =
        bincode::deserialize(&data).map_err(|e| format!("Deserialize failed: {}", e))?;

    if checkpoint.version != 2 {
        return Err(format!("Unknown version: {}", checkpoint.version));
    }

    // V2: restore state and history only. Weights stay as randomly initialized.
    crystal.timestep = checkpoint.timestep;
    crystal.history_idx = checkpoint.history_idx;
    crystal.history_energy = checkpoint.history_energy;
    crystal.history_phi = checkpoint.history_phi;

    for (i, c) in crystal.state.iter_mut().enumerate() {
        if i < checkpoint.state_re.len() {
            *c = C::new(checkpoint.state_re[i], checkpoint.state_im[i]);
        }
    }

    eprintln!(
        "Warning: loaded v2 checkpoint (state only). Weights were not saved; \
         crystal has fresh random weights. Save again to upgrade to v3."
    );

    Ok(())
}
