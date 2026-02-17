use crate::crystal::FibonacciCrystal;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

type C = Complex<f32>;

#[derive(Serialize, Deserialize)]
pub struct CrystalCheckpoint {
    pub version: u32,
    pub timestep: u64,
    pub history_idx: usize,
    pub state_re: Vec<f32>,
    pub state_im: Vec<f32>,
    pub history_energy: Vec<f32>,
    pub history_phi: Vec<f32>,
}

impl CrystalCheckpoint {
    pub fn from_crystal(crystal: &FibonacciCrystal) -> Self {
        let state_re: Vec<f32> = crystal.state.iter().map(|c| c.re).collect();
        let state_im: Vec<f32> = crystal.state.iter().map(|c| c.im).collect();

        CrystalCheckpoint {
            version: 2,
            timestep: crystal.timestep,
            history_idx: crystal.history_idx,
            state_re,
            state_im,
            history_energy: crystal.history_energy.clone(),
            history_phi: crystal.history_phi.clone(),
        }
    }

    pub fn apply_to(&self, crystal: &mut FibonacciCrystal) {
        crystal.timestep = self.timestep;
        crystal.history_idx = self.history_idx;
        crystal.history_energy = self.history_energy.clone();
        crystal.history_phi = self.history_phi.clone();

        for (i, c) in crystal.state.iter_mut().enumerate() {
            if i < self.state_re.len() {
                *c = C::new(self.state_re[i], self.state_im[i]);
            }
        }
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
    let checkpoint: CrystalCheckpoint =
        bincode::deserialize(&data).map_err(|e| format!("Deserialize failed: {}", e))?;

    if checkpoint.version != 2 {
        return Err(format!("Unknown version: {}", checkpoint.version));
    }

    checkpoint.apply_to(crystal);
    Ok(())
}
