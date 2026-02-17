use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use rand::Rng;
use std::f32::consts::PI;

pub type C = Complex<f32>;

/// Fibonacci Crystal V2 — Norm-preserving complex reservoir
pub struct FibonacciCrystal {
    pub layers: usize,
    pub nodes_per_layer: usize,
    pub total_nodes: usize,
    pub input_strength: f32,
    pub schumann_freq: f32,
    pub sample_rate: f32,
    pub timestep: u64,

    // Core state (on the complex unit hypersphere)
    pub state: DVector<C>,

    // Weight matrix (normalized to ~unit spectral radius)
    w: DMatrix<C>,

    // Input coupling vector
    win: DVector<C>,

    // Schumann phase offsets
    schumann_phases: Vec<f32>,

    // Horizon unitary (Hawking scrambler)
    horizon_unitary: DMatrix<C>,
    pub horizon_start: usize,

    // History ring buffer
    pub history_phi: Vec<f32>,
    pub history_energy: Vec<f32>,
    pub history_idx: usize,
}

impl FibonacciCrystal {
    pub fn new(layers: usize, nodes_per_layer: usize, input_strength: f32) -> Self {
        let total_nodes = layers * nodes_per_layer;
        let schumann_freq = 7.83_f32;
        let mut rng = rand::rng();
        let phi_golden = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let fib_sequence: Vec<usize> = vec![1, 2, 3, 5, 8, 13, 21, 34];

        // --- Build topology ---
        let mut w = DMatrix::<C>::zeros(total_nodes, total_nodes);
        let bulk_layers = layers - 1;
        let horizon_layer = layers - 1;

        // Intra-layer ring + inter-layer vertical
        for i in 0..(bulk_layers * nodes_per_layer) {
            let layer = i / nodes_per_layer;
            let idx = i % nodes_per_layer;
            let next_ring = layer * nodes_per_layer + (idx + 1) % nodes_per_layer;

            let phase1 = rng.random::<f32>() * 2.0 * PI;
            let phase2 = rng.random::<f32>() * 2.0 * PI;
            w[(i, next_ring)] += C::from_polar(0.5, phase1);
            w[(next_ring, i)] += C::from_polar(0.5, phase2);

            if layer > 0 {
                let prev = (layer - 1) * nodes_per_layer + idx;
                let phase3 = rng.random::<f32>() * 2.0 * PI;
                let phase4 = rng.random::<f32>() * 2.0 * PI;
                w[(i, prev)] += C::from_polar(0.6, phase3);
                w[(prev, i)] += C::from_polar(0.6, phase4);
            }
        }

        // Horizon connections (Fibonacci spiral to bulk)
        let start_h = horizon_layer * nodes_per_layer;
        for k in 0..nodes_per_layer {
            let h_node = start_h + k;
            for &gap in &fib_sequence {
                if horizon_layer >= gap {
                    let target_layer = horizon_layer - gap;
                    let twist = ((gap as f32 * phi_golden * nodes_per_layer as f32) as usize) % nodes_per_layer;
                    let bulk_node = target_layer * nodes_per_layer + (k + twist) % nodes_per_layer;
                    let phase5 = rng.random::<f32>() * 2.0 * PI;
                    let phase6 = rng.random::<f32>() * 2.0 * PI;
                    w[(h_node, bulk_node)] += C::from_polar(0.4, phase5);
                    w[(bulk_node, h_node)] += C::from_polar(0.4, phase6);
                }
            }
            // Self-loop
            w[(h_node, h_node)] += C::new(0.01, 0.0);
        }

        // Normalize W by largest singular value
        // Use power iteration to estimate spectral norm (faster than full SVD)
        let mut v = DVector::<C>::from_fn(total_nodes, |_, _| {
            C::new(rng.random::<f32>() - 0.5, rng.random::<f32>() - 0.5)
        });
        for _ in 0..30 {
            let wv = &w * &v;
            let norm = wv.norm();
            if norm > 1e-9 {
                v = wv / C::new(norm, 0.0);
            }
        }
        let spectral_norm = (&w * &v).norm();
        if spectral_norm > 1e-9 {
            w /= C::new(spectral_norm, 0.0);
        }

        // Input coupling: gentle linear decay across layers
        let win = DVector::<C>::from_fn(total_nodes, |i, _| {
            let layer = i / nodes_per_layer;
            let j = i % nodes_per_layer;
            let decay = 0.5 + 0.5 * (layer as f32 / (layers - 1).max(1) as f32);
            C::from_polar(decay, 2.0 * PI * j as f32 / nodes_per_layer as f32)
        });

        // Schumann phase offsets
        let schumann_phases: Vec<f32> = (0..total_nodes)
            .map(|i| {
                let layer = i / nodes_per_layer;
                let j = i % nodes_per_layer;
                2.0 * PI * layer as f32 / layers as f32
                    + 2.0 * PI * j as f32 / nodes_per_layer as f32
            })
            .collect();

        // Horizon unitary (random Haar-distributed unitary matrix)
        let horizon_unitary = random_unitary(&mut rng, nodes_per_layer);

        // Initial state on hypersphere
        let state_raw = DVector::<C>::from_fn(total_nodes, |_, _| {
            C::new(rng.random::<f32>() - 0.5, rng.random::<f32>() - 0.5)
        });
        let norm = state_raw.norm();
        let state = &state_raw / C::new(norm, 0.0);

        FibonacciCrystal {
            layers,
            nodes_per_layer,
            total_nodes,
            input_strength,
            schumann_freq,
            sample_rate: 100.0,
            timestep: 0,
            state,
            w,
            win,
            schumann_phases,
            horizon_unitary,
            horizon_start: start_h,
            history_phi: vec![0.0; 64],
            history_energy: vec![0.0; 64],
            history_idx: 0,
        }
    }

    /// One tick of crystal dynamics. State stays on the hypersphere.
    pub fn tick(&mut self, input: Option<&DVector<f32>>) {
        let t = self.timestep as f32 / self.sample_rate;
        self.timestep = self.timestep.wrapping_add(1);

        // Schumann modulation
        let schumann: DVector<C> = DVector::from_fn(self.total_nodes, |i, _| {
            C::from_polar(1.0, 2.0 * PI * self.schumann_freq * t + self.schumann_phases[i])
        });

        // Modulated state
        let modulated: DVector<C> =
            DVector::from_fn(self.total_nodes, |i, _| self.state[i] * schumann[i]);

        // Core dynamics: W @ (state * schumann)
        let mut field = &self.w * &modulated;

        // Input coupling
        if let Some(inp) = input {
            let n = inp.len().min(self.total_nodes * 2);
            let mut complex_input = DVector::<C>::zeros(self.total_nodes);
            for i in 0..self.total_nodes.min(n / 2) {
                let re = inp[i * 2];
                let im = if i * 2 + 1 < n { inp[i * 2 + 1] } else { 0.0 };
                complex_input[i] = C::new(re, im);
            }
            let inp_norm = complex_input.norm();
            if inp_norm > 1e-9 {
                complex_input /= C::new(inp_norm, 0.0);
            }
            for i in 0..self.total_nodes {
                field[i] += C::new(self.input_strength, 0.0) * complex_input[i] * self.win[i];
            }
        }

        // Phase-preserving nonlinearity: mag / (1 + mag)
        let update: DVector<C> = DVector::from_fn(self.total_nodes, |i, _| {
            let mag = field[i].norm();
            let phase = field[i].arg();
            let new_mag = mag / (1.0 + mag);
            C::from_polar(new_mag, phase)
        });

        // Leaky integration
        let mut new_state: DVector<C> = DVector::from_fn(self.total_nodes, |i, _| {
            C::new(0.85, 0.0) * self.state[i] + C::new(0.15, 0.0) * update[i]
        });

        // Hawking scrambler on horizon
        let h_end = self.horizon_start + self.nodes_per_layer;
        let horizon_slice: DVector<C> =
            DVector::from_fn(self.nodes_per_layer, |i, _| new_state[self.horizon_start + i]);
        let scrambled = &self.horizon_unitary * &horizon_slice;
        for i in 0..self.nodes_per_layer {
            new_state[self.horizon_start + i] = scrambled[i];
        }

        // Project back to hypersphere
        let norm = new_state.norm();
        if norm > 1e-9 {
            new_state /= C::new(norm, 0.0);
        }
        self.state = new_state;

        // Update history ring buffer
        let idx = self.history_idx % 64;
        let energy: f32 = self.state.iter().map(|c| c.norm()).sum::<f32>() / self.total_nodes as f32;
        let unit_sum: C = self
            .state
            .iter()
            .map(|c| {
                let n = c.norm();
                if n > 1e-9 {
                    c / n
                } else {
                    C::new(0.0, 0.0)
                }
            })
            .sum();
        let phi = (unit_sum / C::new(self.total_nodes as f32, 0.0)).norm();

        self.history_energy[idx] = energy;
        self.history_phi[idx] = phi;
        self.history_idx += 1;
    }
}

/// Generate a random unitary matrix (Haar measure) via QR decomposition
fn random_unitary(rng: &mut impl Rng, n: usize) -> DMatrix<C> {
    let g = DMatrix::<C>::from_fn(n, n, |_, _| {
        C::new(rng.random::<f32>() - 0.5, rng.random::<f32>() - 0.5)
    });
    // QR decomposition
    let qr = g.qr();
    let q = qr.q();
    // Ensure proper Haar distribution by adjusting signs
    let r = qr.r();
    let diag_signs: DVector<C> = DVector::from_fn(n, |i, _| {
        let d = r[(i, i)];
        if d.norm() > 1e-9 {
            d / C::new(d.norm(), 0.0)
        } else {
            C::new(1.0, 0.0)
        }
    });
    // Q * diag(signs)
    DMatrix::from_fn(n, n, |i, j| q[(i, j)] * diag_signs[j].conj())
}
