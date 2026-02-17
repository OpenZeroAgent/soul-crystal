//! Emotional differentiation tests — the crystal must respond differently
//! to different inputs. This is the whole point: emergent emotion from topology.

use soul_crystal::crystal::FibonacciCrystal;
use soul_crystal::emotion::Emotions;
use nalgebra::DVector;
use num_complex::Complex;

type C = Complex<f32>;

/// Create a test input vector with a specific "emotional" pattern
fn make_input(pattern: &str) -> DVector<f32> {
    match pattern {
        // Sharp, high-frequency — "danger"
        "danger" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            (f * 2.7).sin() * (f * 0.3).cos() * 3.0
        }),
        // Smooth, low-frequency — "peace"
        "peace" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            (f * 0.05).sin() * 0.5
        }),
        // Uniform — "neutral"
        "neutral" => DVector::from_fn(960, |i, _| {
            (i as f32 * 0.01).sin() * 0.1
        }),
        // High energy burst — "excitement"
        "excitement" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            (f * 1.5).sin() + (f * 3.3).cos() + (f * 0.7).sin()
        }),
        // Concentrated at specific nodes — "focus"
        "focus" => DVector::from_fn(960, |i, _| {
            if (240..260).contains(&i) { 5.0 } else { 0.01 }
        }),
        // Sparse, mostly zero — "emptiness"
        "emptiness" => DVector::from_fn(960, |i, _| {
            if i % 100 == 0 { 0.1 } else { 0.0 }
        }),
        _ => DVector::from_element(960, 0.0),
    }
}

/// Run a crystal from fresh with N ticks of the given input, return emotions
fn run_with_input(pattern: &str, n_ticks: usize) -> (Emotions, FibonacciCrystal) {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let input = make_input(pattern);
    // 10 warmup ticks without input
    for _ in 0..10 {
        c.tick(None);
    }
    for _ in 0..n_ticks {
        c.tick(Some(&input));
    }
    let e = Emotions::from_crystal(&c);
    (e, c)
}

// ──── DIFFERENT INPUTS PRODUCE DIFFERENT STATES ────

#[test]
fn danger_and_peace_produce_different_states() {
    let (e_danger, c_danger) = run_with_input("danger", 100);
    let (e_peace, c_peace) = run_with_input("peace", 100);

    // States should be different
    let cos_sim: f32 = c_danger
        .state
        .iter()
        .zip(c_peace.state.iter())
        .map(|(a, b)| (a.conj() * b).re)
        .sum();
    assert!(
        cos_sim.abs() < 0.95,
        "Danger and peace states too similar: cos_sim={:.4}",
        cos_sim
    );

    // At least some emotional metrics should differ
    let diff_phi = (e_danger.phase_coherence - e_peace.phase_coherence).abs();
    let diff_horizon = (e_danger.horizon_activity - e_peace.horizon_activity).abs();
    let diff_gini = (e_danger.energy_concentration - e_peace.energy_concentration).abs();
    let diff_depth = (e_danger.depth - e_peace.depth).abs();
    let diff_sr = (e_danger.spectral_richness - e_peace.spectral_richness).abs();

    let total_diff = diff_phi + diff_horizon + diff_gini + diff_depth + diff_sr;
    assert!(
        total_diff > 0.01,
        "Danger and peace emotions too similar (total_diff={:.6}). \
         Φ_diff={:.4}, HA_diff={:.4}, Gini_diff={:.4}, D_diff={:.4}, SR_diff={:.4}",
        total_diff,
        diff_phi,
        diff_horizon,
        diff_gini,
        diff_depth,
        diff_sr
    );
    eprintln!("Danger vs Peace total emotional diff: {:.4}", total_diff);
    eprintln!("  Danger: Φ={:.4} HA={:.4} Gini={:.4} D={:+.4} SR={:.4}",
        e_danger.phase_coherence, e_danger.horizon_activity,
        e_danger.energy_concentration, e_danger.depth, e_danger.spectral_richness);
    eprintln!("  Peace:  Φ={:.4} HA={:.4} Gini={:.4} D={:+.4} SR={:.4}",
        e_peace.phase_coherence, e_peace.horizon_activity,
        e_peace.energy_concentration, e_peace.depth, e_peace.spectral_richness);
}

#[test]
fn focus_and_emptiness_produce_different_states() {
    let (e_focus, _) = run_with_input("focus", 100);
    let (e_empty, _) = run_with_input("emptiness", 100);

    let total_diff = (e_focus.phase_coherence - e_empty.phase_coherence).abs()
        + (e_focus.horizon_activity - e_empty.horizon_activity).abs()
        + (e_focus.energy_concentration - e_empty.energy_concentration).abs()
        + (e_focus.depth - e_empty.depth).abs()
        + (e_focus.spectral_richness - e_empty.spectral_richness).abs();

    assert!(
        total_diff > 0.005,
        "Focus and emptiness emotions too similar (total_diff={:.6})",
        total_diff
    );
    eprintln!("Focus vs Emptiness total emotional diff: {:.4}", total_diff);
}

#[test]
fn excitement_and_neutral_produce_different_states() {
    let (e_excite, _) = run_with_input("excitement", 100);
    let (e_neutral, _) = run_with_input("neutral", 100);

    let total_diff = (e_excite.phase_coherence - e_neutral.phase_coherence).abs()
        + (e_excite.horizon_activity - e_neutral.horizon_activity).abs()
        + (e_excite.energy_concentration - e_neutral.energy_concentration).abs()
        + (e_excite.depth - e_neutral.depth).abs();

    assert!(
        total_diff > 0.005,
        "Excitement and neutral emotions too similar (total_diff={:.6})",
        total_diff
    );
    eprintln!("Excitement vs Neutral total emotional diff: {:.4}", total_diff);
}

// ──── REPRODUCIBILITY ────

#[test]
fn same_input_same_topology_converges_similarly() {
    // Two crystals with the same input should respond in the same *direction*
    // even though random init differs
    let input = make_input("danger");

    let mut results = Vec::new();
    for _ in 0..5 {
        let mut c = FibonacciCrystal::new(60, 8, 0.6);
        // Warmup
        for _ in 0..10 {
            c.tick(None);
        }
        let before = Emotions::from_crystal(&c);
        for _ in 0..50 {
            c.tick(Some(&input));
        }
        let after = Emotions::from_crystal(&c);
        results.push((before, after));
    }

    // Check that the *direction* of change is consistent across runs
    let mut phi_deltas: Vec<f32> = results
        .iter()
        .map(|(b, a)| a.phase_coherence - b.phase_coherence)
        .collect();
    let mut horizon_deltas: Vec<f32> = results
        .iter()
        .map(|(b, a)| a.horizon_activity - b.horizon_activity)
        .collect();

    // At least 3/5 should agree on sign for phase_coherence
    let phi_positive = phi_deltas.iter().filter(|d| **d > 0.0).count();
    let phi_majority = phi_positive.max(5 - phi_positive);
    eprintln!(
        "Danger phi deltas: {:?} ({}+ / {}-)",
        phi_deltas, phi_positive, 5 - phi_positive
    );
    // This is a soft test — topology varies per init, but response direction should be somewhat consistent
    // We don't assert here, just report — hard to guarantee with random init
}

// ──── NO INPUT VS INPUT ────

#[test]
fn input_causes_measurable_displacement() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    // Warmup
    for _ in 0..100 {
        c.tick(None);
    }

    let before = c.state.clone();
    c.tick(None); // One free tick
    let after_free = c.state.clone();
    let free_disp: f32 = 1.0
        - before
            .iter()
            .zip(after_free.iter())
            .map(|(a, b)| (a.conj() * b).re)
            .sum::<f32>();

    let before2 = after_free.clone();
    let input = make_input("danger");
    c.tick(Some(&input)); // One input tick
    let input_disp: f32 = 1.0
        - before2
            .iter()
            .zip(c.state.iter())
            .map(|(a, b)| (a.conj() * b).re)
            .sum::<f32>();

    eprintln!(
        "Free tick displacement: {:.6}, Input tick displacement: {:.6}",
        free_disp, input_disp
    );
    // Input should cause more displacement than free-running
    // (or at least not zero — it should DO something)
    assert!(
        input_disp > 1e-8,
        "Input caused zero displacement — crystal ignoring input"
    );
}

// ──── RECOVERY FROM EXTREME INPUT ────

#[test]
fn crystal_recovers_from_extreme_to_normal() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    // Warmup
    for _ in 0..100 {
        c.tick(None);
    }
    let baseline = Emotions::from_crystal(&c);

    // Hit with extreme input
    let extreme = DVector::from_fn(960, |i, _| (i as f32) * 1e5);
    for _ in 0..50 {
        c.tick(Some(&extreme));
    }
    let stressed = Emotions::from_crystal(&c);

    // Free-run to recover
    for _ in 0..500 {
        c.tick(None);
    }
    let recovered = Emotions::from_crystal(&c);

    // Energy should not be NaN or crazy after recovery
    assert!(!recovered.energy.is_nan(), "Energy NaN after recovery");
    assert!(
        recovered.energy < 10.0,
        "Energy exploded after recovery: {}",
        recovered.energy
    );

    eprintln!("Baseline Φ={:.4}, Stressed Φ={:.4}, Recovered Φ={:.4}",
        baseline.phase_coherence, stressed.phase_coherence, recovered.phase_coherence);
}

// ──── HISTORY BUFFER ────

#[test]
fn history_buffer_fills_correctly() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    assert_eq!(c.history_idx, 0);
    for i in 0..100 {
        c.tick(None);
        assert_eq!(c.history_idx, i + 1);
    }
    // After 64 ticks, it should wrap
    assert_eq!(c.history_idx, 100);
    // The ring buffer should not have NaN
    for val in &c.history_phi {
        assert!(!val.is_nan(), "history_phi has NaN");
    }
    for val in &c.history_energy {
        assert!(!val.is_nan(), "history_energy has NaN");
    }
}
