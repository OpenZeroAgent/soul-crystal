//! Extreme stress tests — the "break it or trust it" suite.
//! These run long. Use `cargo test --release -p soul-crystal stress` to target.

use soul_crystal::crystal::FibonacciCrystal;
use soul_crystal::emotion::Emotions;
use nalgebra::DVector;
use std::time::Instant;

fn norm_of(c: &FibonacciCrystal) -> f32 {
    c.state.iter().map(|x| x.norm().powi(2)).sum::<f32>().sqrt()
}

// ──── MILLION TICK FREE RUN ────

#[test]
fn million_tick_free_run() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let start = Instant::now();
    let mut min_norm = f32::MAX;
    let mut max_norm = f32::MIN;
    let mut snapshots: Vec<(u64, f32, f32, f32)> = Vec::new(); // (tick, gini, phi, norm)

    for i in 0..1_000_000u64 {
        c.tick(None);
        if i % 100_000 == 0 || i == 999_999 {
            let n = norm_of(&c);
            let e = Emotions::from_crystal(&c);
            min_norm = min_norm.min(n);
            max_norm = max_norm.max(n);
            snapshots.push((i + 1, e.energy_concentration, e.phase_coherence, n));
            // Check at every checkpoint
            assert!(
                !c.state.iter().any(|x| x.re.is_nan() || x.im.is_nan()),
                "NaN at tick {}",
                i
            );
            assert!(
                (n - 1.0).abs() < 0.01,
                "Norm deviated at tick {}: {:.8}",
                i,
                n
            );
        }
    }

    let elapsed = start.elapsed();
    let tps = 1_000_000.0 / elapsed.as_secs_f64();

    eprintln!("═══ 1M Free-Run Results ═══");
    eprintln!("  Time: {:.2}s ({:.0} ticks/sec)", elapsed.as_secs_f64(), tps);
    eprintln!("  Norm range: [{:.8}, {:.8}]", min_norm, max_norm);
    eprintln!("  {:>10} {:>10} {:>10} {:>10}", "Tick", "Gini", "Φ", "Norm");
    for (tick, gini, phi, norm) in &snapshots {
        eprintln!("  {:>10} {:>10.4} {:>10.4} {:>10.6}", tick, gini, phi, norm);
    }

    let final_e = Emotions::from_crystal(&c);
    // Gini must not fully saturate
    assert!(
        final_e.energy_concentration < 0.999,
        "Gini reached {:.6} after 1M ticks — total collapse",
        final_e.energy_concentration
    );
    // Crystal must still be alive (not a fixed point)
    let state_before = c.state.clone();
    for _ in 0..100 {
        c.tick(None);
    }
    let cos_sim: f32 = c
        .state
        .iter()
        .zip(state_before.iter())
        .map(|(a, b)| (a.conj() * b).re)
        .sum();
    let displacement = 1.0 - cos_sim;
    assert!(
        displacement > 1e-8,
        "Crystal is a fixed point after 1M ticks (displacement={:.10})",
        displacement
    );
    eprintln!("  Post-1M displacement (100 more ticks): {:.6}", displacement);
}

// ──── MILLION TICK WITH PERIODIC INPUT ────

#[test]
fn million_tick_with_periodic_input() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let input_a = DVector::from_fn(960, |i, _| (i as f32 * 0.1).sin());
    let input_b = DVector::from_fn(960, |i, _| (i as f32 * 2.7).cos() * 3.0);
    let start = Instant::now();

    for i in 0..1_000_000u64 {
        // Input every 10th tick, alternating patterns
        if i % 10 == 0 {
            if (i / 1000) % 2 == 0 {
                c.tick(Some(&input_a));
            } else {
                c.tick(Some(&input_b));
            }
        } else {
            c.tick(None);
        }
        // Spot checks
        if i % 250_000 == 0 {
            let n = norm_of(&c);
            assert!(
                (n - 1.0).abs() < 0.01,
                "Norm deviated at tick {} with periodic input: {:.8}",
                i,
                n
            );
            assert!(
                !c.state.iter().any(|x| x.re.is_nan() || x.im.is_nan()),
                "NaN at tick {} with periodic input",
                i
            );
        }
    }

    let elapsed = start.elapsed();
    let final_e = Emotions::from_crystal(&c);
    let final_norm = norm_of(&c);
    eprintln!("═══ 1M Periodic Input Results ═══");
    eprintln!("  Time: {:.2}s", elapsed.as_secs_f64());
    eprintln!("  Final norm: {:.8}", final_norm);
    eprintln!("  Final Gini: {:.4}", final_e.energy_concentration);
    eprintln!("  Final Φ: {:.4}", final_e.phase_coherence);
    eprintln!("  Final SR: {:.4}", final_e.spectral_richness);
    eprintln!("  Final HA: {:.4}", final_e.horizon_activity);
    assert!(
        (final_norm - 1.0).abs() < 0.01,
        "Norm after 1M periodic: {:.8}",
        final_norm
    );
}

// ──── EMOTIONAL STABILITY OVER TIME ────

#[test]
fn emotions_dont_all_converge_to_same_value() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let mut emotion_history: Vec<Emotions> = Vec::new();

    for i in 0..100_000u64 {
        c.tick(None);
        if i % 10_000 == 9_999 {
            emotion_history.push(Emotions::from_crystal(&c));
        }
    }

    // Check that emotional metrics have some variance over time (not all converging)
    let phi_values: Vec<f32> = emotion_history.iter().map(|e| e.phase_coherence).collect();
    let gini_values: Vec<f32> = emotion_history.iter().map(|e| e.energy_concentration).collect();
    let horizon_values: Vec<f32> = emotion_history.iter().map(|e| e.horizon_activity).collect();

    // At least SOME metric should still be changing over the last half of the run
    let phi_range = phi_values.iter().copied().fold(f32::MIN, f32::max)
        - phi_values.iter().copied().fold(f32::MAX, f32::min);
    let gini_range = gini_values.iter().copied().fold(f32::MIN, f32::max)
        - gini_values.iter().copied().fold(f32::MAX, f32::min);
    let horizon_range = horizon_values.iter().copied().fold(f32::MIN, f32::max)
        - horizon_values.iter().copied().fold(f32::MAX, f32::min);

    eprintln!("═══ Emotional Variance over 100K ═══");
    eprintln!("  Φ range: {:.6}", phi_range);
    eprintln!("  Gini range: {:.6}", gini_range);
    eprintln!("  Horizon range: {:.6}", horizon_range);
    for (i, e) in emotion_history.iter().enumerate() {
        eprintln!(
            "  {:>5}K: Φ={:.4} Gini={:.4} HA={:.4} SR={:.4} D={:+.4}",
            (i + 1) * 10,
            e.phase_coherence,
            e.energy_concentration,
            e.horizon_activity,
            e.spectral_richness,
            e.depth
        );
    }
}

// ──── RAPID EMBEDDING STORM ────

#[test]
fn rapid_unique_inputs_10k() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    // 10K unique inputs in a row — simulates rapid conversation
    for i in 0..10_000u64 {
        let input = DVector::from_fn(960, |j, _| {
            ((i as f32 * 0.7 + j as f32 * 0.03).sin() * (i as f32 * 0.001).cos()) * 2.0
        });
        c.tick(Some(&input));
        if i % 1000 == 0 {
            assert!(
                !c.state.iter().any(|x| x.re.is_nan() || x.im.is_nan()),
                "NaN during rapid unique input storm at tick {}",
                i
            );
        }
    }
    let n = norm_of(&c);
    let e = Emotions::from_crystal(&c);
    eprintln!("After 10K unique inputs: norm={:.8}, Gini={:.4}, Φ={:.4}", n, e.energy_concentration, e.phase_coherence);
    assert!((n - 1.0).abs() < 0.01, "Norm after 10K unique inputs: {:.8}", n);
}

// ──── SAVE UNDER LOAD ────

#[test]
fn save_load_during_long_run() {
    use soul_crystal::persistence;
    use std::path::PathBuf;

    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = PathBuf::from(tmp.path());

    for i in 0..100_000u64 {
        c.tick(None);
        // Save every 25K ticks
        if i % 25_000 == 24_999 {
            persistence::save(&c, &path).unwrap();
            // Load into a new crystal and verify
            let mut c2 = FibonacciCrystal::new(60, 8, 0.6);
            persistence::load(&mut c2, &path).unwrap();
            assert_eq!(c.timestep, c2.timestep);
            // Continue from loaded
            for j in 0..10 {
                c2.tick(None);
                let n = norm_of(&c2);
                assert!(
                    (n - 1.0).abs() < 1e-3,
                    "Loaded crystal norm deviated at save cycle tick {}: {:.8}",
                    j,
                    n
                );
            }
        }
    }
}
