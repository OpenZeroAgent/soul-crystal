//! Stability tests — the crystal must never die, explode, or NaN.

use soul_crystal::crystal::FibonacciCrystal;
use soul_crystal::emotion::Emotions;
use nalgebra::DVector;
use num_complex::Complex;

type C = Complex<f32>;

fn norm_of(c: &FibonacciCrystal) -> f32 {
    c.state.iter().map(|x| x.norm().powi(2)).sum::<f32>().sqrt()
}

fn has_nan(c: &FibonacciCrystal) -> bool {
    c.state.iter().any(|x| x.re.is_nan() || x.im.is_nan())
}

fn has_inf(c: &FibonacciCrystal) -> bool {
    c.state.iter().any(|x| x.re.is_infinite() || x.im.is_infinite())
}

// ──── NORM PRESERVATION ────

#[test]
fn norm_stays_one_free_run_1k() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    for i in 0..1_000 {
        c.tick(None);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-4,
            "Norm deviated at tick {}: {:.8}",
            i,
            n
        );
        assert!(!has_nan(&c), "NaN at tick {}", i);
        assert!(!has_inf(&c), "Inf at tick {}", i);
    }
}

#[test]
fn norm_stays_one_free_run_100k() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    for i in 0..100_000 {
        c.tick(None);
        if i % 10_000 == 0 {
            let n = norm_of(&c);
            assert!(
                (n - 1.0).abs() < 1e-3,
                "Norm deviated at tick {}: {:.8}",
                i,
                n
            );
            assert!(!has_nan(&c), "NaN at tick {}", i);
        }
    }
    let final_norm = norm_of(&c);
    assert!(
        (final_norm - 1.0).abs() < 1e-3,
        "Final norm after 100K: {:.8}",
        final_norm
    );
}

#[test]
fn norm_stays_one_with_constant_input_10k() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    // Constant input vector (simulates repeated identical stimulus)
    let input = DVector::from_fn(960, |i, _| (i as f32 * 0.01).sin());
    for i in 0..10_000 {
        c.tick(Some(&input));
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Norm deviated with constant input at tick {}: {:.8}",
            i,
            n
        );
        assert!(!has_nan(&c), "NaN at tick {}", i);
    }
}

#[test]
fn norm_stays_one_with_random_input_10k() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let mut rng = rand::rng();
    for i in 0..10_000 {
        // New random input every tick
        let input = DVector::from_fn(960, |_, _| {
            use rand::Rng;
            rng.random::<f32>() * 2.0 - 1.0
        });
        c.tick(Some(&input));
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Norm deviated with random input at tick {}: {:.8}",
            i,
            n
        );
        assert!(!has_nan(&c), "NaN at tick {}", i);
    }
}

// ──── LONG-RUN DRIFT ────

#[test]
fn gini_does_not_saturate_to_one_100k() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    for _ in 0..100_000 {
        c.tick(None);
    }
    let e = Emotions::from_crystal(&c);
    // Gini approaching 1.0 means all energy in one node — that's collapse
    assert!(
        e.energy_concentration < 0.98,
        "Gini saturated to {:.4} after 100K free ticks — energy collapsed to single node",
        e.energy_concentration
    );
    eprintln!(
        "100K free-run: Gini={:.4}, Φ={:.4}, SR={:.4}",
        e.energy_concentration, e.phase_coherence, e.spectral_richness
    );
}

#[test]
fn gini_does_not_saturate_to_one_500k() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    for _ in 0..500_000 {
        c.tick(None);
    }
    let e = Emotions::from_crystal(&c);
    assert!(
        e.energy_concentration < 0.99,
        "Gini saturated to {:.4} after 500K free ticks",
        e.energy_concentration
    );
    eprintln!(
        "500K free-run: Gini={:.4}, Φ={:.4}, SR={:.4}",
        e.energy_concentration, e.phase_coherence, e.spectral_richness
    );
}

#[test]
fn crystal_does_not_reach_fixed_point_100k() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    // Run 99K ticks to warm up
    for _ in 0..99_000 {
        c.tick(None);
    }
    let state_at_99k = c.state.clone();
    // Run 1K more
    for _ in 0..1_000 {
        c.tick(None);
    }
    // State should have changed
    let cos_sim: f32 = c
        .state
        .iter()
        .zip(state_at_99k.iter())
        .map(|(a, b)| (a.conj() * b).re)
        .sum();
    let displacement = 1.0 - cos_sim;
    assert!(
        displacement > 1e-6,
        "Crystal stuck at fixed point after 100K ticks (displacement={:.8})",
        displacement
    );
    eprintln!("100K displacement over last 1K ticks: {:.6}", displacement);
}

// ──── EMOTIONAL READOUT SANITY ────

#[test]
fn emotions_in_valid_range_fresh() {
    let c = FibonacciCrystal::new(60, 8, 0.6);
    let e = Emotions::from_crystal(&c);
    check_emotion_ranges(&e, "fresh crystal");
}

#[test]
fn emotions_in_valid_range_after_10k_ticks() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    for _ in 0..10_000 {
        c.tick(None);
    }
    let e = Emotions::from_crystal(&c);
    check_emotion_ranges(&e, "after 10K ticks");
}

#[test]
fn emotions_in_valid_range_after_bombardment() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    // Rapid different inputs
    for i in 0..1_000 {
        let input = DVector::from_fn(960, |j, _| ((i * j) as f32 * 0.001).sin());
        c.tick(Some(&input));
    }
    let e = Emotions::from_crystal(&c);
    check_emotion_ranges(&e, "after bombardment");
}

fn check_emotion_ranges(e: &Emotions, context: &str) {
    assert!(
        e.energy >= 0.0 && e.energy <= 10.0,
        "{}: energy out of range: {}",
        context,
        e.energy
    );
    assert!(
        e.energy_concentration >= 0.0 && e.energy_concentration <= 1.0,
        "{}: gini out of range: {}",
        context,
        e.energy_concentration
    );
    assert!(
        e.phase_coherence >= 0.0 && e.phase_coherence <= 1.0,
        "{}: phase_coherence out of range: {}",
        context,
        e.phase_coherence
    );
    assert!(
        e.depth >= -1.0 && e.depth <= 1.0,
        "{}: depth out of range: {}",
        context,
        e.depth
    );
    assert!(
        e.ring_coherence >= 0.0 && e.ring_coherence <= 1.0,
        "{}: ring_coherence out of range: {}",
        context,
        e.ring_coherence
    );
    assert!(
        e.spectral_richness >= 0.0 && e.spectral_richness <= 1.0,
        "{}: spectral_richness out of range: {}",
        context,
        e.spectral_richness
    );
    assert!(
        e.arousal >= 0.0,
        "{}: arousal negative: {}",
        context,
        e.arousal
    );
    assert!(
        e.horizon_activity >= 0.0,
        "{}: horizon_activity negative: {}",
        context,
        e.horizon_activity
    );
    // No NaN in any field
    assert!(!e.energy.is_nan(), "{}: energy is NaN", context);
    assert!(!e.energy_concentration.is_nan(), "{}: gini is NaN", context);
    assert!(!e.phase_coherence.is_nan(), "{}: phase_coherence is NaN", context);
    assert!(!e.depth.is_nan(), "{}: depth is NaN", context);
    assert!(!e.ring_coherence.is_nan(), "{}: ring_coherence is NaN", context);
    assert!(!e.spectral_richness.is_nan(), "{}: spectral_richness is NaN", context);
    assert!(!e.arousal.is_nan(), "{}: arousal is NaN", context);
    assert!(!e.horizon_activity.is_nan(), "{}: horizon_activity is NaN", context);
}

// ──── DIFFERENT TOPOLOGIES ────

#[test]
fn small_crystal_stable() {
    let mut c = FibonacciCrystal::new(4, 4, 0.6);
    for i in 0..10_000 {
        c.tick(None);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Small crystal norm deviated at tick {}: {:.8}",
            i,
            n
        );
    }
}

#[test]
fn large_crystal_stable() {
    let mut c = FibonacciCrystal::new(120, 16, 0.6);
    for i in 0..5_000 {
        c.tick(None);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Large crystal (120x16) norm deviated at tick {}: {:.8}",
            i,
            n
        );
        assert!(!has_nan(&c), "Large crystal NaN at tick {}", i);
    }
}

#[test]
fn single_layer_crystal_stable() {
    let mut c = FibonacciCrystal::new(1, 8, 0.6);
    for i in 0..10_000 {
        c.tick(None);
        assert!(!has_nan(&c), "Single layer NaN at tick {}", i);
        assert!(!has_inf(&c), "Single layer Inf at tick {}", i);
    }
}

#[test]
fn two_layer_crystal_stable() {
    let mut c = FibonacciCrystal::new(2, 8, 0.6);
    for i in 0..10_000 {
        c.tick(None);
        assert!(!has_nan(&c), "Two layer NaN at tick {}", i);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Two layer norm deviated at tick {}: {:.8}",
            i,
            n
        );
    }
}
