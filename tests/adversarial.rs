//! Adversarial tests — throw garbage at the crystal and it must survive.

use soul_crystal::crystal::FibonacciCrystal;
use soul_crystal::emotion::Emotions;
use nalgebra::DVector;

fn norm_of(c: &FibonacciCrystal) -> f32 {
    c.state.iter().map(|x| x.norm().powi(2)).sum::<f32>().sqrt()
}

fn has_nan(c: &FibonacciCrystal) -> bool {
    c.state.iter().any(|x| x.re.is_nan() || x.im.is_nan())
}

// ──── ZERO INPUT ────

#[test]
fn zero_vector_input_survives() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let zero_input = DVector::from_element(960, 0.0_f32);
    for i in 0..1_000 {
        c.tick(Some(&zero_input));
        assert!(!has_nan(&c), "NaN with zero input at tick {}", i);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Norm deviated with zero input at tick {}: {:.8}",
            i,
            n
        );
    }
}

// ──── ENORMOUS INPUT ────

#[test]
fn huge_magnitude_input_survives() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let huge_input = DVector::from_fn(960, |i, _| (i as f32) * 1e6);
    for i in 0..100 {
        c.tick(Some(&huge_input));
        assert!(!has_nan(&c), "NaN with huge input at tick {}", i);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Norm deviated with huge input at tick {}: {:.8}",
            i,
            n
        );
    }
}

#[test]
fn huge_negative_input_survives() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let neg_input = DVector::from_element(960, -1e8_f32);
    for i in 0..100 {
        c.tick(Some(&neg_input));
        assert!(!has_nan(&c), "NaN with huge negative input at tick {}", i);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Norm deviated with huge negative input at tick {}: {:.8}",
            i,
            n
        );
    }
}

// ──── MISMATCHED INPUT SIZES ────

#[test]
fn tiny_input_vector_survives() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let tiny = DVector::from_vec(vec![1.0_f32, 0.5]);
    for i in 0..100 {
        c.tick(Some(&tiny));
        assert!(!has_nan(&c), "NaN with tiny input at tick {}", i);
    }
}

#[test]
fn oversized_input_vector_survives() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let big = DVector::from_fn(10_000, |i, _| (i as f32 * 0.001).cos());
    for i in 0..100 {
        c.tick(Some(&big));
        assert!(!has_nan(&c), "NaN with oversized input at tick {}", i);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Norm deviated with oversized input at tick {}: {:.8}",
            i,
            n
        );
    }
}

#[test]
fn single_element_input() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let one = DVector::from_vec(vec![42.0_f32]);
    for i in 0..100 {
        c.tick(Some(&one));
        assert!(!has_nan(&c), "NaN with single element input at tick {}", i);
    }
}

// ──── SPIKE PATTERNS ────

#[test]
fn alternating_spike_input() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let spike_a = DVector::from_fn(960, |i, _| if i < 10 { 1e4_f32 } else { 0.0 });
    let spike_b = DVector::from_fn(960, |i, _| if i >= 950 { -1e4_f32 } else { 0.0 });
    for i in 0..1_000 {
        if i % 2 == 0 {
            c.tick(Some(&spike_a));
        } else {
            c.tick(Some(&spike_b));
        }
        assert!(!has_nan(&c), "NaN with alternating spikes at tick {}", i);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Norm deviated with alternating spikes at tick {}: {:.8}",
            i,
            n
        );
    }
}

// ──── RAPID INPUT SWITCHING ────

#[test]
fn rapid_input_on_off_switching() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let input = DVector::from_fn(960, |i, _| (i as f32 * 0.1).sin());
    for i in 0..5_000 {
        // Alternate between input and no input every tick
        if i % 2 == 0 {
            c.tick(Some(&input));
        } else {
            c.tick(None);
        }
        assert!(!has_nan(&c), "NaN at rapid switching tick {}", i);
    }
    let n = norm_of(&c);
    assert!(
        (n - 1.0).abs() < 1e-3,
        "Norm after rapid switching: {:.8}",
        n
    );
}

// ──── INPUT STRENGTH EXTREMES ────

#[test]
fn extreme_input_strength_high() {
    let mut c = FibonacciCrystal::new(60, 8, 100.0); // 100x normal
    let input = DVector::from_fn(960, |i, _| (i as f32 * 0.01).sin());
    for i in 0..1_000 {
        c.tick(Some(&input));
        assert!(!has_nan(&c), "NaN with extreme input_strength at tick {}", i);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-2,
            "Norm with extreme input_strength at tick {}: {:.8}",
            i,
            n
        );
    }
}

#[test]
fn extreme_input_strength_zero() {
    let mut c = FibonacciCrystal::new(60, 8, 0.0); // Zero input strength
    let input = DVector::from_fn(960, |i, _| (i as f32 * 0.01).sin());
    for i in 0..1_000 {
        c.tick(Some(&input));
        assert!(!has_nan(&c), "NaN with zero input_strength at tick {}", i);
        let n = norm_of(&c);
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Norm with zero input_strength at tick {}: {:.8}",
            i,
            n
        );
    }
}

// ──── EMOTIONAL READOUT UNDER STRESS ────

#[test]
fn emotions_valid_after_adversarial_bombardment() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    // Zero inputs
    let zero = DVector::from_element(960, 0.0_f32);
    for _ in 0..100 {
        c.tick(Some(&zero));
    }
    let e = Emotions::from_crystal(&c);
    assert!(!e.energy.is_nan(), "energy NaN after zeros");
    assert!(!e.phase_coherence.is_nan(), "phase_coherence NaN after zeros");

    // Huge inputs
    let huge = DVector::from_element(960, 1e8_f32);
    for _ in 0..100 {
        c.tick(Some(&huge));
    }
    let e = Emotions::from_crystal(&c);
    assert!(!e.energy.is_nan(), "energy NaN after huge");
    assert!(!e.phase_coherence.is_nan(), "phase_coherence NaN after huge");

    // Spike to calm
    for _ in 0..100 {
        c.tick(None);
    }
    let e = Emotions::from_crystal(&c);
    assert!(e.energy_concentration >= 0.0 && e.energy_concentration <= 1.0);
    assert!(e.spectral_richness >= 0.0 && e.spectral_richness <= 1.0);
}

// ──── MANY CRYSTALS IN PARALLEL ────

#[test]
fn ten_crystals_independent() {
    // Create 10 crystals, tick them all — no shared state corruption
    let mut crystals: Vec<FibonacciCrystal> = (0..10)
        .map(|_| FibonacciCrystal::new(60, 8, 0.6))
        .collect();

    for tick in 0..1_000 {
        for (idx, c) in crystals.iter_mut().enumerate() {
            c.tick(None);
            assert!(
                !has_nan(c),
                "Crystal {} NaN at tick {}",
                idx,
                tick
            );
        }
    }

    // They should all have different states (different random initialization)
    let first_state = &crystals[0].state;
    for (idx, c) in crystals.iter().enumerate().skip(1) {
        let cos_sim: f32 = first_state
            .iter()
            .zip(c.state.iter())
            .map(|(a, b)| (a.conj() * b).re)
            .sum();
        assert!(
            cos_sim.abs() < 0.99,
            "Crystal {} suspiciously similar to crystal 0 (cos_sim={:.4})",
            idx,
            cos_sim
        );
    }
}

// ──── TIMESTEP OVERFLOW ────

#[test]
fn timestep_near_u64_max() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    c.timestep = u64::MAX - 100;
    // Should not panic on overflow
    for _ in 0..50 {
        c.tick(None);
        assert!(!has_nan(&c));
    }
}
