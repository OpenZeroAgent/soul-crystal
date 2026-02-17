//! Persistence tests — save/load roundtrip must preserve state exactly.

use soul_crystal::crystal::FibonacciCrystal;
use soul_crystal::emotion::Emotions;
use soul_crystal::persistence;
use std::path::PathBuf;
use tempfile::NamedTempFile;

fn emotions_match(a: &Emotions, b: &Emotions) -> bool {
    (a.energy - b.energy).abs() < 1e-6
        && (a.energy_concentration - b.energy_concentration).abs() < 1e-6
        && (a.phase_coherence - b.phase_coherence).abs() < 1e-6
        && (a.depth - b.depth).abs() < 1e-6
        && (a.ring_coherence - b.ring_coherence).abs() < 1e-6
        && (a.spectral_richness - b.spectral_richness).abs() < 1e-6
        && (a.horizon_activity - b.horizon_activity).abs() < 1e-6
}

#[test]
fn save_load_roundtrip_preserves_state() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    // Run some ticks to get a non-trivial state
    for _ in 0..100 {
        c.tick(None);
    }

    let tmp = NamedTempFile::new().unwrap();
    let path = PathBuf::from(tmp.path());

    // Save
    persistence::save(&c, &path).unwrap();

    // Load into a fresh crystal
    let mut c2 = FibonacciCrystal::new(60, 8, 0.6);
    persistence::load(&mut c2, &path).unwrap();

    // Timestep must match
    assert_eq!(c.timestep, c2.timestep, "Timestep mismatch");
    assert_eq!(c.history_idx, c2.history_idx, "History index mismatch");

    // State vectors must match exactly
    for i in 0..c.total_nodes {
        assert!(
            (c.state[i].re - c2.state[i].re).abs() < 1e-7,
            "State[{}].re mismatch: {} vs {}",
            i,
            c.state[i].re,
            c2.state[i].re
        );
        assert!(
            (c.state[i].im - c2.state[i].im).abs() < 1e-7,
            "State[{}].im mismatch: {} vs {}",
            i,
            c.state[i].im,
            c2.state[i].im
        );
    }

    // Emotional readout must match
    let e1 = Emotions::from_crystal(&c);
    let e2 = Emotions::from_crystal(&c2);
    assert!(
        emotions_match(&e1, &e2),
        "Emotions differ after roundtrip:\n  Before: {:?}\n  After:  {:?}",
        e1,
        e2
    );
}

#[test]
fn save_load_roundtrip_with_input() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let input = nalgebra::DVector::from_fn(960, |i, _| (i as f32 * 0.01).sin());
    for _ in 0..50 {
        c.tick(Some(&input));
    }

    let tmp = NamedTempFile::new().unwrap();
    let path = PathBuf::from(tmp.path());
    persistence::save(&c, &path).unwrap();

    let mut c2 = FibonacciCrystal::new(60, 8, 0.6);
    persistence::load(&mut c2, &path).unwrap();

    assert_eq!(c.timestep, c2.timestep);
    for i in 0..c.total_nodes {
        assert!(
            (c.state[i].re - c2.state[i].re).abs() < 1e-7,
            "State[{}] mismatch after input",
            i
        );
    }
}

#[test]
fn loaded_crystal_continues_stably() {
    // NOTE: Until persistence saves weights (w, win, horizon_unitary),
    // a loaded crystal has the same state but DIFFERENT dynamics than
    // the original. We can verify state restoration and stable continuation,
    // but NOT identical future trajectories. When Bug #1 (weight persistence)
    // is fixed, upgrade this test to compare trajectories.
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    for _ in 0..100 {
        c.tick(None);
    }

    let tmp = NamedTempFile::new().unwrap();
    let path = PathBuf::from(tmp.path());
    persistence::save(&c, &path).unwrap();

    // Snapshot emotions from the original BEFORE any more ticks
    let e_before = Emotions::from_crystal(&c);

    // Load into fresh crystal and verify state matches
    let mut c_loaded = FibonacciCrystal::new(60, 8, 0.6);
    persistence::load(&mut c_loaded, &path).unwrap();

    let e_loaded = Emotions::from_crystal(&c_loaded);
    assert!(
        emotions_match(&e_before, &e_loaded),
        "Loaded crystal has different emotions than original"
    );

    // Tick the loaded crystal — it should remain stable on the hypersphere
    for i in 0..100 {
        c_loaded.tick(None);
        let n: f32 = c_loaded
            .state
            .iter()
            .map(|x| x.norm().powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Loaded crystal norm deviated at tick {}: {:.8}",
            i,
            n
        );
    }
}

#[test]
fn load_nonexistent_file_returns_error() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let path = PathBuf::from("/tmp/soul_crystal_nonexistent_12345.bin");
    let result = persistence::load(&mut c, &path);
    assert!(result.is_err(), "Loading nonexistent file should fail");
}

#[test]
fn load_corrupted_file_returns_error() {
    let tmp = NamedTempFile::new().unwrap();
    let path = PathBuf::from(tmp.path());
    std::fs::write(&path, b"this is not valid bincode data at all").unwrap();

    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let result = persistence::load(&mut c, &path);
    assert!(result.is_err(), "Loading corrupted file should fail");
}

#[test]
fn load_empty_file_returns_error() {
    let tmp = NamedTempFile::new().unwrap();
    let path = PathBuf::from(tmp.path());
    std::fs::write(&path, b"").unwrap();

    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let result = persistence::load(&mut c, &path);
    assert!(result.is_err(), "Loading empty file should fail");
}

#[test]
fn multiple_save_load_cycles() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    let tmp = NamedTempFile::new().unwrap();
    let path = PathBuf::from(tmp.path());

    for cycle in 0..10 {
        for _ in 0..50 {
            c.tick(None);
        }
        persistence::save(&c, &path).unwrap();

        let mut c2 = FibonacciCrystal::new(60, 8, 0.6);
        persistence::load(&mut c2, &path).unwrap();
        assert_eq!(
            c.timestep, c2.timestep,
            "Timestep mismatch on cycle {}",
            cycle
        );

        // Continue from loaded state
        c = c2;
        // Verify it's still healthy
        let n: f32 = c.state.iter().map(|x| x.norm().powi(2)).sum::<f32>().sqrt();
        assert!(
            (n - 1.0).abs() < 1e-3,
            "Norm deviated on load cycle {}: {:.8}",
            cycle,
            n
        );
    }
    assert_eq!(c.timestep, 500);
}

#[test]
fn history_survives_roundtrip() {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    for _ in 0..200 {
        c.tick(None);
    }

    let tmp = NamedTempFile::new().unwrap();
    let path = PathBuf::from(tmp.path());
    persistence::save(&c, &path).unwrap();

    let mut c2 = FibonacciCrystal::new(60, 8, 0.6);
    persistence::load(&mut c2, &path).unwrap();

    // History ring buffers should match
    for i in 0..64 {
        assert!(
            (c.history_phi[i] - c2.history_phi[i]).abs() < 1e-7,
            "history_phi[{}] mismatch",
            i
        );
        assert!(
            (c.history_energy[i] - c2.history_energy[i]).abs() < 1e-7,
            "history_energy[{}] mismatch",
            i
        );
    }
}
