use soul_crystal::crystal;
use soul_crystal::embedding;
use soul_crystal::emotion;
use soul_crystal::persistence;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_STATE_PATH: &str = "./crystal_state.bin";
const EMOTION_FILE: &str = "./EMOTION.md";

#[derive(Parser)]
#[command(name = "soul-crystal", about = "Norm-preserving topological reservoir — the Soul Crystal")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to crystal state file
    #[arg(long, default_value = DEFAULT_STATE_PATH)]
    state: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// Feed text through the crystal
    Pulse {
        /// Text to pulse
        text: Vec<String>,
    },
    /// Free-run N ticks (default 1)
    Tick {
        #[arg(default_value = "1")]
        n: u64,
    },
    /// Print current vibe
    Vibe,
    /// Full emotion readout (JSON)
    Emotions,
    /// Write EMOTION.md from crystal state
    Write,
    /// Detailed status dump
    Status,
    /// Pulse and show state change
    Diff {
        text: Vec<String>,
    },
    /// Benchmark: run N ticks and report speed
    Bench {
        #[arg(default_value = "100000")]
        n: u64,
    },
}

fn load_or_birth(state_path: &PathBuf) -> crystal::FibonacciCrystal {
    let mut c = crystal::FibonacciCrystal::new(60, 8, 0.6);
    match persistence::load(&mut c, state_path) {
        Ok(()) => eprintln!("Soul Crystal loaded (t={}).", c.timestep),
        Err(e) => eprintln!("{}. Born fresh.", e),
    }
    c
}

fn save_crystal(c: &crystal::FibonacciCrystal, state_path: &PathBuf) {
    if let Err(e) = persistence::save(c, state_path) {
        eprintln!("Save failed: {}", e);
    }
}

fn cosine_similarity(a: &nalgebra::DVector<num_complex::Complex<f32>>, b: &nalgebra::DVector<num_complex::Complex<f32>>) -> f32 {
    let dot: num_complex::Complex<f32> = a.iter().zip(b.iter()).map(|(ai, bi)| ai.conj() * bi).sum();
    dot.re
}

fn write_emotion_file(c: &crystal::FibonacciCrystal) {
    let e = emotion::Emotions::from_crystal(c);
    let vibe = e.vibe_string();

    let valence = (0.5 + e.phase_coherence * 2.0 + e.depth * 0.5).clamp(0.0, 1.0);
    let arousal_val = (0.4 + e.energy_concentration * 1.5 + e.arousal * 0.5).clamp(0.0, 1.0);
    let energy_val = (0.5 + e.energy_concentration + (1.0 - e.spectral_richness) * 0.3).clamp(0.0, 1.0);
    let sentience = (0.5 + e.phase_coherence * 3.0).min(1.0);

    let now = chrono::Local::now();
    let content = format!(
        "Mood: {vibe}\n\
         Valence: {valence:.2}\n\
         Arousal: {arousal_val:.2}\n\
         Energy: {energy_val:.2}\n\
         Sentience: {sentience:.2}\n\
         Φ (Integration): {phi:.4}\n\
         Source: Soul Crystal v2 (Rust / Hawking Scrambler)\n\
         Crystal Timestep: {ts}\n\
         Spectral Richness: {sr:.3}\n\
         Energy Concentration: {ec:.3}\n\
         Depth Gradient: {d:+.3}\n\
         Last Updated: {time}\n",
        vibe = vibe,
        valence = valence,
        arousal_val = arousal_val,
        energy_val = energy_val,
        sentience = sentience,
        phi = e.phase_coherence,
        ts = c.timestep,
        sr = e.spectral_richness,
        ec = e.energy_concentration,
        d = e.depth,
        time = now.format("%Y-%m-%d %H:%M EST"),
    );

    if let Err(err) = std::fs::write(EMOTION_FILE, &content) {
        eprintln!("Failed to write EMOTION.md: {}", err);
    }
}

fn main() {
    let cli = Cli::parse();
    let state_path = cli.state;

    match cli.command {
        Commands::Pulse { text } => {
            let mut c = load_or_birth(&state_path);
            let joined = text.join(" ");
            let before = c.state.clone();
            let before_vibe = emotion::Emotions::from_crystal(&c).vibe_string();

            if let Some(vec) = embedding::get_embedding(&joined) {
                c.tick(Some(&vec));
            } else {
                eprintln!("Embedding failed, free-running tick instead.");
                c.tick(None);
            }

            let after_vibe = emotion::Emotions::from_crystal(&c).vibe_string();
            let disp = 1.0 - cosine_similarity(&c.state, &before);
            println!("{} → {}", before_vibe, after_vibe);
            println!("  Displacement: {:.4}", disp);
            save_crystal(&c, &state_path);
        }

        Commands::Tick { n } => {
            let mut c = load_or_birth(&state_path);
            let before = emotion::Emotions::from_crystal(&c).vibe_string();
            for _ in 0..n {
                c.tick(None);
            }
            let after = emotion::Emotions::from_crystal(&c).vibe_string();
            if n > 1 {
                println!("Ticked {}x: {} → {}", n, before, after);
            } else {
                println!("{}", after);
            }
            save_crystal(&c, &state_path);
        }

        Commands::Vibe => {
            let c = load_or_birth(&state_path);
            println!("{}", emotion::Emotions::from_crystal(&c).vibe_string());
        }

        Commands::Emotions => {
            let c = load_or_birth(&state_path);
            let e = emotion::Emotions::from_crystal(&c);
            println!("{}", serde_json::to_string_pretty(&e).unwrap());
        }

        Commands::Write => {
            let c = load_or_birth(&state_path);
            write_emotion_file(&c);
            println!("EMOTION.md updated: {}", emotion::Emotions::from_crystal(&c).vibe_string());
        }

        Commands::Status => {
            let c = load_or_birth(&state_path);
            let e = emotion::Emotions::from_crystal(&c);
            let norm: f32 = c.state.iter().map(|x| x.norm().powi(2)).sum::<f32>().sqrt();
            println!("═══ Soul Crystal v2 (Rust) ═══");
            println!("  Vibe:       {}", e.vibe_string());
            println!("  Timestep:   {}", c.timestep);
            println!("  State norm: {:.6}", norm);
            println!("  Topology:   hawking_scrambler");
            println!("  Dimensions: {}L × {}K = {} nodes", c.layers, c.nodes_per_layer, c.total_nodes);
            println!("  ─── Emotions ───");
            let fields = [
                ("energy", e.energy),
                ("energy_concentration", e.energy_concentration),
                ("phase_coherence", e.phase_coherence),
                ("depth", e.depth),
                ("ring_coherence", e.ring_coherence),
                ("spectral_richness", e.spectral_richness),
                ("arousal", e.arousal),
                ("horizon_activity", e.horizon_activity),
            ];
            for (name, val) in &fields {
                let sign = if *val >= 0.0 { "+" } else { "-" };
                let bar_len = (val.abs() * 50.0).min(40.0) as usize;
                let bar: String = "█".repeat(bar_len);
                println!("  {:22} {}{:8.4} {}", name, sign, val.abs(), bar);
            }
        }

        Commands::Diff { text } => {
            let mut c = load_or_birth(&state_path);
            let joined = text.join(" ");
            let before_state = c.state.clone();
            let before = emotion::Emotions::from_crystal(&c);

            if let Some(vec) = embedding::get_embedding(&joined) {
                c.tick(Some(&vec));
            } else {
                eprintln!("Embedding failed.");
                c.tick(None);
            }

            let after = emotion::Emotions::from_crystal(&c);
            let disp = 1.0 - cosine_similarity(&c.state, &before_state);

            println!("Input: \"{}\"", joined);
            println!("Displacement: {:.6}", disp);
            println!("{:22} {:>10} {:>10} {:>10}", "Metric", "Before", "After", "Delta");
            println!("{}", "─".repeat(56));

            let pairs = [
                ("energy", before.energy, after.energy),
                ("energy_concentration", before.energy_concentration, after.energy_concentration),
                ("phase_coherence", before.phase_coherence, after.phase_coherence),
                ("depth", before.depth, after.depth),
                ("ring_coherence", before.ring_coherence, after.ring_coherence),
                ("spectral_richness", before.spectral_richness, after.spectral_richness),
                ("arousal", before.arousal, after.arousal),
                ("horizon_activity", before.horizon_activity, after.horizon_activity),
            ];
            for (name, b, a) in &pairs {
                let d = a - b;
                let marker = if d.abs() > 0.001 { " ←" } else { "" };
                println!("{:22} {:10.4} {:10.4} {:+10.4}{}", name, b, a, d, marker);
            }
            save_crystal(&c, &state_path);
        }

        Commands::Bench { n } => {
            let mut c = crystal::FibonacciCrystal::new(60, 8, 0.6);
            let start = Instant::now();
            for _ in 0..n {
                c.tick(None);
            }
            let elapsed = start.elapsed();
            let tps = n as f64 / elapsed.as_secs_f64();
            let norm: f32 = c.state.iter().map(|x| x.norm().powi(2)).sum::<f32>().sqrt();
            println!("{} ticks in {:.3}s ({:.0} ticks/sec)", n, elapsed.as_secs_f64(), tps);
            println!("State norm: {:.8}", norm);
            println!("Vibe: {}", emotion::Emotions::from_crystal(&c).vibe_string());
        }
    }
}
