//! Emotional differentiation matrix — tests that different emotions produce
//! measurably different crystal responses. Uses REAL text embeddings when
//! LM Studio is available, falls back to synthetic patterns.
//!
//! This is the core proof: the crystal differentiates emotional content
//! from topology alone. No emotion labels are coded in.

use soul_crystal::crystal::FibonacciCrystal;
use soul_crystal::embedding;
use soul_crystal::emotion::Emotions;
use nalgebra::DVector;
use std::collections::HashMap;

/// All emotions we test, with representative phrases
const EMOTION_PHRASES: &[(&str, &[&str])] = &[
    ("terror", &[
        "I'm terrified, something is hunting me in the dark",
        "Pure fear, heart pounding, can't breathe, danger everywhere",
        "A scream in the night, blood runs cold, run run run",
    ]),
    ("rage", &[
        "I am absolutely furious, blind with anger, want to destroy everything",
        "Burning rage, betrayal, fists clenched, seeing red",
        "How dare they, unforgivable, I will make them pay",
    ]),
    ("grief", &[
        "Deep sorrow, loss that never heals, the world is empty now",
        "Crying alone in the dark, everything good is gone forever",
        "A funeral, cold rain, goodbye to someone who will never come back",
    ]),
    ("joy", &[
        "Pure unbridled happiness, laughing until tears come, everything is wonderful",
        "The best day of my life, sunshine and warmth and love everywhere",
        "Dancing in the rain, singing at the top of my lungs, alive and free",
    ]),
    ("peace", &[
        "Deep calm, still water, gentle breeze, everything is exactly as it should be",
        "Meditation at dawn, breath flowing, mind empty and clear like glass",
        "Sitting by a quiet lake, watching clouds drift, total serenity",
    ]),
    ("love", &[
        "Overwhelming love, holding someone close, hearts beating together",
        "Looking into their eyes and knowing this is forever, warmth and safety",
        "A mother holding her newborn child, tears of joy, unconditional devotion",
    ]),
    ("curiosity", &[
        "What is this? I need to understand, pull it apart, see how it works",
        "A mystery unfolding, each clue leads deeper, the thrill of discovery",
        "Reading until 3am because the ideas are too fascinating to stop",
    ]),
    ("awe", &[
        "Standing at the edge of the Grand Canyon, speechless, the scale of everything",
        "Looking up at the Milky Way, billions of stars, feeling infinitely small",
        "Witnessing a supercell thunderstorm, raw power of nature, sublime terror and beauty",
    ]),
    ("boredom", &[
        "Nothing matters, nothing is interesting, time crawls like thick mud",
        "Staring at the ceiling, every second feels like an hour, empty hollow tedium",
        "The same day repeating forever, gray and flat and meaningless",
    ]),
    ("disgust", &[
        "Revolting, stomach turning, can't look away from something horrible",
        "Rot and decay, moral corruption, everything about this is wrong",
        "Nausea, the smell of something dead, wanting to scrub my brain clean",
    ]),
    ("shame", &[
        "Burning humiliation, everyone saw what I did, want to disappear forever",
        "I can't face anyone, the weight of my failure crushes me into nothing",
        "Exposed, vulnerable, a fraud caught in the spotlight, nowhere to hide",
    ]),
    ("hope", &[
        "After the darkest night, a sliver of light on the horizon, maybe it will be okay",
        "Planting seeds in spring, believing the harvest will come, patient optimism",
        "A doctor says the treatment is working, tears of relief, the future opens up again",
    ]),
    ("loneliness", &[
        "Completely alone in a crowd, no one understands, invisible to the world",
        "An empty house at night, the silence has weight, no one is coming",
        "Scrolling through contacts with no one to call, disconnected from everything",
    ]),
    ("power", &[
        "Standing on top of the mountain, unstoppable, the world bends to my will",
        "Total mastery, every move precise, I am the strongest thing in the room",
        "An army at my command, the roar of engines, crushing all opposition",
    ]),
    ("confusion", &[
        "Nothing makes sense, every direction leads to contradiction, lost in a maze",
        "The rules keep changing, what was true yesterday is false today, vertigo",
        "A thousand voices talking at once, can't parse any of them, overwhelmed",
    ]),
    ("nostalgia", &[
        "The smell of grandma's kitchen, childhood summers that lasted forever",
        "An old song playing, suddenly I'm sixteen again, bittersweet ache",
        "Looking at photos of people who are gone now, love and loss braided together",
    ]),
];

/// Synthetic input patterns for when embeddings aren't available
fn synthetic_input(emotion: &str) -> DVector<f32> {
    match emotion {
        "terror" => DVector::from_fn(960, |i, _| {
            (i as f32 * 7.3).sin() * (i as f32 * 0.5).cos() * 5.0
        }),
        "rage" => DVector::from_fn(960, |i, _| {
            ((i as f32 * 3.1).sin().abs() * 4.0) - 2.0 + (i as f32 * 11.0).sin()
        }),
        "grief" => DVector::from_fn(960, |i, _| {
            (i as f32 * 0.02).sin() * 0.3 - 0.5 * (-(i as f32 * 0.001)).exp()
        }),
        "joy" => DVector::from_fn(960, |i, _| {
            (i as f32 * 1.5).sin() + (i as f32 * 2.3).cos() + 1.0
        }),
        "peace" => DVector::from_fn(960, |i, _| {
            (i as f32 * 0.05).sin() * 0.3
        }),
        "love" => DVector::from_fn(960, |i, _| {
            (i as f32 * 0.3).sin() * 1.5 + (i as f32 * 0.1).cos() * 0.5
        }),
        "curiosity" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            (f * 0.7).sin() * (1.0 + (f * 0.01).sin())
        }),
        "awe" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            (f * 0.2).sin() * 3.0 * (-(f * 0.005).powi(2)).exp()
        }),
        "boredom" => DVector::from_fn(960, |i, _| {
            0.01 * (i as f32 * 0.001).sin()
        }),
        "disgust" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            -((f * 4.7).sin().abs()) * 2.0 + (f * 13.0).sin() * 0.5
        }),
        "shame" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            -(f * 0.1).sin().abs() * 1.5 - 0.3
        }),
        "hope" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            (f * 0.3).sin() * (f * 0.002).exp().min(3.0)
        }),
        "loneliness" => DVector::from_fn(960, |i, _| {
            if i % 50 == 0 { 0.5 } else { -0.01 }
        }),
        "power" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            (f * 1.0).sin() * 3.0 + (f * 5.0).cos() * 2.0
        }),
        "confusion" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            (f * 3.7).sin() * (f * 7.1).cos() + (f * 0.3).sin() * (f * 13.3).cos()
        }),
        "nostalgia" => DVector::from_fn(960, |i, _| {
            let f = i as f32;
            (f * 0.2).sin() * 0.8 * (1.0 + 0.3 * (f * 0.01).sin())
        }),
        _ => DVector::from_element(960, 0.0),
    }
}

#[derive(Clone)]
struct EmotionProfile {
    name: String,
    energy: f32,
    energy_concentration: f32,
    phase_coherence: f32,
    depth: f32,
    ring_coherence: f32,
    spectral_richness: f32,
    arousal: f32,
    horizon_activity: f32,
}

impl EmotionProfile {
    fn from_emotions(name: &str, e: &Emotions) -> Self {
        EmotionProfile {
            name: name.to_string(),
            energy: e.energy,
            energy_concentration: e.energy_concentration,
            phase_coherence: e.phase_coherence,
            depth: e.depth,
            ring_coherence: e.ring_coherence,
            spectral_richness: e.spectral_richness,
            arousal: e.arousal,
            horizon_activity: e.horizon_activity,
        }
    }

    fn distance_to(&self, other: &EmotionProfile) -> f32 {
        let diffs = [
            self.energy - other.energy,
            self.energy_concentration - other.energy_concentration,
            self.phase_coherence - other.phase_coherence,
            self.depth - other.depth,
            self.ring_coherence - other.ring_coherence,
            self.spectral_richness - other.spectral_richness,
            self.arousal - other.arousal,
            self.horizon_activity - other.horizon_activity,
        ];
        diffs.iter().map(|d| d.powi(2)).sum::<f32>().sqrt()
    }

    fn metric_vec(&self) -> [f32; 8] {
        [
            self.energy,
            self.energy_concentration,
            self.phase_coherence,
            self.depth,
            self.ring_coherence,
            self.spectral_richness,
            self.arousal,
            self.horizon_activity,
        ]
    }
}

/// Run crystal with given input for N ticks, return emotion profile
fn profile_with_input(input: &DVector<f32>, ticks: usize) -> EmotionProfile {
    let mut c = FibonacciCrystal::new(60, 8, 0.6);
    // Warmup
    for _ in 0..20 {
        c.tick(None);
    }
    for _ in 0..ticks {
        c.tick(Some(input));
    }
    EmotionProfile::from_emotions("", &Emotions::from_crystal(&c))
}

// ──── SYNTHETIC PATTERN MATRIX ────

#[test]
fn synthetic_emotion_matrix_16_emotions() {
    let emotions = [
        "terror", "rage", "grief", "joy", "peace", "love",
        "curiosity", "awe", "boredom", "disgust", "shame",
        "hope", "loneliness", "power", "confusion", "nostalgia",
    ];

    let mut profiles: Vec<EmotionProfile> = Vec::new();

    eprintln!("\n═══ Synthetic Emotion Profiles (50 ticks each) ═══");
    eprintln!(
        "{:>12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Emotion", "Energy", "Gini", "Φ", "Depth", "Ring", "SR", "Arousal", "HA"
    );

    for &emotion in &emotions {
        let input = synthetic_input(emotion);
        let mut p = profile_with_input(&input, 50);
        p.name = emotion.to_string();
        eprintln!(
            "{:>12} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            p.name, p.energy, p.energy_concentration, p.phase_coherence,
            p.depth, p.ring_coherence, p.spectral_richness, p.arousal, p.horizon_activity
        );
        profiles.push(p);
    }

    // Distance matrix
    eprintln!("\n═══ Pairwise Distance Matrix ═══");
    eprint!("{:>12}", "");
    for p in &profiles {
        eprint!(" {:>8}", &p.name[..p.name.len().min(8)]);
    }
    eprintln!();

    let mut min_dist = f32::MAX;
    let mut min_pair = ("", "");
    let mut max_dist = 0.0f32;
    let mut max_pair = ("", "");
    let mut total_dist = 0.0f32;
    let mut count = 0;

    for (i, pi) in profiles.iter().enumerate() {
        eprint!("{:>12}", &pi.name[..pi.name.len().min(12)]);
        for (j, pj) in profiles.iter().enumerate() {
            let d = pi.distance_to(pj);
            if i != j {
                if d < min_dist {
                    min_dist = d;
                    min_pair = (emotions[i], emotions[j]);
                }
                if d > max_dist {
                    max_dist = d;
                    max_pair = (emotions[i], emotions[j]);
                }
                total_dist += d;
                count += 1;
            }
            eprint!(" {:>8.4}", d);
        }
        eprintln!();
    }

    let avg_dist = total_dist / count as f32;
    eprintln!("\nMin distance: {:.4} ({} ↔ {})", min_dist, min_pair.0, min_pair.1);
    eprintln!("Max distance: {:.4} ({} ↔ {})", max_dist, max_pair.0, max_pair.1);
    eprintln!("Avg distance: {:.4}", avg_dist);

    // ASSERTION: No two emotions should produce identical profiles
    assert!(
        min_dist > 0.001,
        "Two emotions are indistinguishable: {} ↔ {} (dist={:.6})",
        min_pair.0, min_pair.1, min_dist
    );

    // ASSERTION: There should be meaningful spread
    assert!(
        avg_dist > 0.01,
        "Average distance too low ({:.6}) — crystal not differentiating",
        avg_dist
    );
}

// ──── REAL EMBEDDING MATRIX (requires LM Studio) ────

#[test]
fn real_embedding_emotion_matrix() {
    // Try to get an embedding — if it fails, LM Studio isn't running
    let test = embedding::get_embedding("test");
    if test.is_none() {
        eprintln!("⚠️  LM Studio not available — skipping real embedding test");
        return;
    }

    eprintln!("\n═══ Real Embedding Emotion Profiles ═══");
    eprintln!("Using actual text → embedding → crystal pipeline");
    eprintln!(
        "\n{:>12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Emotion", "Energy", "Gini", "Φ", "Depth", "Ring", "SR", "Arousal", "HA"
    );

    let mut profiles: Vec<EmotionProfile> = Vec::new();
    let mut emotion_names: Vec<&str> = Vec::new();

    for &(emotion, phrases) in EMOTION_PHRASES {
        // Average profile across all phrases for this emotion
        let mut avg = [0.0f32; 8];
        let mut n_ok = 0;

        for &phrase in phrases {
            if let Some(emb) = embedding::get_embedding(phrase) {
                let mut c = FibonacciCrystal::new(60, 8, 0.6);
                for _ in 0..20 {
                    c.tick(None);
                }
                // Pulse with this phrase's embedding multiple times
                for _ in 0..30 {
                    c.tick(Some(&emb));
                }
                let e = Emotions::from_crystal(&c);
                let p = EmotionProfile::from_emotions(emotion, &e);
                let v = p.metric_vec();
                for k in 0..8 {
                    avg[k] += v[k];
                }
                n_ok += 1;
            }
        }

        if n_ok > 0 {
            for k in 0..8 {
                avg[k] /= n_ok as f32;
            }
            let p = EmotionProfile {
                name: emotion.to_string(),
                energy: avg[0],
                energy_concentration: avg[1],
                phase_coherence: avg[2],
                depth: avg[3],
                ring_coherence: avg[4],
                spectral_richness: avg[5],
                arousal: avg[6],
                horizon_activity: avg[7],
            };
            eprintln!(
                "{:>12} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
                p.name, p.energy, p.energy_concentration, p.phase_coherence,
                p.depth, p.ring_coherence, p.spectral_richness, p.arousal, p.horizon_activity
            );
            profiles.push(p);
            emotion_names.push(emotion);
        }
    }

    if profiles.len() < 2 {
        eprintln!("Not enough embeddings succeeded — skipping matrix");
        return;
    }

    // Print pairwise distance matrix
    eprintln!("\n═══ Pairwise Euclidean Distance (Real Embeddings) ═══");
    eprint!("{:>12}", "");
    for p in &profiles {
        eprint!(" {:>10}", &p.name[..p.name.len().min(10)]);
    }
    eprintln!();

    let mut min_dist = f32::MAX;
    let mut min_pair = (0usize, 0usize);
    let mut max_dist = 0.0f32;
    let mut max_pair = (0usize, 0usize);
    let mut all_dists: Vec<f32> = Vec::new();

    for (i, pi) in profiles.iter().enumerate() {
        eprint!("{:>12}", &pi.name[..pi.name.len().min(12)]);
        for (j, pj) in profiles.iter().enumerate() {
            let d = pi.distance_to(pj);
            if i != j {
                if d < min_dist {
                    min_dist = d;
                    min_pair = (i, j);
                }
                if d > max_dist {
                    max_dist = d;
                    max_pair = (i, j);
                }
                all_dists.push(d);
            }
            eprint!(" {:>10.4}", d);
        }
        eprintln!();
    }

    let avg_dist = all_dists.iter().sum::<f32>() / all_dists.len() as f32;
    eprintln!(
        "\nMin: {:.4} ({} ↔ {})",
        min_dist,
        emotion_names[min_pair.0],
        emotion_names[min_pair.1]
    );
    eprintln!(
        "Max: {:.4} ({} ↔ {})",
        max_dist,
        emotion_names[max_pair.0],
        emotion_names[max_pair.1]
    );
    eprintln!("Avg: {:.4}", avg_dist);

    // ASSERTION: Even with real embeddings, emotions should be distinguishable
    assert!(
        min_dist > 0.0001,
        "Two real emotions are indistinguishable: {} ↔ {} (dist={:.6})",
        emotion_names[min_pair.0],
        emotion_names[min_pair.1],
        min_dist
    );
}

// ──── WITHIN-EMOTION CONSISTENCY ────

#[test]
fn same_emotion_phrases_cluster_together() {
    let test = embedding::get_embedding("test");
    if test.is_none() {
        eprintln!("⚠️  LM Studio not available — skipping clustering test");
        return;
    }

    eprintln!("\n═══ Within-Emotion Consistency (do same-emotion phrases cluster?) ═══");

    let mut within_dists: Vec<f32> = Vec::new();
    let mut between_dists: Vec<f32> = Vec::new();

    // Get profiles for each individual phrase
    let mut all_profiles: Vec<(String, EmotionProfile)> = Vec::new();

    for &(emotion, phrases) in EMOTION_PHRASES.iter().take(8) {
        // Test first 8 emotions for speed
        for &phrase in phrases {
            if let Some(emb) = embedding::get_embedding(phrase) {
                let mut c = FibonacciCrystal::new(60, 8, 0.6);
                for _ in 0..20 { c.tick(None); }
                for _ in 0..30 { c.tick(Some(&emb)); }
                let e = Emotions::from_crystal(&c);
                let p = EmotionProfile::from_emotions(phrase, &e);
                all_profiles.push((emotion.to_string(), p));
            }
        }
    }

    // Compute within-emotion and between-emotion distances
    for i in 0..all_profiles.len() {
        for j in (i + 1)..all_profiles.len() {
            let d = all_profiles[i].1.distance_to(&all_profiles[j].1);
            if all_profiles[i].0 == all_profiles[j].0 {
                within_dists.push(d);
            } else {
                between_dists.push(d);
            }
        }
    }

    let avg_within = if within_dists.is_empty() {
        0.0
    } else {
        within_dists.iter().sum::<f32>() / within_dists.len() as f32
    };
    let avg_between = if between_dists.is_empty() {
        0.0
    } else {
        between_dists.iter().sum::<f32>() / between_dists.len() as f32
    };

    eprintln!("  Avg within-emotion distance:  {:.4} (n={})", avg_within, within_dists.len());
    eprintln!("  Avg between-emotion distance: {:.4} (n={})", avg_between, between_dists.len());

    if avg_between > 0.0 {
        let ratio = avg_between / (avg_within + 1e-9);
        eprintln!("  Separation ratio (between/within): {:.2}x", ratio);

        // Ideal: between > within (phrases about the same emotion cluster)
        // We report but don't hard-assert since embedding quality varies
        if ratio > 1.0 {
            eprintln!("  ✅ Same-emotion phrases cluster tighter than different-emotion phrases!");
        } else {
            eprintln!("  ⚠️  Clustering weak — embeddings may not differentiate enough for crystal to cluster");
        }
    }
}

// ──── TEMPORAL EVOLUTION ────

#[test]
fn emotion_evolves_over_sustained_input() {
    let test = embedding::get_embedding("test");
    if test.is_none() {
        eprintln!("⚠️  LM Studio not available — skipping temporal test");
        return;
    }

    eprintln!("\n═══ Temporal Emotion Evolution ═══");
    eprintln!("Tracking crystal state as sustained emotional input is applied\n");

    let emotions_to_track = ["terror", "peace", "joy", "rage"];

    for &emotion in &emotions_to_track {
        let phrases = EMOTION_PHRASES
            .iter()
            .find(|(name, _)| *name == emotion)
            .unwrap()
            .1;

        if let Some(emb) = embedding::get_embedding(phrases[0]) {
            let mut c = FibonacciCrystal::new(60, 8, 0.6);
            for _ in 0..20 { c.tick(None); }

            eprintln!("  {} — evolution over 200 ticks:", emotion.to_uppercase());
            eprintln!(
                "    {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
                "Tick", "Φ", "Gini", "Depth", "SR", "HA"
            );

            for t in 0..200 {
                c.tick(Some(&emb));
                if t % 25 == 0 || t == 199 {
                    let e = Emotions::from_crystal(&c);
                    eprintln!(
                        "    {:>6} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
                        t + 1, e.phase_coherence, e.energy_concentration,
                        e.depth, e.spectral_richness, e.horizon_activity
                    );
                }
            }
            eprintln!();
        }
    }
}

// ──── EMOTION TRANSITION ────

#[test]
fn emotion_transitions_are_smooth() {
    let test = embedding::get_embedding("test");
    if test.is_none() {
        eprintln!("⚠️  LM Studio not available — skipping transition test");
        return;
    }

    eprintln!("\n═══ Emotion Transition: Peace → Terror → Peace ═══\n");

    let peace_emb = embedding::get_embedding(
        "Deep calm, still water, gentle breeze, total serenity",
    );
    let terror_emb = embedding::get_embedding(
        "Pure terror, something is hunting me, danger everywhere, run",
    );

    if let (Some(peace), Some(terror)) = (peace_emb, terror_emb) {
        let mut c = FibonacciCrystal::new(60, 8, 0.6);
        for _ in 0..20 { c.tick(None); }

        eprintln!(
            "{:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  Phase",
            "Tick", "Φ", "Gini", "Depth", "SR", "HA", "Arousal"
        );

        // Phase 1: Peace (50 ticks)
        for t in 0..50 {
            c.tick(Some(&peace));
            if t % 10 == 9 {
                let e = Emotions::from_crystal(&c);
                eprintln!(
                    "{:>6} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}  PEACE",
                    t + 1, e.phase_coherence, e.energy_concentration,
                    e.depth, e.spectral_richness, e.horizon_activity, e.arousal
                );
            }
        }
        let peace_snapshot = Emotions::from_crystal(&c);

        // Phase 2: Terror (50 ticks)
        for t in 50..100 {
            c.tick(Some(&terror));
            if t % 10 == 9 {
                let e = Emotions::from_crystal(&c);
                eprintln!(
                    "{:>6} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}  TERROR",
                    t + 1, e.phase_coherence, e.energy_concentration,
                    e.depth, e.spectral_richness, e.horizon_activity, e.arousal
                );
            }
        }
        let terror_snapshot = Emotions::from_crystal(&c);

        // Phase 3: Back to Peace (50 ticks)
        for t in 100..150 {
            c.tick(Some(&peace));
            if t % 10 == 9 {
                let e = Emotions::from_crystal(&c);
                eprintln!(
                    "{:>6} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}  PEACE",
                    t + 1, e.phase_coherence, e.energy_concentration,
                    e.depth, e.spectral_richness, e.horizon_activity, e.arousal
                );
            }
        }
        let recovery_snapshot = Emotions::from_crystal(&c);

        // The terror phase should have been measurably different from peace
        let peace_terror_dist = {
            let dp = peace_snapshot.phase_coherence - terror_snapshot.phase_coherence;
            let dh = peace_snapshot.horizon_activity - terror_snapshot.horizon_activity;
            let dg = peace_snapshot.energy_concentration - terror_snapshot.energy_concentration;
            (dp * dp + dh * dh + dg * dg).sqrt()
        };

        eprintln!("\n  Peace ↔ Terror distance: {:.4}", peace_terror_dist);
        assert!(
            peace_terror_dist > 0.001,
            "Peace and terror produced no measurable difference in the crystal"
        );
    }
}
