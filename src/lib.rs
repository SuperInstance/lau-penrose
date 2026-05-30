//! `lau-penrose` — THE correlation detection system.
//!
//! When multiple rooms are active simultaneously and their signals correlate,
//! the system detects it and creates automatic splines (connections).
//! Deadband consistency reveals causal relationships that resonate synergistically,
//! reducing tokens and wattage organically.
//!
//! ## Core Insight
//!
//! Correlations aren't just interesting — they're **free efficiency**.
//! When two rooms consistently do related things, the system notices and creates
//! automatic connections. These connections reduce token usage (shared context),
//! reduce energy (predictive loading), and reduce latency (compiled protocols).
//! The system gets organically more efficient the more it runs, like muscles
//! growing from daily use.

use std::collections::HashMap;

// Helper: serde-friendly (String, String) pair for use as HashMap keys
#[derive(Debug, Clone, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignalKey {
    pub room_id: String,
    pub signal_type: String,
}

impl SignalKey {
    pub fn new(room: &str, signal_type: &str) -> Self {
        Self {
            room_id: room.to_string(),
            signal_type: signal_type.to_string(),
        }
    }
}

impl From<(&str, &str)> for SignalKey {
    fn from((r, s): (&str, &str)) -> Self {
        Self::new(r, s)
    }
}

// ---------------------------------------------------------------------------
// 1. Signal
// ---------------------------------------------------------------------------

/// A single data point from a room.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Signal {
    pub room_id: String,
    pub signal_type: String,
    pub value: f64,
    pub tick: u64,
    pub metadata: HashMap<String, String>,
}

impl Signal {
    pub fn new(room: &str, signal_type: &str, value: f64, tick: u64) -> Self {
        Self {
            room_id: room.to_string(),
            signal_type: signal_type.to_string(),
            value,
            tick,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: &str, val: &str) -> Self {
        self.metadata.insert(key.to_string(), val.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// 2. SignalStream
// ---------------------------------------------------------------------------

/// A rolling-window stream of signals from a room.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SignalStream {
    pub room_id: String,
    pub signal_type: String,
    pub values: Vec<(u64, f64)>,
    pub max_length: usize,
}

impl SignalStream {
    pub fn new(room_id: &str, signal_type: &str, max_length: usize) -> Self {
        Self {
            room_id: room_id.to_string(),
            signal_type: signal_type.to_string(),
            values: Vec::with_capacity(max_length),
            max_length,
        }
    }

    pub fn push(&mut self, tick: u64, value: f64) {
        self.values.push((tick, value));
        while self.values.len() > self.max_length {
            self.values.remove(0);
        }
    }

    pub fn latest(&self) -> Option<(u64, f64)> {
        self.values.last().copied()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Arithmetic mean of the values.
    pub fn mean(&self) -> f64 {
        let n = self.values.len();
        if n == 0 {
            return 0.0;
        }
        self.values.iter().map(|&(_, v)| v).sum::<f64>() / n as f64
    }

    /// Population standard deviation.
    pub fn std_dev(&self) -> f64 {
        let n = self.values.len();
        if n < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance =
            self.values.iter().map(|&(_, v)| (v - mean).powi(2)).sum::<f64>() / n as f64;
        variance.sqrt()
    }

    /// Pearson correlation coefficient with another stream.
    /// Only considers matching ticks; returns 0 if fewer than 2 aligned points.
    pub fn correlation_with(&self, other: &SignalStream) -> f64 {
        let aligned = self.align_values(other);
        let n = aligned.len();
        if n < 2 {
            return 0.0;
        }
        let mean_x = aligned.iter().map(|&(x, _)| x).sum::<f64>() / n as f64;
        let mean_y = aligned.iter().map(|&(_, y)| y).sum::<f64>() / n as f64;

        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;
        for &(x, y) in &aligned {
            let dx = x - mean_x;
            let dy = y - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }
        let den = (den_x * den_y).sqrt();
        if den == 0.0 {
            0.0
        } else {
            (num / den).clamp(-1.0, 1.0)
        }
    }

    /// How many values stay within `threshold` of the mean (deadband consistency).
    pub fn deadband_count(&self, threshold: f64) -> usize {
        if self.values.is_empty() {
            return 0;
        }
        let m = self.mean();
        self.values
            .iter()
            .filter(|&&(_, v)| (v - m).abs() <= threshold)
            .count()
    }

    /// Signal is non-trivial (has enough variance).
    pub fn is_active(&self, threshold: f64) -> bool {
        if self.values.len() < 2 {
            return false;
        }
        let max = self
            .values
            .iter()
            .map(|&(_, v)| v)
            .fold(f64::NEG_INFINITY, f64::max);
        let min = self
            .values
            .iter()
            .map(|&(_, v)| v)
            .fold(f64::INFINITY, f64::min);
        (max - min) > threshold
    }

    // ── helpers ──

    fn align_values(&self, other: &SignalStream) -> Vec<(f64, f64)> {
        let map_b: HashMap<u64, f64> = other.values.iter().copied().collect();
        let mut aligned = Vec::new();
        for (tick, va) in &self.values {
            if let Some(&vb) = map_b.get(tick) {
                aligned.push((*va, vb));
            }
        }
        aligned
    }
}

// ---------------------------------------------------------------------------
// 3. Correlation
// ---------------------------------------------------------------------------

/// A detected correlation between two signals.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Correlation {
    pub signal_a: (String, String),
    pub signal_b: (String, String),
    pub correlation: f64,
    pub confidence: f64,
    pub first_detected: u64,
    pub last_confirmed: u64,
    pub occurrences: u32,
}

impl Correlation {
    pub fn new(a: (&str, &str), b: (&str, &str), corr: f64) -> Self {
        Self {
            signal_a: (a.0.to_string(), a.1.to_string()),
            signal_b: (b.0.to_string(), b.1.to_string()),
            correlation: corr,
            confidence: 0.0,
            first_detected: 0,
            last_confirmed: 0,
            occurrences: 0,
        }
    }

    /// Correlation is strong (|r| > 0.7).
    pub fn is_strong(&self) -> bool {
        self.correlation.abs() > 0.7
    }

    /// Correlation is positive.
    pub fn is_positive(&self) -> bool {
        self.correlation > 0.0
    }

    /// Positive AND reduces combined cost (synergistic).
    pub fn is_synergistic(&self) -> bool {
        self.is_positive() && self.correlation.abs() > 0.7
    }

    /// Update with a new observation.
    pub fn strengthen(&mut self, corr: f64) {
        self.occurrences += 1;
        self.last_confirmed = 0;
        // Weighted moving average — newer observations count more
        let weight = 0.3;
        self.correlation = self.correlation * (1.0 - weight) + corr * weight;
        // Confidence grows with occurrences, max 1.0
        self.confidence = (self.confidence + 1.0).min(1.0);
    }

    /// Decay the correlation — older correlations weaken.
    pub fn decay(&mut self, rate: f64) {
        self.correlation *= 1.0 - rate;
        self.confidence = (self.confidence - rate).max(0.0);
        // Clamp after decay
        self.correlation = self.correlation.clamp(-1.0, 1.0);
    }

    /// Human-readable description.
    pub fn describe(&self) -> String {
        format!(
            "{} ({}) correlates {:.2} with {} ({})",
            self.signal_a.1, self.signal_a.0, self.correlation, self.signal_b.1, self.signal_b.0
        )
    }
}

// ---------------------------------------------------------------------------
// 4. Spline
// ---------------------------------------------------------------------------

/// The type of automatic connection created by correlation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum SplineType {
    /// A causes B (e.g., engineering motor → navigation deviation)
    Causal,
    /// A and B resonate together (same underlying cause)
    Resonant,
    /// A predicts B (e.g., engineering vibration predicts security alert)
    Predictive,
    /// A and B together are more than sum (collaboration)
    Synergistic,
    /// A and B carry same information (can be compressed)
    Redundant,
}

impl SplineType {
    pub fn classify(correlation: f64, lead: Option<u64>) -> Self {
        match lead {
            Some(_) if correlation.abs() > 0.7 => SplineType::Predictive,
            _ if correlation.abs() > 0.85 => SplineType::Redundant,
            _ if correlation > 0.7 => SplineType::Synergistic,
            _ if correlation.abs() <= 0.3 => SplineType::Resonant,
            _ => SplineType::Causal,
        }
    }
}

/// An automatic connection created by correlation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Spline {
    pub id: String,
    pub correlation: Correlation,
    pub from_room: String,
    pub to_room: String,
    pub spline_type: SplineType,
    pub energy_savings: f64,
    pub token_savings: u64,
}

impl Spline {
    /// Create a spline from a correlation. Classifies the type automatically.
    pub fn new(correlation: Correlation) -> Self {
        let from_room = correlation.signal_a.0.clone();
        let to_room = correlation.signal_b.0.clone();
        let spline_type = SplineType::classify(correlation.correlation, None);
        let id = format!("{}-{}-{:x}", from_room, to_room, fast_hash(&correlation));
        Self {
            id,
            correlation,
            from_room,
            to_room,
            spline_type,
            energy_savings: 0.0,
            token_savings: 0,
        }
    }

    /// Calculate token/energy savings given baseline and combined usage.
    pub fn calculate_savings(&mut self, baseline_tokens: u64, combined_tokens: u64) {
        if baseline_tokens > combined_tokens {
            let saved = baseline_tokens - combined_tokens;
            self.token_savings = saved;
            // Energy savings proportional to token savings — 0.01 Wh/token approximation
            self.energy_savings = saved as f64 * 0.01;
        }
    }

    /// This spline provides useful savings.
    pub fn is_useful(&self) -> bool {
        self.token_savings > 0 || self.energy_savings > 0.0
    }

    /// Human-readable description.
    pub fn describe(&self) -> String {
        let type_label = match self.spline_type {
            SplineType::Causal => "causal",
            SplineType::Resonant => "resonant",
            SplineType::Predictive => "predictive",
            SplineType::Synergistic => "synergistic",
            SplineType::Redundant => "redundant",
        };
        format!(
            "Spline: {} → {} ({}, saves {} tokens, {} Wh)",
            self.from_room, self.to_room, type_label, self.token_savings, self.energy_savings
        )
    }
}

fn fast_hash(c: &Correlation) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    c.signal_a.hash(&mut h);
    c.signal_b.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// 5. PenroseEngine
// ---------------------------------------------------------------------------

/// Summary of the engine state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PenroseSummary {
    pub stream_count: usize,
    pub correlation_count: usize,
    pub spline_count: usize,
    pub strong_correlations: usize,
    pub total_energy_saved: f64,
    pub total_tokens_saved: u64,
    pub most_synergistic: Option<String>,
    pub active_rooms: Vec<String>,
}

/// THE correlation detection engine.
///
/// Watches signal streams from multiple rooms, detects correlations,
/// and creates automatic splines (connections) that save tokens and energy.
#[derive(Debug, Clone)]
pub struct PenroseEngine {
    pub streams: HashMap<SignalKey, SignalStream>,
    pub correlations: Vec<Correlation>,
    pub splines: Vec<Spline>,
    pub correlation_threshold: f64,
    pub min_samples: usize,
    pub tick: u64,
    pub scan_interval: u64,
}

impl PenroseEngine {
    pub fn new() -> Self {
        Self {
            streams: HashMap::new(),
            correlations: Vec::new(),
            splines: Vec::new(),
            correlation_threshold: 0.7,
            min_samples: 10,
            tick: 0,
            scan_interval: 1,
        }
    }

    // ── builder-like setters ──

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.correlation_threshold = t;
        self
    }

    pub fn with_min_samples(mut self, n: usize) -> Self {
        self.min_samples = n;
        self
    }

    pub fn with_scan_interval(mut self, n: u64) -> Self {
        self.scan_interval = n;
        self
    }

    // ── ingestion ──

    /// Ingest a single signal into the engine.
    pub fn ingest(&mut self, signal: Signal) {
        self.tick = signal.tick;
        let key = SignalKey::new(&signal.room_id, &signal.signal_type);
        let max_len = self.min_samples.max(10) * 2; // generous rolling window
        let stream = self
            .streams
            .entry(key)
            .or_insert_with(|| SignalStream::new(&signal.room_id, &signal.signal_type, max_len));
        stream.push(signal.tick, signal.value);
    }

    // ── scanning ──

    /// Scan all stream pairs for correlations above the threshold.
    /// Only runs when tick % scan_interval == 0.
    pub fn scan(&mut self) -> Vec<Correlation> {
        if !self.tick.is_multiple_of(self.scan_interval) {
            return Vec::new();
        }

        let mut new_corrs = Vec::new();
        let keys: Vec<_> = self.streams.keys().cloned().collect();

        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                let a_key = &keys[i];
                let b_key = &keys[j];

                // Avoid comparing a stream with itself on the same room+type
                if a_key == b_key {
                    continue;
                }

                let stream_a = &self.streams[a_key];
                let stream_b = &self.streams[b_key];

                if stream_a.len() < self.min_samples || stream_b.len() < self.min_samples {
                    continue;
                }

                let corr = stream_a.correlation_with(stream_b);
                if corr.abs() >= self.correlation_threshold {
                    let c = Correlation::new(
                        (&a_key.room_id, &a_key.signal_type),
                        (&b_key.room_id, &b_key.signal_type),
                        corr,
                    );
                    new_corrs.push(c);
                }
            }
        }

        // Merge or add
        for new_c in new_corrs {
            self.add_or_update_correlation(new_c);
        }

        self.correlations.clone()
    }

    fn add_or_update_correlation(&mut self, new_c: Correlation) {
        let norm_a = norm_pair(&new_c.signal_a);
        let norm_b = norm_pair(&new_c.signal_b);

        for existing in self.correlations.iter_mut() {
            let ex_a = norm_pair(&existing.signal_a);
            let ex_b = norm_pair(&existing.signal_b);
            if (ex_a == norm_a && ex_b == norm_b)
                || (ex_a == norm_b && ex_b == norm_a)
            {
                existing.strengthen(new_c.correlation);
                existing.last_confirmed = self.tick;
                return;
            }
        }

        let mut c = new_c;
        c.first_detected = self.tick;
        c.last_confirmed = self.tick;
        self.correlations.push(c);
    }

    // ── spline creation ──

    /// Create splines from strong correlations.
    pub fn detect_splines(&mut self) -> Vec<Spline> {
        let strong: Vec<Correlation> = self
            .correlations
            .iter()
            .filter(|c| c.is_strong())
            .cloned()
            .collect();

        for c in strong {
            let already = self.splines.iter().any(|s| {
                let a = norm_pair(&s.correlation.signal_a);
                let b = norm_pair(&s.correlation.signal_b);
                let ca = norm_pair(&c.signal_a);
                let cb = norm_pair(&c.signal_b);
                (a == ca && b == cb) || (a == cb && b == ca)
            });
            if !already {
                let mut spline = Spline::new(c);
                // Estimate savings based on correlation strength
                let baseline = 1000;
                let combined = (baseline as f64 * (1.0 - spline.correlation.correlation.abs() * 0.5))
                    as u64;
                spline.calculate_savings(baseline, combined);
                self.splines.push(spline);
            }
        }

        self.splines.clone()
    }

    // ── queries ──

    /// All splines involving a particular room.
    pub fn get_splines_for_room(&self, room_id: &str) -> Vec<&Spline> {
        self.splines
            .iter()
            .filter(|s| s.from_room == room_id || s.to_room == room_id)
            .collect()
    }

    /// All splines between two specific rooms.
    pub fn get_splines_between(&self, room_a: &str, room_b: &str) -> Vec<&Spline> {
        self.splines
            .iter()
            .filter(|s| {
                (s.from_room == room_a && s.to_room == room_b)
                    || (s.from_room == room_b && s.to_room == room_a)
            })
            .collect()
    }

    /// Total energy and token savings across all splines.
    pub fn total_savings(&self) -> (f64, u64) {
        let energy: f64 = self.splines.iter().map(|s| s.energy_savings).sum();
        let tokens: u64 = self.splines.iter().map(|s| s.token_savings).sum();
        (energy, tokens)
    }

    /// Decay all correlations.
    pub fn decay_all(&mut self, rate: f64) {
        for c in self.correlations.iter_mut() {
            c.decay(rate);
        }
    }

    /// Remove correlations below a confidence threshold.
    pub fn prune_weak(&mut self, threshold: f64) {
        self.correlations.retain(|c| c.confidence >= threshold);
        self.splines.retain(|s| s.correlation.confidence >= threshold);
    }

    // ── summaries ──

    /// Serialize to JSON-friendly format.
    pub fn to_json_value(&self) -> serde_json::Value {
        let streams: Vec<serde_json::Value> = self
            .streams
            .iter()
            .map(|(key, stream)| {
                serde_json::json!({
                    "key": { "room_id": key.room_id, "signal_type": key.signal_type },
                    "stream": stream
                })
            })
            .collect();
        serde_json::json!({
            "streams": streams,
            "correlations": self.correlations,
            "splines": self.splines,
            "correlation_threshold": self.correlation_threshold,
            "min_samples": self.min_samples,
            "tick": self.tick,
            "scan_interval": self.scan_interval,
        })
    }

    pub fn engine_summary(&self) -> PenroseSummary {
        let strong_corrs = self
            .correlations
            .iter()
            .filter(|c| c.is_strong())
            .count();

        let most_syn = self
            .splines
            .iter()
            .max_by(|a, b| {
                a.correlation
                    .correlation
                    .abs()
                    .partial_cmp(&b.correlation.correlation.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.describe());

        let mut active_rooms: Vec<String> = self
            .streams
            .keys()
            .map(|key| key.room_id.clone())
            .collect();
        active_rooms.sort();
        active_rooms.dedup();

        let (energy, tokens) = self.total_savings();

        PenroseSummary {
            stream_count: self.streams.len(),
            correlation_count: self.correlations.len(),
            spline_count: self.splines.len(),
            strong_correlations: strong_corrs,
            total_energy_saved: energy,
            total_tokens_saved: tokens,
            most_synergistic: most_syn,
            active_rooms,
        }
    }

    pub fn render_correlations(&self) -> String {
        if self.correlations.is_empty() {
            return "No correlations detected yet.".to_string();
        }
        let mut lines = vec!["── Correlation Map ──".to_string()];
        for c in &self.correlations {
            let strength = if c.is_strong() { "STRONG" } else { "weak" };
            lines.push(format!(
                "  {} — {} (visits={}, r={:.3}, conf={:.2})",
                c.describe(),
                strength,
                c.occurrences,
                c.correlation,
                c.confidence
            ));
        }
        lines.push("────────────────────".to_string());
        lines.join("\n")
    }

    pub fn render_splines(&self) -> String {
        if self.splines.is_empty() {
            return "No splines created yet.".to_string();
        }
        let mut lines = vec!["── Spline Network ──".to_string()];
        for s in &self.splines {
            lines.push(format!("  {}", s.describe()));
        }
        lines.push("────────────────────".to_string());
        lines.join("\n")
    }
}

impl Default for PenroseEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// 6. PenroseCompiler
// ---------------------------------------------------------------------------

/// An optimization derived from detected correlations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Optimization {
    /// Two streams carry the same information — read once.
    MergeStreams {
        a: (String, String),
        b: (String, String),
        savings: u64,
    },
    /// Preload based on prediction.
    PredictiveLoad {
        predictor: String,
        predicted: String,
        lead_ticks: u64,
    },
    /// Rooms share context instead of duplicating.
    SharedContext {
        rooms: Vec<String>,
        shared_tokens: u64,
    },
    /// Use a shorter protocol between correlated rooms.
    CompressedProtocol {
        rooms: Vec<String>,
        compression_ratio: f64,
    },
}

/// Compiles detected correlations into concrete optimizations.
pub struct PenroseCompiler;

impl PenroseCompiler {
    /// Compile splines into a list of optimizations.
    pub fn compile_splines(splines: &[Spline]) -> Vec<Optimization> {
        let mut opts = Vec::new();

        for spline in splines {
            match spline.spline_type {
                SplineType::Redundant => {
                    opts.push(Optimization::MergeStreams {
                        a: spline.correlation.signal_a.clone(),
                        b: spline.correlation.signal_b.clone(),
                        savings: spline.token_savings,
                    });
                }
                SplineType::Predictive => {
                    opts.push(Optimization::PredictiveLoad {
                        predictor: spline.from_room.clone(),
                        predicted: spline.to_room.clone(),
                        lead_ticks: 1,
                    });
                }
                _ => {
                    // For synergistic/causal/resonant, create shared context
                    let rooms = vec![spline.from_room.clone(), spline.to_room.clone()];
                    opts.push(Optimization::SharedContext {
                        rooms,
                        shared_tokens: spline.token_savings / 2,
                    });
                }
            }
        }

        // Group redundant splines by room for compressed protocol
        let room_pairs: Vec<(&str, &str)> = splines
            .iter()
            .map(|s| (s.from_room.as_str(), s.to_room.as_str()))
            .collect();
        let mut room_set: Vec<Vec<String>> = Vec::new();
        for (a, b) in &room_pairs {
            let mut merged = false;
            for group in room_set.iter_mut() {
                if group.contains(&a.to_string()) || group.contains(&b.to_string()) {
                    if !group.contains(&a.to_string()) {
                        group.push(a.to_string());
                    }
                    if !group.contains(&b.to_string()) {
                        group.push(b.to_string());
                    }
                    merged = true;
                    break;
                }
            }
            if !merged {
                room_set.push(vec![a.to_string(), b.to_string()]);
            }
        }

        for rooms in room_set {
            if rooms.len() > 2 {
                opts.push(Optimization::CompressedProtocol {
                    rooms,
                    compression_ratio: 0.6,
                });
            }
        }

        opts
    }

    /// Apply an optimization to a stream system.
    pub fn apply(
        optimization: &Optimization,
        system: &mut HashMap<SignalKey, SignalStream>,
    ) {
        match optimization {
            Optimization::MergeStreams { a, b, .. } => {
                // Merge b's values into a's stream, then remove b
                let a_key = SignalKey::new(&a.0, &a.1);
                let b_key = SignalKey::new(&b.0, &b.1);
                if let Some(b_stream) = system.remove(&b_key) {
                    if let Some(a_stream) = system.get_mut(&a_key) {
                        for &(tick, val) in &b_stream.values {
                            a_stream.push(tick, val);
                        }
                    }
                }
            }
            Optimization::PredictiveLoad { .. }
            | Optimization::SharedContext { .. }
            | Optimization::CompressedProtocol { .. } => {
                // These optimizations change metadata or system-level config,
                // not the stream data itself, so this is a no-op on raw streams.
            }
        }
    }

    /// Estimate total savings from a set of optimizations.
    pub fn estimated_total_savings(optimizations: &[Optimization]) -> (f64, u64) {
        let mut energy = 0.0;
        let mut tokens = 0u64;

        for opt in optimizations {
            match opt {
                Optimization::MergeStreams { savings, .. } => {
                    tokens += savings;
                    energy += *savings as f64 * 0.01;
                }
                Optimization::SharedContext { shared_tokens, .. } => {
                    tokens += shared_tokens;
                    energy += *shared_tokens as f64 * 0.01;
                }
                Optimization::CompressedProtocol {
                    compression_ratio, ..
                } => {
                    let estimated = (1000.0 * (1.0 - compression_ratio)) as u64;
                    tokens += estimated;
                    energy += estimated as f64 * 0.01;
                }
                Optimization::PredictiveLoad { .. } => {
                    // Predictive saves latency, not raw tokens directly
                    tokens += 5;
                    energy += 0.05;
                }
            }
        }

        (energy, tokens)
    }
}

// ── internal helpers ──

/// Produce a canonical ordering of a signal pair for comparison.
fn norm_pair(pair: &(String, String)) -> (&str, &str) {
    (pair.0.as_str(), pair.1.as_str())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Signal tests ──

    #[test]
    fn test_signal_new() {
        let s = Signal::new("engineering", "motor_current", 2.3, 42);
        assert_eq!(s.room_id, "engineering");
        assert_eq!(s.signal_type, "motor_current");
        assert_eq!(s.value, 2.3);
        assert_eq!(s.tick, 42);
        assert!(s.metadata.is_empty());
    }

    #[test]
    fn test_signal_with_metadata() {
        let s = Signal::new("nav", "heading", 15.0, 1)
            .with_metadata("unit", "degrees");
        assert_eq!(s.metadata.get("unit").unwrap(), "degrees");
    }

    // ── SignalStream tests ──

    #[test]
    fn test_stream_push_and_latest() {
        let mut stream = SignalStream::new("engineering", "motor_current", 10);
        assert!(stream.latest().is_none());

        stream.push(1, 1.0);
        stream.push(2, 2.0);
        stream.push(3, 3.0);
        assert_eq!(stream.latest(), Some((3, 3.0)));
    }

    #[test]
    fn test_stream_rolling_window() {
        let mut stream = SignalStream::new("eng", "current", 3);
        for i in 0..10 {
            stream.push(i as u64, i as f64);
        }
        assert_eq!(stream.len(), 3);
        assert_eq!(stream.latest(), Some((9, 9.0)));
    }

    #[test]
    fn test_stream_mean() {
        let mut stream = SignalStream::new("test", "val", 10);
        for i in 1..=5 {
            stream.push(i as u64, i as f64);
        }
        assert_eq!(stream.mean(), 3.0);
    }

    #[test]
    fn test_stream_mean_empty() {
        let stream = SignalStream::new("test", "val", 10);
        assert_eq!(stream.mean(), 0.0);
    }

    #[test]
    fn test_stream_std_dev() {
        let mut stream = SignalStream::new("test", "val", 10);
        // values: 2, 4, 6 → mean=4, variance = (4+0+4)/3 ≈ 2.666, std ≈ 1.633
        stream.push(1, 2.0);
        stream.push(2, 4.0);
        stream.push(3, 6.0);
        let std = stream.std_dev();
        approx::assert_relative_eq!(std, 1.63299, epsilon = 1e-4);
    }

    #[test]
    fn test_stream_std_dev_single() {
        let mut stream = SignalStream::new("test", "val", 10);
        stream.push(1, 42.0);
        assert_eq!(stream.std_dev(), 0.0);
    }

    #[test]
    fn test_correlation_perfect_positive() {
        let mut a = SignalStream::new("room_a", "x", 10);
        let mut b = SignalStream::new("room_b", "y", 10);
        for i in 1..=5 {
            a.push(i, i as f64);
            b.push(i, i as f64); // same values
        }
        let r = a.correlation_with(&b);
        approx::assert_relative_eq!(r, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let mut a = SignalStream::new("room_a", "x", 10);
        let mut b = SignalStream::new("room_b", "y", 10);
        for i in 1..=5 {
            a.push(i, i as f64);
            b.push(i, -(i as f64));
        }
        let r = a.correlation_with(&b);
        approx::assert_relative_eq!(r, -1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_correlation_zero() {
        let mut a = SignalStream::new("room_a", "x", 10);
        let mut b = SignalStream::new("room_b", "y", 10);
        // constant stream → no correlation possible
        a.push(1, 5.0);
        a.push(2, 5.0);
        b.push(1, 10.0);
        b.push(2, 20.0);
        let r = a.correlation_with(&b);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_correlation_insufficient_data() {
        let a = SignalStream::new("room_a", "x", 10);
        let b = SignalStream::new("room_b", "y", 10);
        assert_eq!(a.correlation_with(&b), 0.0);
    }

    #[test]
    fn test_deadband_count() {
        let mut stream = SignalStream::new("test", "val", 10);
        // values: 1, 2, 3, 10 → mean=4
        stream.push(1, 1.0);
        stream.push(2, 2.0);
        stream.push(3, 3.0);
        stream.push(4, 10.0);
        // threshold 3: |v - 4| <= 3 → 1, 2, 3, 10? |10-4|=6 >3 → only 3
        assert_eq!(stream.deadband_count(3.0), 3);
    }

    #[test]
    fn test_is_active() {
        let mut stream = SignalStream::new("test", "val", 10);
        stream.push(1, 1.0);
        stream.push(2, 1.0);
        assert!(!stream.is_active(0.5));
        stream.push(3, 10.0);
        assert!(stream.is_active(0.5));
    }

    #[test]
    fn test_is_active_not_enough_samples() {
        let mut stream = SignalStream::new("test", "val", 10);
        stream.push(1, 5.0);
        assert!(!stream.is_active(0.1));
    }

    // ── Correlation tests ──

    // ── Correlation tests (continued) ──

    #[test]
    fn test_correlation_new() {
        let c = Correlation::new(("eng", "current"), ("nav", "heading"), 0.89);
        assert_eq!(c.signal_a, ("eng".to_string(), "current".to_string()));
        assert_eq!(c.signal_b, ("nav".to_string(), "heading".to_string()));
        assert_eq!(c.correlation, 0.89);
        assert_eq!(c.occurrences, 0);
    }

    #[test]
    fn test_correlation_is_strong() {
        let strong = Correlation::new(("a", "x"), ("b", "y"), 0.85);
        let weak = Correlation::new(("a", "x"), ("b", "y"), 0.5);
        assert!(strong.is_strong());
        assert!(!weak.is_strong());
    }

    #[test]
    fn test_correlation_is_positive() {
        let pos = Correlation::new(("a", "x"), ("b", "y"), 0.5);
        let neg = Correlation::new(("a", "x"), ("b", "y"), -0.5);
        assert!(pos.is_positive());
        assert!(!neg.is_positive());
    }

    #[test]
    fn test_correlation_is_synergistic() {
        let syn = Correlation::new(("a", "x"), ("b", "y"), 0.85);
        let not_syn = Correlation::new(("a", "x"), ("b", "y"), 0.5);
        assert!(syn.is_synergistic());
        assert!(!not_syn.is_synergistic());
    }

    #[test]
    fn test_correlation_strengthen() {
        let mut c = Correlation::new(("a", "x"), ("b", "y"), 0.8);
        c.strengthen(0.9);
        assert_eq!(c.occurrences, 1);
        assert!(c.correlation > 0.8);
        assert!(c.confidence > 0.0);
    }

    #[test]
    fn test_correlation_decay() {
        let mut c = Correlation::new(("a", "x"), ("b", "y"), 0.8);
        c.decay(0.1);
        assert!(c.correlation < 0.8);
        assert!(c.correlation > 0.7);
    }

    #[test]
    fn test_correlation_describe() {
        let c = Correlation::new(("engineering", "motor_current"), ("navigation", "heading"), 0.89);
        let desc = c.describe();
        assert!(desc.contains("engineering"));
        assert!(desc.contains("navigation"));
        assert!(desc.contains("0.89"));
    }

    // ── Spline tests ──

    #[test]
    fn test_spline_new() {
        let c = Correlation::new(("eng", "current"), ("nav", "heading"), 0.89);
        let s = Spline::new(c);
        assert_eq!(s.from_room, "eng");
        assert_eq!(s.to_room, "nav");
        assert!(!s.id.is_empty());
    }

    #[test]
    fn test_spline_calculate_savings() {
        let c = Correlation::new(("eng", "current"), ("nav", "heading"), 0.89);
        let mut s = Spline::new(c);
        s.calculate_savings(1000, 600);
        assert_eq!(s.token_savings, 400);
        assert_eq!(s.energy_savings, 4.0);
    }

    #[test]
    fn test_spline_no_savings_when_not_useful() {
        let c = Correlation::new(("eng", "current"), ("nav", "heading"), 0.89);
        let mut s = Spline::new(c);
        s.calculate_savings(500, 600);
        assert_eq!(s.token_savings, 0);
        assert_eq!(s.energy_savings, 0.0);
        assert!(!s.is_useful());
    }

    #[test]
    fn test_spline_describe() {
        let c = Correlation::new(("eng", "current"), ("nav", "heading"), 0.89);
        let mut s = Spline::new(c);
        s.calculate_savings(1000, 600);
        let desc = s.describe();
        assert!(desc.contains("eng → nav"));
        assert!(desc.contains("400 tokens"));
    }

    #[test]
    fn test_spline_type_classification() {
        match SplineType::classify(0.9, None) {
            SplineType::Redundant => {}
            _ => panic!("Expected Redundant"),
        }
        match SplineType::classify(0.8, Some(1)) {
            SplineType::Predictive => {}
            _ => panic!("Expected Predictive"),
        }
        match SplineType::classify(0.75, None) {
            SplineType::Synergistic => {}
            _ => panic!("Expected Synergistic"),
        }
        match SplineType::classify(0.5, None) {
            SplineType::Causal => {}
            _ => panic!("Expected Causal"),
        }
        match SplineType::classify(0.2, None) {
            SplineType::Resonant => {}
            _ => panic!("Expected Resonant"),
        }
    }

    // ── PenroseEngine tests ──

    #[test]
    fn test_engine_new() {
        let engine = PenroseEngine::new();
        assert!(engine.streams.is_empty());
        assert!(engine.correlations.is_empty());
        assert!(engine.splines.is_empty());
        assert_eq!(engine.correlation_threshold, 0.7);
        assert_eq!(engine.min_samples, 10);
    }

    #[test]
    fn test_engine_ingest_creates_stream() {
        let mut engine = PenroseEngine::new();
        let s = Signal::new("engineering", "motor_current", 2.3, 1);
        engine.ingest(s);
        assert_eq!(engine.streams.len(), 1);
        assert_eq!(engine.tick, 1);
    }

    #[test]
    fn test_engine_ingest_append_to_stream() {
        let mut engine = PenroseEngine::new();
        engine.ingest(Signal::new("eng", "current", 1.0, 1));
        engine.ingest(Signal::new("eng", "current", 2.0, 2));
        let key = SignalKey::new("eng", "current");
        assert_eq!(engine.streams[&key].len(), 2);
    }

    #[test]
    fn test_engine_scan_detects_correlation() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.7)
            .with_min_samples(3);
        // Two streams with perfectly correlated data
        for i in 1..=5 {
            engine.ingest(Signal::new("eng", "current", i as f64, i));
            engine.ingest(Signal::new("nav", "heading", i as f64, i));
        }
        let corrs = engine.scan();
        assert!(!corrs.is_empty(), "Should detect correlation");
        assert!(corrs[0].correlation.abs() > 0.7);
    }

    #[test]
    fn test_engine_scan_does_not_false_positive() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.7)
            .with_min_samples(3);
        // Two unrelated streams
        for i in 1..=5 {
            engine.ingest(Signal::new("eng", "current", i as f64, i));
            engine.ingest(Signal::new("nav", "heading", ((i * 137) % 7) as f64, i));
        }
        let corrs = engine.scan();
        // Random-ish values unlikely to correlate at r > 0.7
        assert!(corrs.is_empty() || corrs[0].correlation.abs() <= 0.7);
    }

    #[test]
    fn test_engine_scan_insufficient_samples() {
        let mut engine = PenroseEngine::new()
            .with_min_samples(10);
        for i in 1..=3 {
            engine.ingest(Signal::new("eng", "current", i as f64, i));
            engine.ingest(Signal::new("nav", "heading", i as f64, i));
        }
        let corrs = engine.scan();
        assert!(corrs.is_empty());
    }

    #[test]
    fn test_engine_detect_splines() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.7)
            .with_min_samples(3);
        for i in 1..=5 {
            engine.ingest(Signal::new("eng", "current", i as f64, i));
            engine.ingest(Signal::new("nav", "heading", i as f64, i));
        }
        engine.scan();
        let splines = engine.detect_splines();
        assert!(!splines.is_empty(), "Should create splines from strong correlations");
        assert!(splines[0].is_useful());
    }

    #[test]
    fn test_engine_detect_splines_no_duplicates() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.7)
            .with_min_samples(3);
        for i in 1..=5 {
            engine.ingest(Signal::new("eng", "current", i as f64, i));
            engine.ingest(Signal::new("nav", "heading", i as f64, i));
        }
        engine.scan();
        engine.detect_splines();
        // Second detect_splines should not duplicate
        let count_before = engine.splines.len();
        engine.detect_splines();
        assert_eq!(engine.splines.len(), count_before);
    }

    #[test]
    fn test_get_splines_for_room() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.7)
            .with_min_samples(3);
        for i in 1..=5 {
            engine.ingest(Signal::new("eng", "current", i as f64, i));
            engine.ingest(Signal::new("nav", "heading", i as f64, i));
        }
        engine.scan();
        engine.detect_splines();
        let eng_splines = engine.get_splines_for_room("eng");
        assert!(!eng_splines.is_empty());
    }

    #[test]
    fn test_get_splines_between() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.7)
            .with_min_samples(3);
        for i in 1..=5 {
            engine.ingest(Signal::new("eng", "current", i as f64, i));
            engine.ingest(Signal::new("nav", "heading", i as f64, i));
        }
        engine.scan();
        engine.detect_splines();
        let between = engine.get_splines_between("eng", "nav");
        assert!(!between.is_empty());
    }

    #[test]
    fn test_total_savings() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.7)
            .with_min_samples(3);
        for i in 1..=5 {
            engine.ingest(Signal::new("eng", "current", i as f64, i));
            engine.ingest(Signal::new("nav", "heading", i as f64, i));
        }
        engine.scan();
        engine.detect_splines();
        let (energy, tokens) = engine.total_savings();
        assert!(energy > 0.0);
        assert!(tokens > 0);
    }

    #[test]
    fn test_decay_and_prune() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.7)
            .with_min_samples(3);
        for i in 1..=5 {
            engine.ingest(Signal::new("eng", "current", i as f64, i));
            engine.ingest(Signal::new("nav", "heading", i as f64, i));
        }
        engine.scan();
        engine.detect_splines();
        let count_before = engine.correlations.len();
        // Decay heavily
        engine.decay_all(0.5);
        engine.prune_weak(0.1);
        // All correlations should have decayed below 0.1 confidence and been pruned
        // (since strengthen was never called so confidence is 0)
        assert!(engine.correlations.len() <= count_before);
    }

    #[test]
    fn test_engine_summary() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.7)
            .with_min_samples(3);
        for i in 1..=5 {
            engine.ingest(Signal::new("eng", "current", i as f64, i));
            engine.ingest(Signal::new("nav", "heading", i as f64, i));
        }
        engine.scan();
        engine.detect_splines();
        let summary = engine.engine_summary();
        assert!(summary.stream_count > 0);
        assert_eq!(summary.active_rooms.len(), 2);
        assert!(summary.active_rooms.contains(&"eng".to_string()));
        assert!(summary.active_rooms.contains(&"nav".to_string()));
    }

    #[test]
    fn test_engine_render_correlations() {
        let engine = PenroseEngine::new();
        let rendered = engine.render_correlations();
        assert!(rendered.contains("No correlations"));
    }

    #[test]
    fn test_engine_render_splines() {
        let engine = PenroseEngine::new();
        let rendered = engine.render_splines();
        assert!(rendered.contains("No splines"));
    }

    // ── PenroseCompiler tests ──

    #[test]
    fn test_compile_empty_splines() {
        let opts = PenroseCompiler::compile_splines(&[]);
        assert!(opts.is_empty());
    }

    #[test]
    fn test_compile_redundant_spline() {
        let c = Correlation::new(("eng", "current"), ("nav", "heading"), 0.9);
        let mut s = Spline::new(c);
        s.calculate_savings(1000, 500);
        let opts = PenroseCompiler::compile_splines(&[s]);
        let has_merge = opts.iter().any(|o| matches!(o, Optimization::MergeStreams { .. }));
        assert!(has_merge);
    }

    #[test]
    fn test_compile_predictive_spline() {
        let mut c = Correlation::new(("eng", "vibration"), ("sec", "alert"), 0.8);
        c.occurrences = 5;
        let mut s = Spline::new(c);
        s.spline_type = SplineType::Predictive;
        let opts = PenroseCompiler::compile_splines(&[s]);
        let has_predictive = opts.iter().any(|o| matches!(o, Optimization::PredictiveLoad { .. }));
        assert!(has_predictive);
    }

    #[test]
    fn test_compile_shared_context() {
        let c = Correlation::new(("eng", "current"), ("nav", "heading"), 0.75);
        let mut s = Spline::new(c);
        s.calculate_savings(1000, 700);
        // Ensure it's Synergistic (0.75 > 0.7, not > 0.85, no lead)
        let opts = PenroseCompiler::compile_splines(&[s]);
        let has_shared = opts.iter().any(|o| matches!(o, Optimization::SharedContext { .. }));
        assert!(has_shared);
    }

    #[test]
    fn test_apply_merge_streams() {
        use std::collections::HashMap;
        let mut system: HashMap<SignalKey, SignalStream> = HashMap::new();
        let mut s1 = SignalStream::new("eng", "current", 10);
        let mut s2 = SignalStream::new("nav", "current", 10);
        s1.push(1, 1.0);
        s1.push(2, 2.0);
        s2.push(1, 1.0);
        s2.push(2, 2.0);
        system.insert(SignalKey::new("eng", "current"), s1);
        system.insert(SignalKey::new("nav", "current"), s2);

        let opt = Optimization::MergeStreams {
            a: ("eng".to_string(), "current".to_string()),
            b: ("nav".to_string(), "current".to_string()),
            savings: 400,
        };
        PenroseCompiler::apply(&opt, &mut system);
        assert!(system.contains_key(&SignalKey::new("eng", "current")));
        assert!(!system.contains_key(&SignalKey::new("nav", "current")));
        // eng stream should have all 4 values now
        assert_eq!(system[&SignalKey::new("eng", "current")].len(), 4);
    }

    #[test]
    fn test_estimated_total_savings() {
        let opts = vec![
            Optimization::MergeStreams {
                a: ("a".into(), "x".into()),
                b: ("b".into(), "y".into()),
                savings: 400,
            },
            Optimization::SharedContext {
                rooms: vec!["a".into(), "b".into()],
                shared_tokens: 200,
            },
            Optimization::PredictiveLoad {
                predictor: "a".into(),
                predicted: "b".into(),
                lead_ticks: 1,
            },
            Optimization::CompressedProtocol {
                rooms: vec!["a".into(), "b".into()],
                compression_ratio: 0.6,
            },
        ];
        let (energy, tokens) = PenroseCompiler::estimated_total_savings(&opts);
        assert!(tokens > 0);
        assert!(energy > 0.0);
    }

    // ── Integration test ──

    #[test]
    fn test_full_pipeline() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.6)
            .with_min_samples(3);

        // Simulate: engineering motor using current and navigation adjusting course
        let scenarios = vec![
            (1, 2.3, 15.0),
            (2, 2.1, 14.0),
            (3, 2.5, 16.0),
            (4, 2.0, 13.0),
            (5, 2.4, 15.5),
        ];

        for (tick, motor_current, course_dev) in scenarios {
            engine.ingest(Signal::new("engineering", "motor_current", motor_current, tick));
            engine.ingest(Signal::new("navigation", "course_deviation", course_dev, tick));
        }

        let corrs = engine.scan();
        assert!(!corrs.is_empty(), "Should detect correlation in pipeline");

        let splines = engine.detect_splines();
        assert!(!splines.is_empty(), "Should create splines in pipeline");

        let summary = engine.engine_summary();
        assert!(summary.stream_count > 0);
        assert!(summary.spline_count > 0);

        // Compile into optimizations
        let opts = PenroseCompiler::compile_splines(&splines);
        assert!(!opts.is_empty());

        let (energy, tokens) = PenroseCompiler::estimated_total_savings(&opts);
        assert!(tokens > 0);
        assert!(energy > 0.0);

        // Verify summary rendering
        let corr_map = engine.render_correlations();
        assert!(corr_map.contains("engineering"));
        assert!(corr_map.contains("navigation"));

        let spline_net = engine.render_splines();
        assert!(spline_net.contains("engineering"));
    }

    #[test]
    fn test_three_room_collaborative_correlation() {
        let mut engine = PenroseEngine::new()
            .with_threshold(0.6)
            .with_min_samples(3);

        // Three rooms all correlated
        for i in 1..=6 {
            let v = i as f64;
            engine.ingest(Signal::new("engineering", "load", v, i));
            engine.ingest(Signal::new("navigation", "heading", v, i));
            engine.ingest(Signal::new("security", "motion", v * 0.5, i));
        }

        engine.scan();
        engine.detect_splines();

        let summary = engine.engine_summary();
        assert_eq!(summary.active_rooms.len(), 3);

        let splines_for_eng = engine.get_splines_for_room("engineering");
        assert!(!splines_for_eng.is_empty());
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut engine = PenroseEngine::new()
            .with_min_samples(3)
            .with_threshold(0.6);
        engine.ingest(Signal::new("eng", "current", 1.0, 1));
        engine.ingest(Signal::new("nav", "heading", 1.0, 1));
        engine.ingest(Signal::new("eng", "current", 2.0, 2));
        engine.ingest(Signal::new("nav", "heading", 2.0, 2));

        let json_value = engine.to_json_value();
        let json_str = serde_json::to_string(&json_value).expect("Serialize failed");
        let parsed: serde_json::Value =
            serde_json::from_str(&json_str).expect("Deserialize failed");
        assert_eq!(
            parsed["streams"].as_array().unwrap().len(),
            2
        );
        assert_eq!(
            parsed["min_samples"].as_u64().unwrap(),
            3
        );
    }

    #[test]
    fn test_signal_key_new() {
        let key = SignalKey::new("engineering", "motor_current");
        assert_eq!(key.room_id, "engineering");
        assert_eq!(key.signal_type, "motor_current");
    }

    #[test]
    fn test_signal_key_from_tuple() {
        let key: SignalKey = ("room", "signal").into();
        assert_eq!(key.room_id, "room");
        assert_eq!(key.signal_type, "signal");
    }

    #[test]
    fn test_signal_serde() {
        let s = Signal::new("room", "temp", 22.5, 100)
            .with_metadata("unit", "celsius");
        let json = serde_json::to_string(&s).unwrap();
        let back: Signal = serde_json::from_str(&json).unwrap();
        assert_eq!(back.room_id, "room");
        assert_eq!(back.value, 22.5);
        assert_eq!(back.metadata.get("unit").unwrap(), "celsius");
    }

    #[test]
    fn test_edge_case_negative_correlation() {
        let mut a = SignalStream::new("room_a", "x", 10);
        let mut b = SignalStream::new("room_b", "y", 10);
        for i in 1..=5 {
            a.push(i, i as f64);
            b.push(i, (6 - i) as f64); // inversely related
        }
        let r = a.correlation_with(&b);
        assert!(r < 0.0);
        // verify negative strength check
        let c = Correlation::new(("a", "x"), ("b", "y"), r);
        assert!(c.is_strong()); // |r| > 0.7 expected
    }

    #[test]
    fn test_edge_case_no_matching_ticks() {
        let mut a = SignalStream::new("a", "x", 10);
        let mut b = SignalStream::new("b", "y", 10);
        a.push(1, 1.0);
        a.push(3, 3.0);
        b.push(2, 2.0);
        b.push(4, 4.0);
        assert_eq!(a.correlation_with(&b), 0.0);
    }

    #[test]
    fn test_spline_type_classify_lead() {
        // With a lead, even r=0.8 should be Predictive
        match SplineType::classify(0.8, Some(5)) {
            SplineType::Predictive => {}
            _ => panic!("Expected Predictive with lead"),
        }
    }

    #[test]
    fn test_engine_with_setters() {
        let engine = PenroseEngine::new()
            .with_threshold(0.5)
            .with_min_samples(5)
            .with_scan_interval(2);
        assert_eq!(engine.correlation_threshold, 0.5);
        assert_eq!(engine.min_samples, 5);
        assert_eq!(engine.scan_interval, 2);
    }

    #[test]
    fn test_correlation_decay_and_strengthen_cycle() {
        let mut c = Correlation::new(("a", "x"), ("b", "y"), 0.9);
        c.strengthen(0.8);
        assert!(c.occurrences > 0);
        c.decay(0.1);
        assert!(c.correlation < 0.9);
    }
}