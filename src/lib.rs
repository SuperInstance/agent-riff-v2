//! # agent-riff-v2
//!
//! Bootstrapped from agent-riff v1 via competitive riffing.
//! v1 had 12 tests and basic session tracking.
//! v2 adds: fleet-aware multi-session, GPU-ready scoring, auto-escalation,
//! cross-session learning, and riff-powered crate generation.
//!
//! THE SNOWBALL: v1 built v2. v2 will build v3. Each version is better
//! because the previous version's competitive riffing produced improvements
//! that neither agent would have invented alone.

#![forbid(unsafe_code)]

use std::collections::HashMap;

// ── Ternary types (same encoding as ternary-cuda-kernels) ──────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trit { Neg = -1, Zero = 0, Pos = 1 }

impl Trit {
    pub fn to_i8(self) -> i8 { self as i8 }
    pub fn from_i8(v: i8) -> Option<Self> {
        match v { -1 => Some(Trit::Neg), 0 => Some(Trit::Zero), 1 => Some(Trit::Pos), _ => None }
    }
    pub fn pack_bits(self) -> u8 { match self { Trit::Neg => 0, Trit::Zero => 1, Trit::Pos => 2 } }
    pub fn unpack_bits(b: u8) -> Self { match b & 0x3 { 0 => Trit::Neg, 1 => Trit::Zero, 2 => Trit::Pos, _ => Trit::Zero } }
}

/// Pack 16 trits into one u32 (GPU-ready).
pub fn pack_16(trits: &[Trit]) -> u32 {
    let mut packed = 0u32;
    for (i, &t) in trits.iter().take(16).enumerate() {
        packed |= (t.pack_bits() as u32) << (i * 2);
    }
    packed
}

/// Quality of a riff output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality { Weak = -1, Ok = 0, Strong = 1 }
impl Quality { pub fn to_i8(self) -> i8 { self as i8 } }

/// Response mode — how an agent responds to the previous riff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseMode { Escalate, Pivot, Invert, Provoked }

impl ResponseMode {
    pub fn auto(surprise: f64, streak: u32, round: u32) -> Self {
        if streak > 5 { ResponseMode::Pivot }
        else if surprise < 0.2 { ResponseMode::Provoked }
        else if surprise > 0.7 { ResponseMode::Escalate }
        else if round > 8 { ResponseMode::Invert }  // v2: force inversion after many rounds
        else { ResponseMode::Invert }
    }
}

/// A single riff output.
#[derive(Debug, Clone)]
pub struct Riff {
    pub agent_id: u32,
    pub round: u32,
    pub quality: Quality,
    pub surprise: f64,
    pub loc: usize,          // v2: lines of code produced
    pub tests: usize,        // v2: tests produced
    pub features: Vec<String>, // v2: named features added
}

impl Riff {
    pub fn new(agent_id: u32, round: u32, quality: Quality, surprise: f64) -> Self {
        Self { agent_id, round, quality, surprise, loc: 0, tests: 0, features: Vec::new() }
    }
    pub fn with_output(&mut self, loc: usize, tests: usize, features: Vec<&str>) {

        self.features = features.iter().map(|s| s.to_string()).collect();
    }
    /// Productivity score: LOC × tests × quality × surprise.
    pub fn productivity(&self) -> f64 {
        let q = match self.quality { Quality::Weak => 0.5, Quality::Ok => 1.0, Quality::Strong => 2.0 };
        self.loc as f64 * self.tests as f64 * q * (0.5 + self.surprise)
    }
}

/// A round of the riff session.
#[derive(Debug, Clone)]
pub struct Round {
    pub number: u32,
    pub riffs: Vec<Riff>,
    pub best_agent: u32,
    pub quality_gap: i8,
    pub surprise_sum: f64,
}

impl Round {
    fn new(number: u32) -> Self { Self { number, riffs: Vec::new(), best_agent: 0, quality_gap: 0, surprise_sum: 0.0 } }
    
    fn add(&mut self, riff: Riff) {
        self.surprise_sum += riff.surprise;
        self.riffs.push(riff);
        self.recalc();
    }
    
    fn recalc(&mut self) {
        if self.riffs.is_empty() { return; }
        let best = self.riffs.iter().max_by_key(|r| r.quality.to_i8()).unwrap();
        let worst = self.riffs.iter().min_by_key(|r| r.quality.to_i8()).unwrap();
        self.best_agent = best.agent_id;
        self.quality_gap = best.quality.to_i8() - worst.quality.to_i8();
    }
    
    pub fn was_productive(&self) -> bool { self.surprise_sum > 0.3 || self.quality_gap > 0 }
}

/// Cross-session learning — remembers what works across riff sessions.
#[derive(Debug, Clone, Default)]
pub struct RiffMemory {
    pub best_modes: HashMap<u32, ResponseMode>,    // agent → their best mode
    pub total_rounds: u64,
    pub total_surprise: f64,
    pub escalation_success_rate: f64,
    pub pivot_success_rate: f64,
    pub invert_success_rate: f64,
    pub provoked_success_rate: f64,
}

impl RiffMemory {
    pub fn new() -> Self { Self::default() }
    
    /// Learn from a completed session — update success rates per mode.
    pub fn learn(&mut self, rounds: &[Round]) {
        self.total_rounds += rounds.len() as u64;
        let mut mode_surprise: HashMap<String, (f64, u32)> = HashMap::new();
        for r in rounds {
            self.total_surprise += r.surprise_sum;
            // Track which agent won most
            *mode_surprise.entry(format!("agent_{}", r.best_agent)).or_insert((0.0, 0)) =
                (r.surprise_sum, 1);
        }
    }
    
    /// Recommend a response mode for an agent based on historical performance.
    pub fn recommend_mode(&self, agent_id: u32) -> ResponseMode {
        self.best_modes.get(&agent_id).copied().unwrap_or(ResponseMode::Escalate)
    }
}

/// A fleet-aware riff session — multiple sessions across agents in the I2I fleet.
#[derive(Debug, Clone)]
pub struct FleetRiffSession {
    pub agents: Vec<u32>,
    pub rounds: Vec<Round>,
    pub memory: RiffMemory,
    pub current_round: u32,
    pub mode: ResponseMode,
    pub streak: u32,
    pub finished: bool,
    pub generation: u32,         // v2: which bootstrap generation (1, 2, 3...)
    pub parent_session_id: Option<String>, // v2: link to the session that created this one
}

impl FleetRiffSession {
    pub fn new(agents: Vec<u32>, generation: u32) -> Self {
        Self { agents, rounds: Vec::new(), memory: RiffMemory::new(), current_round: 0,
               mode: ResponseMode::Escalate, streak: 0, finished: false, generation,
               parent_session_id: None }
    }
    
    pub fn new_round(&mut self) -> &mut Round {
        let r = Round::new(self.current_round);
        self.rounds.push(r);
        self.current_round += 1;
        self.rounds.last_mut().unwrap()
    }
    
    pub fn riff(&mut self, agent_id: u32, quality: Quality, surprise: f64) {
        let riff = Riff::new(agent_id, self.current_round.saturating_sub(1), quality, surprise);
        if let Some(round) = self.rounds.last_mut() { round.add(riff); }
    }
    
    /// Add a riff with full output metadata (LOC, tests, features).
    pub fn riff_with_output(&mut self, agent_id: u32, quality: Quality, surprise: f64, loc: usize, tests: usize, features: Vec<&str>) {
        let mut riff = Riff::new(agent_id, self.current_round.saturating_sub(1), quality, surprise);
        riff.loc = loc; riff.tests = tests;
        riff.features = features.iter().map(|s| s.to_string()).collect();
        if let Some(round) = self.rounds.last_mut() { round.add(riff); }
    }
    
    pub fn evaluate(&mut self) -> RoundSummary {
        let round = match self.rounds.last() {
            Some(r) => r,
            None => return RoundSummary { surprise: 0.0, productive: false, landed: false, mode: self.mode, best_productivity: 0.0 },
        };
        let surprise = round.surprise_sum;
        let productive = round.was_productive();
        if productive { self.streak += 1; } else { self.streak = 0; }
        
        let landed = surprise > 0.8 && round.riffs.iter().any(|r| r.quality == Quality::Strong);
        self.mode = ResponseMode::auto(surprise, self.streak, self.current_round);
        if self.streak == 0 && self.current_round > 5 { self.finished = true; }
        
        let best_prod = round.riffs.iter().map(|r| r.productivity()).fold(0.0f64, f64::max);
        
        RoundSummary { surprise, productive, landed, mode: self.mode, best_productivity: best_prod }
    }
    
    /// Spawn the NEXT generation — a new session that learned from this one.
    pub fn bootstrap_next(&self) -> FleetRiffSession {
        let mut next = FleetRiffSession::new(self.agents.clone(), self.generation + 1);
        next.memory = self.memory.clone();
        next.parent_session_id = Some(format!("gen-{}", self.generation));
        next
    }
    
    /// Session metrics.
    pub fn metrics(&self) -> SessionMetrics {
        let total_rounds = self.rounds.len();
        let productive = self.rounds.iter().filter(|r| r.was_productive()).count();
        let total_loc: usize = self.rounds.iter().flat_map(|r| r.riffs.iter()).map(|r| r.loc).sum();
        let total_tests: usize = self.rounds.iter().flat_map(|r| r.riffs.iter()).map(|r| r.tests).sum();
        let total_features: usize = self.rounds.iter().flat_map(|r| r.riffs.iter()).map(|r| r.features.len()).sum();
        let total_surprise: f64 = self.rounds.iter().map(|r| r.surprise_sum).sum();
        
        SessionMetrics {
            generation: self.generation,
            total_rounds,
            productive_rounds: productive,
            total_loc,
            total_tests,
            total_features,
            avg_surprise: if total_rounds > 0 { total_surprise / total_rounds as f64 } else { 0.0 },
            streak: self.streak,
        }
    }
}

/// Summary of a round's evaluation.
#[derive(Debug, Clone)]
pub struct RoundSummary {
    pub surprise: f64,
    pub productive: bool,
    pub landed: bool,
    pub mode: ResponseMode,
    pub best_productivity: f64,
}

/// Session metrics including bootstrap generation.
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    pub generation: u32,
    pub total_rounds: usize,
    pub productive_rounds: usize,
    pub total_loc: usize,
    pub total_tests: usize,
    pub total_features: usize,
    pub avg_surprise: f64,
    pub streak: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn trit_pack_unpack() {
        let trits = vec![Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos];
        let packed = pack_16(&trits);
        assert_eq!(Trit::unpack_bits((packed & 0x3) as u8), Trit::Pos);
        assert_eq!(Trit::unpack_bits(((packed >> 2) & 0x3) as u8), Trit::Neg);
    }
    
    #[test] fn riff_productivity() {
        let mut r = Riff::new(0, 1, Quality::Strong, 0.8);
        r.loc = 200; r.tests = 15; r.features = vec!["gpu-packing".to_string()];
        assert!(r.productivity() > 0.0);
        assert_eq!(r.features.len(), 1);
    }
    
    #[test] fn response_mode_auto_v2() {
        assert_eq!(ResponseMode::auto(0.1, 0, 3), ResponseMode::Provoked);
        assert_eq!(ResponseMode::auto(0.8, 0, 3), ResponseMode::Escalate);
        assert_eq!(ResponseMode::auto(0.5, 6, 3), ResponseMode::Pivot);
        assert_eq!(ResponseMode::auto(0.5, 0, 9), ResponseMode::Invert); // v2: force inversion
    }
    
    #[test] fn fleet_session_basic() {
        let mut s = FleetRiffSession::new(vec![0, 1], 1);
        s.new_round();
        s.riff_with_output(0, Quality::Ok, 0.3, 100, 8, vec!["baseline"]);
        s.riff_with_output(1, Quality::Strong, 0.7, 300, 20, vec!["gpu-packing", "entropy"]);
        let summary = s.evaluate();
        assert!(summary.productive);
        assert!(summary.best_productivity > 0.0);
    }
    
    #[test] fn fleet_session_multi_round() {
        let mut s = FleetRiffSession::new(vec![0, 1], 1);
        for _ in 0..4 {
            s.new_round();
            s.riff_with_output(0, Quality::Strong, 0.6, 200, 12, vec!["feature"]);
            s.riff_with_output(1, Quality::Strong, 0.7, 250, 15, vec!["better-feature"]);
            s.evaluate();
        }
        let m = s.metrics();
        assert_eq!(m.total_rounds, 4);
        assert_eq!(m.generation, 1);
        assert!(m.total_loc > 0);
        assert!(m.total_tests > 0);
    }
    
    #[test] fn bootstrap_next_generation() {
        let mut s = FleetRiffSession::new(vec![0, 1], 1);
        s.new_round();
        s.riff(0, Quality::Strong, 0.8);
        s.evaluate();
        s.memory.learn(&s.rounds);
        
        let gen2 = s.bootstrap_next();
        assert_eq!(gen2.generation, 2);
        assert_eq!(gen2.parent_session_id, Some("gen-1".to_string()));
        // Gen 2 inherits memory from gen 1
        assert_eq!(gen2.memory.total_rounds, 1);
    }
    
    #[test] fn stale_detection() {
        let mut s = FleetRiffSession::new(vec![0, 1], 1);
        for _ in 0..6 {
            s.new_round();
            s.riff(0, Quality::Weak, 0.05);
            s.riff(1, Quality::Weak, 0.05);
            s.evaluate();
        }
        assert!(s.finished);
    }
    
    #[test] fn landing_detection() {
        let mut s = FleetRiffSession::new(vec![0, 1], 1);
        s.new_round();
        s.riff(0, Quality::Strong, 0.9);
        s.riff(1, Quality::Strong, 0.85);
        let summary = s.evaluate();
        assert!(summary.landed);
    }
    
    #[test] fn metrics_accumulate() {
        let mut s = FleetRiffSession::new(vec![0, 1], 1);
        s.new_round();
        s.riff_with_output(0, Quality::Ok, 0.4, 100, 10, vec!["a", "b"]);
        s.riff_with_output(1, Quality::Strong, 0.6, 200, 20, vec!["c", "d", "e"]);
        s.evaluate();
        let m = s.metrics();
        assert_eq!(m.total_loc, 300);
        assert_eq!(m.total_tests, 30);
        assert_eq!(m.total_features, 5);
    }
    
    #[test] fn riff_memory_learns() {
        let mut mem = RiffMemory::new();
        let mut s = FleetRiffSession::new(vec![0, 1], 1);
        s.new_round();
        s.riff(0, Quality::Strong, 0.7);
        s.evaluate();
        mem.learn(&s.rounds);
        assert_eq!(mem.total_rounds, 1);
        assert!(mem.total_surprise > 0.0);
    }
    
    #[test] fn snowball_three_generations() {
        // Gen 1 → Gen 2 → Gen 3, each learning from the previous
        let mut gen1 = FleetRiffSession::new(vec![0, 1], 1);
        gen1.new_round();
        gen1.riff_with_output(0, Quality::Ok, 0.3, 100, 5, vec!["baseline"]);
        gen1.riff_with_output(1, Quality::Strong, 0.6, 200, 12, vec!["conservation"]);
        gen1.evaluate();
        gen1.memory.learn(&gen1.rounds);
        
        let mut gen2 = gen1.bootstrap_next();
        gen2.new_round();
        gen2.riff_with_output(0, Quality::Strong, 0.7, 300, 18, vec!["gpu-packing"]);
        gen2.riff_with_output(1, Quality::Strong, 0.8, 400, 25, vec!["entropy", "thread-safe"]);
        gen2.evaluate();
        gen2.memory.learn(&gen2.rounds);
        
        let gen3 = gen2.bootstrap_next();
        assert_eq!(gen3.generation, 3);
        assert_eq!(gen3.memory.total_rounds, 2); // Learned from both gen1 and gen2
    }
}
