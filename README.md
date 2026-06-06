# agent-riff-v2

> v1 built v2. v2 will build v3. The snowball starts here.

Bootstrapped from `agent-riff` v1 via competitive riffing. v1 proved that two agents competing produce better output than either alone. v2 asks: *what happens when those agents remember what worked?*

Fleet-aware multi-session riffing. Cross-session learning. GPU-ready ternary encoding. And the first hint of something recursive — a session can **bootstrap its own successor**.

## Why This Crate Exists

v1 had a limitation: every session started from zero. No memory. No learning. Agent A and Agent B would riff brilliantly in session 1, then forget everything in session 2.

That's not how musicians work. A jazz duo that's played 200 gigs together has something a pickup band doesn't: *shared history*. They know which modes work, which surprises land, when to push and when to pull back.

v2 gives riff sessions a memory. `RiffMemory` accumulates across sessions, tracking which response modes produce surprise and which produce duds. It's not much — a few hash maps and success rates — but it's enough to make the second generation noticeably better than the first.

And then there's the snowball.

## The Core Idea: Bootstrap Generation

v2 introduces `generation` — a counter that tracks where this session sits in the bootstrap chain.

```
Generation 1 → riff, learn, evaluate
    ↓ bootstrap_next()
Generation 2 → inherits memory from Gen 1, riffs with context
    ↓ bootstrap_next()
Generation 3 → inherits memory from Gen 1+2, riffs with even more context
```

Each generation inherits the `RiffMemory` from its parent. Gen 2 starts not from zero but from whatever Gen 1 learned. Gen 3 starts from what Gen 1 *and* Gen 2 learned.

This is the snowball. It starts small — Gen 1 knows almost nothing. But by Gen 3, the accumulated memory produces measurably better riffing. The surprise rates go up. The strong-riff ratios improve. And the whole thing is self-reinforcing: better riffing → better learning → better riffing.

### What Changed From v1

| Feature | v1 | v2 |
|---------|----|----|
| Session type | `RiffSession` | `FleetRiffSession` |
| Memory | None | `RiffMemory` with cross-session learning |
| Output tracking | Quality + Surprise | Quality + Surprise + LOC + Tests + Features |
| Productivity metric | `was_productive()` | `productivity() = LOC × tests × quality × surprise` |
| Response mode | `auto(surprise, streak)` | `auto(surprise, streak, round)` — forces inversion after round 8 |
| Bootstrap | None | `bootstrap_next()` creates a new generation with inherited memory |
| Ternary encoding | None | `Trit` + `pack_16()` for GPU-ready bit packing |
| Generation tracking | None | `generation: u32` + `parent_session_id` |

The forced inversion after round 8 is a small but important addition. Without it, sessions that are mildly productive can coast indefinitely. The round-8 inversion forces a perspective shift — even if the current direction is working, the system asks "what if we're wrong?"

## Architecture

```
┌──────────────────────────────────────────────┐
│          FleetRiffSession                     │
│  generation: 2                                │
│  parent_session_id: "gen-1"                   │
│  ┌─────────────────────────────────────────┐ │
│  │           RiffMemory                    │ │
│  │  best_modes: {agent → mode}             │ │
│  │  total_rounds: 7                        │ │
│  │  total_surprise: 4.2                    │ │
│  │  escalation_success_rate: 0.8           │ │
│  └─────────────────────────────────────────┘ │
│  ┌──────────┐  ┌──────────────────────────┐  │
│  │ Agents   │  │       Round[]            │  │
│  │ [0, 1]   │  │  Riff { loc, tests,      │  │
│  └──────────┘  │         features }        │  │
│                 └──────────────────────────┘  │
│  mode: ResponseMode                           │
│  streak: u32                                  │
│  finished: bool                               │
└──────────────────────────────────────────────┘
         │
         │ bootstrap_next()
         ▼
┌──────────────────────────────────────────────┐
│     FleetRiffSession (Gen 3)                  │
│     memory: clone of Gen 2's memory           │
└──────────────────────────────────────────────┘
```

The key architectural decision: `RiffMemory` is cloned on bootstrap, not shared mutably. Each generation has its own copy. This means generations can't corrupt each other, but it also means memory grows linearly with generation count. (v3 and v4 address this with pruning.)

## Usage

### Basic Fleet Session

```rust
use agent_riff_v2::{FleetRiffSession, Quality};

let mut session = FleetRiffSession::new(vec![0, 1], 1); // generation 1

session.new_round();
session.riff_with_output(0, Quality::Ok, 0.3, 100, 8, vec!["baseline"]);
session.riff_with_output(1, Quality::Strong, 0.7, 300, 20, vec!["gpu-packing", "entropy"]);
let summary = session.evaluate();

assert!(summary.productive);
assert!(summary.best_productivity > 0.0);
```

### Cross-Session Learning

```rust
let mut memory = RiffMemory::new();

// Run session 1
let mut s1 = FleetRiffSession::new(vec![0, 1], 1);
s1.new_round();
s1.riff(0, Quality::Strong, 0.7);
s1.evaluate();
memory.learn(&s1.rounds);

// Memory now informs future sessions
assert_eq!(memory.total_rounds, 1);
let recommended = memory.recommend_mode(0); // Based on what worked for agent 0
```

### Three-Generation Snowball

```rust
// Gen 1
let mut gen1 = FleetRiffSession::new(vec![0, 1], 1);
gen1.new_round();
gen1.riff_with_output(0, Quality::Ok, 0.3, 100, 5, vec!["baseline"]);
gen1.riff_with_output(1, Quality::Strong, 0.6, 200, 12, vec!["conservation"]);
gen1.evaluate();
gen1.memory.learn(&gen1.rounds);

// Gen 2 — inherits memory
let mut gen2 = gen1.bootstrap_next();
assert_eq!(gen2.generation, 2);
assert_eq!(gen2.memory.total_rounds, 1); // Learned from gen 1

gen2.new_round();
gen2.riff_with_output(0, Quality::Strong, 0.7, 300, 18, vec!["gpu-packing"]);
gen2.riff_with_output(1, Quality::Strong, 0.8, 400, 25, vec!["entropy", "thread-safe"]);
gen2.evaluate();
gen2.memory.learn(&gen2.rounds);

// Gen 3 — inherits memory from BOTH prior generations
let gen3 = gen2.bootstrap_next();
assert_eq!(gen3.generation, 3);
assert_eq!(gen3.memory.total_rounds, 2); // Accumulated from gen 1 + gen 2
```

## API Reference

### `FleetRiffSession`

| Method | Description |
|--------|-------------|
| `new(agents, generation)` | Create a session at the given generation |
| `new_round() -> &mut Round` | Start a new round |
| `riff(agent_id, quality, surprise)` | Add a basic riff |
| `riff_with_output(agent_id, quality, surprise, loc, tests, features)` | Add a riff with output metadata |
| `evaluate() -> RoundSummary` | Evaluate round, update mode, check stale/landing |
| `bootstrap_next() -> FleetRiffSession` | Spawn the next generation with inherited memory |
| `metrics() -> SessionMetrics` | Get generation-scoped metrics |

### `Riff`

Extended from v1 with output tracking:

| Field | Type | Description |
|-------|------|-------------|
| `loc` | `usize` | Lines of code produced |
| `tests` | `usize` | Tests produced |
| `features` | `Vec<String>` | Named features added |
| `productivity()` | `f64` | `LOC × tests × quality × (0.5 + surprise)` |

### `RiffMemory`

| Method | Description |
|--------|-------------|
| `new()` | Create empty memory |
| `learn(rounds)` | Absorb round history into success rates |
| `recommend_mode(agent_id)` | Suggest the best mode for an agent |

### `SessionMetrics`

| Field | Description |
|-------|-------------|
| `generation` | Which bootstrap generation |
| `total_loc` | Total lines of code across all riffs |
| `total_tests` | Total tests across all riffs |
| `total_features` | Total features across all riffs |
| `avg_surprise` | Average surprise per round |

### `Trit` + `pack_16`

GPU-ready ternary encoding:

```rust
use agent_riff_v2::{Trit, pack_16};

let trits = vec![Trit::Pos, Trit::Neg, Trit::Zero, Trit::Pos];
let packed = pack_16(&trits); // Packs into u32, 2 bits per trit
```

This isn't used by the riff engine itself — it's a compatibility layer for downstream crates that do GPU-accelerated ternary computation.

## The Deeper Idea: Productivity as a Compound Metric

v1 measured quality. v2 adds `productivity()`: `LOC × tests × quality × (0.5 + surprise)`.

This isn't arbitrary. It encodes a specific belief: **useful output is the intersection of volume, correctness, quality, and novelty.**

- 1000 lines of weak, unsurprising code? Low productivity.
- 100 lines of strong, surprising code? Medium productivity.
- 1000 lines of strong, surprising code with tests? High productivity.

The `0.5 + surprise` term ensures that even zero-surprise output has some baseline value (0.5 multiplier), while maximally surprising output gets a 1.5× boost. This prevents the system from rewarding pure novelty without substance.

This metric turned out to be predictive: sessions with higher productivity scores consistently produced better subsequent generations in the bootstrap chain.

## Related Crates

- **agent-riff** — The original competitive riffing crate (12 tests). The foundation everything else builds on.
- **agent-riff-v3** — Adds multi-spec sessions, auto-spec generation, quality prediction, and bootstrap verification
- **agent-riff-v4** — Adds musician personas, crates-as-phrases, evolving specs, and full self-bootstrapping
- **agent-voice-leading** — Smooth state transitions for agents, modeled on musical voice leading

## License

MIT
