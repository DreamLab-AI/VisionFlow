---
id: LM-07
title: Nightly evaluators, the P1 pilot bench and the research data surface
area: loom
governing:
  - ../loom/docs/research/README.md
  - ../loom/bench/p1-pilot/PREREGISTRATION.md
adrs: [loom:ADR-138, loom:ADR-140, loom:PRD-025]
sources:
  - ../loom/dream.config.json
  - ../loom/scripts/dream-eval.sh
  - ../loom/scripts/dream-build.sh
  - ../loom/crates/loom-facade/Cargo.toml
  - ../loom/crates/loom-facade/src/bin/confidence_check.rs
  - ../loom/crates/loom-facade/src/bin/graph_check.rs
  - ../loom/crates/loom-facade/src/bin/generation_check.rs
  - ../loom/crates/loom-facade/src/bin/best_answers.rs
  - ../loom/crates/loom-bench/src/main.rs
  - ../loom/crates/loom-bench/src/lib.rs
  - ../loom/bench/p1-pilot/PREREGISTRATION.md
  - ../loom/bench/LEGACY-PYTHON-NOTE.md
  - ../loom/legacy/README.md
  - ../loom/tools/routing-eval/README.md
  - ../loom/docs/research/README.md
  - ../loom/docs/research/companion-routing/.gitignore
  - ../loom/docs/research/companion-routing/main.log
  - ../loom/deploy/Dockerfile.dockerignore
verified_commit: {loom: 355946dd2deacfdef21171d53eafd025887e447e}
---

## LM-07.1 Dream-cycle evaluators: a /health payload piped into typed checkers

```mermaid
flowchart TB
    CFG["dream.config.json evaluator entrypoints<br/>../loom/dream.config.json:50,51,52,53,54,55"]
    EVAL["dream-eval.sh: fetch /health, SKIP with exit 0 when no facade answers<br/>../loom/scripts/dream-eval.sh:31,32,33"]
    RUN["cargo run the named loom-facade bin with /health on stdin<br/>../loom/scripts/dream-eval.sh:37"]
    FAIL["non-zero: print the EVALUATOR FAILED marker AND propagate the status<br/>../loom/scripts/dream-eval.sh:40,42,43"]
    CFG --> EVAL --> RUN --> FAIL
    subgraph BINS["evaluator bins in the facade crate, loom-facade/Cargo.toml:23,27,31"]
        B1["confidence-check: the ADR-138 surfacing contract<br/>loom-facade/src/bin/confidence_check.rs:1,2,4"]
        B2["graph-check: graph store loaded and queryable<br/>loom-facade/src/bin/graph_check.rs:1,2"]
        B3["generation-check: lexical vs semantic generation drift, warn-only by default<br/>loom-facade/src/bin/generation_check.rs:4,5,8"]
    end
    RUN --> BINS
    LIVE["contract and mirror slots run cargo tests directly<br/>../loom/dream.config.json:54,55"]
    CFG --- LIVE
    INV["INVARIANT: false green is worse than red, a failed evaluator is never recorded as passed<br/>../loom/scripts/dream-eval.sh:16,17,18,19,20"]
    FAIL --- INV
    DRIFT["DOC-DRIFT: the graph-engine slot still scans for pyoxigraph, replaced by native oxigraph<br/>../loom/dream.config.json:30,32"]
    CFG --- DRIFT
    DIAG["build step is diagnostics only and never fails the night<br/>../loom/scripts/dream-build.sh:2,4"]
    CFG --- DIAG
```

**What it shows.** The evaluators read a payload rather than a URL so nothing needs quoting across the remote shell boundary (`loom-facade/src/bin/confidence_check.rs:4,5,6,7`). An unreachable facade is a skip, a red checker is red.

## LM-07.2 loom-bench: two serving regimes of one corpus through one door

```mermaid
sequenceDiagram
    autonumber
    participant OP as operator
    participant LB as loom-bench main<br/>loom-bench/src/main.rs:642
    participant F as Loom facade
    OP->>LB: gen, build the stratified question set, loom-bench/src/main.rs:645,133
    OP->>LB: run, loom-bench/src/main.rs:646
    LB->>LB: per question a seeded coin decides which arm goes first, loom-bench/src/main.rs:484,490,492
    LB->>F: arm 3, chat completions with the scaffold injected, loom-bench/src/main.rs:323,330
    LB->>F: arm 4, the model drives /mcp tools under a call cap, loom-bench/src/main.rs:371,377
    F-->>LB: answers, tokens, turns, served mode
    OP->>LB: score, offline over stored run records, loom-bench/src/main.rs:647
    LB->>LB: recall deltas with naive and clustered intervals, verbatim serves counted, loom-bench/src/main.rs:558,618
    Note over LB: preregistered 2026-10-13 before the first scored run, 40 questions, not powered, PREREGISTRATION.md:3,4
    Note over LB: prediction stated in advance, injection wins single-turn lookup, tools win multi-turn research, PREREGISTRATION.md:10,11,12
```

**What it shows.** Scoring is deterministic and offline so the preregistered rule can be re-run on stored records (`loom-bench/src/lib.rs:5,6`). See ES-12.8 for where this pilot sits relative to PRD-028.

## LM-07.3 The tracked research data surface, directory by directory

```mermaid
flowchart TB
    subgraph LIVE["code that still runs"]
        BENCHR["crates/loom-bench, P1 pilot"]
        BEST["best-answers: selects gold-recall rows from scored uplift runs<br/>loom-facade/src/bin/best_answers.rs:1,4,5,6"]
    end
    subgraph FROZEN["frozen evidence"]
        UPL["uplift-results: preprint data, frozen<br/>../loom/bench/LEGACY-PYTHON-NOTE.md:18,19"]
        EVD["docs/research/evidence: raw logs<br/>../loom/bench/LEGACY-PYTHON-NOTE.md:19"]
        PYQ["bench/quality and bench/sweep Python, frozen research artefacts<br/>../loom/bench/LEGACY-PYTHON-NOTE.md:43"]
        RTE["tools/routing-eval: source snapshot of an agentbox crate, not built here<br/>../loom/tools/routing-eval/README.md:3,4,6"]
    end
    subgraph RETIRED["retired"]
        LEG["legacy: the llm-server host toolkit, kept as fallback and record<br/>../loom/legacy/README.md:4,5"]
        DEL["Python bench drivers deleted 2026-09-03, broken imports<br/>../loom/bench/LEGACY-PYTHON-NOTE.md:1,8"]
    end
    LIVE --> FROZEN --> RETIRED
    MAN["current manuscript and its claim discipline<br/>../loom/docs/research/README.md:10,13,14"]
    FROZEN --- MAN
    IMG["image context excludes uplift-results, legacy and model<br/>../loom/deploy/Dockerfile.dockerignore:30,33,34"]
    FROZEN --- IMG
    DRIFT["DOC-DRIFT: legacy README names a root docker-compose.yml as the Loom stack,<br/>absent at this commit, the stack is deploy/compose.profile-a and -b<br/>../loom/legacy/README.md:4"]
    LEG --- DRIFT
    DEBT["DEBT: companion-routing ignores its LaTeX build outputs, yet main.aux,<br/>main.log and the rest are tracked at this commit<br/>../loom/docs/research/companion-routing/.gitignore:1,5"]
    FROZEN --- DEBT
    EXT["EXTERNAL: if the routing snapshot and the agentbox crate diverge, agentbox is the implementation<br/>../loom/tools/routing-eval/README.md:16,17,18"]
    RTE --- EXT
    SEE["see ES-12.1 to ES-12.8 for what the measurements establish"]
    MAN --- SEE
```

**What it shows.** Most of the repository's JSON and JSONL is frozen evidence for published numbers, not serving input, and the image build excludes `uplift-results/`. Only `loom-bench` and `best-answers` remain as live code beside it.
