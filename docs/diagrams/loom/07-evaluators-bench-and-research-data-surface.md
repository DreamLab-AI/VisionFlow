---
id: LM-07
title: Nightly evaluators, the P1 pilot bench and the research data surface
area: loom
governing:
  - ../loom/docs/research/README.md
  - ../loom/bench/p1-pilot/PREREGISTRATION.md
  - ../loom/docs/README.md
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
  - ../loom/docs/README.md
  - ../loom/.gitignore
  - ../loom/tools/paper/live_harness.py
  - ../loom/tools/paper/retry_empties.py
  - ../loom/tools/paper/control_harness.py
  - ../loom/tools/paper/control_rerun.py
  - ../loom/tools/paper/judge_v2.py
  - ../loom/tools/paper/cli_judge.py
  - ../loom/tools/paper/judge_openrouter.py
  - ../loom/tools/paper/analyze.py
  - ../loom/tools/paper/analyze_controls.py
  - ../loom/tools/paper/analyze_controls_v2.py
  - ../loom/tools/paper/render_controls_md.py
  - ../loom/tools/paper/exposure_summary.py
  - ../loom/tools/paper/ontology_scaffold_v1.py
  - ../loom/tools/paper/decompose_exposure.py
  - ../loom/tools/paper/parametric_suppression.py
  - ../loom/tools/paper/semantic_audit.py
  - ../loom/tools/paper/paraphrase_stress.py
  - ../loom/tools/paper/rewrite_v4.py
  - ../loom/crates/loom-scaffold/src/exposure.rs
  - ../loom/tools/ingest/README.md
  - ../loom/crates/loom-scaffold/src/bin/build_concept_records.rs
  - ../loom/crates/loom-vector-ruvector/src/bin/stage_corpus.rs
  - ../loom/crates/loom-facade/src/config.rs
  - ../loom/.github/workflows/rust-contracts.yml
  - ../loom/.claude/expectations/EXPECTATIONS-rust-loom.md
  - ../loom/.claude/evidence/AUDIT-gpt54.md
  - ../loom/.claude/evidence/AUDIT-gpt54-f123.md
  - ../loom/.claude/evidence/EXP-001.evidence.md
  - ../loom/.claude/evidence/EXP-008.evidence.md
  - ../loom/.claude/evidence/EXP-012.evidence.md
  - ../loom/.claude/evidence/EXP-013.evidence.md
  - ../loom/.claude/evidence/EXP-014.evidence.md
  - ../loom/.claude/evidence/EXP-attest.evidence.md
  - ../loom/.claude/evidence/EXP-deploy.evidence.md
  - ../loom/.claude/evidence/EXP-embed.evidence.md
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
        PAPER["tools/paper: the paper's stdlib Python harness, see LM-07.4 and LM-07.5<br/>../loom/docs/README.md:73"]
    end
    subgraph FROZEN["frozen evidence"]
        UPL["uplift-results: preprint data, frozen<br/>../loom/bench/LEGACY-PYTHON-NOTE.md:18,19"]
        EVD["docs/research/evidence: raw logs<br/>../loom/bench/LEGACY-PYTHON-NOTE.md:19"]
        PYQ["bench/quality and bench/sweep Python, frozen research artefacts<br/>../loom/bench/LEGACY-PYTHON-NOTE.md:43"]
        RTE["tools/routing-eval: source snapshot of an agentbox crate, not built here<br/>../loom/tools/routing-eval/README.md:3,4,6"]
        CEV[".claude/evidence: per-expectation executed evidence and the audit, see LM-07.6<br/>../loom/docs/README.md:78,79"]
    end
    subgraph RETIRED["retired"]
        LEG["legacy: the llm-server host toolkit, kept as fallback and record<br/>../loom/legacy/README.md:4,5"]
        DEL["Python bench drivers deleted 2026-09-03, broken imports<br/>../loom/bench/LEGACY-PYTHON-NOTE.md:1,8"]
        ING["tools/ingest: Python ported to Rust bins, the README stays as the channel's doc, see LM-07.6<br/>../loom/tools/ingest/README.md:3,4,5,6"]
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

**What it shows.** Most of the repository's JSON and JSONL is frozen evidence for published numbers, not serving input, and the image build excludes `uplift-results/`. The live code beside it is `loom-bench`, `best-answers` and the Python paper harness in `tools/paper/`, which writes into that same tree.

## LM-07.4 The paper harness: paired live arms, negative controls, cross-family judges, paired statistics

```mermaid
flowchart TB
    subgraph RUN["paper-v2 cohort"]
        LH["live_harness: one model, two serving paths, the production facade arm<br/>and an SSH-tunnelled bare-model arm, paired by question id<br/>../loom/tools/paper/live_harness.py:4,5,7,11,13"]
        RE["retry_empties: pairs that exhausted the 1536-token budget re-run at 4096,<br/>both arms, stale judgements purged<br/>../loom/tools/paper/retry_empties.py:4,5,6,7,11,12,13"]
        CH["control_harness: true, shuffled, masked and seed-disjoint irrelevant arms<br/>over the block the facade's /loom/scaffold serves<br/>../loom/tools/paper/control_harness.py:9,12,15,18,26,38"]
        LH --> RE --> CH
    end
    subgraph JUDGE["judges, blind to arm"]
        JV["judge_v2: the v1 rubric reused, resumable per set, id and arm<br/>../loom/tools/paper/judge_v2.py:4,6,7,13"]
        CJ["cli_judge: the Claude Code and Codex CLIs behind one interface<br/>../loom/tools/paper/cli_judge.py:2,49,77"]
        JO["judge_openrouter: reuses the judge_v2 rubric and HTTP call verbatim,<br/>records judge model, temperature and a sha256 of the graded text<br/>../loom/tools/paper/judge_openrouter.py:4,5,6,7,12,13,33"]
        JV --> CJ
        JO --> JV
    end
    subgraph STATS["paired statistics"]
        AN["analyze: percentile bootstrap, 10,000 resamples, seed 7,<br/>signed-rank, Cliff's delta, Holm<br/>../loom/tools/paper/analyze.py:8,9,10,11,14,15,32"]
        AC["analyze_controls: control contrasts merged into the paper-v2 analysis.json<br/>../loom/tools/paper/analyze_controls.py:2,4,5,13,21"]
        AN --> AC
    end
    RUN --> JUDGE --> STATS
    subgraph V9["paper-v9 remediation cohort"]
        CR["control_rerun: imports control_harness, adds a length-matched fluent_noise arm,<br/>common retry 4096 then 8192, measured placebo exposure per row<br/>../loom/tools/paper/control_rerun.py:9,10,14,22,26,28,35,67"]
        AC2["analyze_controls_v2: cohort on the command line, accounting chain,<br/>Holm over the whole family, common-intersection sensitivity<br/>../loom/tools/paper/analyze_controls_v2.py:5,6,10,13,15,32"]
        RMD["render_controls_md: ANALYSIS.md generated from analysis.json, not transcribed<br/>../loom/tools/paper/render_controls_md.py:2,4,5,6"]
        EXS["exposure_summary: per-arm placebo exposure aggregated from rows.jsonl<br/>../loom/tools/paper/exposure_summary.py:2,5,7"]
        CR --> AC2 --> RMD
        CR --> EXS
    end
    CH -.->|same code| CR
    OUT["outputs: uplift-results/paper-v2 and the dated rerun directory, see LM-07.3<br/>../loom/tools/paper/analyze.py:22,23,24<br/>../loom/tools/paper/control_rerun.py:53"]
    STATS --- OUT
    DRIFT["DOC-DRIFT: judge_v2 says the judge is GPT via OpenRouter, usage --judge-model openai/gpt-4.1,<br/>yet main offers only the two CLI judges and defaults to cli-claude<br/>../loom/tools/paper/judge_v2.py:4,5,16,113,114,115<br/>../loom/tools/paper/judge_openrouter.py:4,5"]
    JV --- DRIFT
    DEBT["DEBT: .gitignore ignores uplift-results, yet the cohort files are tracked,<br/>so new files a re-run writes there stay untracked unless force-added<br/>../loom/.gitignore:6"]
    OUT --- DEBT
```

**What it shows.** The published uplift numbers come from a stdlib Python pipeline beside the Rust workspace, not from `loom-bench`. The v9 cohort re-runs the controls by importing the original harness rather than forking it, so the two cohorts share the scaffold fetch, shuffle, mask and donor search (`tools/paper/control_rerun.py:9,10,11,12`).

## LM-07.5 The post-hoc analyses: exposure decomposition, suppression, semantic audit, paraphrase stress

```mermaid
flowchart TB
    OSV["ontology_scaffold_v1: the retired Python scaffold engine, vendored,<br/>link, seed, expand, serialise, clamp, stdlib only<br/>../loom/tools/paper/ontology_scaffold_v1.py:2,4,12<br/>../loom/tools/paper/decompose_exposure.py:21,22"]
    DX["decompose_exposure: per-item 2x2 exposure against recovery,<br/>measures n01 directly instead of reading it off a negative gain<br/>../loom/tools/paper/decompose_exposure.py:2,9,12,13,14,15,40"]
    PS["parametric_suppression: on unexposed gold, does injection suppress<br/>recall the bare model has, over the 510-question sweep<br/>../loom/tools/paper/parametric_suppression.py:2,5,7,8,9,22,23"]
    OSV --> DX --> PS
    CRX["control_rerun measures placebo exposure with the same gold_hit, see LM-07.4<br/>../loom/tools/paper/control_rerun.py:69"]
    DX --> CRX
    IDX["reads app/data/scaffold-index.json at a 1500-token budget<br/>../loom/tools/paper/decompose_exposure.py:65,69"]
    DX --- IDX
    TENS["TENSION: app/data is ignored and never tracked, so a clean checkout<br/>lacks the index the decomposition regenerates scaffolds from<br/>../loom/.gitignore:1<br/>../loom/tools/paper/decompose_exposure.py:65"]
    IDX --- TENS
    RUST["the matcher is ported to loom-scaffold at semantic, not byte, parity,<br/>and the Python stays a live import of two scripts<br/>../loom/crates/loom-scaffold/src/exposure.rs:1,2,3,4<br/>../loom/bench/LEGACY-PYTHON-NOTE.md:33"]
    DX --- RUST
    SA["semantic_audit: does the lexical title matcher agree with semantic correctness,<br/>units sampled by matcher category<br/>../loom/tools/paper/semantic_audit.py:2,22,26"]
    OPA["OPEN: the adjudicator is an LLM, the human pass is prepared,<br/>120 blind units, and not performed<br/>../loom/tools/paper/semantic_audit.py:13,14,15,16,17"]
    SA --- OPA
    PSX["paraphrase_stress: the vocabulary-mismatch set rebuilt from scratch,<br/>gpt-4.1 via OpenRouter, temperature 0, seed 42<br/>../loom/tools/paper/paraphrase_stress.py:2,12,22"]
    OPB["OPEN: the original stress set was never written to disk and is not recoverable,<br/>so these numbers are a new measurement, not a reproduction<br/>../loom/tools/paper/paraphrase_stress.py:7,8,11,13,14"]
    PSX --- OPB
    RW["rewrite_v4: paragraph rewriter for the paper's prose under mechanical<br/>content invariants, a writing tool, not a measurement<br/>../loom/tools/paper/rewrite_v4.py:2,4,10,11,12"]
```

**What it shows.** The decomposition and suppression analyses regenerate scaffolds with the git-recovered Python engine, not the Rust facade, so they reproduce the v1 sweep's configuration byte for byte (`tools/paper/decompose_exposure.py:21,22,23,24,25`). Two of the analyses say outright that their adjudicator or their data is new rather than the paper's original.

## LM-07.6 The committed proof surfaces: expectations, executed evidence, the audit, the ingest channel's doc

```mermaid
flowchart TB
    EXPS["EXPECTATIONS-rust-loom: EXP-001 to EXP-014, each needs executed evidence<br/>and then an adversarial audit by a different model family<br/>../loom/.claude/expectations/EXPECTATIONS-rust-loom.md:1,3,4,5,6,14,158"]
    subgraph EVD[".claude/evidence, tracked"]
        E1["one file per EXP except EXP-011: verdict, date, SHA, toolchain<br/>../loom/.claude/evidence/EXP-001.evidence.md:1,9,10,11,12"]
        E2["EXP-attest, EXP-deploy and EXP-embed for the ledger, the image and the embedder<br/>../loom/.claude/evidence/EXP-attest.evidence.md:1<br/>../loom/.claude/evidence/EXP-deploy.evidence.md:1<br/>../loom/.claude/evidence/EXP-embed.evidence.md:1"]
        AUD["AUDIT-gpt54: five findings and a post-fix verdict per EXP,<br/>EXP-011 is covered here, not in a file of its own<br/>../loom/.claude/evidence/AUDIT-gpt54.md:10,24,129,143"]
        AUD2["AUDIT-gpt54-f123: the same audit over the F1 to F3 serving features<br/>../loom/.claude/evidence/AUDIT-gpt54-f123.md:1,3"]
    end
    EXPS --> EVD
    RED["EXP-008: recall 0.8160 against the 0.87 floor, gate RED,<br/>LOOM_SEMANTIC_FALLBACK stays default-off, see LM-05.6<br/>../loom/.claude/evidence/EXP-008.evidence.md:3,4,5,6,7<br/>../loom/crates/loom-facade/src/config.rs:35,130"]
    E1 --- RED
    DRIFT["DOC-DRIFT: EXP-attest calls the ledger build and CI time only, not on the serving path,<br/>while the default facade serves it, see LM-01.3 and LM-02.3<br/>../loom/.claude/evidence/EXP-attest.evidence.md:8,9"]
    E2 --- DRIFT
    TENS["TENSION: EXP-011 names the CI gate as what keeps the all-features test green,<br/>hosted CI tests default features only, see LM-06.4<br/>../loom/.claude/expectations/EXPECTATIONS-rust-loom.md:105,106,107,108<br/>../loom/.github/workflows/rust-contracts.yml:53"]
    EXPS --- TENS
    WT["OPEN: EXP-012 to EXP-014 were evidenced against a pre-commit working tree,<br/>so no commit identifies the tree under test<br/>../loom/.claude/evidence/EXP-012.evidence.md:10<br/>../loom/.claude/evidence/EXP-013.evidence.md:10<br/>../loom/.claude/evidence/EXP-014.evidence.md:10"]
    E1 --- WT
    subgraph ING["tools/ingest, a README since the Rust port"]
        IR["build and write halves kept apart: build-concept-records is pure,<br/>stage_corpus rides pg-write, see LM-05.6<br/>../loom/tools/ingest/README.md:3,4,15,19,20"]
        GOLD["byte parity with the retired Python pinned by a golden on every cargo test<br/>../loom/tools/ingest/README.md:39,40,41<br/>../loom/crates/loom-scaffold/src/bin/build_concept_records.rs:417,418"]
        LAW["INVARIANT: one embedder, bge-small-en-v1.5 at 384, and the HNSW index law,<br/>the stager never emits CREATE INDEX<br/>../loom/tools/ingest/README.md:58,64,65,66,67,68<br/>../loom/crates/loom-vector-ruvector/src/bin/stage_corpus.rs:338"]
        IR --> GOLD
        IR --> LAW
    end
    ING -->|README names the gate| RED
    IDX["docs/README indexes both surfaces under EVIDENCE and OPERATIONS<br/>../loom/docs/README.md:62,78,79"]
    EVD --- IDX
```

**What it shows.** The repository carries its own proof trail: a contract per expectation, raw executed evidence per contract, and a cross-family audit that recorded where its first verdicts were refuted (`.claude/evidence/AUDIT-gpt54.md:136,138,139,140`). The ingest README points at the same RED gate the evidence records (`tools/ingest/README.md:82,83`).
