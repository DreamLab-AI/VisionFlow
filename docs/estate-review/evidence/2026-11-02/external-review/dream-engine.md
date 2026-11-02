# External review reconciliation — 2026-11-02

Source baseline: `bfe8c8885a58cc576a18ecb55d4e05e6f3bd37c7`. The external review largely restates the existing DE-02 and AB-23 boundary findings. This source recheck changes no decision or activation status and executes no candidate evaluator.

| External concern | Current-source disposition and acceptance |
| --- | --- |
| Formatter path escape (DE-02.2) | Confirmed. `crates/dream-core/src/candidate.rs:115` searches for the literal substring `#[path`; a spaced Rust attribute is not matched. Formatting runs locally without a filesystem sandbox. The intended ADR-DE-009 footprint rule is stronger than its lexical guard. Test spaced/token variants and formatter-followed baseline modules, including absolute paths outside the worktree, in an isolated fixture. |
| Evaluator mutations versus shipped hash | Confirmed gap in the claim. `candidate.rs:546–558` runs build/evaluate, then reports prepared.tree_hash and prepared.applied_patch without a post-evaluator tree check. A pre-evaluation hash alone cannot prove evaluators judged the bytes later shipped. Require a post-evaluator integrity check and a mutation fixture. |
| Repair excerpt secret withholding (AB-23.5) | Confirmed in the standalone producer. `crates/dream-core/src/source.rs:1320–1336` failing_excerpt calls read_blob and emits verbatim committed context without the source-stage Denylist or has_secret_content filter. The normal gathering path's protection does not govern this path. Route repair context through the same policy and test filename/key-shaped withholding. No real secret was read during this review. |
| Required evaluator gate | Supported. `dream-gate/src/gate.rs` makes missing/nonpassing required evidence veto ACCEPT and marks unproven candidates inconclusive. This is a decision gate over supplied receipts, not an evaluator sandbox or proof of tree immutability. |
| Persist-before-forge lifecycle | Supported ordering, qualified consequence. `dream-core/src/runstate.rs` journals Persisted before forge publication, but explicitly records that outcomes are not journalled and a crash can leave publication incomplete. Do not claim exactly-once draft delivery from the ordering. |
| Permissive engine / AGPL adapter split | Supported ownership boundary. Standalone engine/gate APIs and Agentbox host adapter remain separate. Agentbox roster, SSH health, model routing and deployment claims belong to the host adapter's pinned source and cannot be certified by this standalone source pass. |
| Formatting before the gate | Supported stage ordering in ADR-DE-009; the external description must retain the open escape and tree-integrity findings above. An implemented formatter stage is not a completed confinement claim. |

Acceptance must distinguish source checks, controlled mutation reproductions, actual nightly execution and package/consumer adoption. Existing VisionFlow dream-corpus evidence remains the baseline for earlier reproductions.
