# Dream ledger follow-up — 2026-11-03

Commit `e95d183cd` on `fix/dream-ledger-followup`, based on root branch `fix/external-review-20261103` at `e2033dc787bb9b516a7a98b36a45b57e14c90ae9`.

Only `services/dream-engine/src/{digest,governance}.rs` and owning ADR-2115 changed. No identity-port, custody, environment or VisionClaw files changed.

| Finding | Repair | Regression |
| --- | --- | --- |
| Digest raw pipe split shifts verdict/witness | `digest.rs:121` calls the existing escaping-aware `governance::ledger_cells` (`governance.rs:230`); digest indices account for stripped outer table boundaries. No second parser introduced. | A finding with two escaped pipes must retain full text, ACCEPT verdict and the correct witness. Failed before the repair, passes afterwards. |
| Governance context ignores configured ledger | `governance.rs:277` calls the same validated/defaulted `config::load_dream_config` used by digest collection and reads `cfg.ledger_path`. Invalid/missing config yields no ledger context while independent manifest/gate metadata remains available. | Custom configured path plus conflicting default-path decoy: correct finding and PR must win. Failed before repair, passes afterwards. Invalid config must not silently select the decoy. Existing default-path fixture remains a control through estate defaulting. |

Validation: red run2failed; focused green3passed; full Dream suite196unit+11integration passed, bin/doc suites0tests. Clippy all-targets `-D warnings`, changed Rust-file rustfmt check and rustdoc warnings denied pass. Logs `/tmp/dream-ledger-{red,green,full-tests,clippy,changed-fmt,doc}.txt`.

Whole-crate cargo fmt reformats pre-existing `candidate_flow.rs` and `jev_triage.rs` at the root base. Those unrelated edits were restored; the changed files themselves pass formatting. No blanket full-crate format PASS is claimed.

Deepsec exact source diff review PASS at the configured HIGH threshold: receipt `workspace/.tmp/external-fixes-artifacts/dream-ledger-followup/deepsec/reports/20261103T071244Z/receipt.json`, revision `e95d183cd4978c22da992eb715f1a47fb0f560da`, clean tree; no HIGH/CRITICAL findings. One MEDIUM reports the pre-existing, documented reliance of `plan_decisions` on the relay's admin admission rather than a second engine admin allowlist. No decision-authentication code changed in this commit. Root owns disposition of that existing trust boundary; duplicating admin authority was not introduced as a ledger repair. Gate PASS is not a claim of zero findings. The gate selects the two Rust files, including their inline tests; the prose ADR note was manually checked. No deployment or forum publication was attempted.

Root integrated the commit into `fix/external-review-20261103` at merge `004c1872d`. Ignored review/session evidence and native logs preserved under `workspace/.tmp/external-fixes-artifacts/dream-ledger-followup/`. Child worktree removed; task branch deleted with ordinary `git branch -d` after verifying merged-to-root ancestry and setting its upstream to that integration branch. Final worktree list checked; foreign trees preserved. Root remains responsible for main integration.
