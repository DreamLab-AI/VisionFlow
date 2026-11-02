# Forum, website and dream-engine external-review reconciliation

Completed source rechecks and owner-local ADR qualifications on 2026-11-02. No implementation, browser journey, production account mutation or deployment was performed.

| Owner | Reviewed source | Merged documentation commit | ADRs updated |
| --- | --- | --- | --- |
| nostr-rust-forum | c39f41f104b114d6a3fa29013daaf6fa58b45080 | 944e844 | 2003, 2004, 2006, 2008, 2011, 2014, 2016, 2017, 2023 |
| dreamlab-ai-website | 0e1572801d03a1af6d8bc8861f928da087d69eed | d7e39fb6 | 2005, 2006, 2007, 2010, 2011 |
| dream-engine | bfe8c8885a58cc576a18ecb55d4e05e6f3bd37c7 | ce94bbc | DE-003, DE-008, DE-009 |

Each repository now has `docs/adr/external-review-2026-11-02.md`, with a row for each scoped external concern and acceptance conditions. Relevant ADRs link to that source qualification before Context. Original decisions, verified_commit fields, implementation_status and activation_status are unchanged. The additions are dated qualifications, not whole-document re-verification.

## Dispositions

Forum: confirmed unsigned passkey register/verify requests and random-options versus deterministic-stored PRF salt mismatch; omitted device_proof and f64-worker versus integer-client fields; qualified device rebind to require signed owner-bound device proof and to respect the existing known-principal guard; confirmed admin union/demotion mismatch, invite code/id mismatch and cohort SELECT/UPDATE race; confirmed separate push-drain SELECT/subtract race, current-bump rather than claimed-bump mark, and service-worker IndexedDB versus Rust localStorage cfg mismatch; confirmed cross-tick starvation boundary because trust sweep starts with no cursor; retained source-identical earlier sealed-history and retained-Rc signer findings. Device exact-string policy is shared as a pure helper with independently enforced worker checks, correcting the external implication of entirely duplicated logic. Relay/rationale pipeline architectural strengths remain supported; stage counts and historical live receipts remain revision-bound. Refer to local note for exact paths and lines.

Website: confirmed SEO check of empty staged index after successful commit, unsanitised advice.error ledger sink, disclosure/registry confusion shared with forum ingress merging, missing no-privilege health verification and marking more wraps seen than the capped digest displays. Hand-synced configuration is an explicit accepted trade-off with bounded checks, not an accidental absent generator. Key-split execution is unverified; the existing ephemeral NIP-42 plus unverified extension-hint contract does not prove operational key custody. Missing local encryption ADR recorded against existing ADR-2007; no new decision or ADR-2016 number was invented.

Dream-engine: confirmed literal `#[path` formatter guard misses valid spaced syntax; no post-evaluator comparison before reporting prepared tree hash/patch; failing_excerpt reads committed hunk text without normal source-stage denylist/key-content filtering. Required receipt gate and permissive/AGPL host split are supported. Persisted-before-forge is deliberate at-most-once ordering with possible lost publication, not exactly-once delivery. Host-specific roster/SSH/model/deployment assertions are outside standalone engine source authority.

## Validation and closure

`git diff --check` passed for all three branches. All 17 new ADR links resolve to the owner-local note; original frontmatter was untouched. `inspect-text` reported zero suspicious characters in the three notes. No code test was needed for these documentation-only changes. Every task worktree was checked with `status --porcelain --ignored` before removal, then its merged branch deleted. Foreign worktrees and dream-engine's existing dirty review ledger/untracked review files were preserved. Primary heads after merge are the documentation commits above.

Ledger handling: these additive ADR byte changes make old completion hashes stale until the lead records a scoped additive-note receipt or retains the stale disposition. Do not silently claim a new exhaustive ADR audit. Nothing pushed.
