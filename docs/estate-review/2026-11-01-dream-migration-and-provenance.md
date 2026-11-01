---
title: Dream-engine migration and dependency provenance audit
date: 2026-11-01
status: assessed-with-open-findings
scope: ADR-DE-001, ADR-DE-002, migration contracts and consumed package identity
---

# Dream-engine migration and dependency provenance audit

Rust source matches between the extracted engine and the four engine crates consumed by its adapter. That does not establish complete migration parity: executing the old TypeScript ledger validator reveals four whitespace differences missing from the compatibility fixtures. The adapter's prohibition on registry publication is also a written policy rather than a Cargo restriction, and the permissive packages omit the repository's provenance notice.

This pass assesses ADR-DE-001 and ADR-DE-002 section by section, updates AB-23.3 and DE-01's dependency qualification, and adds DE-05's two migration/publication panels. All nine local dream-engine decisions now have dated section assessments, with implementation and acceptance findings still open. Eighteen other AB-23 panels and the nested engine corpus remain queued. [Candidate](2026-11-01-dream-candidate-and-evidence.md), [recovery](2026-11-01-dream-recovery-and-source-supply.md) and [forge/seam](2026-11-01-dream-forge-and-seam-contracts.md) assessments retain their findings. Owner ADR statuses and production code are unchanged.

## Evidence and limits

Reviewed revisions are dream-engine `815f963b1cbd6999e78144ce5475005ae1dd1d8e`, agentbox `9dfc27b16e82e9786535a55e790710493dd51fdf` and the linked legacy dream-machine `b77093e564ef689923fb523b0e0b18d2bb4a21a4`. [Source hashes](evidence/2026-11-01/dream-migration-source-manifest.json) identify the reviewed inputs. The legacy repo was used only for the ledger validator and its package context; this does not close its separate repository audit.

[Nine compatibility tests](evidence/2026-11-01/dream-migration-ts-parity.log) and [eleven host-night tests](evidence/2026-11-01/dream-migration-host-night-tests.log) pass in clean exports. Host tests use scripted models, recording forges, local evaluators and SSH/SCP shims that refuse non-fixture hosts. They disable source reading, journal delivery, sweeping, digest, governance and forum suggestions. Memory configuration is empty: embedding requests fail before any database operation. These tests cover local orchestration, not real model, remote annexe, forge, memory or deployed-image acceptance.

An additional [seven-case differential probe](evidence/2026-11-01/dream-migration-ledger-differential.json) executes both implementations on identical ledger rows. The [Rust probe](evidence/2026-11-01/dream-migration-ledger-probe.rs) calls the public validator; the [Node driver](evidence/2026-11-01/dream-migration-ledger-oracle.mjs) imports the unmodified, pinned TypeScript source using Node's type stripping. Two cases agree, four expose previously undocumented whitespace differences, and one confirms the documented VETOED-prefix tightening. These are seven compared cases, not seven additional Rust tests.

[Dependency evidence](evidence/2026-11-01/dream-migration-dependency-receipt.json) records 120 packages in the offline core metadata graph and 329 in the adapter graph. Cached archives for dream-core 0.4.0, dream-gate 0.3.0, pairwise-gate 0.1.0 and failure-fingerprint 0.1.0 match the adapter lockfile checksums. All 36 corresponding `src/**/*.rs` files are byte-identical to the inspected engine checkout. This binds those local cached artefacts to the consumer lock; it does not verify a current registry response, every package file or the running daemon.

## Findings and acceptance work

| Finding | Evidence and consequence | Acceptance required |
|---|---|---|
| DE-M1 / medium: ledger compatibility differs on invisible whitespace | `dream-gate/src/verdict.rs:394,438-451` uses Rust trimming; `dream-core/src/ledger/read.rs:215` trims cells similarly. Legacy `packages/ledger/src/rowContract.ts:63,82-87` uses JavaScript trimming and regular-expression whitespace. A U+FEFF-only finding passes Rust but is empty in TypeScript; `x see<U+FEFF>report` passes Rust but violates the old pointer rule. U+0085 yields the reverse outcomes. | Specify one whitespace contract and test both implementations over the relevant Unicode set. Correct the claims at verdict.rs:392 and ledger/read.rs:227-229 that accepted findings and flagged rows match. Treat the empty-looking accepted finding as a data-quality defect; this probe does not demonstrate executable patch acceptance. |
| DE-M2 / medium: the migration gate is not host equivalence | `tests/ts_parity.rs:3-7` says expected results come from reading TypeScript and an independent Python rendition. It checks configuration, ledger and witness examples, not a complete old/new nightly driver. `docs/guide/migrating-from-the-ts-fork.md:43-46` points host retirement at this suite. Host tests exercise the new adapter with substitutions. | Publish a migration matrix identifying compatible, deliberately changed and untested paths. Include differential runtime cases and operational acceptance for scheduling, remote execution, delivery and recovery before claiming host parity or removing rollback. Passing these suites alone cannot justify that broad claim. |
| DE-M3 / medium: “never publish the adapter” has no manifest guard | Agentbox `services/dream-engine/Cargo.toml` has a version and no publish restriction; metadata reports publish=null. Offline `cargo package --list` includes `src/zone_crypto.rs`. No publishing command was run. | Set an explicit publication prohibition and test it in the consumer repository. Cargo documents `publish = false` for preventing accidental registry publication. This finding concerns a missing release control, not an observed upload or disclosure of runtime key material. |
| DE-M4 / low: crate recipients do not receive the repository NOTICE | Each of the five package lists includes its own MIT and Apache licence files, but none includes root NOTICE. That file carries the rUv design-lineage credit and reproduced upstream notice required by the record's stated approach. `dream-core` also excludes its workspace-dependent parity test. | Decide and document which provenance files accompany every released package; test the package contents. Include or link reproducible compatibility fixtures if consumers are expected to rerun them. Omission is a distribution/traceability gap here, not a legal finding that the licences have been breached. |
| DE-M5 / qualification: graph policy is not authorship proof | deny.toml allows licence expressions with a permitted alternative, bans three named host-coupling crates and rejects unknown registry/git sources. CI invokes cargo-deny. Metadata contains no AGPL-only package in the core graph; an LGPL alternative with MIT/Apache options is present and is not itself a contradiction. Cargo-deny is unavailable locally, so no fresh deny result is claimed. | Keep licence scanning, source provenance and release identity as separate evidence. Preserve contribution origin and ownership records for extracted code. A licence label or dependency direction cannot establish the universal “no copied source” assertion. |
| DE-M6 / low: minimum compiler and migration companion remain stale or weakly verified | Workspace rust-version is 1.89, while CI installs stable. No resolved core package declares a higher minimum, but that is not a build on 1.89. Agentbox's extraction ledger still says ADR-OM-NNN no longer loads; the later ADR-2141 re-verification and current config compatibility disagree. | Add a minimum-version build and correct the companion's obsolete migration advice. Keep dated activation evidence distinct from later source pins; this pass establishes no current 0.4.0 image activation. |

[Cargo's manifest reference](https://doc.rust-lang.org/cargo/reference/manifest.html#the-publish-field) supports DE-M3's proposed guard. [Cargo-deny's licence policy documentation](https://embarkstudios.github.io/cargo-deny/checks/licenses/cfg.html) explains allowed licence alternatives; this audit does not interpret every third-party licence or certify legal compliance. [Package lists](evidence/2026-11-01/dream-migration-package-lists.json) reproduce DE-M3/M4 without creating or uploading a registry release.

## ADR-DE-001 section dispositions

| Section or clause | Assessment |
|---|---|
| Context: TypeScript and product-local Rust ancestry | The migration suite, legacy validator and host extraction ledger support this lineage. The historical claim that nobody else could use either implementation was not measured and is not a present adoption result. |
| Decision: standalone Rust workspace and permissive crates | Five members and declared dual licences are present. Resolved source dependencies and the consumed registry sources were checked. |
| Decision: retire TypeScript after host parity | A compatibility suite exists and passes nine tests, but does not run TypeScript or a host. DE-M1/M2 show why it cannot establish full equivalence. Agentbox's Nix packaging describes the old orchestrator as a manual fallback, rather than deleted source. |
| Decision: product-specific code remains behind seams | Core and adapter dependency direction is confirmed. The adapter contains forum/identity/envelope modules. Its no-publication requirement lacks a Cargo guard. Seam limits remain in the preceding audit. |
| Consequences: one engine and reusable loop | The library is separate and the adapter consumes registry releases. A retained legacy fallback means “one engine” is an intended maintenance direction, not a literal claim that only one implementation remains. No second independent production consumer was established here. |
| Consequences: all five crates tested | Existing workspace evidence is linked by prior passes; this pass reruns only compatibility and isolated host-night suites. It also checks package membership and dependencies, not every crate test again. |
| Consequences: in-progress due to host driver and TrainDev | Host driver and historical activation evidence now exist in agentbox; local ADR status does not track those later facts. TrainDev exists as a leaf policy but remains unwired to held-out evaluators in the engine gate, as the candidate assessment records. Neither fact justifies changing owner status during this audit. |

## Migration guide section dispositions

| Section | Assessment |
|---|---|
| What stays | Witness test vectors pass. Ledger compatibility has documented width/prefix changes and the four newly reproduced whitespace differences; “unchanged” is too broad. |
| What is dropped | Dropping the Darwin mutator is a stated choice. The claimed historical uniform 0.985 across 21 variants was not rederived here; it is not new performance evidence. |
| What is new | Manifest, verdict, protected-path, pairwise, fingerprint, seam and optional merge implementations exist, with qualifications in all preceding engine reports. “Implemented” does not mean their known boundaries are accepted. |
| Further additions | Recovery, replay, per-test parsing and dry-streak semantics have code and tests; previous audits retain recovery and evidence limitations. TrainDev integration remains explicitly absent. |
| Order of work | Separation of offline checks, library and host is accurate. The referenced fixture suite is insufficient as the sole host retirement test and is excluded from the published dream-core package. |

## ADR-DE-002 section dispositions

| Section or clause | Assessment |
|---|---|
| Context: upstream MIT lineage and consumer AGPL/envelope | Legacy ledger package declares MIT; NOTICE credits rUv. The adapter manifest declares AGPL-3.0-only and owns zone_crypto. This establishes declarations and placement, not a full provenance or legal audit. |
| Decision: five MIT OR Apache-2.0 crates | All five manifests inherit the dual expression and package lists contain both licence files. The four consumed crate archives match the host lock checksums. |
| Decision: independent implementation, no copied text | NOTICE asserts independence from dream-machine; ADR-2141 describes extraction from product-local Rust. File hashes and a dependency graph cannot prove authorship, permission history or absence of copied text. Preserve contribution-level evidence before presenting that assertion as audited fact. |
| Decision: reproduce upstream MIT notice | Present in the repository NOTICE; absent from each of the five crate file lists. DE-M4 is a release traceability gap. |
| Decision: adapter AGPL, never published; envelope not reusable | Placement and declared licence are verified. The whole adapter remains eligible for registry publication and contains the envelope module in its package list. Written intent is not enforcement; DE-M3 remains open. |
| Decision: permissive crates never depend on AGPL | Current core metadata graph contains no AGPL-only dependency and no consumer dependency. CI policy aims to prevent one, but its fresh execution is unverified in this environment. Licence alternatives must be evaluated as expressions, not banned by substring. |
| Decision: same-licence contributions, origin required | This is a contribution policy. This pass does not establish that every historical contribution carries sufficient origin evidence. |
| Consequences: usable under either licence, physical boundary | Workspace declarations and dependency direction support the intended separation. They are not universal guarantees against accidental future publication or misdeclared third-party source. |
| Consequences: cargo deny in CI blocks disallowed licences | Workflow and policy exist. Local cargo-deny is missing; metadata inspection is explicitly not a substitute execution result. Minimum-version and package-content gaps remain separate. |

## Diagrams, host scope and comparison

AB-23.3 now distinguishes the module boundary, exact consumer pins and missing publication guard. DE-01's licence invariant is qualified by the current dependency graph. DE-05 separates library fixtures from host tests and shows where provenance and publication policy stop. Nested CRT-01 was read in full: its claim that the ADR names only four crates is obsolete, and its absolute downstream licence guarantee exceeds what source placement or CI can establish. Its minimum-compiler concern remains valid. The nested topic was not edited or marked complete in passing.

ADR-2141 was read for extraction, pins and activation context. Its full behavioural, test-accounting and operational obligations remain a separate host assessment; reading the record and rerunning eleven night tests does not complete that work. Current tests disable several night-level services. Its extraction companion's obsolete config rejection is recorded as an owner finding. No live daemon, Nix image, remote annexe or old nightly script was run.

[REEF's current primary README](https://github.com/Human-Agent-Society/reef) describes both model-weight training and prompt/skill optimisation, with interactions and feedback feeding selection and versioned delivery. Dream-engine's local comparison page describes a narrower tree of candidate agent configurations and should be revised before using it for product positioning. The engine here is a repository-patch toolkit whose host supplies execution and delivery; its credible advantage would need evidence of useful changes, reproducibility and low operator effort. This is an inference from the two descriptions, not a comparative benchmark or a ranking.

For adopters, measure time from a clean checkout to the first reproducible draft, compatibility failures on existing ledgers/configs, review effort, and the percentage of accepted experiments delivered with a matching commit and recoverable evidence. A documented host migration recipe, executable compatibility corpus and package provenance receipt would make that evaluation possible. Training weights is a different use case and should not be conflated with repository maintenance.

[Reproduction notes](evidence/2026-11-01/dream-migration-reproduction.md), [corpus check](evidence/2026-11-01/dream-migration-corpus-regeneration.log), [final verification](evidence/2026-11-01/dream-migration-final-corpus-check.log) and [retained artefacts](evidence/2026-11-01/dream-migration-retention.json) support this pass. The structural gate covers 180 topics and 1,596 panels, with 24,556 verified citations, nine documented artefact exceptions and 41 declared revisions. The wider estate audit remains active.
