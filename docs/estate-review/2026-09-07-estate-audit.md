---
title: VisionFlow estate source audit and closeout reconciliation
date: 2026-09-07
status: audited-with-explicit-evidence-limits
type: explanation
---

# VisionFlow estate audit — 2026-09-07

**Execution follow-up:** [2026-09-07 source repairs, tests and runtime limits](2026-09-07-estate-closeout-execution.md) updates this dated audit view. The master register owns current task dispositions; earlier observations remain historical.

**The estate has substantial implemented components, but component success does not establish a complete federated system.** The external review identifies real boundaries—shared-UID credentials, several persistence authorities, partial federation and target-specific XR—but overstates what diagram metadata proves and repeats defects that the current source has already repaired. The most useful closeout priorities are specific failures at publication, identity and acknowledgement boundaries.

This audit updates the diagrams, reconciles current ADRs with source, preserves historical/imported decision scope, and extends the existing closeout programme and master TODO. It does not implement the production repairs in that roadmap or certify a deployed estate. The actual directory is `docs/diagrams`, not `docs/disgrams`.

## Scope, identity and evidence

The [workspace census](evidence/2026-09-07/estate-inventory.md) discovers physical Git checkouts from workspace roots and available declared submodules, deduplicates symlink aliases, inventories tracked and nonignored ADR candidates, and records source hashes. It distinguishes the 14-repository health roster from consumed dependencies and historical checkouts. Neighbouring projects are listed without treating workspace co-location as adoption. The census finds 1,113 ADR candidates across 66 physical checkouts: 269 operative candidates, 290 historical, 313 imported, 82 support, six ontology-content, nine skill-local and 144 outside-estate records. Operative records are reconciled in the source-review tables; the other categories retain their scope and lineage. Ignored files and unregistered nested checkouts are outside discovery; this is an explicit inventory boundary.

The primary diagrams cover nine repositories. Loom, Dream Engine, the WasmVOWL demo, prose-sanitiser and diagram-ir are also on the health roster; RuVector, RuView and archived Logseq have dependency or historical roles. The [repository map](../architecture/repository-map.md) now distinguishes those scopes. Local source, registry packages, Nix/image declarations and running binaries remain different identities.

There were extensive uncommitted diagram and implementation changes at the start, and another editor continued changing diagrams during the audit. Those changes were preserved. HEAD is the committed base, not a hash of the inspected working tree. Use the [machine-readable census and source hashes](evidence/2026-09-07/estate-inventory.json), each lane's source evidence, and the [validation receipt](evidence/2026-09-07/validation.json). Hashes detect change; they do not archive missing historical bytes.

| Evidence class | What this audit establishes | What it does not establish |
|---|---|---|
| Source review | A named entry point, branch, state transition or mismatch exists in inspected bytes | That a running service uses those bytes |
| Local test / counterexample | The named test or isolated production-query probe gives the recorded result | Production occurrence, complete integration or hardware conformance |
| Prior receipt, hash revalidated | Previously inspected implementation bytes remain unchanged | A fresh runtime measurement or wider acceptance than the original receipt |
| Diagram/index validation | Metadata, source paths, Mermaid syntax and bounded rendering; citation warnings separately reported | Semantic truth, owner ratification or deployment |
| Declared ADR axes | The record's stated decision, implementation and activation posture | Independent acceptance of every claim in that record |

## What the external review gets right—and what needs correction

| Review claim | Current source-grounded assessment |
|---|---|
| 1,300+ diagrams are automatically verified against commits | The index counts blocks and reads author-supplied revisions. Hosted CI skips sibling source paths. Citation checks are local and warning-only. The generator now says **declared source revisions** and repository-qualifies ADR numbers. It never proved semantic correctness. |
| Multiple stores imply a need for distributed 2PC | Multiple authorities are real, but 2PC is not established as a requirement. The concrete defects are non-atomic generation publication, incomplete erasure/restore membership and receipts that can disagree with applied state. Specify each authority and reconciliation boundary. |
| Authentication variety itself proves a large vulnerability | Multiple auth realms are implemented. Threat assessment must follow authority and caller paths: shared-UID custody, fallback identity, session/HTTP/WS gates and optional feature combinations. Protocol count alone does not establish an exploit. |
| PTX rewriting is a regex hack likely to segfault | Current code uses one shared span parser with structural validation. Seventeen PTX policy tests pass. Header policy still does not prove the instruction set, driver, kernels or GPU ABI compatible; no runtime segmentation fault was demonstrated by this audit. |
| 212-byte SimParams and 52-byte frames imply fragile unchecked layouts | They are documented ABI contracts with checks. The 52 bytes are a node record inside a versioned envelope, not the total WebSocket frame. Target execution remains necessary. |
| GPU fallback is inconsistent | Correct when scoped by actor: some computations fall back, SSSP is GPU-only, and APSP is disabled. Updated diagrams identify the actual failure behaviour instead of citing removed fallback code. |
| Agentbox confines one agent from another | It does not supply per-agent UID isolation. Seccomp/capability controls are supplemental; a same-UID process can still reach owner-readable credentials. The deny gate covers 46 syscall names plus AF_ALG, but a custom profile replaces Docker's default selection rather than inheriting an assumed second profile. ADR-2040 also has sanctioned unauthenticated wayvnc/x11vnc listeners reachable by sibling containers; host loopback publishing does not fulfil its universal authenticated-listener rule. |
| Learning fails when an LLM omits an error phrase | Current capture reads structured transcript tool results, including `is_error`, and skips unknown outcomes. It remains dependent on retained transcript coverage and consumer admission; text-guessing is not an accurate account of the primary recorder. |
| ES-08.9 proves HNSW recall degrades under bulk deletes | That panel describes missing cross-store erasure propagation. The historical recall incident concerned parallel index construction; tombstone handling, erasure propagation and measured production recall are separate obligations. Preserve dated FAIL and later PASS evidence without presenting either as today's measurement. |
| Federation is wholly fictional | Forum types, transports/planners, relay admission and several point-to-point paths exist. Outbound join points, configuration consumption and a complete agent→human→applied journey remain incomplete. Neither “fully wired mesh” nor “nothing implemented” matches the source. |
| BrokerActor absence proves governance absence | The old actor was left behind; the current broker kernel, REST/ACSP paths and durable decision-case store exist. End-to-end correlation, legal concurrent transitions and external application receipts still need work. |
| Godot and browser clients are struggling for XR parity | The browser immersive-session surface was removed; capability probes remain. Native Godot desktop compatibility rendering is the supported configuration in source; Quest APK and target acceptance remain separate. Stored-action freshness checks now exist, contrary to a stale governing paragraph corrected in this audit. |

Detailed evidence: [VisionClaw](2026-09-07-visionclaw-audit.md), [Agentbox/Loom](2026-09-07-agentbox-audit.md), [federation and publishing](2026-09-07-federation-audit.md), [imported decision scope](2026-09-07-imported-adr-scope.md).

## Findings that determine sprint order

These priorities are engineering judgement based on the observed consequences. A reproduced local defect is not automatically a deployed incident.

| Finding | Evidence and consequence | Closeout route |
|---|---|---|
| EA-01 Public inference can expose private identifiers | The active visionGraph producer's local fixture retains a private grandparent identifier in inferred RDF. Stronger knowledgeGraph working-tree controls do not repair a different producer automatically. See federation F-03 and its probe. | First: establish the actual publishing implementation and test private ancestors across every public export. CP-02/04. |
| EA-02 Ontology publication and ACL probing violate the failure promise | `ontology_pull.rs:345-373` writes sequentially; manifest-last does not roll back earlier files. `exists(...).await.unwrap_or(false)` treats an ACL read failure as absence. ADR-2106 now states partial implementation of the full contract. | Atomic generation activation or a proven equivalent; fail closed on indeterminate ACL existence; failure injection and restart evidence. CP-02/04/08. |
| EA-03 Trust audit can record a transition that did not happen | Exact production queries in SQLite: guarded UPDATE changes zero rows, audit INSERT commits, actual TL3 remains TL3 while audit claims TL1→TL0. Batch error rollback does not fix successful no-op writes. | Conditional state/audit transaction with concurrency tests. CP-04/05. |
| EA-04 Governance receipts do not yet prove one legal applied decision | Request reference is optional; case UPDATE lacks an observed-state predicate; statement success alone does not establish affected rows or external application. Missing-case insertion may be rejected by the FK, so that outcome is not claimed as reproduced. | Full correlation, guarded transitions, replay and external application evidence. CP-05/09. |
| Existing G-5/G-6/G-17: custody and identity | Shared UID remains the boundary; break-glass optional scope/expiry checks exist but durable custody/rotation acceptance is incomplete. Fail-closed minting is distinct from shell fallback identity. | Separate authority and named custody policy; verify denial/rotation/revocation and loaded configuration before expanding exposure. CP-04/08. |
| EA-05 and existing G-19: served generation and reproducibility | Loom has a generation endpoint; Agentbox does not consume/verify it. Loom also depends on a sibling RuVector checkout, whose source identity differs from Agentbox's declared Postgres image and RuView's registry crates. | Client attestation, cache identity/mismatch handling and reproducible dependency inputs. CP-01/03/08. |
| Existing G-16/G-1: numerical and release proof | PTX text tests do not close the recorded LOF oracle failure or prove an artefact excludes dev-auth. CI shell checks and boot guards are real but have bounded scope. | Preserve LOF as unresolved until target oracle rerun; test the produced release artefact and negative auth cases. CP-06/08. |
| Existing D-1/DR-1+DR-2/M-3: learning and activation | Historical recall results and source fixes are not a current loaded-index or Nix-process receipt. Darwin bounds diagnostic checks two bounds on the CLI path; promoted-lineage count is explicitly unobserved there. | Bind corpus/model/index/configuration and candidate/evaluator/output identities; rebuild then verify the actual process. CP-07/08. |
| EA-07 Optional sensing authentication/execution claims exceed code | RuView package verification uses a digest of public inputs rather than Ed25519 verification; elapsed-time budget is checked after synchronous execution. Source is not evidence of safe untrusted execution. | Keep profile excluded until real authentication and bounded nontermination tests pass. CP-04/06. |
| EA-06 and existing G-18: evidence and store membership | Repeated ADR numbers collided in coverage; defaults were described as all possible persistence. Optional Redis session persistence is implemented in VisionClaw and must be accounted for when enabled. | Repository-qualified record keys, effective-profile inventory, source hashes and acceptance receipts; do not collapse different proof classes. CP-01/08/09. |

## ADR reconciliation

The census gives every discovered record a repository/path identity and a scope disposition. Current source assessments cover 99 VisionClaw ledger records, 74 Agentbox ledger records, 31 federation/publishing ledger records, four Loom design decisions, the eleven VisionFlow operative/engineering records below, four Dream Engine records and the separate WasmVOWL proposal. Support registers and frozen records are not counted as adopted decisions. The [sensing reconciliation](2026-09-07-sensing-audit.md) and [imported-pack review](2026-09-07-imported-adr-scope.md) preserve component and consumer limits.

Historical lineage remains in the [VisionClaw](closeout/visionclaw-history.md), [Agentbox](closeout/agentbox-history.md), [Logseq](closeout/logseq-decision-lineage.md) and [upstream](closeout/upstream-packs.md) companions. These are justified routing and scope decisions, not a claim that every historical or upstream acceptance promise has been independently executed. In particular, unused imported quantum, coherence, example and application packs do not become requirements of the deployed memory service. Newly demonstrated callers reopen that scope.

| VisionFlow record | Core source checked | Disposition |
|---|---|---|
| ADR-2001 | `scripts/adr-index-gen.cjs`, `.github/workflows/adr-index.yml`, living ledger and archive | Baseline/ledger split and metadata gate exist. Archived Context remains historical. |
| ADR-2002 | `website/build.sh:18-33`, `scripts/website-assets.mjs` | Copy-only staging and required-asset verification exist; asset regression tests pass. No new deployment claim. |
| ADR-2003 | `.github/workflows/deploy.yml` | Build/verification precede Pages artifact upload/deploy. Gate success is not current live-site verification. |
| ADR-2004 | `scripts/check-diagram-text.js`, `scripts/diagram-render/render.mjs`, topic generator | Distinct render/visibility and source-review boundaries retained; misleading index wording and ADR identity aggregation corrected. |
| ADR-2005 | `scripts/drift-counter/drift-counter.mjs:68-119`, allowlist | Pin mismatch correctly fails in this workspace; isolated checkout at the reviewed pin passes the gate suite. |
| ADR-2006 | health roster, `scripts/generate-release-manifest.sh`, `scripts/check-fixture-drift.sh` | Canon coordinates a declared estate; neither repository co-location nor manifest emission proves a shipped system. |
| ADR-2007 | review/census/sprint artefacts and master TODO | Extended, still proposed/partial/staged. Owner ratification and complete-system journeys are not invented. |
| ADR-2008 | `scripts/estate-health.mjs`, `.github/workflows/estate-health.yml`, site consumer | Nightly collection/offline reading is wired in source. Snapshot age and source readability remain observable states, not guarantees of current service health. |
| ADR-2009 | `dream.config.json`, evaluator scripts | Four current rotation slots omit webgl-mesh; source inspection does not establish browser behaviour. |
| engineering ADR-004 | `scripts/harness-audit.sh:251-272,331-332`, template/schema files | Pair de-duplication and real source checks exist; regression suite passes. Planned hooks are not promoted to enforcement. |
| engineering ADR-005 | harness schema, Agentbox kind allowlist and memory admission | Mandate-at-grant proposal remains speculative. Existing protected memory namespaces do not supply all grant channels. |

| Supporting record | Core source checked | Disposition |
|---|---|---|
| Dream Engine ADR-0001 | `packages/compile/src/index.ts:242-282`, CLI/workspace boundaries | Compiler emits candidate/evaluation/promotion rules. Toolkit and Agentbox runtime are separate execution paths. |
| Dream Engine ADR-0002 | `packages/cli/src/entrypoint.ts:32`, `index.ts:322-382` | Liveness classifier distinguishes blocked/silent/live. Arbitrary nonempty output is not a semantic evaluator verdict. |
| Dream Engine ADR-0003 | `packages/cli/src/darwinBounds.ts`, CLI Darwin branch | Bounds checker rejects empty parsing and excessive depth/candidate count; CLI passes promotion count zero explicitly, leaving that bound unchecked. Diagnostic exit is not proof of nightly admission enforcement. |
| Dream Engine ADR-DL-001 | compiler/CLI versus `services/dream-engine` candidate/gate source | Outer-loop proposal remains proposed. Later inner-loop source progress qualifies the older “absent” premise, without establishing a trustworthy deployed optimisation target. |
| WasmVOWL ADR-001 | `modern/src/hooks/useWasmSimulation.ts:84-98`, Rust parser `:51-56`, Map store | Hook sends nodes/edges while parser requires class/classes. All seven earlier source hashes remain unchanged. Proposal remains partial/inactive; earlier test failures are preserved as historical results, not rerun results. |
| prose-sanitiser / diagram-ir | Cargo workspace/package and binaries | Distinct tool packages on the roster; no local ADR ledger discovered. Their availability is not evidence that all diagram/publication gates invoke them. |

## Validation and limits

The [validation receipt](evidence/2026-09-07/validation.json) holds commands, exit codes and log hashes. Focused tests include PTX policy (17), Agentbox learning/admission (57), forum relay (239), pod OIDC (28), VOWL (136), website (68), knowledgeGraph (85), Dream Engine compiler/CLI (81), and Agentbox Dream Engine library re-verification (155). ADR-2081 initially failed its source-provenance gate; the old failure is retained alongside the tested correction and synchronised 74-record index. VisionGraph reports 57 passing and one failing test: the existing dirty `_misc` corpus has eight pages where a test expects at least fourteen. This audit preserves that result and the authored deletions instead of weakening the test or restoring unrelated files.

The earlier RuView/WasmVOWL evidence revalidation checks 265 source references: 255 are byte-identical and ten changed references are ADR documents with later annexes. No implementation source in that set changed. The [revalidation receipt](evidence/2026-09-07/prior-evidence-revalidation.json) and script preserve each mapping; old test results remain old.

No new GPU-kernel execution, headset test, device sensing, published-package resolution, deployed binary inspection, cross-store erasure, restore drill or complete federated user journey was performed. These are explicit acceptance tasks in the [closeout sprint](closeout/2026-09-07-sprint.md), not hidden failures or presumed successes. The final diagram snapshot contains 130 topics and 1,398 diagrams. The citation check reports no warnings, and every diagram source matches its rendered input with a present SVG inside the width limit. The initial three render failures and subsequent repair receipts are retained; snapshot equality is not a semantic or deployment attestation. The index regression suite checks repository-qualified ADR identity and declared-revision wording.

The [master TODO](../../../project/docs/TODO-unified.md) is the row-level work queue. This audit supplies the evidence and dependencies needed to finish its closeout work without converting documentation progress into a claim that the system is complete.

## Publication and staged consolidation — 2026-09-07

The authorised publication includes VisionClaw's staged removal of the duplicate diagram corpus and its old generator/validator. `project/docs/diagrams/README.md` redirects readers to the VisionFlow canon; the README hero image moves to `docs/assets/hero/linkedInEcosystem.png`, with its consumer link updated. Git retains the removed corpus history. This consolidates documentation ownership without changing a runtime service.

The 10:46 UTC validation receipt remains a dated 1,398-diagram snapshot. A concurrent editor subsequently added AB-06.9; the publication index contains 1,399 diagrams and its citation check passes without warnings. Another session is rebuilding generated SVGs. That later work does not retroactively change the earlier render receipt or imply a fresh complete render result here.
