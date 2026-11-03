---
title: External estate review repairs and release handoff
date: 2026-11-03
status: in-progress
type: reference
---

# External review repairs

This implementation pass follows the [source reconciliation](2026-11-02-external-review-reconciliation.md).
The fourteen supplied remediation groups are tracked below. A source merge, a
registry release and a deployed consumer are separate outcomes. Historical
diagram and ADR verification stamps have not been swept forward.

## Implementation register

| Group | Source result | Remaining acceptance |
| --- | --- | --- |
| 1 Passkeys | Forum `7852fe8`: both clients sign registration; registration and login retain the issued PRF salt. | Deploy workers/clients. Existing broken registrations require re-enrolment; no speculative identity migration ran. |
| 2 Device keys | Forum `7852fe8`: device proof, integer response types and guarded re-enrolment implemented. | Deploy and exercise an actual browser ceremony. |
| 3 SEO publisher | Website `c3849abb`: inspect committed paths after commit, scrub errors before logging, and enforce changed-line limits using Git's parsed patch statistics. | Activate the nightly source. |
| 4 Dream engine | Dream-core `bf3a11d`: stdin-only formatting, confined explicit writes, post-evaluator tree checks with a fresh index, denied-path/whole-blob withholding in repair, shared Ontomark credential detection and NUL-delimited tracked-path parsing. Core 0.4.1 published and adopted by Agentbox `ba4e8cc2a`. | Image rebuild. Final-state identity checks do not sandbox arbitrary evaluator execution. |
| 5 Solid | Solid `72d3f45`: owner-only whole-history Git export with correct routing, provenance WAC/SHA binding, malformed ACL denial and canonical-payment read failure handling. Additional numeric and credential-copy findings repaired. | Publication held: full-diff review found additional payment/ACL issues now under repair. Alpha.19 remains unpublished. Consumer adoption and rebuild follow verification. Historical backups are not erased. |
| 6 SideStr | `7f046444`: duplicate withdrawals require sufficient independent payment, snapshot bytecode hashes are checked, and header/clock failures do not evict otherwise valid mempool transactions. Core 0.4.3 published. | Consume the new core and rebuild the private EVM application. No running chain was changed. |
| 7 Ontology contracts | Ontomark `6c3f693`: resource IRIs remain immutable and versionIRI metadata is excluded from consequence deltas. Agentbox explicit promote digest guard merged in `ba4e8cc2a`. VisionClaw's tested branch refuses to sign unsupported grouped proposals. | Finish VisionClaw consumer pins and merge its held branch after the outstanding review boundary is resolved. Grouped application is not implemented. |
| 8 Membership/push | Forum `7852fe8`: relay authority, invite lookup/cohort updates, atomic drain/sequence accounting and logout registration state repaired; trust sweep uses a persisted cursor. | Deploy and validate migrations/real subscriptions. This does not promote every part of ADR-2014 to complete. |
| 9 Inbox | Website `c3849abb` plus forum `7852fe8`: registry presence is distinct from configuration disclosure, privilege drift is checked, and only successfully displayed wraps are marked seen. | Deploy both sides of the additive `registered` response field. |
| 10 XR/presence | VisionClaw `6b811f7a1` and Codescape `d2c3c3b`: release upgrade, drag/ack, holder, link-budget/provenance, portal scaling, topology grab and reliable presence-transition fixes tested in isolated worktrees. | Branches held: registry pins and review follow-up remain. The review worker requires hardware-key re-authentication. Live HMD and operator spring work were not changed. |
| 11 VOWL/explorer | VOWL `2aff199`: bounded legacy subdivision, finite parameters, reload reset and retryable immutable npm publication. KG `4809ea9772e`: transferred-view accounting repaired. Official Rust 1.99.0 build/tests passed; GitHub release bundle uploaded and adopted by VG `f7186a7d2`. | npm publication failed with E404 (package/permission boundary). Owner must correct registry publishing configuration, then rerun the release against the existing immutable bundle. Local 1.99.1 bundle is test evidence only. |
| 12 Custody | Agentbox W3b implementation in progress under the existing identity-port contract. | Owner-run browser-key ceremony, custody provisioning, host socket permissions, flag-on rehearsal and activation remain. No key export, key mint or running-container cutover performed. |
| 13 Pins/certificates | Ontomark `6c3f693` stages certificates inside the generation transaction; VG `fa7604d65` consumes nested certificates. `rdfs-certificate` 0.1.1, `ontomark-compile` 0.6.4 and CLI 0.7.3 published. VG `f7186a7d2` and Agentbox `ba4e8cc2a` pin the released CLI bytes. | Solid and VisionClaw registry adoption remain; deployment is separate. Certificate output outside the generation is now refused early. |
| 14 ADR hygiene | Owning repair notes retain historical anchors and qualify activation. Agentbox's retired precedent-bridge packaging fallback removed in `ba4e8cc2a`. Review skill lessons merged in `cb08c91a8` (131-skill lint passes, 29 warnings/33 existing suppressions). | Focused diagram successor topics and final audit receipts in progress. Owner decision authority is unchanged. |

## Evidence and review limits

Executed command logs and review receipts are retained under
[evidence/2026-11-03/external-fixes](evidence/2026-11-03/external-fixes/).
Counts are snapshots of specific test runs, not measures of estate completeness.

- Website: 227 script tests and the production build/prerender/discovery checks
  pass after installing locked dependencies in its isolated worktree. The final
  three-file follow-up review reports no new findings. Its aggregate export
  retains two earlier findings: the patch-line count defect was reproduced and
  repaired; the claimed generated-file integrity failure did not reproduce.
  ESLint ignored the selected script paths, so its exit code is not lint evidence.
- Dream-core: workspace tests, Clippy, formatting, rustdoc and the registry
  package dry run pass. Initial review findings concerning index flags, token
  coverage and planner redaction have regression coverage. Final follow-up
  review reported one new filename-parsing bug, reproduced and repaired before
  release; workspace gates passed again. The aggregate export carried the three
  historical findings with refreshed metadata. Its per-run comment records the
  actual new finding. The small final parser repair was checked locally rather
  than presented as covered by the earlier external receipt.
- Agentbox: 68 ontology tests and the serial operations suite pass. The real
  registry Dream consumer passes its full suite (including 11 whole-night
  tests), Clippy and rustdoc. Nix source parsing and release-byte checks pass;
  no image build was performed. The reviewer reports two MEDIUM and one BUG
  outside the repaired mechanisms: existing unauthenticated desktop listeners,
  missing general-dispatch signer-roster admission, and fallback governance
  persistence paths/overwritten case provenance. These are explicit follow-ups,
  not new regressions attributed to the small changed hunks.
- Forum: 1,120 initial native tests, worker Clippy and five-package WASM checks
  pass. Its external review timed out and is not a security PASS.
- Solid: final core 535, server library 121 (one ignored), Git 5, payment routes
  6, payment-read 1 and provenance 9 tests pass, with Clippy and rustdoc clean.
- Ontomark: compiler 512 and certificate 12 tests pass; VOWL: 156 native tests
  pass (one ignored) and five real-WASM tests pass. VG publisher 17 and KG
  transfer 11 tests pass. VG release adoption also passes seven migration and
  four notice-publication tests plus TypeScript and the production Vite build.
- VisionClaw: 1,809 library tests pass (11 ignored), XR 503 (two ignored), vault
  342, browser 5, release-evidence 3, and isolated Godot 556 (three renderer
  cases pending). Codescape: 299 pass (seven ignored); a watcher concurrency
  failure passed in isolation and in the serial suite. These results precede
  the new login-verifier finding and do not close it.

## Explicit open boundaries

The VisionClaw worker reported a pre-existing login verifier accepting a signed
Nostr event without the route's full request-bound, fresh, single-use contract.
Because returning saved API keys makes that authentication boundary material,
the branch is held pending its fix and review. The worker was then stopped by an
explicit hardware-key authentication restriction; it has not been retried or
rerouted through another model. Other review observations (timestamp handling,
relationship bounds and an unrelated spring constraint remapping path) remain
separate findings, not silently claimed repaired by the external-list changes.

LOOM-A-01 still lacks a canonical producer/verifier contract for the signed
DistillJobResult, lease, provider and revocation rules. No bespoke signed-byte
format was invented to declare that integration complete. The three spending
ledgers also represent distinct accounting domains; their existence does not
authorise merging balances without the settlement/reconciliation contract.

## Runtime observation

At 06:23:30 UTC the CPU package sensors reported 39/46 C and the GPUs 72/46/44 C.
The busy A6000 reported 82% utilisation; all GPUs reported thermal slowdown
inactive. This was a read-only thermal check, not deployment acceptance. No
runtime rebuild, HMD restart, live funds movement or custody cutover occurred.
The temporary Nix fetch/parse containers had no workspace bind mounts and were
removed after use; they were tooling checks, not project image rebuilds.
