---
title: Forum ADR ledger and sprint ADRs
date: 2026-11-02
status: assessed
type: explanation
---

# Forum ADR ledger and sprint ADRs

This report assesses 15 nostr-rust-forum decision records section by section at forum `c39f41f104b1`: twelve ledger ADRs under `docs/adr/` (2001, 2002, 2005, 2007, 2009, 2010, 2011, 2012, 2022, 2025, 2027, 2029) and the three sprint-resident ADRs 090, 091 and 092. Across 101 rows (86 sections plus one frontmatter row per record), 39 rows are implemented as written, 30 are partly implemented or carry stale facts and citations, 10 are contradicted, 5 are proposals that are correctly not yet built, and 17 cannot be checked from source. The most important finding is ADR-2027. It is recorded as accepted and partially implemented, and its Phase 3 notes describe shipped code. None of that code is on `main` at the pinned revision: commit `7096903` says it landed docs only, and the implementing branch is not an ancestor. ADR-2007 is contradicted by the code it governs: its exact `solid-pod-rs` pin and lockstep kit versions have both gone. ADR-2011 contradicts itself over whether an agent's declared tier stands when no task properties exist, and the code follows its Consequences rather than Decision 2.

## Findings

**F1: ADR-2027 describes code that is not on `main`.** *Trigger:* the acceptance record says Phase 2 "landed flagged off", and the Phase 3 notes describe `gift_wrap_rumor_with_signer` in core 1.0.0-beta.20, the `RECIPIENT_KEYS_ENABLED` flag, the `public_inbox` opt-in and a verbatim-consent test. *Evidence:* none of these strings occur under `crates/` at `c39f41f`. Core is `1.0.0-beta.19` (`crates/nostr-bbs-core/Cargo.toml:3`). The gift-wrap gate still checks the raw `p` key against the ingress list or the whitelist, with no inbox-key-to-owner resolution (`crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1140-1143`). Ancestor `7096903` states "DOCS ONLY: the implementing code … is on feat/upgrade-phase-0-1 and is NOT in this commit". That branch's commit `7df61ae` is not an ancestor of `c39f41f`. The consent text exists only in docs (`docs/deployment-profiles.md:31`). *Consequence:* `implementation_status: partial` and "Decision 2's precondition is met in code" are false for the revision consumers pin. A reader of `main` would believe the inbox-key relay change exists. *Close:* mark the record against the branch, or set `implementation_status` to `none` on `main` until `feat/upgrade-phase-0-1` merges, and restamp `verified_commit` on merge.

**F2: ADR-2007 is contradicted but still recorded as accepted, complete and live.** *Evidence:* `solid-pod-rs = { version = "0.5.0-alpha.18", … }` is a caret requirement, adopted under estate register ER-35 by an owner decision on 2026-10-18 (`Cargo.toml:181-185`). Internal crates no longer share one version: `1.0.0-beta.19`, `.16`, `.10` and `.15` (`Cargo.toml:188-194`). The stale "published to crates.io as 1.0.0-beta.3" comment remains (`Cargo.toml:187`). `superseded_by` is empty. *Consequence:* the ledger's Invariant ("exact `=` pin") is no longer the rule, and nothing in the ADR corpus records the replacement decision. `Cargo.lock` still fixes the resolved version (`solid-pod-rs 0.5.0-alpha.18`, registry). *Close:* record ER-35 as a superseding ADR, or amend ADR-2007 and its living-doc invariant, and fix the comment.

**F3: ADR-2011 contradicts itself on a declared tier without task properties.** *Evidence:* `(Some(d), None) => d` (`crates/nostr-bbs-core/src/governance.rs:527`). A request that declares a `risk_tier` while neither the panel nor the request declares properties keeps its own tier, even below the advertised default. Decision 2 says `risk_tier` "carries no authority alone". The third Consequences bullet ("a panel that declares nothing behaves as before except that its unlabelled requests fold to the advertised default") approves exactly what the code does. The code follows Consequences, and Decision 2 overstates. The Activation section carries the residual risk as an open MEDIUM from the deepsec gate. **Tension:** an agent publishing against an undeclared panel can self-tier `low` under an advertised `medium` default. *Close:* choose one reading. Either make that arm `d.max(advertised_default)` or narrow Decision 2's sentence; then add the counter-example to the effective-tier table test.

**F4: ADR-2005 is not amended for ADR-2022's exception.** *Evidence:* ADR-2022 admits a config-declared recipient without a whitelist row (`nip_handlers.rs:1136-1143`). That fires ADR-2005's `review_trigger` ("deliver gift-wraps to non-members") and falsifies its consequence "Non-members can never receive gift-wrapped DMs through this relay". ADR-2005's `superseded_by` and text are unchanged. Its three citations are stale: `:103-110` is now `:138-146`, `:510-518` is now `:1135-1153`, and `:58-64` is now `:58-76`. *Close:* add an amendment to ADR-2005 that names ADR-2022 as the explicit exception, and re-resolve the citations.

**F5: ADR-2002's "nothing retired" posture is stale for NIP-04/NIP-44.** *Evidence:* `nip44.rs:1-6` and `nip04.rs:1-7` (core) state that the hand-rolled implementations were "absorbed into" upstream `nostr`, and the modules delegate (`nip44.rs:14-18`). No canary verdict is recorded (`docs/BASELINE-architecture.md:161`). The decision's letter holds: no module file was deleted, and Schnorr remains in core on `k256` (`keys.rs:5`, `event.rs:9`). The context and consequences ("nostr-bbs-core remains the sole authority", "every crypto change … must still land in nostr-bbs-core") no longer describe NIP-04/44. The canary's documented command `-p nostr-upstream-canary` does not match the package `nostr-bbs-upstream-canary` (`crates/nostr-bbs-upstream-canary/src/lib.rs:14-15` against its `Cargo.toml:2`). CI only compiles the canary for wasm32 (`.github/workflows/ci.yml:187-188`) and runs its smokes natively (`:113-114`). *Close:* record the delegation as a partial Shape A (or a separate decision), keep the wasm32 run receipt the 2026-09-04 acceptance condition asks for, and fix the command. The Verification citation `Cargo.toml:34-37` is now `:57-60`.

**F6: ADR-2010's "Remaining" list is stale.** *Evidence:* the application stages it calls unimplemented now exist on the forum side. `ReceiptStage` carries `consumer-received`, `applied`, `not-applied` and `applied-manually` (`crates/nostr-bbs-core/src/governance.rs:938-1013`). `can_advance_stage` enforces monotonicity (`:1080-1100`). The auth worker accepts `POST /api/governance/receipts/{id}/application` (`crates/nostr-bbs-auth-worker/src/governance_api.rs:1109-1130`). The 2026-09-07 audit's gap ("accepts an absent request event reference") is closed in the projection SQL, which requires the receipt's `request_event_id` to equal the case's `nostr_event_id` (`crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs:186-193`). The unmarked-`e` fallback of 2026-10-02 is present (`receipts.rs:128-130`). *Close:* append an acceptance-progress section that names ADR-2011 Decision 5 as the implementer. Agentbox adoption of the contract was not traced.

**F7: ADR-2009 `.meta` sidecars are checked against the wrong ACL.** *Evidence:* only `.acl` paths reach `handle_acl_request` (`crates/nostr-bbs-pod-worker/src/lib.rs:118-120`, `:992-1004`), which uses the protected resource from `effective_acl_target` (`lib.rs:1648-1653`). A `.meta` path falls to the standard check (`lib.rs:1015-1016`). There `coerce_required_mode_for_acl` keeps only the mode (`acl.rs:85-90`, `.1`), and the ACL is resolved for the sidecar path itself. `find_effective_acl` probes the sequence from `acl_probe_sequence`: the path's own `{resource}.acl` and then enclosing container sidecars (`acl.rs:180-205,285-296`). So `/x.meta` is decided by `/x.meta.acl` or a container ACL, never `/x.acl`. **Open:** whether this over- or under-grants depends on how a per-resource ACL differs from the container default. Source alone does not settle it. *Close:* evaluate against `effective_acl_target(path, mode).0` in the standard path, or route `.meta` through the sidecar handler; add a test in which `/x.acl` and the container ACL disagree.

**F8: ADR-2025 Decision 2 holds in two readers, not in core.** *Evidence:* the client and zone-migrate readers map every failure to a fixed string (`crates/nostr-bbs-forum-client/src/zone_crypto/mod.rs:683,692`; `crates/nostr-bbs-zone-migrate/src/grants.rs:90,102`). Core does not:
- `SealVerification(e.to_string())` (`gift_wrap.rs:475`, `:729`) can carry `IdMismatch`, which prints the decrypted seal's declared `id`, an unconstrained string (`event.rs:19`).
- `AuthorMismatch` prints the decrypted rumor pubkey, a plain sender-chosen `String` (`event.rs:63`), beside the seal pubkey (`gift_wrap.rs:90`, `:498`).
- `InvalidKind` echoes decrypted kind numbers.

The seal pubkey's hex decode at `:478` is not a leak path: `verify_event_strict` at `:475` has already rejected a non-hex pubkey (`event.rs:213-221`).

The canary test injects only into the rumor's `created_at` (`gift_wrap.rs:1478-1496`). *Consequence:* sender-controlled decrypted metadata, not message content, can reach a log line. *Close:* map these to fixed strings and extend the canary to the seal `id` and the rumor `pubkey`.

**F9: ADR-2012 D4's pin facts are stale again.** *Evidence:* the forum pin is the caret `0.5.0-alpha.18` at `Cargo.toml:185`. It is neither `=0.5.0-alpha.7` at `:155` (Decision 4) nor `=0.5.0-alpha.10` at `:166` (the Disposition). The host, VisionClaw, pins `0.5.0-alpha.17` (VisionClaw `86cdd76`, `Cargo.toml:246`), and the host's `FsPaymentStore` no longer exists in VisionClaw source (it appears only in docs and VisionClaw ADR-2111). The Verification list still asks for a `sidestr/sign/` known answer that amended D5 struck. Amendment citations have drifted:

| Cited | Now at |
|---|---|
| deposit script check `pay_ledger.rs:558` | `:585-587` |
| unit refusal `:637` | `:643` |
| 410 `:810` | `:818` |
| known answers `deposit_address.rs:160` | `:200-225` |
| `crates/nostr-bbs-core/Cargo.toml:41` | `:49` (optional, behind `did`) |

The D1 view itself is not built: `read_balance` returns the stored number (`pay_ledger.rs:411-417`). *Close:* refresh D4 and the Verification list on reopening, as the Disposition already asks.

**F10: ADR-2022 Decision 2 says "renders"; the one consumer mirrors.** *Evidence:* dreamlab-ai-website `0e15728` holds `INGRESS_RECIPIENTS` as a literal (`forum-config/deploy/relay-worker.wrangler.toml:55`). It is kept equal by a test that renders with the kit's `ingress_relay_var` (`forum-config/tests/ingress_mirror.rs:46-56`). Nothing in this repository calls `ingress_relay_var` outside tests. **Drift**, low. The fail-closed parse (`crates/nostr-bbs-config/src/ingress.rs:147-164`) and the registry-wins merge (`crates/nostr-bbs-relay-worker/src/agent_disclosure.rs:68-91`) hold.

**F11: ADR-2001's citation discipline is not maintained.** Its consequence requires each change to update `file:line` and `verified_commit` "in the same change". The records assessed here carry drifted citations (F4, F5, F9, F12), and `docs/BASELINE-architecture.md:6` stamps only some sections at `0065dd9`. **Debt**: a ledger-wide citation re-resolution is the closing action.

**F12: ADR-2011 implementation-note line numbers have drifted; every symbol still exists.** **Drift** in `crates/nostr-bbs-forum-client/src/`:

| File | Symbol moves |
|---|---|
| `utils/governance_view.rs` | `compute_boundary` 225→227, `rationale_satisfied` 57→58, `decision_content` 77→78, `visible_probe` 292→315, `card_sections` 445→652, `is_decidable_by` 399→429, `relative_age_label` 304→327, `is_overdue` 328→351 |
| `pages/governance.rs` | `ActionCardData` 446→592, `card_body_parts` 480→639, `assemble_card` 652→823, `ActionRow` 695→867, `ReadOnlyActionRow` 1028→1252, `GovernancePage` 87→68, `DecisionChainRow` 1117→1378 |
| `stores/receipts.rs` | `parse_receipts` 62→134, `reduce_case` 106→199, `stage_label` 132→275 |
| `stores/panel_registry.rs` | `resolve_panel_for` 356→554 |

The relay citations have also moved:

| Cited | Now at |
|---|---|
| `nip42.rs:383` | `:416` |
| `nip_handlers.rs:4144` | `:5808` |
| `wrangler.toml:48-49` | `crates/nostr-bbs-relay-worker/wrangler.toml:61-62` |
| `handle_req` at `nip_handlers.rs:1288-1331` | `:1770ff`, gates still precede the insert |

**F13: cited security-gate receipts are not committed.** **Drift.** ADR-2011's Activation cites `.deepsec-gate/reports/20260914T211502Z/receipt.json` and a `.tmp/nrf-deepsec-head-341c5d2/…` receipt. Neither exists at `c39f41f`. The `.claude/evidence/*` receipts and `docs/estate-closeout/2026-09-05/*.json` it and ADR-2010/090–092 cite are committed.

**F14: the receipts API's `applied` flag drops back to false at `consumer-received`.** *Evidence:* `ReceiptStage::is_applied` is true for `projection-committed`, `applied` and `applied-manually` only (`crates/nostr-bbs-core/src/governance.rs:1034-1039`). The receipts API publishes it as `"applied"` (`crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs:844`). *Consequence:* a receipt reads `applied: true` at `projection-committed`, then `applied: false` once the mutation owner reports `consumer-received`, then `true` again at `applied`. The flag therefore conflates "projected" with "applied externally", the distinction ADR-2010 exists to keep, and it moves backwards as the ladder advances. The separate `isApplicationStage` field (`:849`) lets a careful reader recover the meaning. **Debt**. *Close:* split the flag into `projected` and `appliedExternally`, or define `applied` over the ladder rank; add a test that the flag never regresses along the ladder.

## ADR-2001 section assessment

Classification: operative decision (accepted, complete, live).

| Section | Disposition |
|---|---|
| Frontmatter (accepted/complete/live) | **Implemented** for the structure; the review trigger has fired many times since (31 ledger records now). |
| Context | **Cannot check from source**: historical description of the 2026-08 corpus. |
| Decision | **Implemented.** Living docs exist (`docs/BASELINE-architecture.md`, `docs/IDENTITY-keys-and-trust.md`). The archive holds 24 ADRs plus a README tombstone and the legacy index (`docs/archive/adr/README.md:1-25`). `PREAMBLE.md`, `TEMPLATE.md` and `scripts/adr-index-gen.js` exist. |
| Consequences | **Partly implemented.** Lookup order and tombstone mapping hold (`docs/archive/adr/README.md:15-25`), and sprint 090–092 keep full text with archived stubs. The "update `file:line` and `verified_commit` in the same change" cost is not being paid (F11). |
| Verification | **Cannot check from source**: a historical archive-cut record at `23be587`; `docs/adr/` has since grown well past "only TEMPLATE, PREAMBLE, this record and README". |
| Closeout extension — 2026-09-04 | **Partly implemented.** Lineage is resolvable. The acceptance condition (each operative claim bound to a scoped receipt; living docs amended with each record) is unmet for the records in F2, F3 and F9. The cross-repo links resolve (`docs/estate-review/forum-decisions.md` exists in VisionFlow). |

## ADR-2002 section assessment

Classification: operative decision (accepted, partial, staged).

| Section | Disposition |
|---|---|
| Frontmatter (accepted/partial/staged) | **Partly implemented**: no canary verdict recorded (`docs/BASELINE-architecture.md:161`), so the trigger has not fired; "partial" now also covers delegated NIP-04/44 (F5). |
| Context | **Partly implemented.** `nostr = "0.44.7"` with `nip04/44/59/98` (`Cargo.toml:85`). `nip04.rs` and `nip44.rs` are no longer hand-rolled (F5). `gift_wrap.rs`, `keys.rs`, `signer.rs` and `nip98.rs` remain core code. |
| Decision | **Partly implemented.** The unlinked canary exists (`Cargo.toml:57-60`, `crates/nostr-bbs-upstream-canary/Cargo.toml:13`) and no module file was deleted. Schnorr is still core-owned on `k256` (`keys.rs:5`). NIP-04/44 crypto was delegated upstream without a recorded verdict (F5). |
| Consequences | **Contradicted** in part: "every crypto change … must still land in `nostr-bbs-core`, not upstream" no longer holds for NIP-04/44 (`nip44.rs:1-18`). The duplicate-build cost holds. |
| Verification | **Partly implemented**: `Cargo.toml:34-37` is stale (now `:57-60`); `lib.rs:1-30` holds the PASS/FAIL rationale; the module list holds. |
| Closeout extension — 2026-09-04 | **Implemented** as an assessment: CI runs the native workspace tests and a wasm32 `cargo check` only (`.github/workflows/ci.yml:113-114,187-188`); no wasm32 execution receipt exists. |

## ADR-2005 section assessment

Classification: operative decision (accepted, complete, live), amended in effect by ADR-2022.

| Section | Disposition |
|---|---|
| Frontmatter (accepted/complete/live) | **Partly implemented**: the review trigger fired with ADR-2022 and was not acted on (F4). |
| Context | **Implemented**: kind 1059 is author-ephemeral (`nip_handlers.rs:47-49`). |
| Decision | **Partly implemented.** The recipient comes from the first `p` tag (`nip_handlers.rs:138-146`, `filter.rs:360-366`). Admission requires a whitelisted recipient **or** a config-declared ingress recipient (`:1135-1153`). Kind 1059 is absent from the ban-gated set (`:58-76`). The extraction is pure. |
| Consequences | **Contradicted** in one bullet: non-members declared under `[[ingress]]` do receive wraps (F4). The ban-gating bullet holds. |
| Verification | **Partly implemented**: all three citations are stale (F4); the claims they support hold. |
| Closeout extension — 2026-09-04 | **Implemented** for the read-visibility separation: `gift_wrap_readable_by` (`nip_handlers.rs:156-164`) and the REQ/COUNT 1059 gate (`:1835-1843`). The acceptance tests named there are **cannot check from source**. |

## ADR-2007 section assessment

Classification: operative decision (accepted, complete, live), contradicted by later code.

| Section | Disposition |
|---|---|
| Frontmatter (accepted/complete/live) | **Contradicted**: the pin and lockstep it records are gone (F2). |
| Context | **Cannot check from source**: rationale at `7795af5`; the pinned crate is now alpha.18. |
| Decision | **Contradicted**: caret `0.5.0-alpha.18` (`Cargo.toml:181-185`), internal versions beta.10 to beta.19 (`:188-194`). The 64 KiB ACL cap holds (`crates/nostr-bbs-pod-worker/src/acl.rs:106`). |
| Consequences | **Contradicted**: an uptake is now a `cargo update` within the caret range, not a manual edit (`Cargo.toml:182-184`). The reconciliation debt still stands (`:187`). |
| Verification | **Contradicted**: `Cargo.toml:155` is no longer the pin. `crates/nostr-bbs-core/Cargo.toml:3` reads `1.0.0-beta.19`. |
| Closeout extension — 2026-09-04 | **Contradicted**: it reasserts "exact =0.5.0-alpha.7 core-only pin". |

## ADR-2009 section assessment

Classification: operative decision (accepted, complete, live).

| Section | Disposition |
|---|---|
| Frontmatter (accepted/complete/live) | **Partly implemented**: `.meta` handling is incomplete (F7). |
| Context | **Implemented**: the rationale matches `acl.rs:69-84`. |
| Decision | **Partly implemented.** The mode funnels through the shared helper (`acl.rs:33-35,85-90`; upstream `solid-pod-rs` `9b6143c` `crates/solid-pod-rs/src/wac/mod.rs:217-256`). `.acl` reads and writes both elevate to Control on the protected resource (`lib.rs:1648-1660`). `build_delegation_acl` exists (`acl.rs:402`), and the 64 KiB cap is enforced (`acl.rs:106,112,138`). `.meta` is checked against the sidecar path's ACL, not the protected resource's (F7). |
| Consequences | **Implemented** for mode coercion and the cap divergence. |
| Verification | **Implemented**: `acl.rs:85-90`, `:27-35`, `:8-13` and `:14-17` still support their claims. |
| Closeout extension — 2026-09-04 | **Cannot check from source** for deployed-version equality. The path-normalisation and cache tests it asks for are not traced. |

## ADR-2010 section assessment

Classification: operative proposal (proposed, partial, staged).

| Section | Disposition |
|---|---|
| Frontmatter (proposed/partial/staged) | **Implemented** as a posture; the relay and the forum's application stages are built (F6). |
| Context | **Implemented**: OK is sent after `save_event` and projection follows (`nip_handlers.rs:1552-1553,1598-1640`). |
| Proposed decision | **Partly implemented**: stages, correlation set, atomic batch and idempotent replay exist (`receipts.rs:115-196,452-508,555-580`). The API's `applied` flag regresses along the ladder (F14). The UI requirements (rationale, conflicting in-flight actions, retry) were not traced and are left unassessed. |
| Consequences | **Cannot check from source**: an inter-team agreement. |
| Verification and current gaps | **Partly implemented**: historical; the gaps it lists are now partly closed (F6). |
| Closeout extension — 2026-09-04 | **Cannot check from source**: the acceptance journeys are runtime. |
| History consumer closeout extension — 2026-09-05 | **Cannot check from source**: it points to the VisionFlow decision-history review. |
| Disclosure closeout extension — 2026-09-05 | **Partly implemented**: explicit `Unavailable` and empty badge states exist (`crates/nostr-bbs-forum-client/src/stores/badges.rs:170,189,197-223`). The reported EOSE-on-disconnected-socket defect was not traced. |
| Acceptance progress — 2026-09-05 | **Partly implemented**: migration `0005_governance_receipts.sql` and the route (`crates/nostr-bbs-relay-worker/src/lib.rs:340`) exist, and `is_applied` is defined (`receipts.rs:395-400`). The "consumer-received … not implemented" item is stale (F6). Test counts were not re-run. |
| Estate audit — 2026-09-07 | **Implemented** as remediated: the absent-request and zero-row gaps are closed by `PROJECTION_DECISION_SQL` and the `changes() = 1` chain (`receipts.rs:183-196`, `:578`). |
| Execution qualification — 2026-09-07 | **Implemented**: guarded transitions and row-count dependence (`receipts.rs:183-196,555-580`). |
| Live signed journey and acceptance criterion — 2026-10-02 | **Partly implemented**: the unmarked-`e` fallback is present (`receipts.rs:128-130`). The live receipt and deploy runs are **cannot check from source**. |
| Disposition — 2026-10-02 | **Cannot check from source**: it depends on a live high-tier case. |

## ADR-2011 section assessment

Classification: operative decision (accepted, partial, live).

| Section | Disposition |
|---|---|
| Frontmatter (accepted/partial/live) | **Implemented** for `partial`: probe blindness is partial as stated. "Live" is **cannot check from source**. |
| Context | **Implemented**: NIP-11 defaults (`crates/nostr-bbs-relay-worker/wrangler.toml:61-62`; the cited `:48-49` is stale). |
| Decision | **Partly implemented.** Points 1, 3, 4, 5 and 6 are implemented: `TaskProperties` and a per-axis `max` over `Ord` enums declared loosest-first (`governance.rs:228-275,390,451-457`); floors (`:481-499`); `rationale_required` before storage (`nip_handlers.rs:1361-1377`, `governance.rs:710,764`); `system:` refusal (`nip_handlers.rs:195-212,505-518`); ladder and side receipts (`governance.rs:938-1013,1080`); `case_delegations` (`lib.rs:1047`, `nip_handlers.rs:2869,2904`); HMAC selection (`governance.rs:595-615`); probe tag honoured only from the probe agent (`nip_handlers.rs:401-403`) and match-nothing on `#probe` (`filter.rs:204-207`). Point 2's "no authority alone" sentence is not what the code does for a declared tier without properties, which follows the Consequences instead (F3). |
| Consequences | **Implemented**: absent `tp-*` parses to `None` (`governance.rs:463`), and the `probe` tag remains on the raw signed event. Role revocation lower-cases the pubkey, matches `lower(pubkey)` and returns 404 when no row changed (`crates/nostr-bbs-auth-worker/src/governance_api.rs:753-769`). |
| Implementation notes | **Partly implemented**: see the subsections. |
| Client (`nostr-bbs-forum-client`) | **Partly implemented**: every named symbol exists, `with_relay_tier` included (`utils/governance_view.rs:280`). Line numbers have drifted (F12). |
| Out-of-scope security fixes carried on this change | **Implemented**: `handle_req` gates precede the subscription insert (`nip_handlers.rs:1826-1863`), and `authorize_event` consults `protected_read_permitted` (`nip_handlers.rs:2262-2285`, `nip42.rs:85`). The cited lines are stale (F12). |
| Known limits (client) | **Implemented**: the receipts route is admin-only (`receipts.rs:869`) and no 31400 publisher exists in the client. |
| Verification | **Cannot check from source**: test runs and deepsec verdicts were not re-run. The named test modules exist (`nip_handlers.rs:5808`, `nip42.rs:416`). |
| Activation — 2026-10-02 | **Cannot check from source** for publication, deploy and M4 run claims. `CALIBRATION_SELECTION_KEY` unset still degrades to an unkeyed hash with a warning (`nip_handlers.rs:2655-2666,2755-2763`), as stated. Cited deepsec receipts are uncommitted (F13). |

## ADR-2012 section assessment

Classification: operative proposal (proposed, partial, inactive) with amended, built sub-decisions.

| Section | Disposition |
|---|---|
| Frontmatter (proposed/partial/inactive) | **Implemented** as a posture: D2, D3 and D6 amendments are built and D1 (the view) is not. "Inactive" is **cannot check from source** (the website compatibility record lists `ed67732` as integrated, which is a documentation claim). |
| Context | **Cannot check from source**: historical. `D1PaymentStore`, `credit_atomic` and the `payments.rs:148/161/183` lines no longer exist; `payments.rs:1-22` now describes the outpoint rule. |
| Decision | **Partly implemented.** D1 view: not implemented (proposed); `read_balance` reads a stored number (`pay_ledger.rs:411-417`). D1 amendment: implemented (`pay_credits` at `:110`, `SQL_APPLY_CREDIT` at `:168`, auth-worker `0004_pay_credits.sql`). D2 amendment: implemented (`debit` at `:434`; no `credit_atomic` anywhere). D3: implemented (410 at `:818`). D4: contradicted; the facts are stale and the pin is a caret range (F9). D5: `derive_subkey` is HMAC-SHA256 as stated (`crates/nostr-bbs-core/src/keys.rs:252-267`). D6: implemented (script check `:585-587`, lower-cased txid `:369`, `txo:<chain>:…` `:374-376`, unit refusal `:643`, known answers `deposit_address.rs:200-225`, guard `tests/pay_credit_guard.rs` with three tests); citations drifted (F9). |
| Consequences | **Partly implemented**: the 410s and evidence-only credit hold. `sidestr-node` dependence is not built (proposed). |
| Verification | **Partly implemented**: every named test exists (`pay_ledger/tests.rs:284,317,345,393,577,831`; `deposit_address.rs:200,208,225`). Ratification items remain proposed, and the `sidestr/sign/` item is inconsistent with amended D5 (F9). |
| Disposition — 2026-10-02 | **Partly implemented**: its pin restatement (`=0.5.0-alpha.10` at `:166`) is itself stale (F9). The parking rationale is **cannot check from source**. |

## ADR-2022 section assessment

Classification: operative decision (accepted, complete, live).

| Section | Disposition |
|---|---|
| Frontmatter (accepted/complete/live) | **Implemented** for completeness; "live" is **cannot check from source**. |
| Context | **Implemented**: the gate refuses a non-whitelisted recipient (`nip_handlers.rs:1145-1153`). |
| Decision | **Partly implemented.** These hold: the schema `ingress` field (`crates/nostr-bbs-config/src/schema.rs:79`) and roles (`ingress.rs:51-53`); `ingress_relay_var` (`:129`) and the fail-closed `parse_ingress_relay_var` (`:147-164`); the gift-wrap-only admission (`nip_handlers.rs:1135-1143`); `ingressRecipient` (`crates/nostr-bbs-relay-worker/src/whitelist.rs:188-190`); the registry-wins merge (`agent_disclosure.rs:68-91`). The deploy pipeline mirrors the value rather than rendering it (F10). |
| Consequences | **Implemented**: declaring grants no cohort (it is consulted only at `nip_handlers.rs:1141`). |
| Verification | **Cannot check from source**: test runs and the live `ingressRecipient` reading were not re-run. |

## ADR-2025 section assessment

Classification: operative decision (accepted, complete, live).

| Section | Disposition |
|---|---|
| Frontmatter (accepted/complete/live) | **Partly implemented**: see F8. |
| Context | **Cannot check from source**: a historical comparison with anagram. |
| Decision | **Partly implemented.** D1 is implemented: the wrap is verified first in all three readers (`gift_wrap.rs:452,712`; `zone_crypto/mod.rs:683`; `grants.rs:90`), with the two new variants at `gift_wrap.rs:83-84,549-550`. D2 is implemented for the client and zone-migrate readers but not core (F8). D3 is implemented: rumors parse as `UnsignedEvent` (`gift_wrap.rs:204-206`), and the foreign-sig test exists at `:1531`. D4 is implemented (`crates/nostr-bbs-forum-client/index.html:13-14`, `components/media_embed.rs:202`). |
| Consequences | **Partly implemented**: no `#[non_exhaustive]` on the enums, as stated. The "console lines no longer carry anything derived from plaintext" claim fails for core error text (F8). |
| Verification | **Partly implemented**: every named test exists (`gift_wrap.rs:1462,1478,1500,1515,1531,1548,1572`; `zone_crypto/mod.rs:926,937`; `grants.rs:316`). The counts were not re-run, and the canary covers only `created_at` (F8). |
| Close-out 2026-10-30 | **Implemented** for the source anchors (`gift_wrap.rs:83,549,1462`; `zone_crypto/mod.rs:926`). The live wasm fetch and deploy are **cannot check from source**. |

## ADR-2027 section assessment

Classification: operative decision (accepted, partial, inactive); implementation absent at the pinned revision.

| Section | Disposition |
|---|---|
| Frontmatter (accepted/partial/inactive) | **Contradicted** for `partial` on `main` (F1). |
| Context | **Implemented** where checkable: wraps are published only to the configured relay (`crates/nostr-bbs-forum-client/src/dm/mod.rs:481-497`), and the DM module publishes kind 10050 (`:389`). The assessment doc exists (`docs/interop/nostr-chat-interop-assessment-2026-10-20.md`). |
| Decision | **Not implemented (proposed)** on `main`: no foreign-relay routing, `public_inbox`, inbox keys or ratchet. D3 holds trivially: channels remain 40/41/42. |
| Consequences | **Not implemented (proposed).** |
| Verification | **Implemented** as a statement ("None: no implementation") for `main`. |
| Acceptance record (2026-10-20) | **Contradicted**: "Decision 2's precondition is met in code" is false at `c39f41f` (F1). The consent text appears only in `docs/deployment-profiles.md:31`. |
| Implementation notes recorded during Phase 3 (2026-10-20) | **Contradicted**: `gift_wrap_rumor_with_signer` and core `1.0.0-beta.20` do not exist at `c39f41f` (F1). |

## ADR-2029 section assessment

Classification: operative proposal (proposed, none, inactive).

| Section | Disposition |
|---|---|
| Frontmatter (proposed/none/inactive) | **Implemented**: no `CHANNEL_DIALECT` or `ChannelDialect` exists in `crates/`. |
| Context | **Implemented**, with line drift. Ban gate omits kind 9 (`nip_handlers.rs:69`). Zone gate (`:1414`). Counters (`:1558`, cited `:1556`). Zone read decision (`zone_read_permitted` at `:2114`, cited `:2354`). Kind-40 creator lookup (`:2549`). Sealed-only drift exemption (`:1718`; constant `:44`). "NIP-29 TODO" (`:1300`, cited `:1299`). `supported_nips` advertises 29 (`nip11.rs:207`). Protected-read constant (`nip42.rs:44`). Kind-42 profile stats (`profile_stats.rs:11,41`). First-`e`-tag channel (`crates/nostr-bbs-core/src/thread.rs:69-70`, `pages/message_jump.rs:66-70`). Unconditional delete-then-insert for addressable events (`storage.rs:97-140`), so a late older event or a same-second tie leaves two rows. Config validation via script (`ci.yml:118-124`) and a wasm `cargo check` without a trunk build (`ci.yml:187-188`). `groups.rs` has no callers. |
| Decision | **Not implemented (proposed).** |
| Consequences | **Not implemented (proposed).** |
| Verification | **Implemented** as a statement ("None: no implementation"). |
| Amendment (ADR-2030, proposed) | **Not implemented (proposed)**: no `nostr-bbs-groups` crate exists. |

## ADR-090 section assessment

Classification: operative decision (sprint-resident canonical text; archive stub at `docs/archive/adr/ADR-090-…`).

| Section | Disposition |
|---|---|
| Frontmatter (Status: Accepted) | **Implemented.** |
| Context | **Cannot check from source**: a historical bug list. |
| Decision | **Implemented.** Rule 1: `current_app_path` delegates to `paths::app_path` (`crates/nostr-bbs-forum-client/src/app.rs:82-84`), and raw `pathname.get()` callers go through `login_redirect_for`, which strips the base first (`app.rs:1453-1455,1534-1536`; `utils/paths.rs:274-285`). Rule 2: returnTo carries base-relative paths. Rule 3: the service worker is registered at an absolute URL with an explicit scope (`main.rs:56-57`). Rule 4: `safe_return_to` validates against the base (`app.rs:95-97`, `paths.rs:263-265`). |
| Consequences | **Implemented** for the single helper. The bug closures are **cannot check from source**. |
| Closeout extension — 2026-09-05 | **Implemented** as remediated by the following section: textual stripping is replaced by segment-boundary `strip_base` (`paths.rs:65`). |
| Acceptance progress — 2026-09-05 | **Implemented**: `MAX_RETURN_TO_LEN` 512 (`paths.rs:36`), four decode rounds (`:41`), 32 tests in `utils::paths`. Browser receipts are **cannot check from source**. |

## ADR-091 section assessment

Classification: operative decision (sprint-resident canonical text).

| Section | Disposition |
|---|---|
| Frontmatter (Status: Accepted) | **Implemented.** |
| Context | **Cannot check from source**: historical. |
| Decision | **Partly implemented.** There is no stored counter, and `count_for` derives from `channel_messages` (`crates/nostr-bbs-forum-client/src/stores/channels.rs:37-40,159-162`), returning `u32` rather than `usize`. No `total_messages()` exists on `ChannelStore`. The forums page reads `channel_messages` directly (`pages/forums.rs:111`). |
| Consequences | **Implemented**: `CachedData` no longer carries `message_counts` (`channels.rs:37-45`). |
| Closeout extension — 2026-09-05 | **Implemented** as remediated by the following section. |
| Acceptance progress — 2026-09-05 | **Implemented**: `retain_present` and `absent_from` (`utils/reconcile.rs:54,65`) are used by `pages/channel.rs:57,623,658`, with 13 tests. The browser deletion check is **cannot check from source** (not driven). |

## ADR-092 section assessment

Classification: operative decision (sprint-resident canonical text).

| Section | Disposition |
|---|---|
| Frontmatter (Status: Accepted) | **Implemented.** |
| Context | **Cannot check from source**: historical. |
| Decision | **Partly implemented.** `ensure_subscribed` was removed from `ChannelPage` (`pages/channel.rs:348`) in favour of observed bootstrap phases. It survives at `pages/message_jump.rs:177` and `pages/section.rs:230`. The reactive title fallback re-runs on `store.channels` and matches slugs by section (`pages/channel.rs:340-346`). |
| Consequences | **Cannot check from source**: a parity claim. |
| Closeout extension — 2026-09-05 | **Implemented** as remediated: the replay applies the tombstone check (`pages/channel.rs:479-487`). |
| Acceptance progress — 2026-09-05 | **Implemented**: `derive_phase` and `BootstrapFailure` (`utils/bootstrap.rs:54,67,143`), 16 tests, and the failure banner (`pages/channel.rs:1095`). The browser receipts are **cannot check from source**. |

## Verification

*Evidence class:* source tracing at pinned revisions only. Each claim was read with `git show <sha>:<path>` and `git grep -n … <sha>`:
- nostr-rust-forum `c39f41f104b1` (primary);
- solid-pod-rs `9b6143ce55b2` (the `effective_acl_target` helper; its workspace declares `0.5.0-alpha.18`, but identity with the crates.io artefact in the forum's `Cargo.lock` is not established);
- dreamlab-ai-website `0e1572801d03` (the ingress binding);
- VisionClaw `86cdd7635582` (the host `solid-pod-rs` pin);
- agentbox `c57d9c0508f4` (the `FsPaymentStore` search).

*Not verified:* no test, build, browser, D1, Worker or deployment was run. Test counts, deepsec verdicts, run IDs and live fetches quoted in the ADRs are recorded as **cannot check from source**. ADR-2027's absence of implementation rests on string search at `c39f41f`, on ancestor commit `7096903`'s message and on `7df61ae` not being an ancestor. The unmerged branch is context, not implementation, and was not assessed. Agentbox's adoption of ADR-2010's receipt contract, ADR-2010's UI requirements, ADR-2009's path-normalisation and cache tests, and the badge EOSE defect were not traced. Each is named in the row it limits.

*Generator:* no diagram topic was assigned to this worker, so `node scripts/diagram-index-gen.cjs … --strict-citations --render` was not applicable and not run. `scripts/adr-index-gen.js` was not run against the product tree. No product code was edited.

*Counts:* 15 records and 101 rows (86 sections plus 15 frontmatter rows), tallied by each row's leading disposition:

| Disposition | Rows |
|---|---|
| Implemented | 39 |
| Partly implemented / stale | 30 |
| Contradicted | 10 |
| Not implemented (proposed) | 5 |
| Cannot check from source | 17 |

There are 14 findings.
