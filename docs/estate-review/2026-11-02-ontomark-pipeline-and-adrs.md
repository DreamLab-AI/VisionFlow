---
title: Ontomark pipeline topics and ontomark and website ADRs
date: 2026-11-02
status: assessed
type: explanation
---

# Ontomark pipeline topics and ontomark and website ADRs

All 49 panels of ontomark topics OM-00 to OM-08 were assessed at ontomark `4b9d15df4805`. 22 are accurate, 19 were corrected, 8 are accurate with limits, and none is left inaccurate. All nine topics were restamped and pass the strict gate. Thirteen ADRs were assessed section by section: five ontomark and eight dreamlab-ai-website. **The most important finding is that the shared layout checks (ADR-OM-011) changed behaviour that four topics still described as open.** Journals can no longer load twice, repository discovery is canonical, `migrate plan` refuses an absent role, and `LayoutError` exits 2. In the ADRs, the most important finding is that website ADR-2007's "single encrypted zone" is contradicted by configuration that encrypts three zones. Three panels were wrong even before this window: OM-01.6 and OM-00.2 called `Vault::get` and `get_mut` dead code although the guarded writes use them; OM-04.1 said three artefacts were missing from C3 although C3 lists them; and OM-00.5 said agentbox ADR-2136 still named `--vault all`, which contradicted OM-06.5.

## Findings

### Topic findings

| Finding | Trigger and consequence | Evidence (ontomark `4b9d15df4805` unless noted) | What closes it |
| --- | --- | --- | --- |
| OMP-01: layout fixes closed tensions the topics still described as open | ontomark commits after `560e50c` added `Layout::validate`, merged the built-in exclusions with the profile's, rejected a symlinked journal root and canonicalised vocabulary discovery. OM-01.1, OM-01.2, OM-05.4 and OM-05.2/05.3 still described the earlier faults, so a reader would have thought journals could double-load and `--repo .` could miss parents. | `crates/ontomark-compile/src/input/page.rs:485-491`, `:525`; `crates/ontomark-compile/src/profile.rs:530-538`; `crates/ontomark-compile/src/profile/layout.rs:190-212`; `crates/ontomark-cli/src/main.rs:460-463`, `:654`; `crates/ontomark-compile/src/build/mod.rs:559`; `crates/ontomark-compile/src/input/vocabulary.rs:507-523`; `crates/ontomark-cli/src/migrate_cmd.rs:144-146` | **Closed** by this restamp: the panels were corrected. |
| OMP-02: `load_with_walk` trusts the caller's walk (**Tension**) | `load_excluding` now merges `EXCLUDED_DIRS`, but the public `Vault::load_with_walk` takes any `PageWalk`. A library caller that walks without excluding `journals/` would still load journals twice. Every production caller passes `Layout::excluded_dirs`. | `crates/ontomark-compile/src/input/page.rs:506`, `:535`; `crates/ontomark-compile/src/build/mod.rs:569-583` | Validate the walk's exclusions inside `load_with_walk`, or make it crate-private. |
| OMP-03: the lexical `Layout::kind_of` has no production caller (**Debt**) | It is documented as a legacy helper that cannot tell custom folders apart. The build uses `kind_of_at`. The function survives only in doc examples and tests. | `crates/ontomark-compile/src/profile.rs:593-596`; `crates/ontomark-compile/src/build/mod.rs:499-503`, `:579-581` | Deprecate or remove it. |
| OMP-04: the crate doc's exit-code list omits `LayoutError` (**Drift**) | `main` maps `LayoutError` (escaping or overlapping roles) to exit 2, but the contractual exit-code paragraph does not name it. | `crates/ontomark-cli/src/main.rs:31-42`, `:460-463` | Add the layout refusals to the crate doc. |
| OMP-05: outline-form "Current Landscape" now yields an empty prose entry with no error (**Invariant** recorded in OM-04.5) | The landscape regex no longer accepts the `- ### Current Landscape` bullet. The safeguard is upstream: visionGraph's publish now fails on body residue. A page in outline form that reaches `prose_index` by another route loses its `cl` entry silently. | `crates/ontomark-compile/src/build/indexes.rs:173-185`, `:269`; visionGraph `556a0156b5c3` `.github/workflows/publish.yml:244-254` | Accepted as designed. A count of pages with a landscape heading but no `cl` would make the loss visible. |
| OMP-06: dead-code claim was wrong before this window | OM-01.6 and OM-00.2 listed `Vault::get` and `get_mut` as having no production caller. `write/create.rs` and `write/apply.rs` use them, and both callers predate `560e50c` (`5b32ba0`). | `crates/ontomark-compile/src/write/create.rs:252`; `crates/ontomark-compile/src/write/apply.rs:368` | **Closed** by correction. |
| OMP-07: OM-04.1 contradicted C3 and OM-03.7 | OM-04.1 said `_domain-index.json`, `context/v1.jsonld` and `data/.generation.json` were not in C3. C3 lists all three, and OM-03.7 already said so. | VisionFlow `5f8b1da` `docs/engineering/sovereign-corpus-contracts.md:123`, `:126`, `:153` | **Closed** by correction. |
| OMP-08: OM-00.5 contradicted OM-06.5 on ADR-2136 | OM-00.5 said agentbox ADR-2136 still named the `--vault all` machine bundle. ADR-2136 names `build --scope machine`. The only stale wording left is ontomark's own provisional recipe. | agentbox `c57d9c0508f4` `docs/adr/ADR-2136-pin-the-shared-ontology-toolchain.md:84`; `docs/DEPRECATION.md:470` | OM-00.5 is **closed** by correction. The `DEPRECATION.md:470` recipe remains **Drift**. |
| OMP-09: visionGraph moved to ontomark v0.7.0 and made body residue fatal | OM-05.5 showed v0.6.0 and a warning-only residue step. | visionGraph `556a0156b5c3` `.github/workflows/publish.yml:49`, `:248-254` | **Closed** by correction. |
| OMP-10: external-source acceptance extended the ingest contract | `accept` now takes an optional source. On that path it refuses a mismatched or re-versioned source and adds `source`/`counts`. The engine maps the unreachable `SourceVersion` error to a catch-all `ingest_failed`. The note-only sequence in OM-06.4 is unchanged. | `crates/ontomark-ingest/src/accept.rs:307-376`, `:659-674`; `crates/ontomark-ingest/src/error.rs:52`; `crates/ontomark-wasm/src/engine.rs:808-810` | **Closed** by correction of the OM-06.4 citations. |

No `sources:` entry in OM-00 to OM-08 names a generated or uncommitted file. Every source resolves at the declared revisions.

### ADR findings

| Finding | Trigger and consequence | Evidence | What closes it |
| --- | --- | --- | --- |
| ADR-F1: website ADR-2007's encryption decision is contradicted | The Decision, Consequences, Verification and title state that exactly one zone (zone3) is encrypted. At the pin, zones 2, 3 and 4 carry `encrypted = true`, the master gate is on, and the deploy mirror agrees. The ADR says a change to `encrypted` re-opens it, but nothing amended it. | dreamlab-ai-website `0e1572801d03` `forum-config/dreamlab.toml:33-37`, `:123`, `:137`, `:148`; `.github/workflows/deploy.yml:105-107`; `docs/IDENTITY-zones.md:76-77` against `:180` | A website ADR that supersedes the encryption clause, and a corrected Invariant 3 in `IDENTITY-zones.md`. |
| ADR-F2: website ADR-2004 states stale pin values | The lockstep rule is implemented and enforced by `scripts/lib/pin-parity.mjs`. The Decision literal (`1.0.0-beta.9`, `a7544687…`) fails a grep: the crates are `=1.0.0-beta.16` and `KIT_REF` is `9c36b60…` at three sites. | `forum-config/Cargo.toml:50-53`; `.github/workflows/deploy.yml:130`; `.github/workflows/workers-deploy.yml:44`; `.github/workflows/rust-ci.yml:20` | State the rule and point to `docs/architecture/kit-compatibility-record.md` for the values. |
| ADR-F3: `IDENTITY-zones.md` was edited without a restamp | This breaks ADR-2001's rule that `verified_commit` is re-recorded with each change. | `docs/IDENTITY-zones.md:6`, `:76-81` | Restamp after re-verification. |
| ADR-F4: line-number drift in all eight website ADRs | The bodies cite `dc06748`/`d852f61` lines in `deploy.yml`, `nostr.ts`, `AIChatFab.tsx` and `dreamlab.toml` that have since moved (current lines are in the tables below). The text is scoped "At `dc06748`", so this is drift, not falsehood. | Tables below | A restamp pass. |
| ADR-F5: minor website annex drift | ADR-2002 says "dreamlab-ai.com-only CORS", but relay and search allow three origins. ADR-2003's browser receipt predates the current kit. ADR-2008 omits the monitor co-recipient. | `forum-config/deploy/relay-worker.wrangler.toml:31`; `src/components/AIChatFab.tsx:38`, `:351` | Annex notes. |
| ADR-F6: ontomark ADR-OM-001 D6 `Signer` is undelivered | No `trait Signer` exists in `crates/` or `packages/`. `did:nostr` appears only as roster attribution. | ontomark `crates/ontomark-compile/src/build/attribution.rs:3`, `:14` | Implement the trait, or mark D6 deferred. |
| ADR-F7: ADR-OM-001 names APIs that differ from the code | The CLI package is `ontomark`, not `ontomark-cli`. The ADR's `IriClaimed` is `IriTaken`, beside `EmptySlug` and `InvalidIri`. No named "phantom parents" or "first-claimant" adapters exist; only an explicit compatibility mode does. `docs/api/WP-C.md` has no numbered divergence list. | `crates/ontomark-cli/Cargo.toml:2`; `crates/ontomark-model/src/mint.rs:15-44`; `crates/ontomark-compile/src/build/mod.rs:682` | A status note that corrects the names and records where the adapters went. |
| ADR-F8: ADR-OM-003 §2/§2a claims minting | Ingest never calls `mint`, and the crate says nothing is minted. The ADR's "only parse, model, reason" dependency set has also changed: there is no `ontomark-reason`, and `strsim`, `unicode-normalization` and `unicode-general-category` were added. | `crates/ontomark-ingest/src/judge.rs:22`, `:646`; `crates/ontomark-ingest/Cargo.toml` `[dependencies]` | Amend §1, §2 and §2a. |
| ADR-F9: ADR-OM-006 D4's export trigger has fired | D4 says nothing is exported, and that any export will use `oa:` selectors. `ontomark ingest` now writes `evidence.json` with quotes and plain `{start,end}` spans, and no `oa:` selector exists. | `crates/ontomark-cli/src/ingest_cmd.rs:822-837`; `crates/ontomark-ingest/src/accept.rs:30-55` | Emit Web Annotation selectors, or amend D4 in ADR-OM-006 or ADR-OM-010. |
| ADR-F10: ADR-OM-011's open `public_gate` is confirmed | `FrontmatterGrammar.public_gate` is declared and defaulted but consumed nowhere. | `crates/ontomark-compile/src/profile.rs:386`, `:402` | Consume it or remove it, under its own policy decision. |

## ADR-OM-001 section assessment

Classification: **historical** sprint record, amended by two continuation sections. Its `implementation_status: none / inactive` describes the record, not the code.

| Section | Disposition |
| --- | --- |
| ADR-OM-001: Decisions taken in the first ontomark sprint | Header. The amendments to eight estate ADRs and the census baseline `de33ca538` cannot be checked from ontomark source. |
| Context | Narrative. Nothing to check in code. |
| Decisions | **Implemented with gaps.** D1: the seven crates exist, but the CLI package is `ontomark` (`crates/ontomark-cli/Cargo.toml:2`; ADR-F7). D2: `Cargo.toml:23` dual licence and `NOTICE:2-9`. D3: `crates/whelk-el/src/lib.rs:1-8`, Whelk licences, rayon optional (`crates/whelk-el/Cargo.toml:17`, `:24`). D5: `profiles/estate`. **D6 contradicted: no `Signer` trait** (ADR-F6). D7: reversible migrate (`crates/ontomark-cli/src/migrate_cmd.rs:192-198`). D8: `docs/DEPRECATION.md` exists. D4, D9 and D10 cannot be checked from source. |
| Decisions taken during the sprint | **Mostly implemented.** D11: `Resolution::Unresolved` (`crates/ontomark-model/src/model.rs:60-72`), but the named adapters are absent (ADR-F7). D12: `mint_iri` refuses `TitleTaken`/`NoRule` (`crates/ontomark-model/src/mint.rs:129-145`), with naming drift. D13/D20: profile-default redaction (`crates/ontomark-compile/src/build/mod.rs:82-84`), parity-only switch (`crates/ontomark-cli/src/main.rs:443-450`), marker (`crates/ontomark-compile/src/build/redact.rs:60`). D14: `ontomark-ngg1` has no dependencies and is re-exported (`crates/ontomark-compile/src/lib.rs:69`). D15 holds in visionGraph `556a0156b5c3` (`.github/workflows/publish.yml:49-51`, `:107-112`). D21: no divergence list in `docs/api/WP-C.md` (ADR-F7). |
| Facts established by the census (2026-10-11) | Historical snapshot of other repositories. Cannot be checked from ontomark. |
| Consequences | Partly checkable. `docs/DEPRECATION.md:67` is the §2.1 CLI compatibility contract. The golden tag and worktree discipline cannot be checked from source. |
| Open | Open items. No mechanism claim. |
| Shared graph and plugin distribution continuation | Implemented. Defers to ADR-OM-002. `docs/UPSTREAMS.md` exists. |
| Compiler and nightly continuation | Implemented. `model-projection` is opt-in (`crates/ontomark-compile/Cargo.toml:20`, `:97-98`). `dream.config.json` has `autoMerge: false` and its gates (`dream.config.json:30-45`, `:63`, `:68`). Crate versions and the yank cannot be checked from source. |

## ADR-OM-002 section assessment

Classification: **operative**, complete and live.

| Section | Disposition |
| --- | --- |
| ADR-OM-002: shared graph code and separate Obsidian distribution | Header. "0.1.0 shipped" is historical: `packages/graph/package.json:2-3` is 0.2.0. |
| Context | Narrative. |
| Decision | **Implemented within ontomark.** `packages/graph/src/index.js` exports `validateGraph`, `seedPositions`, `neighbourhood`, `mountGraph` and `hierarchyGraph` (`:7`, `:18`, `:28`, `:38`, `:117`). Canvas 2D (`:47`) with a system font (`:71`, `:80`). There is no fetch and no second codec in `packages/graph/src`. The plugin's use was read at ontomark-obsidian HEAD `2a6c44e`, which is outside the pin set. The visionGraph publisher was not traced. |
| Verification and consequences | Cannot be checked from source: CI runs, browser acceptance and live receipts. |
| Status note, 31 October 2026 | Owner status record. Matches the frontmatter (`complete`, `live`). |

## ADR-OM-003 section assessment

Classification: **operative**. Accepted, partial (engine scope), staged.

| Section | Disposition |
| --- | --- |
| ADR-OM-003: native, host-driven assisted ingest | Header. "0.1.0 published" is historical: `crates/ontomark-ingest/Cargo.toml:4` is 0.2.0. |
| Context | Narrative. |
| Decision | Container heading. |
| 1. A pure, two-phase core in a new crate | **Implemented with drift.** `plan`/`accept` are at `crates/ontomark-ingest/src/lib.rs:220`, `:233`. No `std::fs`, `std::net`, `std::process` or `std::thread` in `src`. Prompts are data (`crates/ontomark-ingest/src/contract.rs:16`). UTF-16 spans are at `crates/ontomark-ingest/src/accept.rs:27-35`. The dependency set differs from the ADR (ADR-F8). `plan_source`/`accept_source` were added under ADR-OM-010. |
| 2. First slice: facts only, against the existing vocabulary | Facts-only scope holds. **Contradicted** on identity minting (ADR-F8). |
| 2a. Extraction contract: names in, grounded identifiers out | Grounding is implemented: ungrounded names are withheld (`crates/ontomark-ingest/src/accept.rs:93-144`). "Mints … through `mint`" is **contradicted** (ADR-F8). |
| 3. What is carried across from OntoCast | Implemented: `strsim` plus in-crate Jaccard and sequence ratio (`crates/ontomark-ingest/src/similarity.rs:70`, `:90`, `:143`). |
| 4. Confidence and review | Implemented: `CONFIDENCE = 0.5` and `METHOD = "inferred"` (`crates/ontomark-ingest/src/accept.rs:20-25`). Apply and undo belong to the host. |
| 5. The offline contract is amended, not bypassed | Plugin requirement. Cannot be checked from ontomark. The core makes no model call. |
| 6. Ownership | Implemented for the core and wasm layers (`crates/ontomark-wasm/src/engine.rs:159`, `:733`, `:753`). |
| 7. Reference reading and licence cautions | Reference list. |
| Consequences | OntoCast attribution at `THIRD-PARTY-NOTICES.md:9` and `docs/UPSTREAMS.md:13`. The plugin repin is outside the pin set. |
| Deviations recorded during implementation | In-crate Jaccard and a `unicode-general-category` dependency confirmed. Score rounding was not traced line by line. |
| Hardening after independent review | `RequestMismatch` wire code (`crates/ontomark-wasm/src/engine.rs:805`) and the withheld reasons (`crates/ontomark-ingest/src/accept.rs:93-144`) are present. Items 4, 5 and 7 were checked by symbol, not by behaviour. |
| Verification required before activation | Cannot be checked from source: runtime and test receipts. |
| Open questions | Open. |
| Engine boundary (12 October 2026) | Implemented: `IngestPlan`/`IngestAccept` (`crates/ontomark-wasm/src/engine.rs:159-181`) and the wire codes (`:799-813`). A catch-all `ingest_failed` (`:810`) is newer than the ADR's list. |
| Close-out 2026-10-30 | Holds: `engine.rs:168-179` still falls inside the request declarations. crates.io state cannot be checked from source. |
| Acceptance, 31 October 2026 | Status record. Matches the frontmatter. |

## ADR-OM-006 section assessment

Classification: **operative**. Accepted, live, partial. `verified_paths: []` leaves its staleness gate unarmed.

| Section | Disposition |
| --- | --- |
| ADR-OM-006: what Ontomark takes from Dot Triples, and the bridge between them | Header. |
| Context | Narrative about external projects. |
| Decisions | D1 implemented in ontomark-obsidian HEAD, which is outside the pin set. D2 implemented: `bridges/dot-triples/dot-triples-to-ontomark.rq` (CONSTRUCT at `:36`) and parity test `crates/ontomark-wasm/tests/bridge_dot_triples.rs:123`. D3 implemented (`.rq:13-18`, `:54-58`). **D4 contradicted** by the ingest export (ADR-F9). D5 is stated in the plugin README, outside the pin set. |
| Attribution | Implemented: `THIRD-PARTY-NOTICES.md:11-13` and `docs/UPSTREAMS.md:374-376`. |
| Consequences | Holds as far as source shows. The bridge is opt-in, and its test pins the vocabulary. The shared Oxigraph store is a runtime claim. |

## ADR-OM-011 section assessment

Classification: **operative**. Accepted, complete, staged. Code under `crates/` is unchanged between its `verified_commit` `395c18b` and the pin.

| Section | Disposition |
| --- | --- |
| ADR-OM-011 — Shared filesystem layout checks precede role selection | Header. |
| Context | Narrative. Each defect is closed under Decision. |
| Decision | **Implemented.** `resolve_roots` (`crates/ontomark-compile/src/profile/layout.rs:140-165`) refuses invalid, escaping and overlapping roots. `validate` is at `:190-192` and `kind_of_at` at `:199-212`. Callers: `crates/ontomark-cli/src/main.rs:654`, `crates/ontomark-compile/src/build/mod.rs:559` and `:579-581`, `crates/ontomark-compile/src/migrate/plan.rs:462-471`. Exit 2 at `crates/ontomark-cli/src/main.rs:460-463`. Additive exclusions at `crates/ontomark-compile/src/profile.rs:530-538` and `crates/ontomark-compile/src/input/page.rs:485-491`. Linked-root refusal at `input/page.rs:445-455`, `:525`. Rollback refuses `..` (`crates/ontomark-cli/src/migrate_cmd.rs:193-198`). Canonical discovery at `crates/ontomark-compile/src/input/vocabulary.rs:507-523`. Adoption by VisionClaw cannot be checked here. |
| Consequences | Implemented where checkable. A `.` root is valid. The non-sandbox caveat is documented (`layout.rs:185`). Required-key enforcement is unchanged (`profile.rs:367-369`). |
| Verification | Test files are present (`crates/ontomark-cli/tests/layout_contract.rs`, `layout.rs:215-271`). Not run. |
| Journal-root follow-up — 2026-11-01 | Implemented: the `journal_boundary_*` tests are in `crates/ontomark-compile/tests/migrate.rs`. The open `public_gate` item is confirmed (ADR-F10). |

## Website ADR-2001 section assessment

Classification: **operative** process decision.

| Section | Disposition |
| --- | --- |
| ADR-2001 — Consolidate the ADR corpus into living ground-truth docs | Title. |
| Context | Historical. Raw-hex Schnorr auth still holds (`src/lib/nostr.ts:22`, `:31`). The hand-sync comment has moved to `.github/workflows/deploy.yml:64`. Implements, with stale line numbers. |
| Decision | Implements: `docs/BASELINE-architecture.md`, `docs/IDENTITY-zones.md`, `docs/adr/TEMPLATE.md`, `scripts/adr-index-gen.cjs`, and the legacy records under `docs/archive/adr/`. |
| Consequences | **Partly contradicted:** `docs/IDENTITY-zones.md:6` was not restamped after edits (ADR-F3). |
| Verification | Dated record. Cannot be checked as written. |
| Closeout extension — 2026-09-04 | Dated. The index now lists 11 records. The archive disposition map is still open. |

## Website ADR-2002 section assessment

Classification: **operative**.

| Section | Disposition |
| --- | --- |
| ADR-2002 — Serve the origin from GitHub Pages and the backend from Cloudflare Workers | Title. |
| Context | Rationale. Cannot be checked from source. |
| Decision | Implements: Pages publish (`.github/workflows/deploy.yml:405-413`, `CNAME:1`), workers (`.github/workflows/workers-deploy.yml:120-132`), gated Cloudflare Pages (`deploy.yml:442`), `__ENV__` injection (`deploy.yml:288`, `:360`). Cited lines are stale (ADR-F4). |
| Consequences | Implements in source: the 404 shim (`deploy.yml:367-393`), the mirror publish (`deploy.yml:428-438`) and scoped CORS (`forum-config/deploy/relay-worker.wrangler.toml:30-31`). CORS behaviour is runtime. |
| Verification | Holds; lines have moved. |
| Closeout extension — 2026-09-04 | Consistent. |
| Acceptance progress — 2026-09-05 | Implements: the effective-endpoints gate (`.github/workflows/test-and-lint.yml:78`, `:171-176`) and deploys that require it (`deploy.yml:149`). Browser receipts are runtime and were not re-checked. Minor CORS drift (ADR-F5). |

## Website ADR-2003 section assessment

Classification: **operative**.

| Section | Disposition |
| --- | --- |
| ADR-2003 — Ship three polyglot frontends merged under one origin | Title. |
| Decision | Implements: three builds into one `dist/` (`.github/workflows/deploy.yml:177`, `:244`, `:335`) and two `__ENV__` dialects (`:288`, `:360`). Cited lines are stale. |
| Consequences | Implements. The wasm-bindgen pin is now derived from the kit lockfile (`.github/workflows/workers-deploy.yml:195-205`), so the mechanism has changed. |
| Verification | Holds; lines have moved. |
| Closeout extension — 2026-09-04 | Consistent. |
| Acceptance progress — 2026-09-05 | Dated runtime evidence at kit `a7544687`. The kit is now `9c36b60` (`deploy.yml:130`). |

## Website ADR-2004 section assessment

Classification: **operative**.

| Section | Disposition |
| --- | --- |
| ADR-2004 — Pin the upstream kit by crates.io version and git SHA moved in lockstep | Title. |
| Context | Consistent (`.github/workflows/workers-deploy.yml:12-18`). |
| Decision | The rule is implemented. **The literal values are stale** (ADR-F2). |
| Consequences | The stale comments it names were fixed. The multi-file bump still holds. |
| Verification | **Fails as written**: a grep for `1.0.0-beta.9` finds nothing. |
| Closeout extension — 2026-09-04 | Dated. Closed by the next section. |
| Acceptance progress — 2026-09-05 | Implements: exact pins (`forum-config/Cargo.toml:50-53`), lockfile parity (`scripts/lib/pin-parity.mjs:86-103`), wired at `.github/workflows/ci.yml:225` and `test-and-lint.yml:73`. |

## Website ADR-2005 section assessment

Classification: **operative** (`implementation_status: partial`).

| Section | Disposition |
| --- | --- |
| ADR-2005 — Accept hand-synced config mirrors; defer the single-source generator | Title. |
| Context | Consistent: there is no generator in the tree. |
| Decision | Implements: hand-sync comment and values at `.github/workflows/deploy.yml:64-66`, `:73-74`, `:107`. |
| Consequences | Holds, and is now machine-checked. |
| Verification | Holds; lines have moved. |
| Closeout extension — 2026-09-04 | Consistent. |
| Acceptance progress — 2026-09-05 | Implements: `scripts/lib/config-mirrors.mjs` enumerates the mirrors, wired at `.github/workflows/ci.yml:251-261`. The mirror count has grown beyond seven. |

## Website ADR-2006 section assessment

Classification: **operative**.

| Section | Disposition |
| --- | --- |
| ADR-2006 — Identify principals by raw-hex Schnorr pubkey over NIP-42; defer DID/Multikey to the kit | Title. |
| Context | Consistent. |
| Decision | Implements: `KIND_NIP42_AUTH = 22242` (`src/lib/nostr.ts:31`). No NIP-98 or Multikey. `did:nostr` appears only in marketing prose. Signing sites have moved. |
| Consequences | Holds. |
| Verification | Holds. |
| Closeout extension — 2026-09-04 | Consistent. |
| Acceptance progress — 2026-09-05 | Implements: NIP-07 is read-only (`src/components/AIChatFab.tsx:227-237`), and hints are labelled unverified (`src/lib/chat-turns.ts:34-38`). |

## Website ADR-2007 section assessment

Classification: **operative** (accepted, live, not superseded). **Its encryption decision is contradicted** (ADR-F1).

| Section | Disposition |
| --- | --- |
| ADR-2007 — Gate access with four zones on dual-accept cohorts and a single encrypted zone | **Title contradicted**: three zones are encrypted. |
| Context | Consistent (`forum-config/dreamlab.toml:115-119`). |
| Decision | Four zones and dual-accept arms are implemented (`forum-config/dreamlab.toml:101-151`). "Exactly one zone encrypted" is **contradicted** (`:123`, `:137`, `:148`). |
| Consequences | **Contradicted**: the flags were changed without re-opening the ADR. |
| Verification | **Fails at the pin.** |
| Closeout extension — 2026-09-04 | Consistent. |
| Acceptance progress — 2026-09-05 | The four-site comparison is implemented (`scripts/lib/config-mirrors.mjs:257`, `:262`). It checks that the sites agree, not what the ADR decided, so all four sites changed together and the check passed. |

## Website ADR-2008 section assessment

Classification: **operative**.

| Section | Disposition |
| --- | --- |
| ADR-2008 — Route Talk-to-AI over Nostr gift-wrap and read replies from open relays | Title. |
| Context | Consistent (`src/components/AIChatFab.tsx:41-44`). |
| Decision | Implements: the gift-wrap to the Jarvis key, the reply relays and the sender pin (`AIChatFab.tsx:32-33`, `:45`, `:349`). The monitor copy is unrecorded (ADR-F5). |
| Consequences | Holds. The agent fan-out cannot be checked from this repository. |
| Verification | Holds; lines have moved. |
| Closeout extension — 2026-09-04 | Closed by the next section. |
| Acceptance progress — 2026-09-05 | Implements: `[req:]` correlation (`src/lib/chat-turns.ts:90`, `:115-130`) and routing (`AIChatFab.tsx:270-275`). |

## Panel dispositions

| Panel | Disposition | Evidence |
| --- | --- | --- |
| OM-00.1 | accurate | Every duplicated-rule row re-read. `EXCLUDED_DIRS` (`input/page.rs:59`) and VisionClaw `FURNITURE_DIRS` (`vault-core/src/page.rs:50` at `86cdd76`) still match. |
| OM-00.2 | corrected | `Vault::get`/`get_mut` are used by `write/create.rs:252` and `write/apply.rs:368` (OMP-06). The crate-doc range became `main.rs:29-42`. |
| OM-00.3 | accurate | VisionClaw `vault` origins re-read at `86cdd76`. The reconcile targets hold at the pin. |
| OM-00.4 | accurate-with-limits | Step order matches `build/mod.rs:550-1073`. The parity receipt (`docs/HANDOFF.md:27-29`) is a recorded run that I did not repeat. |
| OM-00.5 | corrected | OPEN4: ADR-2136 now names `build --scope machine` (agentbox `c57d9c0` `ADR-2136…md:84`). Only `DEPRECATION.md:470` is stale (OMP-08). |
| OM-01.1 | corrected | Built-in exclusions are merged (`input/page.rs:485-490`). The linked journal root is refused (`:525`). Layout check at `main.rs:654` and `build/mod.rs:559`. The old Tension is replaced by OMP-02. |
| OM-01.2 | corrected | Caller citation re-derived (`input/page.rs:491`). The reconciliation paragraph was rewritten to the closed state (OMP-01). |
| OM-01.3 | accurate | Frontmatter split and gates unchanged (`input/frontmatter.rs:128-169`, `check/validate.rs:806-843`). |
| OM-01.4 | accurate | 11 `CredentialKind` patterns (`input/secrets.rs:115-157`). Body and frontmatter scans at `check/validate.rs:305`, `:323` and `build/publish.rs:228`, `:240`. |
| OM-01.5 | accurate | Slug and consistency functions unchanged. `whelk_inconsistent` still has no production caller. |
| OM-01.6 | corrected | Canonical discovery (`input/vocabulary.rs:507-523`), `Fence`-matched code regions (`input/code.rs:108-125`), and the get/get_mut callers (OMP-06). |
| OM-02.1 | accurate | `ir/model.rs` is unchanged since the stamp. Resolvers re-read. |
| OM-02.2 | accurate | Selectors and their callers re-read (`build/mod.rs:763-769`, `:866`, `:876`). |
| OM-02.3 | corrected | n3: machine scope builds more corpora (`build/mod.rs:656`, `ir/projection.rs:763-780`). |
| OM-02.4 | accurate | Redaction order re-read (`build/mod.rs:575-686`). |
| OM-02.5 | corrected | The walk excludes the built-ins **plus** the profile's folders, not one or the other. |
| OM-02.6 | corrected | Grep scope restated at the pin, including `ontomark-source`. Dead-code rows re-checked. |
| OM-03.1 | corrected | `layout.validate` runs first (`build/mod.rs:559`). `LayoutError` exits 2 from `main` (`main.rs:460-463`). |
| OM-03.2 | corrected | The role comes from `kind_of_at` (`profile/layout.rs:199-212`). `kind_of` Debt added (OMP-03). |
| OM-03.3 | accurate | One walk and two reads per page hold (`ir/projection.rs:118`, `input/page.rs:518`). |
| OM-03.4 | accurate | Staging and credential gate unchanged (`build/publish.rs:200-271`). |
| OM-03.5 | accurate-with-limits | Two-rename swap at `build/mod.rs:306-327`. Behaviour on failure between the renames is runtime. |
| OM-03.6 | accurate | Generation identity re-read (`build/mod.rs:933-1024`, `build/generation.rs:90-243`). |
| OM-03.7 | accurate | C3 map re-read at VisionFlow `5f8b1da` (`sovereign-corpus-contracts.md:110-165`). |
| OM-03.8 | accurate | Certificate path at `build/mod.rs:700-739`. The certificate is still written before the consistency gate. |
| OM-04.1 | corrected | C3 lists the three artefacts (OMP-07). |
| OM-04.2 | accurate | `turtle.rs` unchanged. Gate graph at `main.rs:702-703`. |
| OM-04.3 | accurate-with-limits | `reason.rs` unchanged. Migrate caller at `migrate/compile.rs:138`. The extraction boundary was reproduced by the 2026-11-01 audit, not by me. |
| OM-04.4 | accurate | `graph_tiers.rs` and `ontomark-ngg1` unchanged. |
| OM-04.5 | corrected | Landscape heading form (OMP-05, `build/indexes.rs:173-185`). |
| OM-04.6 | accurate | `webvowl.rs` and `okf.rs` unchanged. |
| OM-04.7 | accurate | Test-only functions re-grepped at the pin. |
| OM-05.1 | corrected | `ingest` verb added (`main.rs:137-140`, `:659`). The title is no longer "historical". |
| OM-05.2 | corrected | `MissingVault` or `LayoutError` exits 2 (`main.rs:457-466`). |
| OM-05.3 | corrected | Crate doc range `main.rs:31-42`. `LayoutError` added. Drift OMP-04. |
| OM-05.4 | corrected | Canonical discovery, layout check before any verb, and the `migrate plan` role check (`migrate_cmd.rs:144-146`). |
| OM-05.5 | corrected | visionGraph `556a015` pins v0.7.0 and fails on body residue (OMP-09). The VisionClaw workflow was re-read at `86cdd76`. |
| OM-05.6 | corrected | `ingest` is another verb that C2 lacks (`main.rs:137-146`). |
| OM-06.1 | accurate-with-limits | Scope refusals at `build/mod.rs:476-523`. The byte-identity invariants rest on named tests (`tests/machine_scope.rs:143`, `:244`) that I did not run. |
| OM-06.2 | accurate-with-limits | `ir/projection.rs` unchanged since the stamp. The twin rule rests on named tests that I did not run. |
| OM-06.3 | accurate-with-limits | Emitter at `build/records.rs`. Loom promotion re-read at `ba21a68` (`loom-vector-ruvector/src/artifact.rs:154-290`). The live Loom reload is runtime. |
| OM-06.4 | corrected | `accept` checks re-derived (`accept.rs:307-376`, `:659-674`). Ranges at `contract.rs:296-350`. `ingest_failed` catch-all added (OMP-10). |
| OM-06.5 | accurate | Loom readers re-read at `ba21a68`. ADR-2136 lines 5-7, 80, 84 and 97-102 re-read at agentbox `c57d9c0`. |
| OM-07.1 | accurate | The approve-only digest equality check still holds (`local-process-manager.js:400` against `:444-451` at agentbox `c57d9c0`). The only agentbox change in the window removed one environment allowlist entry. |
| OM-07.2 | accurate | `write/create.rs` unchanged. The title-derived target is at `:258`. |
| OM-07.3 | accurate-with-limits | Separate re-read and rename at `write/apply.rs:395-415`. The interleaving is a source-level concern. |
| OM-08.1 | accurate | All cited files are unchanged from `0caae5d`. Five redirects for sources (`ontomark-source/src/url.rs:60`), none for the model (`ingest_cmd.rs:418`). |
| OM-08.2 | accurate-with-limits | The mention check is lexical (`accept.rs:533`). The negated-relation pass was reproduced by the 2026-10-31 audit, not by me. |
| OM-08.3 | accurate | The request id omits the endpoint (`ingest_cmd.rs:605-615`). The manifest directory is removed before writing (`:771-772`). |

## Verification

Evidence class: **source tracing at the pinned revisions**. No product tests, builds or runtime probes were run, and no product code was edited.

Pinned revisions read: ontomark `4b9d15df4805` (full `4b9d15df48056971321a62b8c4e905f4536c74db`); VisionClaw `86cdd7635582`; agentbox `c57d9c0508f4` (through `../project/agentbox`); Loom `ba21a689a57d`; visionGraph `556a0156b5c3`; dreamlab-ai-website `0e1572801d03`; VisionFlow worktree HEAD `5f8b1da92ce5`. ontomark-obsidian HEAD `2a6c44e` was read for some ADR claims and is outside the pin set, as the tables state.

Method for the restamp. Every citation in OM-00 to OM-08 (about 2,400 anchors, including comma lists and bare continuations) was mapped from its old stamp to the pin. Where a file's blob is unchanged, the lines are identical. Where it changed, an anchor was carried over only if its line content is identical at the new position, matched as a whole sequence rather than by offset. Nine anchors fell on changed lines and were re-derived by symbol with `git grep -n`. Each panel's claims were then re-read against the anchors at the pin. The semantic diffs of every changed file (`input/page.rs`, `build/mod.rs`, `build/indexes.rs`, `input/code.rs`, `input/vocabulary.rs`, `profile.rs`, `profile/layout.rs`, CLI `main.rs` and `migrate_cmd.rs`, ingest `accept.rs`/`contract.rs`/`error.rs`, wasm `engine.rs`, visionGraph `publish.yml`, agentbox `local-process-manager.js` and ADR-2109) were read to find behaviour changes. Caller and dead-code claims were re-grepped across `crates/` at the pin. Every topic's `verified_commit` was restamped for every repository it cites. OM-00 gained an `agentbox` key for the ADR-2136 citation.

Gate commands, run from the worktree root. Each exited 0 after the edits:

```sh
node scripts/diagram-index-gen.cjs docs/diagrams --only 00-anomaly-register-and-reconciliation.md --check --cite-check --strict-citations --render   # 0; 236 verified, 5/5 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 01-input-layer-vault-pages-and-gates.md --check --cite-check --strict-citations --render      # 0; 161 verified, 6/6
node scripts/diagram-index-gen.cjs docs/diagrams --only 02-ir-corpus-projection-and-redaction.md --check --cite-check --strict-citations --render     # 0; 198 verified, 6/6
node scripts/diagram-index-gen.cjs docs/diagrams --only 03-build-run-gates-staging-and-generation.md --check --cite-check --strict-citations --render # 0; 226 verified, 8/8
node scripts/diagram-index-gen.cjs docs/diagrams --only 04-emitters-turtle-indexes-tiers-and-contract-c3.md --check --cite-check --strict-citations --render # 0; 164 verified, 7/7
node scripts/diagram-index-gen.cjs docs/diagrams --only 05-cli-commands-vault-selector-and-exit-codes.md --check --cite-check --strict-citations --render # 0; 145 verified, 6/6
node scripts/diagram-index-gen.cjs docs/diagrams --only 06-machine-scope-and-ingest-contract.md --check --cite-check --strict-citations --render     # 0; 131 verified, 5/5
node scripts/diagram-index-gen.cjs docs/diagrams --only 07-governed-create-and-amend.md --check --cite-check --strict-citations --render             # 0; 49 verified, 3/3
node scripts/diagram-index-gen.cjs docs/diagrams --only 08-external-source-ingest.md --check --cite-check --strict-citations --render                # 0; 59 verified, 3/3
```

ADR assessment. The website ADRs (2001–2008) and ontomark ADR-OM-001, 002, 003, 006 and 011 were traced by two read-only sub-reviewers working at the same pins under this contract. I checked their output myself: every heading in the tables was compared verbatim with `git show <sha>:<path> | grep '^#'`. I re-checked the decisive findings at the pin: the three `encrypted = true` zones in `forum-config/dreamlab.toml`, the `=1.0.0-beta.16` pins, no `trait Signer`, no `mint` in `ontomark-ingest`, no `oa:` selectors, and the ingest dependency list.

Not verified: tests named as pinning invariants (machine scope, register, conformance, layout contract); browser, deploy and CI receipts quoted in the ADR annexes; crates.io and GitHub settings; VisionClaw's adoption of the layout checks; the visionGraph publisher's renderer removal; and the Obsidian host beyond the files cited. Topics OM-09 to OM-12 were not touched.
