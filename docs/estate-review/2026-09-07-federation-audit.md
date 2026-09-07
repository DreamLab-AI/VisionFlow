---
title: Federation, storage, publisher and ontology-engine audit
date: 2026-09-07
status: source-and-local-tests-reviewed
type: explanation
---

# Federation, storage, publisher and ontology-engine audit

This audit checks the six repository working trees against their **31 current ADR records**, including three proposed records. It also reviews their diagram families and the estate identifier, identity and pod diagrams. It preserves all pre-existing dirty work. The external review correctly identifies incomplete cross-service authority, publication and erasure contracts, but several of its concrete explanations are stale or stronger than their evidence.

The most consequential new finding is in the forum's repaired trust sweep: its transaction can commit an audit row describing a demotion that the optimistic UPDATE did not perform. A separate, reproduced publication finding remains in the active visionGraph producer, although the extracted knowledgeGraph working tree now has substantially stronger validation and visibility controls. Neither finding is a claim of a reproduced production incident.

## Scope and evidence identity

Repository HEAD identifies committed history. A dirty working tree requires additional content identity; neither HEAD nor a diagram's `verified_commit` proves that the current bytes, a released dependency and a running service agree. No remote deployment, DNS, live account, private resource or financial endpoint was exercised. External standards maturity and vulnerability-feed currency are not re-certified here.

| Repository | HEAD at inspection | Working tree at inspection |
|---|---|---|
| `solid-pod-rs` | `1d9da527076e733d6a5571f474a573c16e5a6047` | 3 status entries; pre-existing changes retained |
| `nostr-rust-forum` | `d48a7a54612388a5c62b55d7ed87d425d205eedf` | 2 status entries; pre-existing changes retained |
| `dreamlab-ai-website` | `9a3dd88306b5414c3522ff9909294b811733b938` | 0 status entries; clean |
| `visionGraph` | `9e308164cc477567941e079c67049bd2ea9e7a05` | 14 status entries; pre-existing changes retained |
| `knowledgeGraph` | `2791111fc4ae301fdc5843ed2ad88b2e67d643fb` | 32 status entries; pre-existing changes retained |
| `vowl-wasm` | `65e2d1e784bf5eb04b3cbc122d36d6926d889c22` | 0 status entries; clean |

The active ledgers are `solid-pod-rs/crates/solid-pod-rs/docs/adr` (7), `nostr-rust-forum/docs/adr` (10), `dreamlab-ai-website/docs/adr` (8), `visionGraph/docs/adr` (2), and `knowledgeGraph/docs/adr` (4). `vowl-wasm` has no local ADR ledger. Frozen archives, the forum's retained sprint copies, and corpus pages named ADR-008/ADR-012 remain history or content; they are not additional current decisions. In particular, knowledgeGraph's absent ADR-NG-001 is classified historical-absent, not reconstructed authority. WasmVOWL and embedded explorer checkouts are separate consumers, not aliases for `vowl-wasm`.

## Findings and corrections to the external review

### F-01 — Trust batching fixes the old defect but still permits a false audit on conflict

The current [sweep](../../../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs) uses a stable `(COALESCE(last_active_at, 0), pubkey)` keyset, named row outcomes and a D1 batch. `trust.rs:406-417` explicitly removes the old per-pubkey demotion entry point. NF-03.9 and NF-10.8 were wrong to continue describing OFFSET skipping and ignored execution failures as current behaviour; those descriptions are corrected.

`commit_demotion_d1` (`trust_sweep.rs:428-489`) guards the UPDATE on the previously read level, then issues an **unconditional** audit INSERT in the same batch. It examines `meta.changes` only after that batch returns. A successful no-op UPDATE is not a SQL error. A local SQLite probe using the extracted production statements, with a snapshot expecting TL1 while the stored row is TL3, produced:

```text
UPDATE changes = 0
actual stored trust level = 3
audit rows committed = 1  (claims TL1 -> TL0)
```

The code then reports a conflict, so the demotion counter need not overcount; the inconsistency is the audit row already committed. This is distinct from SQL-error rollback, which the new batch fixes. The UPDATE also does not predicate on newly changed exemption or activity fields. Closeout must condition audit creation on a successful guarded state change inside the same transaction and test concurrent level, exemption and activity updates. Severity: high audit-integrity priority; no unauthorised live demotion was reproduced.

### F-02 — Relay projection receipts exist; full governance delivery remains partial

The [receipt implementation](../../../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs) records full event identity, stage and replay count, and batches decision insertion, case update and receipt transition (`:489-552`). This is materially stronger than the old separate ignored writes. `nip_handlers.rs:855-928` still stores the envelope and sends OK before projection; OK is a storage acknowledgement.

Correlation (`receipts.rs:164-180`) requires a non-empty case `d` tag but leaves `request_event_id` optional. Planning (`nip_handlers.rs:205-221`) retains an absent-case default. The `broker_decisions` schema has a foreign key to `broker_cases` (`lib.rs:727`), so enforcement of that constraint may reject an absent-case insertion: the fallback is **not** proof that an orphan commits. The case UPDATE has no observed-state predicate, and the adapter checks statement success rather than affected rows. Source review therefore leaves request/case/operation correlation, concurrent terminal decisions and zero-row semantics open. The failure branch may itself fail to record a receipt, and retryable stages do not prove an autonomous replay/recovery job is running. Consumer-received and external applied/rejected receipts remain cross-repository acceptance work. The diagram now draws those stages as the proposed continuation.

### F-03 — The publication boundary differs between the active vault and extracted publisher

In [visionGraph's parser](../../../visionGraph/pipeline/jsonld_parser.py), `is_public=page_block.get("vc:public", False)` (`:205`) preserves a string `"false"` as a truthy value. [Its build](../../../visionGraph/pipeline/build.py) logs validation errors and continues (`:40-46`), computes closure over the parsed pages (`:62-68`) and passes that closure to the inferred exporter. [The inferred exporter](../../../visionGraph/pipeline/reason.py) iterates closure entries without a public filter (`:171-208`). A fresh synthetic public-child/private-parent/private-grandparent probe emitted one inferred triple containing the private-grandparent IRI. This exercises code directly without reading or publishing private user content.

The [knowledgeGraph working-tree pipeline](../../../knowledgeGraph/pipeline/build.py) now has strict rejection, an input census, typed publication flags, cross-export visibility filtering, identity-set validation and an output manifest. Its 85 tests pass. These pre-existing changes do not automatically repair visionGraph or establish deployed consumer identity. Its release gate intentionally keeps the independent count pin **and** a committed identity set, catching equal-count substitutions. Compiler and graph-shape checks must not be described as formal ontology consistency or factual accuracy.

The visionGraph suite currently reports 57 passed and 1 failed: `test_real_corpus_publishes_nothing_from_misc` asserts at least 14 `_misc` files but the dirty corpus currently has 8. The user's existing deletions were preserved; this test failure is a corpus-layout expectation mismatch, not evidence that private publication prevention passed or failed. ADR-VG-001/002 remain proposed, partial and inactive.

### F-04 — Pod security fixes are tier- and version-specific

The standalone pod source implements typed `PolicyOutcome` handling (`server/src/lib.rs:1944-1978`), audience-sensitive caching (`:1196`, GET audience calculation `:1245`) and staged provenance receipts (`:1152-1169`, write hook `:3462-3674`). `default = []` remains in the server manifest: the default has no git-mark provenance. Resource storage followed by mark/sidecar work is still not a multi-store transaction; receipts expose reached stages rather than manufacture atomicity.

The [OIDC verifier](../../../solid-pod-rs/crates/solid-pod-rs/src/oidc/mod.rs) advertises ES256/RS256 (`:184`), disables audience validation (`:812`) and extracts top-level WebID/URL-shaped subject (`:864`). All 28 current compatibility-matrix tests pass. This grounds the selected implementation profile without asserting current external standards conformance. The NIP-98 seam is process-local unless a consumer supplies shared replay state.

The [forum pod edge](../../../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/lib.rs) still emits public cache policy by path (`:290-302`) and pins core-only `=0.5.0-alpha.7` in the root manifest (`Cargo.toml:155`). The standalone source is alpha.9, with a 2026-09-06 closeout release recorded in its changelog. SP-09.7's old claim that no version had been bumped and the consumer could not yet adopt it was stale. This audit has not queried the registry: the changelog is release intent/history, not fresh registry attestation. Adoption needs resolved package identity **and** caller wiring for policy, cache and replay seams.

Other existing source findings remain relevant: standalone COPY authorises destination write but not source read (`server/src/lib.rs:2571-2631`), and glob GET authorises the folder before collecting children (`:2634` onward). These are source-level boundary findings; this pass did not run an exploit or claim they occur in a currently exposed deployment. Do not close all WAC delivery work from the standard GET cache tests alone.

### F-05 — Federation has implemented seams and deliberate refusals

The forum implements peer/kind admission gates (`relay_do/nip_handlers.rs:559-563,2202-2254`), while the inspected mesh transport remains a separate unwired boundary. A dependency named mesh does not establish a running federation transport. Operational agentbox/VisionClaw crossings, however, are no longer merely fragile matching conventions: both derive from `agentbox/schema/federation-kinds.json`, with typed constructors and deliberate refusals. Agentbox ADR-2061 is the kind-map decision; VisionClaw ADR-2061 is an unrelated analytics-kernel decision. ES-03 now uses repository-qualified ADRs and distinguishes the implemented operational mapping from remaining RDF namespace joins.

ES-08.9's pod memory deletion call graph has no reverse vector-erasure dispatch. That is a **missing erasure propagation** finding. It establishes neither bulk HNSW deletion nor index degradation. An embedding survives only if a corresponding row exists and no independent erasure runs; this audit did not establish that join in a live store. The external review's recall-degradation interpretation is unsupported by this diagram.

ES-08.11 now qualifies ontology boot pull: checksums are checked before storage writes, and writing the manifest last supports retry after a partial write. It does not give atomic multi-file activation or isolate readers from mixed content. A network failure before writes leaves old content intact; a storage failure during writes need not.

### F-06 — Website source guards are strong local evidence, not deployment evidence

The website currently exact-pins four kit requirements to `=1.0.0-beta.10`; deploy, workers-deploy and rust-ci all declare KIT_REF `931898a3d82da5dbf573b6b6dccdc76513046875`. The 68 Vitest guard tests pass, including deliberate pin/mirror/endpoint corruption cases. The historical ADR body references beta.9 and an older SHA, while its acceptance progress records the newer controls. Review current manifests and gates before copying historical literals.

The deployed architecture is described in source as three frontend build outputs and separate Pages/Workers planes; this audit verified workflow wiring, not current DNS or deployment. Zone declarations remain four zones, dual-accept locked cohorts, and only family encrypted. Raw Schnorr pubkeys/NIP-42 remain the React relay identity boundary. Talk-to-AI verifies the expected sender and uses configurable reply relays; browser cancellation, reconnect and authority changes still require consumer journey evidence. DW-06's missing `scripts/seed/seed-forum.mjs` source was corrected to the existing `scripts/seed-forum.mjs`.

## Every current ADR disposition

“Aligned” means the scoped source decision is supported, not complete-system or deployed acceptance. “Qualified” means the core choice holds with a material boundary or stale statement. “Partial” means an explicit acceptance gap remains. Historical status axes are not silently overwritten.

### solid-pod-rs

| Record | Current audit disposition | Code and remaining acceptance |
|---|---|---|
| [ADR-2001](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2001-corpus-consolidation.md) | Aligned, documentation scope | Living baseline and seven-record ledger exist; frozen five-record archive retained. Source/feature/deployment identity is separate. |
| [ADR-2002](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2002-wac-access-model.md) | Qualified: WAC and native cache fix | `src/wac/` and server `enforce_read_ctx`/`set_cache_policy`; edge public caching and COPY/glob exceptions remain (F-04). |
| [ADR-2003](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2003-solid-oidc-01-defer-lws10.md) | Aligned deferral, compatibility limits explicit | `src/oidc/mod.rs:184,812,864`; 28 matrix tests pass. Audience checks and unsupported identity shapes are not inferred from OIDC naming. |
| [ADR-2004](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2004-provenance-off-by-default.md) | Aligned opt-in, partial delivery atomicity | `solid-pod-rs-server/Cargo.toml:123,150`; server `git_mark_write` and `ProvenanceReceipt` stages. Write success does not imply mark/sidecar commit. |
| [ADR-2005](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2005-fail-closed-untrusted-parsing.md) | Qualified: native typed failure boundary | `src/wac/resolver.rs`, `server/src/lib.rs:1944-1978`; malformed/unavailable policy differs from absence. All consumer adapters must adopt the seam. |
| [ADR-2006](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2006-nip98-replaystore-seam.md) | Aligned seam, cross-tier protection incomplete | `src/auth/replay_store.rs`, `src/auth/replay.rs`; process-local cache contract and capacity error do not establish replica/restart replay resistance. |
| [ADR-2007](../../../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2007-mempool-single-url-no-fallback.md) | Aligned selected-endpoint contract | `solid-pod-rs-server/src/mempool.rs:52-56` and `MempoolHttpClient`; single configured base, public testnet4 default, no fallback. No network call made. |

### nostr-rust-forum

| Record | Current audit disposition | Code and remaining acceptance |
|---|---|---|
| [ADR-2001](../../../nostr-rust-forum/docs/adr/ADR-2001-corpus-consolidation.md) | Aligned, documentation scope | Two governing docs, ten current records and frozen 24-record archive; retained sprint copies are lineage, not extra authority. |
| [ADR-2002](../../../nostr-rust-forum/docs/adr/ADR-2002-canary-first-upstream-nostr-absorption.md) | Partial, staged absorption | `nostr-bbs-upstream-canary` exists separately; core crypto remains. No new wasm32 canary verdict or deletion approval established. |
| [ADR-2003](../../../nostr-rust-forum/docs/adr/ADR-2003-derive-subkey-raw-hmac-js-parity.md) | Aligned raw HMAC contract | `nostr-bbs-core/src/keys.rs:251-257` uses HmacSha256 over root/tag. Native parity fixtures are distinct from executing deployed JS. |
| [ADR-2004](../../../nostr-rust-forum/docs/adr/ADR-2004-device-keys-default-off-dual-worker-gate.md) | Qualified: independent worker gates, shared predicate | `auth-worker/src/devices.rs:36,91-104` and `relay_do/nip_handlers.rs:14,1649-1663` call the shared core predicate independently. Literal “rather than shared” is stale; default-off exact-string contract holds. |
| [ADR-2005](../../../nostr-rust-forum/docs/adr/ADR-2005-gift-wrap-recipient-tag-admission.md) | Aligned recipient admission | `nip_handlers.rs:114,527-536` selects first recipient tag and whitelists recipient. Admission is not complete private delivery/revocation evidence. |
| [ADR-2006](../../../nostr-rust-forum/docs/adr/ADR-2006-trust-demotion-cron-tl3-never.md) | Partial audit integrity after repaired paging | `trust_sweep.rs` keyset/outcomes/batch fix is real; exact-query conflict probe leaves false audit (F-01). 239 relay tests pass. |
| [ADR-2007](../../../nostr-rust-forum/docs/adr/ADR-2007-kit-exact-pin-beta-channel.md) | Aligned exact pod pin, historical kit literal | Root `Cargo.toml:155` remains exact alpha.7/core. Website resolved beta.10 and standalone alpha.9 are distinct identities. |
| [ADR-2008](../../../nostr-rust-forum/docs/adr/ADR-2008-bbs-signer-holds-key-directly.md) | Aligned backend-specific custody | `bbs-client/src/signer.rs` stores a signer object; local PrfSigner paths hold SecretKey while NIP-07 delegates. Browser storage/JS copies require separate logout evidence. |
| [ADR-2009](../../../nostr-rust-forum/docs/adr/ADR-2009-acl-sidecar-coerce-to-control.md) | Qualified sidecar Control boundary | `pod-worker/src/lib.rs:959,1581-1603` uses sidecar coercion. Effective policy and response caching remain separate acceptance conditions. |
| [ADR-2010](../../../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md) | Proposed, partial, inactive remains accurate | `relay_do/receipts.rs` batches local projection; optional request correlation and absent external applied receipts remain (F-02). |

### dreamlab-ai-website

| Record | Current audit disposition | Code and remaining acceptance |
|---|---|---|
| [ADR-2001](../../../dreamlab-ai-website/docs/adr/ADR-2001-corpus-consolidation.md) | Aligned ledger scope | Eight ledger records and two governing docs exist; frozen archive is history. 68 guard tests are source checks, not deployment certification. |
| [ADR-2002](../../../dreamlab-ai-website/docs/adr/ADR-2002-split-hosting-pages-workers.md) | Qualified split deployment | `.github/workflows/deploy.yml` assembles Pages artifact; Cloudflare Pages path remains variable-gated. No current remote DNS/activation checked. |
| [ADR-2003](../../../dreamlab-ai-website/docs/adr/ADR-2003-three-frontends-one-origin.md) | Aligned three-build layout | `deploy.yml` builds React root, Trunk forum and Trunk BBS; separate environment projections. Browser journey not rerun. |
| [ADR-2004](../../../dreamlab-ai-website/docs/adr/ADR-2004-kit-pin-version-and-sha-lockstep.md) | Aligned current guards; old literals historical | `forum-config/Cargo.toml:49-52`, three KIT_REF declarations and `scripts/lib/pin-parity.mjs`; 19 dedicated parity tests pass. |
| [ADR-2005](../../../dreamlab-ai-website/docs/adr/ADR-2005-config-hand-synced-mirrors.md) | Qualified accepted manual mirrors | `forum-config/dreamlab.toml`, deploy mirrors, `scripts/lib/config-mirrors.mjs`; 22 guard tests pass. Validation does not make mirrors generated. |
| [ADR-2006](../../../dreamlab-ai-website/docs/adr/ADR-2006-raw-schnorr-nip42-identity.md) | Aligned raw-key React identity | `src/lib/nostr.ts` NIP-42 challenge events; no adopted DID/Multikey verification inferred. Kit/pod NIP-98 is a separate boundary. |
| [ADR-2007](../../../dreamlab-ai-website/docs/adr/ADR-2007-four-zone-dual-accept-cohorts.md) | Aligned authored policy; runtime acceptance pending | `dreamlab.toml:95-142` has four zones, dual cohorts, only zone3 encrypted; authoring and mirror guards do not prove live grant revocation. |
| [ADR-2008](../../../dreamlab-ai-website/docs/adr/ADR-2008-talk-to-ai-nostr-dm-routing.md) | Qualified transport and expected sender | `AIChatFab.tsx:37-44,344-348,443` wires recipient/reply relays/sender checks. Full browser disconnect and agent reply journey not rerun. |

### visionGraph

| Record | Current audit disposition | Code and remaining acceptance |
|---|---|---|
| [ADR-VG-001](../../../visionGraph/docs/adr/ADR-VG-001-publication-policy-boundaries.md) | Proposed, partial, inactive; reproduced gap | `pipeline/jsonld_parser.py:205`, `build.py:40-68`, `reason.py:171-208`; synthetic private ancestry appears in inferred output (F-03). |
| [ADR-VG-002](../../../visionGraph/docs/adr/ADR-VG-002-generation-and-consumer-identity.md) | Proposed, partial, inactive | Producer/export/embedded consumer identity still needs a common activation receipt; knowledgeGraph manifest work does not establish this producer contract. |

### knowledgeGraph

| Record | Current audit disposition | Code and remaining acceptance |
|---|---|---|
| [ADR-2001](../../../knowledgeGraph/docs/adr/ADR-2001-corpus-consolidation.md) | Aligned corpus routing; dirty implementation qualified | Frozen ADR-named corpus pages remain; historical-absent ADR-NG-001 is routed, not fabricated. Current `pipeline/manifest.py` records generation identity. |
| [ADR-2002](../../../knowledgeGraph/docs/adr/ADR-2002-corpus-is-the-ontology.md) | Qualified compiler/publication implementation | `pipeline/jsonld_parser.py`, `validate.py`, `visibility.py`, strict `build.py`; 85 tests pass. No formal reasoner or factual-quality guarantee follows. |
| [ADR-2003](../../../knowledgeGraph/docs/adr/ADR-2003-expected-classes-tripwire.md) | Aligned count plus identity gate in working tree | `.github/workflows/build.yml`, `pipeline/release_gate.py:109-125,246-254`, committed `pipeline/contracts/`; equal-count tamper tests pass. |
| [ADR-2004](../../../knowledgeGraph/docs/adr/ADR-2004-corpus-page-identity-immutable.md) | Qualified identity preservation and generation mapping | `jsonld_to_webvowl.py::_remap_iri`, `visibility.py::terminal_slug`, committed identity set. Historical identities retained; no upstream inferred-URN convergence certified. |

### vowl-wasm

There is no current ADR file to classify. `Cargo.toml` declares version 0.1.2 and default features empty. `src/ontology/parser.rs:75-91` consumes merged class entries with string IDs/IRI/label; it does not prove every separate WebVOWL attribute projection is compatible. The NGG1 binding, interaction and markdown loader are optional, while the published npm recipe enables a selected feature set. The native default-feature library suite passes 136 tests. Browser-rendering, zero-copy buffer lifetime, optional-feature and every embedded consumer build remain separate evidence scopes.

## Verification performed in this pass

| Command / probe | Result | Scope limit |
|---|---|---|
| Forum `cargo test --locked --offline -p nostr-bbs-relay-worker --lib` | 239 passed | Native local logic; not a deployed D1 worker |
| Pod `cargo test --locked --offline -p solid-pod-rs --test oidc_compat_matrix --features oidc,dpop-replay-cache` | 28 passed | Selected OIDC compatibility contract; no network |
| Website `./node_modules/.bin/vitest run scripts/__tests__` | 68 passed in 4 files | Pin, mirror, endpoint and dream-gate source guards |
| knowledgeGraph `.venv/bin/python -m pytest pipeline/tests -q` | 85 passed | Pre-existing working-tree pipeline changes |
| visionGraph existing knowledgeGraph venv Python, `-m pytest pipeline/tests -q` | 57 passed, 1 failed | `_misc` corpus-count expectation mismatch in dirty tree |
| vowl-wasm `cargo test --locked --offline --lib` | 136 passed | Native default features only |
| Extracted trust UPDATE + audit INSERT, in-memory SQLite | Zero-row UPDATE, preserved TL3, one false audit row | Models exact SQL transaction semantics; not D1 deployment |
| visionGraph `compute_closure` / `emit_inferred_ttl` synthetic ancestry | 1 inferred triple; private-grandparent IRI present | Synthetic identifiers only; no live corpus published |

The initial website command used Node's test runner for Vitest-authored files and failed in the harness; the corrected Vitest run above passed. The system Python lacked pytest; the existing project venv was used. These setup corrections are not application regressions. All 11 changed topic files passed the Mermaid render and width gates: 101 diagrams rendered. Two sequence-note punctuation errors found by the first render were corrected; both affected topics then rendered 10/10. The renderer checks syntax/output width, not source semantics.

## Closeout roadmap additions

| Priority | Accountable boundary | Concrete acceptance condition | Dependency |
|---|---|---|---|
| P1 | Forum trust/state owner | Condition audit creation on a successful guarded state transition inside the transaction; inject zero-row conflict, concurrent exemption/activity edits and restart | ADR-2006 adapter semantics; retain keyset paging |
| P1 | Publisher and authoring owners | Define typed inclusion and inferred visibility policy in visionGraph; common malformed/private/namespace fixture across every exporter and consumer | ADR-VG-001; preserve dirty authored content |
| P1 | Forum + authority consumer + mutation owner | Require request/case/operation correlation; transactional state guards and durable replay; expose signed/accepted/projected/received/applied distinctions | ADR-2010 remains proposed until agreed |
| P1 | Pod/edge release owners | Publish/resolve exact safe code, adopt typed policy/audience seams in edge, test private response through real caches and revocation; cover COPY/glob boundaries | Source/package/feature/deployment identity |
| P1 | Memory and pod owners | Identify row/resource join, dispatch durable erasure/tombstone, verify retrieval denial and recovery across both stores | Erasure contract, not HNSW tuning |
| P2 | Ontology delivery owner | Stage a complete generation and atomically activate it, or specify/test reader consistency during manifest-last updates | Content hashes plus generation identity |
| P2 | Identity/release owners | Test replay across restart/replicas, signer rotation, browser cancellation and grant removal per consumed version | Exact boundary and custody inventory |
| P2 | Explorer owners | Bind package/features/worker/renderer to output generation and run browser schema/buffer-lifetime tests | Distinguish vowl-wasm, WasmVOWL and embedded copies |

No production code was changed or service restarted. The proposed sprint should close these observable acceptance conditions; a generated coverage matrix or historical `complete` status is insufficient to close them.

## Reproducible source fingerprints

SHA-256 values identify the exact inspected source bytes, including relevant dirty knowledgeGraph changes. They supplement the repository HEADs above.

| Source | SHA-256 |
|---|---|
| `nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs` | `537bbc92dba6657ff93546afb103daddf5c7abeb6a109f25906bd818f7d25753` |
| `nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs` | `871f0c18b8848c7239d9a97277ae681fbe275618c29d662d6465b9d538512023` |
| `nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs` | `db8bce97148bc281b0b51ac9da224e30d9df91352fa8b352f738a877bf44f4b8` |
| `nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs` | `32c2e7b7c97b407dba22cb1a2ba0242aacb07261734b85eb4ef799ca7eb67cf8` |
| `nostr-rust-forum/crates/nostr-bbs-pod-worker/src/lib.rs` | `76dca2c644ffea5185e6a60c50c626bb585e69273cf8e9acb485f1a21f094742` |
| `solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs` | `4446d641b36f6cc7df5ec552455c1736cbf57a55a78876b153053749e7d2bf87` |
| `solid-pod-rs/crates/solid-pod-rs/src/oidc/mod.rs` | `da1c9343cec368b6724e02d74f87d4bc13e1d8d2797a03b6e5d06e0c41d0dc1e` |
| `visionGraph/pipeline/build.py` | `4afa04f86c3d6bf435765ca8ffea64f83101182e79920410c0cb0fce753ca05f` |
| `visionGraph/pipeline/jsonld_parser.py` | `709a54466616727909af665c4eebd05fad478a5b4ee03a77b4abfd20d1fc8a3c` |
| `visionGraph/pipeline/reason.py` | `f82cb9ee9f7d21c79532bb42446bd16fc8f080fbd953f357279c35132f20bfeb` |
| `knowledgeGraph/pipeline/build.py` | `97101a32a401cbb2b9a141a27f3a8b25e6b0a7ee7c6b077955c14ffc85e32bbf` |
| `knowledgeGraph/pipeline/visibility.py` | `09c702a40ccd0e1dbc3759b447a97249edfae89f41c733e849a9c94c47ebd50b` |
| `knowledgeGraph/pipeline/release_gate.py` | `74f4e861ae1a3fc0042aca77d1b5a25ccd068371c776a75a200dd8f3493e4b86` |
| `knowledgeGraph/.github/workflows/build.yml` | `c39bc6f129581ff8e9dcb329b34d16ffecdf41aab9abfc6c66b6796b2037f048` |
| `dreamlab-ai-website/forum-config/Cargo.toml` | `fd17099bec9979754b5495e8c84f8ac5d38da417547f5449c8455b7f1e33458c` |
| `dreamlab-ai-website/scripts/lib/pin-parity.mjs` | `32e0706d424aa0867ab8121b4a556169d88edad3a769d83a694a016a87114b13` |
| `vowl-wasm/Cargo.toml` | `e020bf21c3f8b4d7d507c2a0bf235ce1bb2afa41cbd83685d7a24d46bfc40b41` |
| `vowl-wasm/src/ontology/parser.rs` | `f04c7024be97eb423fb9f34b2a140aeb9ed50ff1417bdcf7d78bf94c8cd2342a` |

## Trust conflict probe recipe

The probe extracts the first matching SQL string from the production file, removing Rust backslash-newline continuations. It runs only against a fresh in-memory SQLite database.

```python
import pathlib, re, sqlite3
source = pathlib.Path("../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs").read_text()
def sql(prefix):
    match = re.search(r'"(' + re.escape(prefix) + r'.*?)"', source, re.S)
    return re.sub(r'\\\n\s*', '', match[1])
db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE whitelist(pubkey TEXT, trust_level INT, trust_level_updated_at INT);
CREATE TABLE admin_log(actor_pubkey TEXT, action TEXT, target_pubkey TEXT,
 previous_value TEXT, new_value TEXT, reason TEXT, created_at INT);
INSERT INTO whitelist VALUES('key', 3, 0);
""")
with db:
    changed = db.execute(sql("UPDATE whitelist SET trust_level = ?1"),
                         (0, 123, "key", 1)).rowcount
    db.execute(sql("INSERT INTO admin_log "),
               ("system", "trust_level_change", "key", "1", "0",
                "auto-demotion (hysteresis)", 123))
print(changed, db.execute("SELECT trust_level FROM whitelist").fetchone(),
      db.execute("SELECT COUNT(*) FROM admin_log").fetchone())
# 0 (3,) (1,)
```
