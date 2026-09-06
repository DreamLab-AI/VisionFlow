# VisionFlow estate — crates.io publication survey (2026-09-06)

Read-only survey. 116 `Cargo.toml` files found across the 12 named repos (after excluding
`target/`, vendored `cargo-registry-*` snapshots under `voice-stack/unmute/volumes`, and
`node_modules`); ~100 carry a `[package]` table and are tabled below. `cargo publish --dry-run
--allow-dirty` was run for every plausible CANDIDATE (never bare `cargo publish`; nothing was
actually uploaded).

## Headline counts

| Class | Count | Notes |
|---|---|---|
| PUBLISHED-CURRENT | 15 | repo version == crates.io max_version |
| PUBLISHED-STALE | 9 | repo version AHEAD of registry — needs a release |
| REGISTRY-AHEAD (ad-hoc class) | 10 | repo version BEHIND registry (opposite problem) — all `wifi-densepose-*`, workspace pinned at 0.3.0 while individual crates have been released past it |
| CANDIDATE — clean, dry-run passed | 14 | see table below |
| CANDIDATE — blocked | 7 | reusable in principle, blocked by a fixable issue |
| PROJECT-SPECIFIC | ~39 | one-line reasons in tables |
| EXCLUDED-BY-POLICY | 1 standalone package + 1 private-module case | `agentbox-secret-backup`; the bespoke WebCrypto envelope inside `randlehow-pipeline` |

**Two crate-name collisions found**: `webvowl-wasm` is defined independently in both
`WasmVOWL/rust-wasm` (v1.0.0) and `knowledgeGraph/explorer/rust-wasm` (v0.3.4) — the two have
diverged (different feature sets, different `repository` fields) and only one can ever claim the
name on crates.io. Separately, `skill-tools` (agentbox) collides with an unrelated, already-published
crate of the same name on crates.io — a rename (e.g. `agentbox-skill-tools`) is required regardless
of its other blocker.

**`webcrypto-envelope`**: no standalone crate/Cargo.toml exists in this checkout. Git history shows
it started as a standalone crate (commit `427e08e`) and was folded into `randlehow-pipeline` as a
private module (`crates/randlehow-pipeline/src/envelope/{mod,aead,kdf,seal}.rs`, commit `1680596`).
That module's own doc comment states it must never be offered as a reusable building block. A second
client-side copy exists at `dreamlab-cumbria/.../road/site/src/envelope.rs`. Both sit inside
`publish = false` crates, so there is no live risk of accidental publication — correctly scoped
EXCLUDED-BY-POLICY in spirit even though there's no separate package to classify.

---

## 1. VisionClaw core (`/home/devuser/workspace/project`) — crates/*, xr-client, client/scene-effects, nntp-stack/lounge, scripts/whelk-rs, voronoi-graphics

23 manifests surveyed (1 virtual: `nntp-stack/lounge/Cargo.toml`).

| Repo/Path | Name | Version | publish | License | Desc/Repo/Readme | crates.io max | Blocking deps | Docs | README | Tests | Classification | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Cargo.toml` (root) | visionclaw-server | 0.1.0 | absent | AGPL-3.0-only | Y/N/N | not found | 7 unversioned path + 1 unversioned git (whelk) | crate: N; item: partial | Y (not wired) | Y (223 hits) | PROJECT-SPECIFIC | 12-min monolith server; not reusable |
| `xr-client/rust/Cargo.toml` | visionclaw-xr-gdext | 0.2.1 | absent | MIT | Y/N/N | not found | visionclaw-xr-presence path, unversioned | Y (989 `///`) | N | Y (11 files) | PROJECT-SPECIFIC | godot-rust/Quest 3 toolchain coupling |
| `xr-client/rust/perf-regression/Cargo.toml` | xr-perf-regression | 0.1.0 | **false** | MIT | Y/N/N | not found | none | Y | N | minimal | PROJECT-SPECIFIC | author already set publish=false |
| `crates/visionclaw-gpu/Cargo.toml` | visionclaw-gpu | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | visionclaw-domain path, unversioned | Y (261 `///`) | N | Y | PROJECT-SPECIFIC | CUDA/PTX tied to kernel layout |
| `crates/visionclaw-domain/Cargo.toml` | visionclaw-domain | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | none | Y (874 `///`) | N | Y | **CANDIDATE** | dry-run **passed**; missing readme |
| `crates/vault-migrate/Cargo.toml` | vault-migrate | 0.1.0 | absent | AGPL-3.0-only | Y/N/N | not found | none | Y (165 `///`) | N | Y | **CANDIDATE** | no crypto despite name (verified by grep); dry-run **passed**; missing repository, readme |
| `crates/visionclaw-adapters/Cargo.toml` | visionclaw-adapters | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | domain path + whelk git, both unversioned | Y (373 `///`) | N | Y | PROJECT-SPECIFIC | Oxigraph+whelk wrapper, blocked deps |
| `crates/visionclaw-actors/Cargo.toml` | visionclaw-actors | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | domain path, unversioned | partial | N | Y | PROJECT-SPECIFIC | Actix supervision bound to domain types |
| `crates/visionclaw-protocol/Cargo.toml` | visionclaw-protocol | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | domain path, unversioned | partial | N | Y | PROJECT-SPECIFIC | wire codec hard-wired to domain types |
| `crates/visionclaw-integration-tests/Cargo.toml` | visionclaw-integration-tests | 0.1.0 | **false** | AGPL-3.0-only | Y/N/N | not found | none | Y | N | Y | PROJECT-SPECIFIC | black-box probes, no standalone value |
| `crates/visionclaw-xr-presence/Cargo.toml` | visionclaw-xr-presence | 0.2.1 | absent | MIT | Y/N/N | not found | none | Y (96 `///`) | N | Y | **CANDIDATE** | dry-run **passed**; missing repository, readme; `fuzz/` sibling noted only |
| `crates/visionclaw-ontology/Cargo.toml` | visionclaw-ontology | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | domain path + whelk git, both unversioned | Y (879 `///`) | N | Y | PROJECT-SPECIFIC | OWL/EL pipeline coupled to domain ports |
| `crates/visionclaw-contracts/Cargo.toml` | visionclaw-contracts | 0.1.0 | absent | AGPL-3.0-only | Y/Y/Y | not found | none | Y (191 `///`) | Y (wired) | Y | **CANDIDATE** | cleanest candidate, full metadata present; dry-run **passed** |
| `crates/visionclaw-analytics-oracle/Cargo.toml` | visionclaw-analytics-oracle | 0.1.0 | absent | AGPL-3.0-only | Y/N/N | not found | none | Y (50 `///`) | N | Y | **CANDIDATE** | zero deps, pure std; dry-run **passed**; missing repository, readme |
| `scripts/whelk-rs/Cargo.toml` | whelk | 0.1.0 | absent | **none** | N/N/Y | **0.2.0** | none | N | Y | minimal | not ours | vendored 3rd-party copy (Jim Balhoff/INCATools); registry is AHEAD of this vendored copy; root already pulls the real dep via git fork |
| `client/crates/scene-effects/Cargo.toml` | scene-effects | 0.1.0 | absent | **none** | Y/N/N | not found | none | Y (128 `///`) | N | minimal | **CANDIDATE** | missing license (would be **rejected** by real registry, not just warned), repository, readme; not covered by any workspace members/exclude — dry-run in place failed on workspace membership, isolated copy dry-run **passed** |
| `agentbox/skills/wasm-js/templates/voronoi-graphics/Cargo.toml` | voronoi-graphics | 0.1.0 | absent | MIT | Y/N/N | not found | none | partial (18 `///`/6 items) | N | Y | **CANDIDATE** | same workspace-membership issue as scene-effects; isolated dry-run **passed**; missing repository, readme |
| `nntp-stack/lounge/Cargo.toml` | *(virtual)* | — | — | — | — | — | — | — | Y | — | n/a | workspace root, MIT/0.1.0 inherited |
| `nntp-stack/lounge/ui/Cargo.toml` | lounge-ui | 0.1.0 | absent | MIT | N/N/N | not found | lounge-domain path, unversioned | partial | N | — | PROJECT-SPECIFIC | wasm32-only Leptos SPA |
| `nntp-stack/lounge/crates/server/Cargo.toml` | lounge-server | 0.1.0 | absent | MIT | N/N/N | not found | 3 sibling path deps, unversioned | partial | N | Y | PROJECT-SPECIFIC | Axum orchestration binary |
| `nntp-stack/lounge/crates/vectorlite/Cargo.toml` | lounge-vectorlite | 0.1.0 | absent | MIT | N/N/N | not found | none | Y | N | Y | **CANDIDATE** | dry-run **passed**; missing description, repository, readme |
| `nntp-stack/lounge/crates/clients/Cargo.toml` | lounge-clients | 0.1.0 | absent | MIT | N/N/N | not found | lounge-domain path, unversioned | Y | N | Y | PROJECT-SPECIFIC | generic-looking HTTP client wrapper, blocked by unversioned path dep |
| `nntp-stack/lounge/crates/domain/Cargo.toml` | lounge-domain | 0.1.0 | absent | MIT | N/N/N | not found | none | Y | N | Y | **CANDIDATE** | dry-run **passed**; missing description, repository, readme |

**Slice summary**: 9 CANDIDATE (all dry-run passed), 12 PROJECT-SPECIFIC (mostly unversioned path/git deps to sibling domain crates), 0 published, 0 excluded-by-policy. `vault-migrate` inspected specifically for hidden crypto — none found. `whelk-rs` is vendored third-party code, not this estate's own IP.

---

## 2. agentbox (`/home/devuser/workspace/project/agentbox`) — setup, crates, services

12 manifests, 11 with `[package]`.

| Repo/Path | Name | Version | publish | License | Desc/Repo/Readme | crates.io max | Blocking deps | Docs | README | Tests | Classification | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `setup/Cargo.toml` | *(virtual)* | — | — | — | — | — | — | — | — | — | n/a | members=["server"] |
| `setup/server/Cargo.toml` | agentbox-setup | 0.1.0 | absent | **none** | Y/N/N | not found | none | N | N | N | PROJECT-SPECIFIC | onboarding wizard; `rust_embed` folder attr escapes crate dir |
| `crates/headroom-napi/Cargo.toml` | headroom-napi | 0.1.0 | absent | AGPL-3.0-only | Y/N/N | not found | none | Y (item) / N (crate) | N | Y | PROJECT-SPECIFIC | napi-rs cdylib only, no rlib — not Rust-consumable; distributed via npm |
| `services/dream-engine/Cargo.toml` | dream-engine | 0.1.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | not found | none | Y (652 `///`) | Y | Y | PROJECT-SPECIFIC | wired to HP annexe SSH, RuVector, ledger format; dry-run **passed** anyway |
| `services/secret-backup/Cargo.toml` | agentbox-secret-backup | 0.1.0 | **false** | AGPL-3.0-only | Y/Y/Y | not found | none | Y (34 `///`) | Y | Y | **EXCLUDED-BY-POLICY** | delegates all crypto to `age` (X25519/scrypt/ChaCha20-Poly1305) — no bespoke primitive found on inspection, but classified excluded per explicit owner direction; also already `publish=false` |
| `services/agentbox-ops/Cargo.toml` | agentbox-ops | 0.1.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | not found | none | Y (636 `///`) | Y | Y | PROJECT-SPECIFIC | 14-binary container-internal daemon grab-bag; dry-run **passed** |
| `services/podcast-ingest/Cargo.toml` | podcast-ingest | 0.1.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | not found | none | Y (356 `///`) | Y | Y | PROJECT-SPECIFIC | wired to ontology/Loom/ledger substrates; dry-run **passed** |
| `services/ontology-tools/Cargo.toml` | ontology-tools | 0.1.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | not found | none | Y (300 `///`) | Y | Y | **CANDIDATE** | clean-room markdown/OWL2-functional-syntax parser + validator; **all metadata already present**; dry-run **passed** cleanly |
| `services/nostr-pod-bridge/Cargo.toml` | nostr-pod-bridge | 0.1.0 | absent | AGPL-3.0-only | Y/Y/Y | not found | `nostr-bbs-core`, `solid-pod-rs-nostr` path, unversioned; `[patch.crates-io]` local | Y (494 `///`) | Y | Y | PROJECT-SPECIFIC (candidate once unblocked) | delegates all crypto to first-party crates correctly; blocked purely by unversioned sibling-monorepo path deps |
| `services/skill-tools/Cargo.toml` | skill-tools | 0.1.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.2.1 (unrelated crate)** | none (all versioned) | Y (460 `///`) | Y | Y | **CANDIDATE — blocked** | BM25 search + Wardley Map + docs-alignment validators, clean-room; dry-run **FAILED**: `include_str!` reaches outside crate root into `agentbox/skills/ui-ux-pro-max-skill/...` (~24 compile errors on verification build); also **name collision** with an unrelated published `skill-tools` crate — needs rename (e.g. `agentbox-skill-tools`) regardless |
| `services/agentbox-manifest/Cargo.toml` | agentbox-manifest | 0.1.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | not found | none | Y (196 `///`) | Y | Y | PROJECT-SPECIFIC | mirrors one specific shell script's subcommand/exit-code contract; dry-run **passed** |
| `services/agentbox-mcp/Cargo.toml` | agentbox-mcp | 0.1.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | not found | none | Y (169 `///`) | Y | Y | PROJECT-SPECIFIC | matches agentbox's own mcp.json tool names; dry-run **passed** |

**Slice summary**: 1 clean CANDIDATE (`ontology-tools`), 1 blocked CANDIDATE (`skill-tools` — fixable, plus needs rename), 8 PROJECT-SPECIFIC, 1 EXCLUDED-BY-POLICY (`secret-backup`, per owner direction). Nothing yet on crates.io under its own name.

---

## 3. loom + nostr-rust-forum

### loom (`/home/devuser/workspace/loom`) — 8 crates, virtual workspace root

| Repo/Path | Name | Version | publish | License | Desc/Repo/Readme | crates.io max | Blocking deps | Docs | README | Tests | Classification | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `crates/loom-domain` | loom-domain | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | none | Y (147 `///`) | N | Y | **CANDIDATE** | pure types/ports/errors, zero I/O; dry-run **passed**. Publish this first — everything else depends on it. |
| `crates/loom-scaffold` | loom-scaffold | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | loom-domain, unversioned | Y (113 `///`) | N | Y | **CANDIDATE — blocked** | LLM-free lexical matcher, genuinely clean-room; dry-run **failed** on unversioned loom-domain — fixable once that's published |
| `crates/loom-vector-ruvector` | loom-vector-ruvector | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | loom-domain + `ruvector-core` (unpublished sibling workspace) + loom-embed-xinference, all unversioned | Y (18 `///`) | N | Y | **CANDIDATE — blocked** | two-deep blocker (also needs ruvector-core published) |
| `crates/loom-graph-oxigraph` | loom-graph-oxigraph | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | loom-domain, unversioned | Y (28 `///`) | N | Y | **CANDIDATE — blocked** | read-only clamped-SPARQL adapter |
| `crates/loom-attest-proofgate` | loom-attest-proofgate | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | loom-domain + ruvector-core, unversioned | Y (25 `///`) | N | tests/ dir | **CANDIDATE — blocked** | sha2 chain-hash ledger, not bespoke crypto |
| `crates/loom-embed-xinference` | loom-embed-xinference | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | loom-domain, unversioned | Y (12 `///`) | N | Y | **CANDIDATE — blocked** | thin reqwest wrapper for Xinference embeddings |
| `crates/loom-backend-openai` | loom-backend-openai | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | loom-domain, unversioned | Y (19 `///`) | N | tests/ dir | **CANDIDATE — blocked** | generic OpenAI-compatible chat adapter |
| `crates/loom-facade` | loom-facade | 0.1.0 | absent | AGPL-3.0-only | Y/Y/N | not found | 6 sibling path deps, unversioned | Y (184 `///`) | N | Y | PROJECT-SPECIFIC | the Loom application itself (axum/tower + dream-cycle bins), not a library |

**Loom summary**: 1 clean CANDIDATE (`loom-domain`), 6 blocked CANDIDATEs (all fail identically on `loom-domain`'s unversioned path dep — a single fix (publish loom-domain, then add `version=` to the six dependents) unblocks the whole crate family), 1 PROJECT-SPECIFIC. None on crates.io yet. No hand-rolled crypto (the one crypto-touching crate uses `sha2` for a chain hash, not an envelope).

### nostr-rust-forum (`/home/devuser/workspace/nostr-rust-forum`) — 14 crates, virtual workspace root

| Repo/Path | Name | Version | publish | License | Desc/Repo/Readme | crates.io max | Blocking deps | Docs | README | Tests | Classification | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `crates/nostr-bbs-core` | nostr-bbs-core | 1.0.0-beta.9 | absent | AGPL-3.0-only | Y/Y/N (readme field/file missing) | **1.0.0-beta.9** | none | Y (943 `///`) | N | Y (20 tests + NIP-44/19 vectors) | **PUBLISHED-CURRENT** | **Crypto verdict: not hand-rolled.** Schnorr wraps `k256::schnorr` directly; NIP-04/NIP-44 hand-rolled pre-refactor, now delegated to upstream `nostr` crate (module docs record the spec bug this fixed). Matches CLAUDE.md's own approved combo. |
| `crates/nostr-bbs-config` | nostr-bbs-config | 1.0.0-beta.9 | absent | AGPL-3.0-only | Y/Y/Y | **1.0.0-beta.9** | none | Y (92 `///`) | Y | Y | **PUBLISHED-CURRENT** | pure TOML/serde schema, best-documented |
| `crates/nostr-bbs-mesh` | nostr-bbs-mesh | 1.0.0-beta.9 | absent | AGPL-3.0-only | Y/Y/Y | **1.0.0-beta.9** | none | Y (93 `///`) | Y | Y | **PUBLISHED-CURRENT** | federation mesh stub, sha2/hex only |
| `crates/nostr-bbs-rate-limit` | nostr-bbs-rate-limit | 1.0.0-beta.9 | absent | AGPL-3.0-only | Y/Y/N | **1.0.0-beta.9** | none | Y (23 `///`) | N | N | **PUBLISHED-CURRENT** | CF-KV rate limiter, reusable for any CF Worker |
| `crates/nostr-bbs-ascii` | nostr-bbs-ascii | 1.0.0-beta.9 | absent | AGPL-3.0-only | Y/Y/N | 1.0.0-beta.**8** | none | Y (21 `///`) | N | Y | **PUBLISHED-STALE** | repo 1 beta ahead of registry; wasm-safe, dependency-free |
| `crates/nostr-bbs-setup-skill` | nostr-bbs-setup-skill | 1.0.0-beta.9 | absent | AGPL-3.0-only | Y/Y/Y | 1.0.0-beta.**8** | none | Y (8 `///`) | Y | Y | **PUBLISHED-STALE** | repo 1 beta ahead of registry |
| `crates/nostr-bbs-bbs-client` | nostr-bbs-bbs-client | 1.0.0-beta.9 | **false** | AGPL-3.0-only | Y/Y/Y (not wired) | not found | n/a | Y (826 `///`) | Y | Y | PROJECT-SPECIFIC | Leptos CSR terminal client wired to this forum's config/relay |
| `crates/nostr-bbs-forum-client` | nostr-bbs-forum-client | 1.0.0-beta.9 | **false** | AGPL-3.0-only | Y/Y/N | 1.0.0-beta.**2** (orphaned) | n/a | partial (bin-only) | N | Y | PROJECT-SPECIFIC | **orphaned stale publish**: live on crates.io at beta.2 from before publish=false was set; consider yanking |
| `crates/nostr-bbs-pod-worker` | nostr-bbs-pod-worker | 1.0.0-beta.9 | **false** | AGPL-3.0-only | Y/Y/N | 1.0.0-beta.**2** (orphaned) | n/a | Y (566 `///`) | N | Y | PROJECT-SPECIFIC | same orphaned-publish anomaly |
| `crates/nostr-bbs-search-worker` | nostr-bbs-search-worker | 1.0.0-beta.9 | **false** | AGPL-3.0-only | Y/Y/N | 1.0.0-beta.**2** (orphaned) | Y (54 `///`) | N | Y | PROJECT-SPECIFIC | same orphaned-publish anomaly |
| `crates/nostr-bbs-auth-worker` | nostr-bbs-auth-worker | 1.0.0-beta.9 | **false** | AGPL-3.0-only | Y/Y/N | 1.0.0-beta.**2** (orphaned) | Y (620 `///`) | N | Y | PROJECT-SPECIFIC | same orphaned-publish anomaly, WebAuthn/D1-specific |
| `crates/nostr-bbs-preview-worker` | nostr-bbs-preview-worker | 1.0.0-beta.9 | **false** | AGPL-3.0-only | Y/Y/N | 1.0.0-beta.**2** (orphaned) | Y (83 `///`) | N | Y | PROJECT-SPECIFIC | same orphaned-publish anomaly |
| `crates/nostr-bbs-relay-worker` | nostr-bbs-relay-worker | 1.0.0-beta.9 | **false** | AGPL-3.0-only | Y/Y/N | 1.0.0-beta.**2** (orphaned) | Y (444 `///`) | N | Y | PROJECT-SPECIFIC | same orphaned-publish anomaly |
| `crates/nostr-bbs-upstream-canary` | nostr-bbs-upstream-canary | 1.0.0-beta.9 | **false** | AGPL-3.0-only | Y/Y/N | not found | Y (8 `///`) | N | Y | PROJECT-SPECIFIC | internal WASM-build-matrix canary spike |

**nostr-rust-forum summary**: 4 PUBLISHED-CURRENT, 2 PUBLISHED-STALE (one beta behind), 8 PROJECT-SPECIFIC, 0 EXCLUDED, 0 fresh CANDIDATE (the reusable primitive is already published). **Anomaly for the owner**: 6 crates (5 CF Workers + forum-client) are live on crates.io at a stale `1.0.0-beta.2` with `publish=false` now set — orphaned, un-updatable published artifacts; worth a yank decision.

---

## 4. solid-pod-rs + dreamlab-ai-website/forum-config + prose-sanitiser

| Repo/Path | Name | Version | publish | License | Desc/Repo/Readme | crates.io max | Blocking deps | Docs | README | Tests | Classification | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `solid-pod-rs/Cargo.toml` | *(virtual)* | — | — | — | — | — | — | — | — | — | n/a | 8 members |
| `crates/solid-pod-rs` | solid-pod-rs | 0.5.0-alpha.8 | absent | AGPL-3.0-only | Y/Y/Y | **0.5.0-alpha.8** | none blocking (dev-only unversioned path deps, stripped on publish) | Y | Y | Y (58 files, 35 unit) | **PUBLISHED-CURRENT** | |
| `crates/solid-pod-rs-forge` | solid-pod-rs-forge | 0.5.0-alpha.8 | absent | AGPL-3.0-only | Y/Y/Y | 0.5.0-alpha.**7** | none | Y | Y | Y | **PUBLISHED-STALE** | one pre-release behind |
| `crates/solid-pod-rs-git` | solid-pod-rs-git | 0.5.0-alpha.8 | absent | AGPL-3.0-only | Y/Y/Y | 0.5.0-alpha.**7** | none | Y | Y | Y | **PUBLISHED-STALE** | " |
| `crates/solid-pod-rs-nostr` | solid-pod-rs-nostr | 0.5.0-alpha.8 | absent | AGPL-3.0-only | Y/Y/Y | 0.5.0-alpha.**7** | none | Y | Y | Y | **PUBLISHED-STALE** | crypto verdict: verifies NIP-01 events via `k256` Schnorr `verify_raw` (RustCrypto) — wire-format logic, not bespoke crypto; not excluded |
| `crates/solid-pod-rs-activitypub` | solid-pod-rs-activitypub | 0.5.0-alpha.8 | absent | AGPL-3.0-only | Y/Y/Y | 0.5.0-alpha.**7** | none | Y | Y | Y | **PUBLISHED-STALE** | " |
| `crates/solid-pod-rs-idp` | solid-pod-rs-idp | 0.5.0-alpha.8 | absent | AGPL-3.0-only | Y/Y/Y | 0.5.0-alpha.**7** | none | Y | Y | Y | **PUBLISHED-STALE** | " |
| `crates/solid-pod-rs-server` | solid-pod-rs-server | 0.5.0-alpha.8 | absent | AGPL-3.0-only | Y/Y/Y | 0.5.0-alpha.**7** | none | Y | Y | Y | **PUBLISHED-STALE** | generic drop-in server binary, already published |
| `crates/solid-pod-rs-didkey` | solid-pod-rs-didkey | 0.5.0-alpha.8 | absent | AGPL-3.0-only | Y/Y/Y | 0.5.0-alpha.**7** | none | Y | Y | Y | **PUBLISHED-STALE** | crypto verdict: hand-parses compact-JWS envelope but delegates signature verification to `ed25519-dalek`/`p256`/`k256` — standard JWT parsing, not a bespoke envelope; not excluded |
| `crates/solid-pod-rs/fuzz` | solid-pod-rs-fuzz | 0.0.0 | **false** | absent | N/N/N | never published | path dep, no version (irrelevant, publish=false) | N | N | 1 fuzz target | n/a | cargo-fuzz harness only |
| `dreamlab-ai-website/forum-config` | dreamlab-forum-config | 3.0.0-rc11 | **false** | AGPL-3.0-only | Y/N/N (readme file present, no key) | not found | none (all pinned crates.io versions) | Y | Y | Y | PROJECT-SPECIFIC | site-specific branding/config overlay pinning nostr-bbs-* to one deployment |
| `prose-sanitiser/Cargo.toml` | *(virtual)* | — | — | — | — | — | — | — | — | — | n/a | 7 members |
| `crates/core` | prose-sanitiser-core | 0.1.1 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.1.1** | none | Y | Y | Y | **PUBLISHED-CURRENT** | no I/O, no subprocesses |
| `crates/unicode` | prose-sanitiser-unicode | 0.1.1 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.1.1** | none | Y | Y | Y | **PUBLISHED-CURRENT** | invisible-Unicode/homoglyph detector — already the general-purpose crate the brief expected as a candidate |
| `crates/uk` | prose-sanitiser-uk | 0.1.1 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.1.1** | none | Y | Y | Y | **PUBLISHED-CURRENT** | UK-English rules — already published |
| `crates/slop` | prose-sanitiser-slop | 0.1.1 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.1.1** | none | Y | Y | Y | **PUBLISHED-CURRENT** | AI-tell scanner — already published |
| `crates/cli` | prose-sanitiser | 0.1.1 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.1.1** | none | Y | Y | Y | **PUBLISHED-CURRENT** | CLI binaries, published under bare name |
| `crates/server` | prose-sanitiser-server | 0.1.1 | **false** | MIT OR Apache-2.0 | Y/Y/Y | not found | none | Y | Y | Y | PROJECT-SPECIFIC | deployment surface by design |
| `crates/media` | prose-sanitiser-media | 0.1.1 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.1.1** | none | Y | Y | Y | **PUBLISHED-CURRENT** | tracked RUSTSEC-2023-0071 (transitive `rsa` via `c2pa`), doesn't block publishability |

**Slice summary**: 7 PUBLISHED-CURRENT, 7 PUBLISHED-STALE (all solid-pod-rs siblings except the root crate, uniformly one alpha behind), 3 PROJECT-SPECIFIC, 0 CANDIDATE, 0 EXCLUDED. Every crate the brief expected to be a strong candidate (`prose-sanitiser-uk/-unicode/-slop`) is **already published and current** — nothing new to publish there. Both crypto-adjacent solid-pod-rs crates confirmed as legitimate trusted-crate wrappers, not excluded.

---

## 5. diagram-ir, dreamlab-cumbria, WasmVOWL, RuView (wifi-densepose-rs), knowledgeGraph

| Repo/Path | Name | Version | publish | License | Desc/Repo/Readme | crates.io max | Blocking deps | Docs | README | Tests | Classification | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `diagram-ir/Cargo.toml` | diagram-ir | 0.1.0 | absent | MIT OR Apache-2.0 | Y/Y/~ (file, no key) | **0.1.0** | none | Y (186 `///`) | Y | Y | **PUBLISHED-CURRENT** | add explicit `readme=` key for crates.io rendering |
| `dreamlab-cumbria/.../pipeline-rs/Cargo.toml` | *(virtual)* | — | — | — | — | — | — | — | — | — | n/a | 1 member |
| `.../randlehow-pipeline` | randlehow-pipeline | 0.1.0 | **false** | **absent** | Y/N/N | not found | none | Y (631 `///`) | N | Y | PROJECT-SPECIFIC | contains the bespoke WebCrypto AES-GCM/PBKDF2 envelope as a **private module**, doc-commented as never to be offered as a reusable building block — EXCLUDED-BY-POLICY in spirit, not a separate package |
| `.../road/site/Cargo.toml` | randlehow | 0.1.0 | **false** | **absent** | Y/N/N | not found | none | Y (154 `///`) | N | Y | PROJECT-SPECIFIC | single deployed Leptos client; also has a client-side `envelope.rs` WebCrypto wrapper |
| `.../models-rs/Cargo.toml` | *(virtual)* | — | — | MIT (workspace) | — | — | — | — | — | — | n/a | 1 member |
| `.../fairfield-energy` | fairfield-energy | 0.1.0 | **false** | MIT | Y/N/N | not found | none | Y (378 `///`) | N | Y | PROJECT-SPECIFIC | one specific site's hourly energy/battery model |
| `WasmVOWL/rust-wasm/Cargo.toml` | webvowl-wasm | 1.0.0 | absent | MIT | Y/Y/~ (file, no key) | not found | none | Y (238 `///`) | Y | Y | **CANDIDATE** | dry-run **passed** (3 warnings only); **name collision** with knowledgeGraph's `webvowl-wasm` (see below) — rename before publishing either |
| `knowledgeGraph/explorer/rust-wasm/Cargo.toml` | webvowl-wasm | 0.3.4 | absent | MIT | Y/Y (upstream repo, misattributed)/~ | not found (same name) | none | Y (1206 `///`) | Y | Y | **CANDIDATE** | dry-run **passed**; `repository` field wrongly points at upstream VisualDataWeb/WebVOWL project, not this fork; diverged feature set (rayon/wee_alloc) from WasmVOWL's copy — same name collision |
| `.../wifi-densepose-rs/Cargo.toml` | *(virtual)* | — | 0.3.0 (ws) | MIT OR Apache-2.0 | — | — | — | — | — | — | n/a | 15 members (wasm-edge excluded from workspace) |
| `.../patches/ruvector-crv` | ruvector-crv | 0.1.1 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.1.1** (matches) | none | Y | Y | Y | not ours | vendored local copy of an already-published crate, unused (no `[patch]` wiring, not referenced by any member) — stale vendor drop, not original work |
| `.../wifi-densepose-wifiscan` | wifi-densepose-wifiscan | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**2** | none | Y (707 `///`) | Y | Y | REGISTRY-AHEAD | repo behind registry by 0.0.2; self-contained, reusable shape |
| `.../wifi-densepose-ruvector` | wifi-densepose-ruvector | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**3** | none | Y (636 `///`) | Y | Y | REGISTRY-AHEAD | behind by 0.0.3 |
| `.../wifi-densepose-wasm-edge` | wifi-densepose-wasm-edge | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/N | not found | none | Y (2092 `///`) | N | Y (65 tests) | **CANDIDATE** | excluded from workspace, no_std/wasm32-only; dry-run on host target fails as expected (no_std), **passed** with `--target wasm32-unknown-unknown`; needs README; 50+ oddly-prefixed modules suggest scope worth reviewing before treating as one public crate |
| `.../wifi-densepose-sensing-server` | wifi-densepose-sensing-server | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**4** | none (path dep versioned) | Y (760 `///`) | Y | Y | PROJECT-SPECIFIC / REGISTRY-AHEAD | one Axum deployment; also behind registry by 0.0.4 |
| `.../wifi-densepose-cli` | wifi-densepose-cli | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**1** | none (versioned) | Y (103 `///`) | Y | Y | PROJECT-SPECIFIC / REGISTRY-AHEAD | this project's own CLI; behind by 0.0.1 |
| `.../wifi-densepose-db` | wifi-densepose-db | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.3.0** | none (empty deps) | 1-line stub | Y | N | **PUBLISHED-CURRENT** | published crate is itself a stub |
| `.../wifi-densepose-nn` | wifi-densepose-nn | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**2** | none | Y (347 `///`) | Y | Y | REGISTRY-AHEAD | behind by 0.0.2; reusable ONNX/tch/candle inference layer |
| `.../wifi-densepose-mat` | wifi-densepose-mat | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**2** | none (versioned) | Y (2468 `///`, docs.rs metadata set) | Y | Y | REGISTRY-AHEAD | best-documented crate surveyed; behind by 0.0.2 |
| `.../wifi-densepose-hardware` | wifi-densepose-hardware | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**2** | none | Y (480 `///`) | Y | Y | PROJECT-SPECIFIC / REGISTRY-AHEAD | ESP32/Intel 5300/Atheros CSI hardware-specific; behind by 0.0.2 |
| `.../wifi-densepose-train` | wifi-densepose-train | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**3** | none (versioned) | Y (1398 `///`) | Y | Y | REGISTRY-AHEAD | behind by 0.0.3 |
| `.../wifi-densepose-core` | wifi-densepose-core | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**2** | none | Y (569 `///`) | Y | Y | REGISTRY-AHEAD | most clearly reusable crate of the set; behind by 0.0.2 |
| `.../wifi-densepose-wasm` | wifi-densepose-wasm | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**1** | none (versioned) | Y (210 `///`) | Y | N (wasm-bindgen-test dev-dep) | REGISTRY-AHEAD | behind by 0.0.1 |
| `.../wifi-densepose-signal` | wifi-densepose-signal | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**6** | none | Y (1427 `///`) | Y | Y | REGISTRY-AHEAD | largest drift (0.0.6); FFT/CSI signal math, self-contained |
| `.../wifi-densepose-vitals` | wifi-densepose-vitals | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | 0.3.**3** | none | Y (179 `///`) | Y | Y | REGISTRY-AHEAD | behind by 0.0.3 |
| `.../wifi-densepose-api` | wifi-densepose-api | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.3.0** | none (empty deps) | 1-line stub | Y | N | **PUBLISHED-CURRENT** | stub crate |
| `.../wifi-densepose-config` | wifi-densepose-config | 0.3.0 | absent | MIT OR Apache-2.0 | Y/Y/Y | **0.3.0** | none (empty deps) | 1-line stub | Y | N | **PUBLISHED-CURRENT** | stub crate |
| `knowledgeGraph/explorer/rust-wasm` | *(see WasmVOWL row pair above — webvowl-wasm 0.3.4)* | | | | | | | | | | | |

**Slice summary**: 4 PUBLISHED-CURRENT (diagram-ir + 3 wifi-densepose stubs), 10 REGISTRY-AHEAD (systemic — the whole wifi-densepose-rs workspace is pinned at `0.3.0` while individual crates have been released past it by 0.0.1–0.0.6; a process gap, not 10 separate issues), 3 CANDIDATE (2 `webvowl-wasm` copies that **collide on name** and 1 `wifi-densepose-wasm-edge`, wasm-target-only), 6 PROJECT-SPECIFIC, 1 vendored non-original crate (`ruvector-crv`). No standalone EXCLUDED-BY-POLICY package; the `webcrypto-envelope` logic is confirmed as a private, `publish=false`-scoped module inside `randlehow-pipeline` (plus a client-side duplicate in `road/site`) — correctly never exposed as a package.

---

## Top candidates ranked by "closest to a real publish"

1. **`visionclaw-contracts`** (project/crates) — full metadata already present, dry-run clean. Ready now.
2. **`ontology-tools`** (agentbox/services) — full metadata already present, dry-run clean. Ready now.
3. **`loom-domain`** — dry-run clean; publishing it unblocks 6 sibling loom crates in one move. Needs only a readme.
4. **`visionclaw-domain`**, **`visionclaw-analytics-oracle`**, **`visionclaw-xr-presence`**, **`vault-migrate`** — all dry-run clean, need repository/readme metadata only.
5. **`lounge-domain`**, **`lounge-vectorlite`** — dry-run clean, need description/repository/readme.
6. **`scene-effects`**, **`voronoi-graphics`** — dry-run clean once given an isolated workspace; `scene-effects` additionally needs a `license` field before the real registry would accept it (local dry-run only warns).
7. **`webvowl-wasm` × 2** and **`skill-tools`** — dry-run clean/fixable but each has a **name collision** that must be resolved (rename one WasmVOWL fork; rename `skill-tools` to avoid the existing unrelated crate) before either can go live; `skill-tools` also needs its out-of-tree `include_str!` fixed.
8. **`wifi-densepose-wasm-edge`** — dry-run clean on the wasm32 target; needs a README and a scope review (50+ densely-prefixed modules).

## Process items for the owner (not per-crate)

- **7 `solid-pod-rs-*` crates** are one alpha pre-release behind their already-published sibling `solid-pod-rs` (repo 0.5.0-alpha.8, registry alpha.7) — a straightforward re-publish catches them up.
- **2 `nostr-bbs-*` crates** (`-ascii`, `-setup-skill`) are one beta behind for the same reason.
- **10 `wifi-densepose-*` crates** have the opposite problem — the workspace-pinned repo version (0.3.0) is behind what's already on crates.io (0.3.1–0.3.6) — a version-bump process gap on this repo specifically.
- **6 nostr-rust-forum CF-Worker/client crates** are orphaned on crates.io at a stale `1.0.0-beta.2` from before `publish=false` was set on them; consider yanking those old versions.
- **`whelk-rs`** and **`ruvector-crv`** vendored copies are third-party/already-published code sitting in-tree — not this estate's IP, excluded from consideration entirely.
