# VisionClaw closeout execution — 2026-09-07

This records implementation and executable acceptance after the estate audit. It does not claim that the running VisionClaw service has been rebuilt or that headset/phone acceptance occurred. Implementation `1ad881cab`, federation follow-up `e29911405`, and documentation/workflow follow-ups `5350edbb6` / `81929f1f3` / `dd82a07b0` are pushed to `dreamlab-github/main`.

## Results by allocated register ID

| ID | Disposition | Implemented or verified result; exact remaining boundary |
|---|---|---|
| EA-02 | Implemented generation boundary; live rollout pending | Actual `PublishedStorage` wraps the pinned FsBackend: stage all five immutable resources/sidecars, fsync, then atomically replace the active pointer. The manifest exposes a generation ID; the browser schema parser pins JSON-LD and Turtle URLs to it. Canonical ACLs also govern pinned URLs. Tests cover all eight pre-activation failures, restart, concurrent activation/pinned reads, traversal and symlink rejection, and an actual Solid-handler private-ACL refusal. Canonical unpinned clients can straddle activation; post-rename root-fsync failure leaves complete content but uncertain durable activation. Old generations are retained; GC/retention and deployed rollout remain. |
| EA-06 | Coordinator implemented; destructive adapters and acceptance remain | `data_reconciliation.rs` supplies explicit effective-store membership (including optional Redis/pod/payments), durable SQLite operation IDs, per-store receipts and retry. Membership/adapter disagreement refuses before work. Injected receipt failure proves retry uses the same operation ID; backends must implement idempotency. No production erase adapters or subject selection are wired, so no private erasure/restoration is claimed. This remaining adapter work is an implementation gap, not an unavailable external dependency. |
| G-1 | Boot gap fixed; actual release negative probes pass | All non-debug builds now reject findings, including release/dev-auth. A real default-feature release artefact rejects `SETTINGS_AUTH_BYPASS=0` and `ALLOW_INSECURE_DEFAULTS=0` with exit 2 before startup; its bytes contain no dev-session-token marker. The first no-default binary build exposes an unrelated pre-existing ungated GPU import. The tested release artefact predates the later storage changes; receipt records its exact hash. Every produced deployment image and ADR ratification remain separate acceptance. |
| G-2 | Implemented and unit-tested; deployment pending | Missing declared profile produces `MissingDeclaredProfile`; no supported flag match produces `UnnamedEffectiveProfile`. Non-debug artefacts cannot bind either case. The existing accepted profiles and illegal public-read/disclosure pair remain covered. |
| G-7 | Checkpoint and recovery boundary implemented; operational recovery remains | The actual open Oxigraph writer supports opt-in `ONTOLOGY_BACKUP_DIR` checkpointing outside its data tree, failing configured startup on checkpoint failure. An on-disk test inserts authored provenance, backs up, removes the fixture source, and reopens the checkpoint with the authored quad intact. This uses the database backup API, not a live directory copy. Cross-store erasure/restore adapters remain outstanding under EA-06. Off-device retention and restoration of a selected deployed snapshot have not been exercised. |
| G-8 | Implemented and unit-tested | Authenticated signers may consume at most 1,000 events per 120-second replay window before the 100,000 global-entry ceiling. Admission occurs after signature/request verification under the same lock as the global claim. It never evicts live event IDs. Tests verify signer fairness, retained replay entries and expiry. This is process-local admission, not Sybil resistance or cross-replica replay protection. |
| G-11 | Signed graph upgrade and default-off sessions implemented; other transport rollout remains | REST signing already existed. The graph browser now signs the public HTTP GET upgrade URL, offers a base64url NIP-98 subprotocol, and aborts before socket construction for absent/declined signing. Server validates URL/method/signature/replay before upgrade and negotiates only public protocols, never echoing the token. Fresh per-request nonces prevent same-second signature collisions. Legacy session issuance/validation/refresh default off; explicit `VISIONCLAW_LEGACY_SESSIONS=1` restores only real unexpired sessions for migration. Remaining MCP/session clients and deployed proxy/reconnect acceptance are not certified. |
| G-16 | Original numerical defect closed on the governed GPU fixture | The actual A6000 test first reproduced the old failure. Correct neighbour k-distance alone reduced the error to 0.071 but still failed. Including kth-distance ties then passes the original 1e-3 bar at max delta 4.759e-7, with identical >95th-percentile set. All four actual GPU oracle tests pass, zero skips. The test's old diagnostic that intentionally required the broken formula was removed; numerical acceptance was not loosened. Neighbour buffers remain capped at 32 and searches are radius/grid scoped. Extra neighbourhood recomputation preserves the launch ABI but requires production-scale latency measurement before claiming performance parity. |
| V-2 | Both source follow-ups implemented; deployed observation remains | Actual CUDA execution proves positive/negative hard clamps, stopped outward velocity and preserved pinned positions. A separate connected-node AABB excludes isolated outliers and is consumed by the unbounded shell-radius path. The all-node AABB still bounds the spatial index. Actual CUDA tests verify both extents and the no-connected-node empty result. No running scene/headset acceptance is claimed. |
| V-3 | Implemented and regression-tested | Settings authentication reuses the server-side identity extension populated by middleware, preserving the stored user's privilege bit. A genuinely signed `/api/bots/update` token can be consumed by middleware then extracted in that request; presenting it in a new request still fails as replay. No client header is trusted as an identity extension. |
| V-4 | Remaining prefix defect fixed and tested | `GitHubClient::get_full_path` accepts an existing base only at an exact path or directory boundary. `knowledge/pages-extra` no longer accidentally matches `knowledge/pages`. Tests also preserve literal percent filenames and root/empty inputs. The earlier DAG assertion and URL percent-encoding fixes remain intact. |
| DOC-1 | Stale premise reconciled | Broker/workflow mention count was not a count of phantom live endpoints. Broker inbox/case/decide routes are registered in `broker_inbox_handler::configure_routes`; workflow routes are already explicitly design-stage/unregistered. Added a dated route audit to the reference rather than deleting real API documentation. |
| DOC-2 | Corrected | Removed the two stale literal kernel counts and the unsubstantiated 4 ms dispatch claim from `actor-hierarchy.md`. Pointed the count to the actual CUDA source/build scope; preserved other authors' dirty edits. |
| DOC-3 | Corrected | Installation now uses the Docker Compose v2 plugin and `docker compose version`, matching `scripts/launch.sh`. No package installation was performed on the host. |
| G-24 | Root causes triaged; hosted rerun remains required | Original documentation CI failed seven ADR validation errors, not documentation-quality scoring. The current local ADR validator passes. Original XR run had Rust success, cancelled GUT and advisory APK failure: templates were copied without their version directory and the project Gradle template was absent. Workflow setup now installs both paths, preserving root's action SHA pins. No new hosted APK or GUT success is claimed. |
| L-2 | External live model dependency | A second independently served model family and its diversity canary traffic are not available through this VisionClaw source checkout. A GPU being visible does not prove a second model service is loaded. No synthetic model-family evidence was created. |
| L-3 | Live service unavailable | Read-only `http://visionclaw-server:4000/api/health` fails DNS resolution in this container (curl exit 6). Source clamp and materialised result caps are present; a local unit fixture does not prove the deployed clamp fired. A reachable deployed ontology door and its event receipt remain required. |
| L-4 | Live sink evidence found; full failure-channel acceptance remains | The real `/var/lib/agentbox/telemetry/ontology-retrieval.jsonl` contains 28 records: 16 canaries and 12 cache hits, latest 2026-09-06T22:02:54.416Z. This disproves the board's absolute “no fire evidence” premise. There are no `fail_open` records in this receipt; do not claim that channel was observed. Only aggregate event metadata was copied. |
| L-5 / X-6 | Headset acceptance blocked by unavailable runtime/target | 229 local XR Rust library tests pass. `godot` and `adb` do not resolve locally, and no headset session or target APK installation was performed. Desktop GPU access is not a headset/compositor/controller receipt. |
| Frozen XR APK | Held target acceptance remains | The CI template path repair addresses a concrete export failure, not release signing, installed APK execution, GUT completion or on-headset stereo acceptance. No frozen architecture scope or RuView profile was silently activated. |

## Executable evidence

Evidence lives in [execution-2026-09-07](../evidence/execution-2026-09-07/). Exact commands:

- Server: `cargo test --locked --offline --lib --no-default-features --features solid-pod-embed -- --test-threads=1` — **1,364 passed, six ignored** (before the final handshake echo test and federation follow-up). Includes new ACL, authentication reuse/replay, profile, path-boundary and signer-admission tests.
- GPU: `cargo test --locked --offline -p visionclaw-gpu --test analytics_oracle_conformance -- --ignored --nocapture --test-threads=1` — **four passed, zero skipped/ignored**, after preserving the failing baseline and formula-only intermediate receipt.
- Hard clamp: `nvcc -std=c++17 tests/gpu/integrate_bounds.cu -o target/vc-integrate-bounds && target/vc-integrate-bounds` — actual CUDA success for all three axes and pinned-node preservation.
- Client: `npm test -- --run src/services/api/__tests__/authInterceptor.test.ts` — **11 passed**; an additional 49 signer/generation tests pass in `vc-client-sunset.log`.
- XR: from `xr-client/rust`, `cargo test --locked --offline --lib --all-features` — **229 passed**; no Godot scene/headset claim.
- ADR: `node scripts/adr-index-gen.js docs/adr --check` — local declaration/provenance gate passes before the implementation commit; re-verification after that commit is recorded separately.

The release artefact identity and its two exit-2 negative probes are recorded in `vc-release-probes.json`; final source/index/render receipts are recorded alongside it. No user data was erased, credentials rotated, service restarted or remote storage restored during these tests.


## Final source and runtime qualification

The default-feature `visionclaw-server` target passes `cargo check --locked --offline`.
The final focused handshake suite passes three tests: direct URL, forwarded public
URL and no authentication-token echo. The federation lane later ran the combined
suite at `e29911405`: 1,369 passed, six ignored (its separate receipt is authoritative
for that combined snapshot). Client signing/generation tests pass 49 cases.
The current-source VisionClaw citation gate passes 35 topics / 439 diagrams with
zero warnings; changed render receipts are recorded separately.

L-3 / E-5 recovery was investigated beyond a failed DNS probe. Docker has no active
or stopped VisionClaw container. The retained production image is
`sha256:9d32decdec09a35f4f9a2bdb5e01a8d17986db550d7fe8cf0636eec7815313ef`,
created 2026-08-26. Its inspected `/app/start.sh` starts Rust on loopback port 4001
behind nginx port 3001. Agentbox's ontology bridge defaults to
`http://visionclaw-server:4000`, a concrete deployment-port mismatch. Existing
`visionclaw-data`/`visionclaw-logs` volumes and `visionclaw_network` survive, but
no declared security profile or owner identity accompanies the available environment.
Required key variables exist; their values were not copied. Activating this old
writer over retained data would neither test the current implementation nor establish
its intended authority. Root accepted this as a concrete blocked activation; no
old service, private data writer or Agentbox session was restarted. A fresh synthetic
service would not close the deployed KG/clamp requirement.

## EA-06 adapter and subject-authority boundary

The journal is executable coordination infrastructure; it has no production
`ReconciliationBackend` implementation or live destructive caller. These are the
specific unresolved mappings, rather than a claim that synthetic adapters complete
erasure:

| Selected store | Concrete authority/subject issue still required |
|---|---|
| Oxigraph | `ADR-2016` forbids destructive provenance graph updates. A user's key, authored entities, attributed activities and referenced objects are different subjects. Redaction/crypto-shred policy must specify what is erased while retaining authorised audit facts. The new checkpoint test addresses recovery, not erasure. |
| SQLite enrichment | `enrichment_proposals.case_id/source_iri` and `enrichment_decisions.broker_pubkey` are separate identities. Removing rows by broker key would also remove governed decisions about other subjects. Case ownership and receipt retention must be defined. |
| SQLite roles/settings | `RoleStore` keys canonical pubkeys and bootstraps an Owner from environment; deleting a role row can be undone at restart. Settings can be global. Erasure must specify principal revocation versus shared settings and owner replacement. |
| SQLite KPI/liveness | Aggregate/event observations are not uniformly keyed by an erasure subject. Attribution retention and derived aggregate recomputation are not implemented by the journal. |
| Nostr memory/optional Redis | In-memory users and Redis token→pubkey, user-data and pubkey→token keys need one revocation generation plus restart rules. Default-off legacy validation reduces acceptance but does not erase stored records. |
| Solid pod | Public ontology generations, personal pod paths and canonical ACLs have different owners. The selected subject must map to owned resources without deleting shared ontology or another user's references. Retained generations also require erasure-aware retention. |
| GitHub authored content | Current tree edits do not erase Git history, forks or release assets. Repository/path ownership and a history policy are required; no history rewrite was attempted. |
| External agent memory | The 384-dimensional MCP memory service is outside the local Rust transaction. Its subject selector, deletion receipt and restore authority must be provided by its owning API. |
| Credential custody | Revocation/rotation and backups have operator-owned custody. Deleting a local reference does not revoke a key or erase sealed backup copies. No credentials were rotated. |
| Optional payment ledger/exchange | Accounting events and payment identifiers require explicit retention/subject rules and external provider receipts. No accounting deletion adapter was fabricated. |

The selected-store manifest deliberately includes these surfaces even where an
adapter is missing, so reconciliation refuses rather than silently dropping them.
Actual adapter implementation and the above authority decisions remain roadmap work.


The release negatives were also repeated with all six flags matching an explicitly
declared `demo-open` profile. Both forbidden variables set to `0` still return
exit 2 from `enforce_release_env_hygiene`, naming the offending variable. The
receipt `vc-release-valid-profile-probes.json` preserves this isolated negative
configuration; no listener or private data writer starts. This avoids relying on
an undeclared profile as an unrelated refusal reason.


## Final render and declaration receipts

Twenty changed VisionClaw topics contain 254 diagrams. The first render passed
252 and exposed two Mermaid semicolon errors in VC-03; the final VC-03 rerender
passes 17/17, closing both failures. ES-04 passes 6/6. ES-08's analogous note-text
parse error was corrected; final ES-08 and ES-10 renders pass 11/11 and 10/10.
The earlier failure receipts remain alongside their replacements:
`vc-changed-render.log`, `vc-es-final-render2.log`, and
`es08-es10-final-render.log`. Thus all 281 diagrams in the changed VC topics and
three estate topics have passing final render receipts.

Both current-source and revision-pinned VisionClaw citation checks pass all
35 topics / 439 diagrams with zero diagnostics. Changed topic stamps use
`dd82a07b0`, which includes the final workflow edits. The final local ADR gate
passes 99 records. ADR-2061 is complete/staged for the four measured kernel
fixtures; ADR-2037 remains proposed with partial/staged implementation. Root
owns the whole-estate final snapshot and master-register status decisions.
