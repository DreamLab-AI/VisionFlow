# TODO/roadmap register evidence pack — sprint 2026-09-01..2026-09-06 (consolidated)

Consolidates three parallel research passes (per-repo detail in `todo-evidence-agentbox.md`,
`todo-evidence-small-repos-and-panel.md`, and the VisionClaw pass folded into this file).
Verified directly at HEAD in each repo unless marked "(commit-message evidence only)".
Repos: VisionClaw = `/home/devuser/workspace/project` (excl. `agentbox/`); agentbox =
`/home/devuser/workspace/project/agentbox`; VisionFlow = `/home/devuser/workspace/VisionFlow`.

## Headline corrections

1. **The closeout doc's "nothing was committed" caveat is now stale everywhere it was checked.**
   agentbox, VisionClaw, nostr-rust-forum, solid-pod-rs, dreamlab-ai-website, prose-sanitiser and
   (as of this write-up — re-verified live) **VisionFlow itself** all show the described
   2026-09-05 closeout work landed in dated commits (VisionFlow: `3db4785`, working tree clean,
   confirmed by a fresh `git status -s` just now — the `??`/`M` files listed in this session's
   opening git-status snapshot have since been committed by another mesh agent working the same
   estate concurrently). Treat every "uncommitted" claim in the closeout doc as time-of-writing,
   not current.
2. **Two genuine live contradictions surfaced** (see §6) that should block any register edit that
   assumes a clean pass: the RuVector recall-gate numbers, and the dev-auth boot-guard status.
3. **Three rows have their evidence in the wrong repo entirely** (G-7, DOC-4, and the T-4/C-11
   citation drift) — both agentbox and VisionClaw independently mint `ADR-2002`, `ADR-2016`,
   `ADR-2017` for unrelated topics; the register needs repo-qualified IDs.

## 1. docs/TODO-unified.md — 40 open rows + carried entries

| ID | Repo | Verdict | Evidence |
|---|---|---|---|
| M-1 | agentbox | **RESOLVED**\* | `agentbox/docs/developer/hp-peer-node.md`: real probe recorded — `:9096` NIP-98 door HP→ml went 401 `pubkey_not_allowed` → 200 after allowlisting; `:7777` relay ml→HP key correctly rejected by `nostr-pod-bridge` while unlisted. This is exactly the "observation under real traffic" the row was waiting on. *(One pass found no sprint commit touching this doc and called it UNCHANGED — flagged in §6, but two independent passes read the doc content directly and found the recorded results.)* |
| G-5 | agentbox | PARTIAL | `agentbox.toml:117` `[sovereign_mesh.operator].pubkey_hex` still literally the shared visionclaw-server key — the split itself has **not** landed. New `services/secret-backup` crate (age-encrypted, ADR-2027, commit `7905d2a64`) gives custody/rotation tooling but doesn't perform the split. |
| G-6 | agentbox | PARTIAL | `management-api/lib/agent-identity.js` `mint` CLI now fails **closed**, non-zero exit (ADR-2044). But `config/entrypoint-unified.sh:~900-922` still literally preserves `AGENTBOX_AGENT_DID="${AGENTBOX_AGENT_DID:-did:nostr:local}"` on a failed/invalid mint — boot is **not** aborted. The row's literal ask (abort boot) is still open; only the CLI's own exit code was hardened. |
| M-2 | agentbox | UNCHANGED | `agentbox.toml` relay section still `bind=127.0.0.1`, `expose=false`; no rail address (`10.10.10.0/30`) baked in. Citation `flake.nix:2269-2270` has drifted with file growth. |
| T-1 | agentbox | UNCHANGED | `[sovereign_mesh.mobile_bridge].enabled = false`; decision not (re)made this sprint. |
| M-3 | agentbox | PARTIAL | `08e817f39`/`ddd1f1ec8` — real round-3 rebuild + "resolve three rebuild-blocking bugs found by ./agentbox.sh rebuild" on **ml**. No receipt found for the HP repeat — that half is unverified. |
| M-4 | agentbox | PARTIAL | `mcp/nostr-bridge/relay-consumer.js` now has a real `RelayConsumer` class (`subscribe`/`unsubscribe`/`fanout` modes) — the row's "no code consumers" is no longer true. But `AGENTBOX_RELAY_FANOUT` still defaults `'off'` (`management-api/server.js:~1328`, was cited `:1308`) — fan-out itself is still off by default; wiring from `[mesh]` toml keys to the constructor not traced. |
| G-3 | agentbox | **RESOLVED**\*\* | `117f94bf5` shipped `schema/federation-kinds.json` + a JS contract test + a Rust `include_str!` parity test, wired into CI, backed by ADR-2061 (`activation_status: live`). *(A literal grep for the string `sha12` finds nothing — the row's citation terminology is superseded by the new schema-file approach, not evidence the fixture is missing.)* |
| M-5 | agentbox | UNCHANGED | `services/nostr-pod-bridge/Cargo.toml` still path-deps `~/dream-annexe/{nostr-rust-forum,solid-pod-rs}`; confirmed from the other side too — neither nostr-rust-forum nor solid-pod-rs references an annexe sibling at all, this is purely an agentbox-side posture decision, not made. |
| L-6+L-1 | agentbox | UNCHANGED | No mobile e2e or envelope-canary artefacts found this window — inherently a live-session/manual-observation row that a repo grep can't move. |
| G-4 | agentbox | RESOLVED (code) / open (governance) | `config/hooks/nostr-live-mirror.cjs`: `redactForEgress(body)` now runs before send, fail-closed on null (commit `11804ba4b`, "single egress redaction contract, relay admission fix, deepsec gate"). ADR-2026 (the governing record) is still `proposed`/`inactive`, and a stale "fail-open everywhere" doc-comment remains nearby. |
| M-6 | agentbox | **RESOLVED**\* | Same `hp-peer-node.md` evidence as M-1 covers both halves of this row's ask: successful handshake **and** allowlist rejection of an unlisted key. *(Same caveat as M-1 — see §6.)* |
| LM-1 | loom (not agentbox) | PARTIAL | Not resolved in agentbox itself, but the underlying work is real and substantial in `loom` (`c86f462`): `crates/loom-domain/src/grounding.rs` (new), `routes/grounding.rs` (+226), `routes/health.rs` (+53), a 453-line `exp014_grounding_contract.rs` test — implements ADR-138's D1-D4 confidence-surfacing contract. ADR-138 itself is still `Status: Proposed` (not ratified), so treat as implemented-but-unratified rather than fully closed. |
| G-7 | **STALE-CITATION** (ID collision) | UNCHANGED in substance | The row's "ADR-2016/2017" in agentbox are unrelated (Wilson-bound / consumer-ordering memory-learning topics). The real target is **VisionClaw's** `docs/adr/ADR-2016-provenance-append-only.md` / `ADR-2017-write-master-backup-posture.md`, both still `implementation_status: partial` — no backup/erasure path landed. VisionClaw opened 4 new *proposed* ADRs (2102-2105) re-scoping the problem rather than closing it. Agentbox's new `services/secret-backup` (ADR-2027) is secrets custody, not provenance backup, and isn't wired into `flake.nix` yet. |
| G-9 | agentbox | **RESOLVED** | ADR-2070 ratified this sprint (`796d85fcf`): declares Loom raw `:8085` a deliberate, bounded, named egress door — the audit the row asked for. |
| G-10 | agentbox | PARTIAL | ADR-2062 extends the exposure gate to container-internal `0.0.0.0` listeners; literal `0.0.0.0` bind count in `flake.nix` dropped from the row's cited nine to **4**. Raw CDP `9222` sits on a separate sidecar, explicitly out of this gate's scope — no dedicated cross-cutting threat-model doc exists yet. |
| G-11 | VisionClaw | UNCHANGED | Session-bearer sunset still blocked on React client per-request signing; not touched this sprint. |
| D-1 | agentbox | PARTIAL — **see contradiction §6** | `fa024cc08` root-causes the recall FAIL as the non-concurrent-vs-parallel HNSW build bug and ships a serial-rebuild fix, claiming self-recall 189/200 PASS. `LEARNING-memory.md` still doesn't define the observation-window length. **Do not flip `feed_routing` off this evidence alone** — see §6. |
| E-1 | VisionClaw | UNCHANGED | No ComfyUI container on `visionclaw_network`. |
| E-4 | agentbox | UNCHANGED | `lib/nagual-qe.nix` still has no `sqlx`/`SqlSafeStr` version pin — upstream blocker unresolved. |
| DR-1+DR-2 | nostr-rust-forum / VisionFlow | **UNCHANGED — row's premise is wrong** | `486ec5a` (the cited "bench fix... committed") touches feature-gate/identity/D1-migration/trust-sweep work — **zero hits** for `dream.config.json`, any `bench`/`perf` evaluator path. The actual bench commits (`d0bc337`, `f18b471`) **predate** the sprint window entirely. `docs/dream-cycle/LEDGER.md` still ends at the same 7 rows (2026-08-15→08-21, all INCONCLUSIVE) — `/dream` was **not** re-run. No `ADR-0057` exists anywhere in VisionFlow (only a coincidental hash substring). |
| G-1 | VisionClaw | PARTIAL — **see contradiction §6** | New `ADR-2086-ci-gate-dev-auth-excluded-from-release.md` adds a CI gate targeting exactly this hazard. But a direct grep of `src/main.rs:169` still shows `#[cfg(any(debug_assertions, feature = "dev-auth"))] fn enforce_release_env_hygiene() {}` — the no-op stub, verbatim unchanged. ADR-2037 `implementation_status: none`, unchanged. Governance now tracks the gap explicitly; the code-level hazard itself does not appear fixed — confirm before closing. |
| G-2 | VisionClaw | PARTIAL (mostly landed) | `assert_effective_profile_or_exit(...)` confirmed wired into `main.rs` (~line 872-876) **before** `HttpServer::new`/`.bind()` — commit `b0bc275f6`, ADR-2038 now `implementation_status: partial`, `activation_status: live`. New ADR-2043 separately closes the illegal `RBAC_PUBLIC_READS=1` + `PUBKEY_VISIBILITY_FILTER=0` combo. Remaining gap (ADR-2038's own text): an *undeclared/unnamed* profile combination still raises no finding and still binds. |
| G-8 | VisionClaw | UNCHANGED | `src/utils/nip98.rs`: `REPLAY_CACHE_MAX_ENTRIES = 100_000` fail-closed via `ReplayCacheFull`, still TTL/capacity-only — no per-pubkey admission control added. |
| G-13 | VisionClaw | **RESOLVED** | `crates/visionclaw-gpu/src/ptx_loader.rs` now delegates to a single shared, span-based `.version`-rewrite implementation (comment cites "ADR-2030 closeout"), used by both the loader and `build.rs` — the fixed-window splice bug the row named is gone. |
| G-14 | VisionClaw | **UNCHANGED, materially worse** | `supersedes: []` still present on the overwhelming majority of ADR-20xx records — counts ranged 91/92 to 98/101 across two independent recounts (methodology differences, not a real discrepancy: both agree it's >90% empty). Corpus grew from the row's cited 42 records to **92-101** (new `ADR-2043`-`ADR-2105`-ish range from Wave 3/closeout, minted without lineage backfill) — the absolute gap widened. Cross-repo ID collision confirmed still live (agentbox `ADR-2031`/`ADR-2002` vs VisionClaw's own, unrelated `ADR-2031`/`ADR-2002`). |
| G-15 | VisionClaw | UNCHANGED | No status-axis-lattice schema/doc change found. |
| L-2 | VisionClaw | UNCHANGED | No second-model-family diversity canary artefact. |
| L-3 | VisionClaw | UNCHANGED | `docs/archive/adr/ADR-117-server-side-sparql-clamp.md` still archived, no clamp-fire event log. |
| L-4 | VisionClaw | UNCHANGED | `docs/archive/adr/ADR-119-verifiable-liveness-telemetry.md` still archived, no telemetry-fire evidence. |
| L-5 | VisionClaw | UNCHANGED | Closeout XR pass reports "218 local XR library tests pass; this does not certify Godot runtime or deployment" — on-headset XR still not evidenced. |
| T-4 | agentbox / solid-pod-rs | UNCHANGED | solid-pod-rs `40f160c` adds an OIDC compat matrix and **ADR-2003 "defer LWS-1.0"** — documents *why* the issuer stays deferred, doesn't un-defer it. `integrations.solid_pod_rs.enable_mcp = false` unchanged. All 8 rows in agentbox backlog.md's deferred-decisions table unchanged. |
| DOC-1 | VisionClaw | UNCHANGED | `docs/reference/rest-api.md`: broker/workflow mention count reproduces at exactly **34**, matching the row — no edits landed. |
| DOC-2 | VisionClaw | UNCHANGED (drift only) | `__global__` count now **82** (was 83, negligible drift, not a fix); `docs/explanation/actor-hierarchy.md:601` still says "37" — mismatch persists. |
| DOC-3 | VisionClaw | UNCHANGED | `docs/tutorials/installation.md:66-92` still walks a docker-compose v1 binary install; `scripts/launch.sh` uses v2 `docker compose` syntax throughout — mismatch unchanged. |
| DOC-4 | agentbox (**STALE-CITATION**: repo mismatch) | PARTIAL | The row's "ADR-2002" is agentbox's `ADR-2002-aoe-token-auth-boundary.md`, not VisionClaw's (VisionClaw's own ADR-2002 is the unrelated nip98 replay-cache record — same G-DEFECT-2 collision as G-7). Agentbox's ADR-2002 got a fresh `verified_commit` (`796d85fcf`) this sprint but is **still `activation_status: staged`** — the row's actual ask (flip to active, reconcile with AB-2009) is still open. |
| C-11 | VisionClaw / agentbox | **STALE-CITATION** (counts), substance unchanged | `project`: now **10 local branches / 7 worktrees** (row cited 27/24). `agentbox`: 10 local / 50 total. The three named branches (`refactor/kg-node-rename`, `report/soundings-qe-audit`, `impl/khive-investigation`) confirmed **remote-only** — matches "local-deleted, remote-kept" description exactly; the remote keep/drop decision itself is still open. |
| V-1 | visionGraph (neither repo checked) | UNVERIFIED | No `DUPLICATE_IRI` found via grep in this pass; visionGraph has active September commits consistent with ongoing work, but the specific pipeline-red artefact wasn't independently located. |
| V-2 | VisionClaw | PARTIAL | Core fix confirmed: commit `9423abdb3` (2026-09-02 12:28Z) "anchor the isolated-node peripheral shell to the configured bounds, not the live AABB" — matches the row almost verbatim. Named follow-ups (`integrate_pass_kernel` hard clamp, connected-only AABB) **not found** — still open. |
| V-3 | VisionClaw | UNCHANGED | `nip98.rs` `TokenReplayed` mechanism unchanged; no route-specific fix for double-verification on `POST /api/bots/update`. |
| V-4 | VisionClaw | PARTIAL (+ stale line citation) | The self-contradictory `dag_rank_tests` assertion is addressed via ADR-2035 (per one pass); a related-but-distinct fix (`8e0ee41bb`, percent-encoding repo paths) also landed 2026-09-02. `GitHubClient::get_full_path`'s own prefix heuristic is **unchanged**. Citation line drifted 4563→~4655-4700 as the file grew (confirm exact current state before closing). |
| V-5 | visionGraph | UNVERIFIED | Not checked this pass — requires inspecting rendered diagram PNGs/note file content, out of repo-grep scope. |
| **Frozen (§7)** | — | Not independently checked | ADR-073..085 window, ADR-122/123, RVF file store, XR APK cross-build — out of scope this pass; frozen by design regardless. |

\* M-1/M-6 verdict has a genuine cross-pass discrepancy — see §6.
\*\* G-3 verdict depends on accepting ADR-2061's schema-file approach as satisfying the row's literal "sha12" ask — see §6.

## 2. agentbox/docs/developer/backlog.md

No independent open rows beyond TODO-unified.md; confirmed still-open at HEAD:
- **Deferred operator decisions** table (relay exposure, mobile bridge, multi-user, git pods/gateway,
  payments, Solid OIDC, pod MCP surface, kernel pip) — all 8 rows present unchanged, = T-4.
- **External blockers** table (ComfyUI, Ollama sidecar, Nagual QE sqlx pin) unchanged, = E-1/E-4.
  **KG elevation** (`visionclaw-server:4000` unreachable) is also still listed here but has **no
  corresponding TODO-unified.md row at all** — register gap, see §5.
- "Done" section entries confirmed accurate historically; no regressions found.

## 3. VisionFlow/docs/roadmap.md (2026-05-22, Phases 0-3)

Stale/superseded planning doc. `git log --since=2026-09-01 -- docs/roadmap.md` is empty — untouched
directly, but its intent has been overtaken:
- **Phase 0** (honesty/traceability): superseded by the entire estate-review/closeout corpus
  (`ADR-2007-estate-closeout-evidence-roadmap.md` + `docs/estate-review/`), now committed at `3db4785`.
  VisionFlow's own sprint work (drift counter repaired, release roster 6→14, harness audit
  de-duplicated 82.5%→79.5%) lands the same intent.
- **Phase 1** (mesh contract — IS-Envelope, event-kind registry, DID service fields, NIP status):
  **unchanged** — the federation-identifier work that did land (agentbox ADR-2025/2061 protocol
  registry) is adjacent but doesn't touch DID document service fields or a canonical envelope owner.
- **Phase 2** (end-to-end proof): partially advanced — M-1/M-6 evidence is a real agentbox↔agentbox
  (ml↔HP) smoke test, not the full agentbox→relay→forum→VisionClaw chain the roadmap specifies; the
  cross-substrate fixture sync gate maps directly onto still-open G-3-style concerns.
- **Phase 3** (operational readiness): release-manifest work advanced (roster 6→14 with provenance);
  pod-tier migration, unified health dashboard, backup/DR runbooks — **no evidence found, unchanged**.

## 4. docs/ROADMAP-consultant-panel-2026-08-31.md — 18 ranked actions vs. sprint

| # | Action | Status this sprint |
|---|---|---|
| 1 | Arm staleness gate (verified_paths + full SHA + CI) | PARTIAL — many records now carry full 40-char `verified_commit`/`verified_paths` (e.g. ADR-2038, ADR-2002); no confirmation CI now *rejects* short SHAs/empty paths. |
| 2 | Govern dev-auth build flag | PARTIAL — new ADR-2086 CI gate; code-level stub (`main.rs:169`) unchanged — see §6. |
| 3 | Close/deadline staged AoE token gap | UNCHANGED — agentbox ADR-2002 still `staged` (= DOC-4). |
| 4 | Boot-time profile selector + illegal-combo abort | **RESOLVED** — see G-2. |
| 5 | Reconcile single-boundary vs nine sanctioned doors | PARTIAL — `0.0.0.0` binds down to 4 (G-10); no unified threat-model doc. |
| 6 | Govern session-mirror cloud egress | **RESOLVED** (code) — see G-4; ADR-2026 governance still pending. |
| 7 | Provenance durability + estate erasure | UNCHANGED — VisionClaw ADR-2016/2017 still partial (= G-7). |
| 8 | Cross-repo federation contract + CI | **RESOLVED** — agentbox `117f94bf5` (= G-3). |
| 9 | Sunset/harden replayable session bearers | UNVERIFIED this pass (= G-11). |
| 10 | Replay-cache DoS mitigation | UNCHANGED — no per-pubkey admission (= G-8). |
| 11 | Key custody/rotation/break-glass records | PARTIAL — `services/secret-backup` (ADR-2027) landed, not wired into `flake.nix`; publisher-key split itself unverified (= G-5). |
| 12 | Kill `did:nostr:local` fail-open | PARTIAL — CLI hardened, entrypoint shell still fail-opens (= G-6). |
| 13 | Audit/authenticate Loom `:8084` door | **RESOLVED** — ADR-2070 (= G-9). |
| 14 | Fix complete-with-defect records (node-ID guard, PTX, doc-comment) | **RESOLVED** for PTX (= G-13); node-ID guard/VC-2035 doc-comment not independently re-checked. |
| 15 | Status-axis coherence lattice | UNCHANGED (= G-15). |
| 16 | Type + populate supersession; namespace IDs | UNCHANGED, **materially worse** (= G-14). |
| 17 | Tombstone 2031; split bundled records; govern living-doc invariants | Not independently re-checked at file level; closeout doc's XR section references "ADR-2031 disposed as a preserved tombstone" for VisionClaw's own 2031 — the cross-repo collision (agentbox has a different ADR-2031) is unresolved regardless. |
| 18 | Shared generator/CI parity across repos | UNVERIFIED. |

**Tally:** 5 RESOLVED (#4, #6, #8, #13, #14-PTX-only), 6 PARTIAL, 4 UNCHANGED (verified), 3 UNVERIFIED this pass.

## 5. VisionFlow closeout roadmap (CP-01..09) — open gates

All nine packages remain open by the doc's own exit criteria ("The programme remains open until
these conditions hold across the complete estate"). This material is now **committed** (`3db4785`,
clean working tree, re-verified live). Per-package headline gap:

- **CP-01** Decision & release identity — 509 of 540 decision candidates still lack the dated
  evidence-backed closeout marker.
- **CP-02** Corpus/semantic publication — WasmVOWL explorer has "failing frontend state
  initialisation and schema mismatch despite passing native Rust tests."
- **CP-03** Grounded execution (Loom) — serving activation vs. library tests still distinguished,
  not closed; production Loom corpus lacks an `embeddingModel` stamp, HP serves split
  lexical/semantic generations (escalated finding).
- **CP-04** Identity/authority/storage — solid-pod-rs "does not validate token `aud` and silently
  substitutes `sub` for a malformed `webid`" (escalated finding); session-lifecycle gaps carried
  from G-11.
- **CP-05** Human judgement/governance — ACSP case correlation, forum receipt/recovery need "a
  governance receipt decision, archive dispositions and browser/deployment journeys."
- **CP-06** Runtime/embodied interaction — no headset/browser acceptance run ("not run — dev stack
  not running; forbidden in-container" for VisionClaw); XR hierarchy ADR-2033 partial, one test
  fails against current acceptance criteria.
- **CP-07** Memory/improvement — **open with a live regression**: the recall-gate contradiction in
  §6 below is exactly this package's blocking issue.
- **CP-08** Delivery/recovery — prose-sanitiser 0.1.2 publication still pending.
- **CP-09** Complete-system acceptance — open by construction, depends on CP-01..08, not started.

Doc's own "Still open after this pass" line (verbatim): "everything that needs hardware, a headset,
hosted CI, production deployments or a human signer; the WasmVOWL consumer stall; the RuView records
classified evidence-review-required; publication of prose-sanitiser 0.1.2."

## 6. Contradictions and flags for owner decision (read before editing the register)

1. **RuVector recall-gate contradiction — blocks D-1.** Agentbox's committed fix (`fa024cc08`,
   `docs/LEARNING-memory.md`) claims the HNSW parallel-build bug is root-caused and a serial
   rebuild restores a PASS (self-recall 189/200, true-recall 115/120). The closeout doc's later
   "Learning / memory lane addendum" reports a protocol-conformant run against the **deployed**
   corpus giving self-recall **164/200** (band ≥175) and true-recall **96/120** (band ≥102) — worse
   than the original failure — and the new gate **refuses (exit 3)**. These cannot both describe
   current production state. Re-run `./agentbox.sh ruvector recall` against the live corpus before
   trusting either number or touching `feed_routing` (D-1). Matches this session's own CLAUDE.md
   guidance that a parallel HNSW rebuild leaves ~20% of rows unreachable — the fix may not have been
   applied to the currently-deployed index yet.
2. **G-1 / dev-auth stub contradiction.** One evidence pass reads the new `ADR-2086` CI gate as
   closing the hazard; a direct grep of `src/main.rs:169` at HEAD shows the `#[cfg(any(debug_assertions,
   feature = "dev-auth"))] fn enforce_release_env_hygiene() {}` no-op stub **verbatim unchanged**.
   The CI gate is real and new; the code-level hazard it's meant to catch does not look fixed.
   Someone should confirm which is true before marking G-1 resolved.
3. **M-1/M-6 discrepancy.** Two independent passes read `agentbox/docs/developer/hp-peer-node.md`
   directly and found recorded real-traffic test results (401→200 handshake, allowlist rejection) —
   both rows' literal asks. A third pass, checking only `git log --stat` on the named sprint
   commits, found no commit touching that path and called both rows unchanged. Likely explanation:
   the doc was updated in a commit outside the named commit list (or by another concurrent mesh
   agent, as happened with VisionFlow's closeout docs — see Headline corrections). Spot-check
   `git -C agentbox log -- docs/developer/hp-peer-node.md` before closing these rows.
4. **G-3 terminology drift.** The row's citation grep-target ("sha12") no longer appears in the
   codebase; the actual cross-repo identity-parity mechanism shipped under ADR-2061's schema-file
   approach instead. Recommend rewording the row rather than treating it as unresolved.
5. **VisionFlow's own "nothing committed" status flipped mid-session.** The session's opening git
   status showed dozens of `M`/`??` files in VisionFlow (estate-review, estate-closeout, ADR-2007).
   A fresh check just now shows a clean working tree at `3db4785` — another mesh agent committed
   this work while this research pass was running. Any register edit should re-check current state
   immediately before landing, not rely on either snapshot.

## New items found (not currently tracked as TODO-unified.md rows)

1. **RuVector recall-gate regression** (see §6.1) — a live, currently-failing gate, more urgent
   than D-1's dormant flag; deserves its own row or an explicit D-1 rewrite blocking on it.
2. **KG-elevation blocker** (`visionclaw-server:4000` unreachable) — present in agentbox
   backlog.md's external-blockers table, absent from TODO-unified.md entirely.
3. **New unfixed GPU defect**: VisionClaw's analytics LOF kernel fails 702× against a CPU oracle on
   an A6000 (documented in a sprint commit message, not fixed).
4. **`services/secret-backup` (agentbox, ADR-2027)** built and tested but not wired into
   `flake.nix` — real capability outside the reproducible build.
5. **dreamlab-ai-website commit `000a40e`** — substantial new checker/pin-parity/chat-turns work,
   distinct from the already-resolved PR #49 kit-pin-guard row, not cited by any current TODO row.
6. **Escalated findings from the closeout mesh pass**, each a candidate row: nostr-rust-forum badge
   panel "defeated by the relay client firing EOSE while disconnected"; VisionClaw's browser decoder
   "auto-detected V2 on foreign opcodes" (guarded, but noted); WasmVOWL "`/graph` requests neither
   WASM nor the binary tier"; agentbox's "baked dream evaluators now all read as `required`"
   (config-drift risk).
7. **VisionClaw ADR corpus growth 59→92-101 records** minted this sprint without lineage backfill —
   makes G-14 proportionally worse; worth noting explicitly if G-14 stays open.
8. **Loom ADR-138 confidence-surfacing contract is implemented and tested but still `Status:
   Proposed`** — ready for ratification, which would let LM-1 close cleanly.
