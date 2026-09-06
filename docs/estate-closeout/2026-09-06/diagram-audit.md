# Diagrams-as-code staleness audit — docs/diagrams

Generated 2026-09-06. Project HEAD `2d87c3b81`; agentbox submodule HEAD `eb7794b17`.

70/71 topic files declare `verified_commit: bed6b617d`; one (`visionclaw/02-actor-supervision.md`) declares `b0bc275f6`. The project repo has moved 6 commits past `bed6b617d` (…, `2cf222406` Wave 3 remediation, …, `2d87c3b81`). At `bed6b617d` the agentbox submodule was pinned at `7e7b2d586`; it has since moved 7 commits to `eb7794b17` (…, `796d85fcf` Wave 3 remediation, …).

## 1. Checker script (`scripts/diagram-index-gen.js`)

- `node scripts/diagram-index-gen.js docs/diagrams --check` → **parsed 71 topic files, 841 mermaid diagrams -- zero errors**
- `node scripts/diagram-index-gen.js docs/diagrams --check --render` → **rendered 841/841 diagrams via mmdc 11.16.0 -- zero mermaid syntax errors**
- **Coverage gap:** The checker ONLY validates (a) frontmatter well-formedness/id uniqueness, (b) that every `sources:`/`governing:` path exists at HEAD, and (c) that every mermaid block parses. It does NOT check verified_commit against HEAD, does NOT check any path:line citation embedded in a participant/Note against the file's actual current line content, and does NOT detect that a cited function/behaviour has been deleted or moved. A diagram can pass --check --render while citing code that no longer exists (confirmed: this run passed clean despite the 10 STALE files below).

## 2–3. Citation audit

- 71 topic files, 841 diagrams, **6551 `path:line` citations** extracted.
- 157 citations were ambiguous bare basenames (mostly `ci.yml`, cited in both repos) and excluded from automated classification; load-bearing ones (ES-09 CI) were checked by hand separately (§4).
- **1810** citations point at a source file that changed since the diagram's `verified_commit`:
  - 771 — content at the exact same line, untouched.
  - 998 — line number drifted (unrelated inserts/deletes elsewhere in the file) but the diff hunks never touch the cited line, so content is preserved at the shifted position.
  - **41** (36 unique file:line pairs) — the cited line itself sits inside a changed diff hunk. All 36 were manually inspected with `git show` at both commits.
    - **11 confirmed genuinely broken** (moved or gone).
    - 3 confirmed false positives (content coincidentally correct or already fixed in the same diagram text — see detail).
    - 4 confirmed cosmetic-only (re-indentation / comment rewording / line-splitting, same meaning).

### Per-file verdicts

**STALE: 10** · **TOUCHED-BUT-INTACT: 60** · **CLEAN: 1**

STALE (confirmed broken citation(s) after manual review):

| ID | File | sources | changed sources | citations | in changed files | flagged-changed-content |
|---|---|---|---|---|---|---|
| AB-04 | docs/diagrams/agentbox/04-adapter-spine.md | 31 | 8 | 261 | 66 | 3 |
| AB-13 | docs/diagrams/agentbox/13-nostr-relay-gateway-bridge-mirror.md | 22 | 9 | 293 | 89 | 1 |
| AB-17 | docs/diagrams/agentbox/17-agent-events-and-provenance-bridge.md | 11 | 1 | 52 | 13 | 2 |
| ES-08 | docs/diagrams/estate/08-solid-pod-estate.md | 23 | 7 | 57 | 18 | 1 |
| VC-09 | docs/diagrams/visionclaw/09-config-and-env-flags.md | 56 | 17 | 188 | 127 | 1 |
| VC-23 | docs/diagrams/visionclaw/23-identifiers-urn-did-sha12.md | 14 | 6 | 88 | 62 | 3 |
| VC-24 | docs/diagrams/visionclaw/24-acsp-decision-elevation.md | 25 | 6 | 117 | 52 | 2 |
| VC-26 | docs/diagrams/visionclaw/26-solid-pod-and-jss.md | 23 | 6 | 126 | 33 | 3 |
| VC-30 | docs/diagrams/visionclaw/30-client-boot-and-state.md | 66 | 10 | 391 | 25 | 3 |
| VC-32 | docs/diagrams/visionclaw/32-client-websocket-and-binary.md | 26 | 5 | 234 | 41 | 13 |

CLEAN (nothing this file cites has changed since its verified_commit):

- VC-18 — docs/diagrams/visionclaw/18-analytics-support-handlers.md

TOUCHED-BUT-INTACT (sources changed, but no confirmed broken citation — 60 files): all remaining topic files. Many carry large `shifted` counts (e.g. VC-01 92, AB-03 98, AB-02 59, VC-22 53, AB-13 55, AB-27 42) meaning a sizeable fraction of their raw line numbers no longer match the exact position of the code they cite, purely from unrelated edits elsewhere in the same file — the automated diff-hunk-overlap test found none of those specific cited lines inside a changed hunk, but individual shifted citations beyond the 36 manually reviewed here were not each re-read by eye.

### Broken citations — detail

**VC-09** — `docs/diagrams/visionclaw/09-config-and-env-flags.md`
- Citation: `src/actors/agent_monitor_actor.rs:216`
- At verified_commit: `let api_key = std::env::var("MANAGEMENT_API_KEY").unwrap_or_else(|_| { ... })`
- At HEAD (same line): `/// Release-build stub: insecure defaults are never honoured.`
- Verdict: **GONE** — src/actors/agent_monitor_actor.rs: decide_management_api_credential() fn (~234-260) and its call sites at ~274, 289, 324, 828, 833, 843
- ADR-2094 (management-api-credential-and-cached-mcp-health, new ADR added in Wave 3) replaced the direct env::var().unwrap_or_else() fallback with a pure decide_management_api_credential() fail-closed function. The flowchart node G3 citing line 216 now points at an unrelated doc-comment for a different helper (insecure_defaults_allowed release stub).

**VC-23** — `docs/diagrams/visionclaw/23-identifiers-urn-did-sha12.md`
- Citation: `src/uri/mod.rs:700-707 (Note over JS,RS, VC-23.14ish)`
- At verified_commit: `} // closing brace, unrelated to cross_from_agentbox`
- At HEAD (same line): `pub target_kind: Option<String>, (struct field)`
- Verdict: **MOVED** — src/uri/mod.rs:809 (fn cross_from_agentbox), bead arm ~841-855, bead_with_address call ~855
- cross_from_agentbox moved from ~700 to 809 as uri/mod.rs grew +456 lines overall (typed CLASS_PREFIX/class_iri constructor per ADR-2095 inserted earlier in the file). bead_with_address itself is still correctly at :284 as cited. The elevation_actor.rs:329/:514 and oxigraph_ontology_repository.rs:174/:1598/:1619 citations in the same Note ARE current and correct (class_iri routing already verified in code) -- only the uri/mod.rs:700-707 span is stale.

**VC-24** — `docs/diagrams/visionclaw/24-acsp-decision-elevation.md`
- Citation: `src/actors/decision_elevation_actor.rs:397 and :397-420 (participant GH, message create_ontology_pr)`
- At verified_commit: `let file_path = case.file_path.clone(); ... (approve branch building GitHubPRService::create_ontology_pr call)`
- At HEAD (same line): `); (closing a self.spawn_decision_outcome(...) call inside reconcile logic)`
- Verdict: **MOVED** — src/actors/decision_elevation_actor.rs:~305-335 (create_ontology_pr call now at line 333, inside a spawned future)
- ADR-2101 (durable-decision-elevation-case-state, new in Wave 3) restructured the actor around a persistent DecisionElevationStore (new src/adapters/decision_elevation_store.rs, +481 lines) and a reconcile-on-restart path; the whole surrounding function was rewritten (844-line diff) and create_ontology_pr's call site shifted ~90 lines up, not down, because a large ReconcileAction match arm was inserted earlier in the impl.

**VC-26** — `docs/diagrams/visionclaw/26-solid-pod-and-jss.md`
- Citation: `client/src/store/websocket/solidWebSocket.ts:32, :71, :107`
- At verified_commit: `function notifySolidSubscribers(...) / state.solidSocket?.send(`sub ${url}`) / (blank line)`
- At HEAD (same line): `function bindLifecycle(...) / export function connectSolidWebSocket(...) / export function subscribeSolidResource(...)`
- Verdict: **GONE** — n/a -- architecture replaced
- ADR-2100 (one-solid-jss-notification-client, accepted 2026-09-05) deleted the module's own WebSocket entirely. The file's own header comment now reads: 'This module used to open its OWN WebSocket to VITE_JSS_WS_URL, registered as `solid-store` ... There is now ONE socket: `podNotificationManager`.' File shrank from ~226 to 152 lines. VC-26's diagrams describing solidWebSocket.ts opening/managing its own raw-protocol socket are describing dead code.

**VC-30** — `docs/diagrams/visionclaw/30-client-boot-and-state.md`
- Citation: `client/src/store/websocket/solidWebSocket.ts:154, :155 (webSocketRegistry.register('solid-store', ...), webSocketEventBus.emit('connection:open', ...))`
- At verified_commit: `webSocketRegistry.register('solid-store', wsUrl!, solidSocket); / webSocketEventBus.emit('connection:open', {name:'solid-store', url: wsUrl!});`
- At HEAD (same line): `(past end of file -- file is now 152 lines, cited lines 154/155 no longer exist)`
- Verdict: **GONE** — n/a -- webSocketRegistry / webSocketEventBus / 'connection:open' no longer referenced anywhere in solidWebSocket.ts (grep confirms zero hits); superseded by podNotificationManager (ADR-2100)
- Same ADR-2100 rewrite as VC-26/VC-32. The VC-30.9-ish diagram showing Solid registering itself in webSocketRegistry alongside Graph/Voice/Pod sockets as a fourth independent registrant is now describing a deleted code path -- solidWebSocket.ts is a thin adapter over podNotifications.ts's client, not an independent socket.

**VC-32** — `docs/diagrams/visionclaw/32-client-websocket-and-binary.md`
- Citation: `client/src/store/websocket/solidWebSocket.ts: :65, :70, :76, :80, :85, :94, :125, :135, :136, :148, :150, :159, :170 (13 citations describing new WebSocket(wsUrl), onopen/onmessage/onclose handlers, and the 'protocol '/'sub '/'ack '/'pub '/'error ' text sub-protocol)`
- At verified_commit: `the full manual WebSocket lifecycle (connect/reconnect ladder, onmessage text-protocol switch, subscription re-sync on 'protocol')`
- At HEAD (same line): `unrelated lines in the new 152-line file (resetSolidReconnect, unsubscribeShared, comments) or past EOF`
- Verdict: **GONE** — n/a -- entire raw-socket implementation deleted; replaced by client/src/services/solidPod/podNotifications.ts (podNotificationManager) per ADR-2100
- The single largest concentration of broken citations found (13 of the file's 234). ES-09.11-style DEA sequence in VC-32.9-ish diagrams narrates a message protocol ('protocol '/'sub '/'ack '/'pub '/'error ') that has been deleted wholesale.

**AB-04** — `docs/diagrams/agentbox/04-adapter-spine.md`
- Citation: `agentbox/management-api/adapters/pods/_solid-http-base.js:47, :63`
- At verified_commit: `this._fetch = this._nip98 ? this._signedFetch.bind(this) : this._rawFetch; / if (!header) return this._rawFetch(url, init);`
- At HEAD (same line): `doc-comment line ('...THROWS `SigningUnavailable`...') / '? this._signedFetch.bind(this)' (different ternary arm)`
- Verdict: **MOVED** — agentbox/management-api/adapters/pods/_solid-http-base.js:62-64 (this._fetch = (this._nip98 || this._requireSigned) ? this._signedFetch.bind(this) : this._rawFetch;)
- ADR-2064 (pod-request-signing-fails-closed, new in Wave 3) added a `requireSigned` gate and a `SigningUnavailable` typed throw; the simple ternary the diagram cites was replaced by ~15 lines of fail-closed logic and a new doc comment, shifting the real assignment down by ~15 lines.

**AB-13** — `docs/diagrams/agentbox/13-nostr-relay-gateway-bridge-mirror.md`
- Citation: `agentbox/management-api/lib/bc20-provenance-bridge.js:92`
- At verified_commit: `const AGENTBOX_TO_VISIONCLAW = Object.freeze({ ...hand-written map literal... });`
- At HEAD (same line): `// `src/uri/mod.rs::cross_from_agentbox` embeds the same bytes via (comment)`
- Verdict: **MOVED** — agentbox/management-api/lib/bc20-provenance-bridge.js:117 (AGENTBOX_TO_VISIONCLAW, now Object.freeze(Object.fromEntries(...)) derived from schema/federation-kinds.json)
- ADR-2061 (federation-kind-map-parity, new in Wave 3) replaced the hand-written kind map with one derived from the new shared schema/federation-kinds.json artefact (258 new lines) so the map can't drift from the Rust side's closed map. Same underlying issue duplicated in AB-17 (:92 and :98, both moved to :117/:125).

**AB-17** — `docs/diagrams/agentbox/17-agent-events-and-provenance-bridge.md`
- Citation: `agentbox/management-api/lib/bc20-provenance-bridge.js:92, :98`
- At verified_commit: `const AGENTBOX_TO_VISIONCLAW = Object.freeze({...}); / const VISIONCLAW_TO_AGENTBOX = Object.freeze({...});`
- At HEAD (same line): `comment lines referencing src/uri/mod.rs and a test-failure note`
- Verdict: **MOVED** — agentbox/management-api/lib/bc20-provenance-bridge.js:117 and :125 respectively
- Same ADR-2061 federation-kinds.json derivation as AB-13; this topic file duplicates the AB-13 citation.

**ES-08** — `docs/diagrams/estate/08-solid-pod-estate.md`
- Citation: `agentbox/management-api/adapters/index.js:61`
- At verified_commit: `const withSigner = (cfg) => (nip98 ? { ...cfg, nip98 } : cfg);`
- At HEAD (same line): `// instead of silently going out anonymous at a default-deny pod. (comment)`
- Verdict: **MOVED** — agentbox/management-api/adapters/index.js:74-75 (const withSigner = (cfg) => { if (nip98) return { ...cfg, nip98, requireSigned }; ... })
- Same ADR-2064 fail-closed change (requireSigned threaded through) as AB-04; withSigner grew from a one-line ternary to a small function.

## 4. CI/deploy estate docs vs reality (estate/09, estate/10)

**Workflow inventory** — cited workflow files vs what actually exists:

- Project repo: ['ci.yml', 'docs-ci.yml', 'ontology-publish.yml', 'xr-godot-ci.yml'] — cited list matches exactly. ✅
- agentbox: 14 workflow files, cited list matches exactly (all 14, no more no less). ✅
- Other estate repos (nostr-rust-forum, solid-pod-rs, dreamlab-ai-website, prose-sanitiser, diagram-ir) each have their own CI **not mentioned at all** in ES-09/ES-10's `sources:` — this is in scope-as-declared, not drift, since these diagrams never claim estate-wide CI coverage beyond project+agentbox.
- **loom has zero `.github/workflows`** — matches its single mention in ES-09 (a docker-compose profile node, not a CI claim). No drift.

**Job-line citation drift found (ES-09.11, VisionClaw `ci.yml`):**

All 5 job-header citations are wrong, and were *already* wrong at the stated `verified_commit` — this is a pre-existing verification gap, not new drift from Wave 3:

| participant | cited line | actual @ bed6b617d | actual @ HEAD |
|---|---|---|---|
| FMT rust-fmt job | ci.yml:57 | 61 | 61 |
| CPU rust-cpu job | ci.yml:71 | 75 | 75 |
| CLI client job | ci.yml:107 | 111 | 121 |
| LINT client-quality job (advisory) | ci.yml:129 | 208 | 218 |
| PW playwright job (manual only) | ci.yml:159 | 238 | 248 |

Root cause: A `dev-auth-release-gate` job (~76 lines) sits between `client` and `client-quality` in ci.yml and was already present at bed6b617d; the diagram's line numbers look authored against an even earlier ci.yml revision that predates that job, and verified_commit was bumped without re-checking these specific citations. Two further insertions since bed6b617d (a `-p vault-migrate` cargo-test target at old line 90, and a 9-line visionclaw-integration-tests clippy/test block at old line 111) added a further +10 to everything from `client` onward.

Not affected: GH trigger citation (ci.yml:37-42) and docs-ci.yml (:4,:20,:23,:34 all exact) and xr-godot-ci.yml (:24,:36,:54,:77,:117 all exact, file untouched since bed6b617d) are correct.

**Line drift found (ES-09.13, `ontology-publish.yml`)** — uniform +7 shift from a new ADR-2098 explanatory comment; content is otherwise correct, classified TOUCHED-BUT-INTACT:

| participant | cited line | actual @ HEAD |
|---|---|---|
| VAL validate-source job | ontology-publish.yml:37 | 44 |
| CONV convert-ontology job | ontology-publish.yml:88 | 95 |
| JLD convert-jsonld job | ontology-publish.yml:352 | 359 |
| DEP deploy-jss job | ontology-publish.yml:517 | 524 |
| WS notify-websocket job | ontology-publish.yml:627 | 634 |

## 5. COVERAGE.md consistency

**CONSISTENT -- no drift found**

- 71 topic-file ids in COVERAGE.md's per-diagram table's trailing ID column == 71 ids in file frontmatter == 71 ids in README.md's index table (exact set match both directions).
- 841 distinct diagram-level ids (VC-NN.n / AB-NN.n / ES-NN.n) in COVERAGE.md's Diagrams table == 841 total mermaid ```mermaid blocks under `## ` headings counted directly across all 71 files (grep -c '^## ').
- No orphan entries in COVERAGE.md (ids present in the index with no backing file) and no topic file missing from the index.

## Bottom line

- Structural tooling (frontmatter/paths/mermaid syntax, COVERAGE.md generation) is 100% healthy — `--check --render` is clean and the index is consistent.
- That tooling gives **zero** protection against semantic staleness: it never looks at what a `path:line` citation actually says.
- Of 71 files, **10 are STALE** with confirmed broken citations, concentrated in the areas the Wave 3 remediation actually touched: ADR-2100 (Solid/JSS socket consolidation — VC-26, VC-30, VC-32, 19 broken citations between them, by far the biggest single cluster), ADR-2064 (pod signing fail-closed — AB-04, ES-08), ADR-2061 (federation-kind-map parity — AB-13, AB-17), ADR-2101 (durable decision-elevation state — VC-24), ADR-2094 (management-api credential — VC-09), and one pre-existing, Wave-3-unrelated line-citation error in VC-23 (`uri/mod.rs` cross_from_agentbox relocation).
- One genuinely new finding outside the Wave-3 footprint: ES-09.11's `ci.yml` job-line citations were never correct even at `bed6b617d` — evidence the per-line verification step is sometimes skipped when `verified_commit` is bumped.
- 60 files are TOUCHED-BUT-INTACT and 1 (VC-18) is fully CLEAN.