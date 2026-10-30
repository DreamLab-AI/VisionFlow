**The bundle supports runtime-evidence presence for 24 live promotions, but does not establish estate-wide acceptance or production security.** Several excerpts are truncated. PASS below means the supplied close-out contains a concrete runtime observation; it does **not** mean I independently verified it or that every acceptance criterion passed.

For file references below, the directory is `docs/adr/` within the named repository.

**A — Live promotions**

| Repository / file | Result | Quoted evidence |
|---|---|---|
| VisionFlow / `ADR-2019-forge-and-mirror-on-in-shipped-pods.md` | PASS | “`GET http://127.0.0.1:8484/forge` answers 200 with the forge page.” |
| project / `ADR-2114-corpus-ingest-is-a-corpussource.md` | PASS | “`GET /api/graph/data` serves 9,688 nodes.” Boot also logged “`Processing 0 changed files (10375 unchanged)`”. |
| project / `ADR-2116-ontology-proposals-are-forum-action-requests-not-prs.md` | PASS | “`POST /api/ontology-agent/propose`, sent with no credentials, answered `{"success":false,"error":"route_retired", …`”. |
| project / `ADR-2117-retire-corpus-format-migration.md` | PASS | Installed binary answers “`vault migrate --fences-to-properties` with `error: unrecognized subcommand 'migrate'`”. |
| project / `ADR-2118-append-space-earth-domain-identities.md` | PASS | “`GET /api/graph/data` (9,688 nodes) contains `domain-root-space-science-and-systems` … and `domain-root-earth-observation-and-geospatial-sensing`”. |
| project / `ADR-2133-live-memory-cloud-serves-a-stratified-sidecar-snapshot.md` | PASS | “`GET /api/memory-cloud/health` returned snapshot `33aff23bb078`, `sidecar.reachable: true`”. |
| project / `ADR-2148-enable-local-typed-decisions.md` | PASS | “`POST /api/jev/edge` … returned `{"used":false,"reason":"low-confidence"}`, HTTP 200, in 0.24 s.” |
| project / `ADR-2156-dev-runtime-profile-builds-incrementally-without-lto.md` | PASS* | Wrapper log: “`Building from published inputs: source_sha=772039bd172027af034d64befacca1ebcf0bbc04 source_dirty=0`”, followed by “``Finished `dev-runtime` profile [optimized] target(s) in 1m 00s``”. |
| agentbox / `ADR-2033-deepsec-security-gate.md` | PASS | “`deepsec --version` → `2.3.10`”; executed diff receipt records “`deepsec_exit: 1`, `result: BLOCK`, `candidates: 13`”. |
| agentbox / `ADR-2057-close-the-capability-gate-gaps.md` | PASS | “`supervisorctl status podcast-cron` → `RUNNING pid 980`”. |
| agentbox / `ADR-2065-one-writer-for-the-pod-inbox.md` | PASS | “`tr '\0' '\n' < /proc/446/environ` … → `AGENTBOX_POD_INBOX_WRITER=rust-pod-bridge`”; “`ps -eo pid,args` → pid 453 … `nostr-pod-bridge`”. |
| agentbox / `ADR-2068-root-session-runs-its-gated-session-hooks.md` | PASS | Boot log: “`[ontology] ontology-monitor hook already registered`”; “`jq '.env, .hooks.SessionEnd' ~/.claude/settings.json` … → `AGENTBOX_ONTOLOGY_MONITOR=1`”. |
| agentbox / `ADR-2072-briefing-workflow-route-surface.md` | PASS | “`curl -X POST :9090/v1/briefs` … → 400 `body must have required property 'content'`”; control route returned 404. |
| agentbox / `ADR-2082-orchestration-proxy-behind-governed-memory-server.md` | PASS | “`node /opt/agentbox/mcp/servers/ruvector-mcp.cjs` … JSON-RPC `initialize` then `tools/list` → 56 tools.” |
| agentbox / `ADR-2083-skills-estate-authoring-contract-and-generated-discovery.md` | PASS | “`readlink ~/.claude/skills/browser` → `/opt/agentbox/skills/browser`”; corresponding Codex link also observed. |
| agentbox / `ADR-2088-grill-before-acting-on-forum-suggestions.md` | PASS | “`grep -n '^junkiejarvis' /etc/agentbox.toml` → `50:junkiejarvis = true`”; runtime log: “`junkiejarvis watching — gift-wrapped DMs (#p) + kind-42 mentions`”. |
| agentbox / `ADR-2091-live-skill-router.md` | PASS | “`jq -r 'select(.ts>="2026-10-30T14:23")|.outcome' ~/.claude/skill-route.jsonl \| sort \| uniq -c` → `47 routed`, `14 skipped`”. |
| agentbox / `ADR-2108-ontology-bridge-retired-vault-baked-by-nix.md` | PASS | “`docker logs agentbox 2>&1 \| grep '\[5d/8\] vault'` … → `[5d/8] vault OK — vault 0.1.0 (/opt/agentbox/bin/vault)`”. |
| agentbox / `ADR-2132-incremental-image-delivery.md` | PASS | “`docker inspect agentbox --format '{{.Image}}'` … return `sha256:9776d2fc…`”; matching activation receipt supplies source SHA, activation time and stage timings. |
| agentbox / `ADR-2147-retain-codex-session-state.md` | PASS | “`findmnt -no SOURCE,FSTYPE,TARGET ~/.codex` → `/dev/nvme0n1p1[/docker/volumes/agentbox-codex-home/_data] ext4 ~/.codex`”; 46 session files predate replacement. |
| nostr-rust-forum / `ADR-2016-end-to-end-encrypted-zones.md` | PASS | “`curl -sL https://dreamlab-ai.com/community/` … serves `window.__ENV__` with `ENCRYPTION_ENABLED:"true"`”. |
| nostr-rust-forum / `ADR-2025-nip59-unwrap-verifies-the-wrap-and-never-echoes-plaintext.md` | PASS | Live WASM “fetched 2026-10-30T19:30Z (6,782,374 bytes), contains the string `gift wrap verification failed` once (`grep -c -a`)”. |
| dreamlab-ai-website / `ADR-2010-seo-watch-nightly.md` | PASS | “`supervisorctl status seo-watch-cron` → `RUNNING pid 989`”; dated nightly ledger result accompanies it. |
| codescape / `ADR-0001-extraction-from-visionclaw.md` | PASS | “`GET http://127.0.0.1:4100/api/health` inside it returned `"engine":"cuda"`” and `"nodes":94249,"settled":true`. |

\* ADR-2156 supplies executed build logs, but **omits the actual rebuild invocation**. Under a literal requirement that every note reproduce a concrete command *and* its output, this is **FAIL for command completeness**. It also expressly substitutes a three-crate rebuild for the prescribed single-file measurement.

Material qualifications:

- VisionFlow ADR-2019 promotes the combined forge/mirror record while admitting the mirror never starts. Runtime evidence exists for the forge only; the equivalent solid-pod-rs records remain staged.
- ADR-2114’s “0 changed files” proves a boot using cached corpus state, not necessarily fresh parsing through the new implementation.
- Route mounting, process startup, registration links and configuration gates do not prove successful workflows. This particularly limits agentbox ADR-2072/2088 and forum ADR-2016.
- `grep -c -a` counts **matching lines**, not string occurrences. Forum ADR-2025’s “once” is unsupported by that command alone. A string in WASM also does not demonstrate execution of the rejection path.
- Anonymous dev responses do not demonstrate production authorisation. The close-outs generally acknowledge this correctly.

**A — Acceptance authority**

Only one literal frontmatter `proposed → accepted` transition is shown:

- **agentbox `ADR-2146-activity-sink-hook.md`: provisionally PASS for the supplied excerpt.** No owner-reservation wording is shown. Its justification is “under the close-out rubric (code landed on main)”. The original body is absent, so absence of a reservation is **not verified**.

One additional semantic acceptance must be audited despite lacking YAML:

- **ontomark `ADR-OM-003-native-assisted-ingest.md`: acceptance not substantiated.** It changes “Proposed” to “Accepted” and says, “**The owner took the decision by shipping it**.” Publication under `jjohare` establishes publication, not necessarily ratification. No explicit owner reservation is visible, so the excerpt alone cannot justify an owner-reservation FAIL; the complete body is needed.

The two acknowledged owner-reserved records are correctly left proposed:

- project ADR-2134 and ADR-2138 each say: “**This record reserves acceptance to the owner, and the close-out does not take owner decisions on the owner's behalf.**”
- Neither supplied final diff contains a frontmatter promotion to accepted. **PASS for the correction**, not evidence that the initial promotion was legitimate.

**B and C — What the output establishes**

- **“All ADR ledgers pass at main”: not established.** B shows six successful checks, covering **355 ADRs**, at six stated SHAs. D claims **11 ledgers**; codescape, ontomark, visionGraph, sidestr-rs and visiontouch-rs have no gate receipts here. The output does not itself establish branch identity. Agentbox explicitly says README was neither written **nor compared**. These gates also do not validate runtime truth or owner approval.
- **“Forum main is fixed but production KIT_REF 9c36b60 still has crypto-bigint 0.5.5”: supported narrowly.** C reports HEAD resolving `crypto-bigint 0.7.10` / `k256 0.14.0`, versus `0.5.5` / `0.13.4` at the pinned kit. A supplies the corresponding successful deployment claim. This supports the **lockfile difference and deployment pin**, not proof that every affected dependency is eliminated or that the vulnerable code is present/reachable in deployed binaries. C omits package names and the actual extraction command, weakening independent interpretation.
- D2’s assertion that the manual secrets workflow **blocks deployment** is unsupported and in tension with A’s successful deployment and D’s “Workers deploy and Pages are green”.

**D — Ten priority checks**

These are commands for the lead to run, **not commands I executed**. Each uses only the permitted read-only tools. Where a claim has several parts, the command targets its most questionable, locally testable part.

1. CLAIM: Ontomark contributed one fully closed ADR, taking its open count from 1 to 0, despite OM-003 remaining staged. | CMD: `git -C ~/workspace/ontomark show 560e50c:docs/adr/ADR-OM-003-native-assisted-ingest.md` | EXPECT: To support full closure under D’s stated rule, the record must be accepted, complete and **live**, or explicitly excluded as superseded. `Activation: staged` refutes the claimed closure.

2. CLAIM: The failed manual secrets workflow blocks the forum production repin/deployment. | CMD: `git -C ~/workspace/dreamlab-ai-website grep -n -E 'PRF_SERVER_SECRET|ADMIN_PUBKEYS|needs:|workflow_run:|workflow_dispatch:|Set CF Worker secrets' a67cb2a8 -- .github/workflows` | EXPECT: An explicit deployment dependency on successful secret provisioning, or a mandatory deployment preflight requiring the missing GitHub secret. Separate independent workflows do not establish the claimed blocker.

3. CLAIM: Agentbox `6500582df` differs from the running source `e786197cc` only in `docs/`. | CMD: `git -C ~/workspace/agentbox show --format=fuller --name-only e786197cc..6500582df` | EXPECT: Every listed changed path is under `docs/`; any source, configuration, workflow or skill path contradicts the “only docs” explanation.

4. CLAIM: Four governance calls at lines 221, 420, 478 and 668 still invoke the removed vault verbs. | CMD: `git -C ~/workspace/agentbox show e786197cc:management-api/lib/ontology-apply.js | sed -n '145,160p;210,235p;410,435p;468,490p;658,680p'` | EXPECT: `vaultBin()` resolves the vault executable, and the four cited regions invoke `find`, `edit`, `create` and `apply`, respectively; no ontomark redirection.

5. CLAIM: The audit journal first becomes invalid at line 35, column 145. | CMD: `jq -c . ~/workspace/events/2026-10-12.jsonl` | EXPECT: Valid preceding JSON records followed by a parse error reporting line 35, column 145. This does not independently verify the hash chain or prove interleaved appends caused the corruption.

6. CLAIM: VisionClaw at `7c953a40b` still resolves core beta.17, solid-pod-rs alpha.17, crypto-bigint 0.5.5 and k256 0.13.4. | CMD: `git -C ~/workspace/project show 7c953a40b:Cargo.lock | sed -n '/^name = "nostr-bbs-core"$/,+5p;/^name = "solid-pod-rs"$/,+5p;/^name = "crypto-bigint"$/,+5p;/^name = "k256"$/,+5p'` | EXPECT: Package blocks containing versions `1.0.0-beta.17`, `0.5.0-alpha.17`, `0.5.5` and `0.13.4`, respectively. Additional versions must also be reported rather than silently discarded.

7. CLAIM: The browser podkey expiry is 2027-01-11, established by `8fad68285`. | CMD: `git -C ~/workspace/agentbox show 8fad68285:browsercontainer/podkey.pin` | EXPECT: An explicit expiry of `2027-01-11`, or timestamp fields unambiguously yielding that date. An opaque pin alone does not support the expiry claim.

8. CLAIM: visionGraph `licensing/NOTICE` credits none of SLSO, SWEET or SAC-CG. | CMD: `git -C ~/workspace/visionGraph show 7b0875f:licensing/NOTICE` | EXPECT: No attribution to those sources or their expanded names. This checks the notice claim only; it cannot establish that per-page attribution is legally insufficient.

9. CLAIM: The supposedly verified `~660`-commit board uses the stated current heads, despite B reporting different heads for three repositories. | CMD: `grep -n -E 'Heads:|660 commits|four-reader|Landing 2026-10-30' ~/workspace/project/docs/TODO-unified.md` | EXPECT: The board still names VisionClaw `7c953a40b`, agentbox `6500582df` and VisionFlow `4d361b5`, with a clear snapshot date or explanation distinguishing them from B’s `8692b12ba`, `eaa847291` and `5eaa5b7`. Without that distinction, “current main” is ambiguous.

10. CLAIM: The register contains exactly 174 parked rows, each with a file-level gap and reopening trigger. | CMD: `sed -n '/^## Parked records by repository/,$p' ~/workspace/VisionFlow/docs/registers/park-2026-10-30.md` | EXPECT: Repository sections containing 174 actual record rows with identifiable gaps and triggers. The supplied D excerpt ends with two empty headings and provides no evidence for this count.

Two contradictions already warrant correction without further commands: **ontomark staged versus fully closed**, and **“every ledger” versus six receipts**. The broader count of 23 closures inherits the ontomark discrepancy unless the census uses an unstated exception.

**Privacy visible in A and D**

- **LAN IP:** A exposes `<lan-redacted>` in agentbox ADR-2109. Loopback `127.0.0.1` and wildcard `0.0.0.0` are not private LAN host addresses.
- **Expanded home paths in A:** `~/workspace/QR-WASM/.deepsec-gate/reports/20261027T212112Z/receipt.json`, `~/workspace/VisionFlow/docs/engineering/templates`, `~/.claude/settings.json`, `~/workspace/project/agentbox/config/hooks/ontology-monitor.cjs`, `~/workspace/.mcp.json`, `~/.codex` and `~/.codex/packages`.
- **Credential material:** No actual secret value, private key or bearer token is visible. Secret names, an `auth.json` mount reference, missing-secret state, signer configuration and a Gemini 401 are operational disclosures, not credential values.
- **Other potentially private context:** A/D expose internal service names, image and filesystem identifiers, personal repository/account names, session identifiers, family-zone naming, and the Trust/NHS residential dates. D’s home paths use `~`; I see no expanded `/home/...` path there.
