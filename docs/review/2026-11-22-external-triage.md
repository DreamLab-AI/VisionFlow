# External review triage, 2026-11-22

An outside reviewer read seven corpus areas (agentbox, nostr-rust-forum, dreamlab-ai-website, solid-pod-rs and others) and listed contradictions inside the corpus, maintenance concerns, and code defects that the corpus exposes. Each claim below was checked again: corpus claims against the topic files, and code claims against the **current HEAD** of each source repository, not against the panel's stamp. Topics are not edited here. Corpus fixes go through `diagrams-as-code` re-authoring and re-stamping. Code fixes belong in the source repository.

Corpus rows re-checked at both local `34cb5b5` and `origin/main` (which carries the 2026-11-22 consolidation and the NF re-stamps to `fd31193`, the source of the reviewer's forum stamps); every section-1 row holds at both. HEADs checked: agentbox `59b1aca1c`, solid-pod-rs `eb22d21` (0.5.0-alpha.24), nostr-rust-forum `4d75679` and `fd31193` (the website's pinned KIT_REF; the poker, invite and moderation code is unchanged between the two), dreamlab-ai-website `e255b2b1`.

Verdicts: **CONFIRMED** means it holds at HEAD. **FIXED** means it was already repaired, so the corpus is stale. **PARTLY** means part of the claim is wrong. **WRONG** means the claim does not hold. Where a verdict depends on reading code rather than running it, the evidence says so.

## 1. Contradictions inside the corpus

| Claim | Verdict | Evidence | Action |
|---|---|---|---|
| AB-16.11 says live mirror is "UNREDACTED" and ADR-2026 is open; AB-13 and AB-08 say it is resolved | PARTLY | `agentbox/16-…:444,451` still say UNREDACTED and open. The RESOLVED note is in AB-13.17 (`agentbox/13-…:545`), not AB-13.9 as the reviewer said. AB-08.14 (`:536`) states it as an INVARIANT, not RESOLVED. AB-16 is stamped `0693c761`, which is newer than AB-13's `e458b308`. The contradiction is real. | Re-author AB-16.11 against agentbox HEAD |
| AB-06.2 and AB-06.7 list browsercontainer, gui-tools and xr-runtime as 0.0.0.0 LAN-sanctioned; AB-10.1 lists only three publishes | CONFIRMED | AB-06 at `e458b308` still lists all three on `0.0.0.0`. AB-10.1 at `0693c761` records `AGENTBOX_SIDECAR_BIND` defaulting to `127.0.0.1`. | Re-author AB-06 (panels .2 and .7) |
| AB-04.13 uses `vault edit page`; AB-25 says edits moved to `ontomark` | CONFIRMED | `agentbox/04-adapter-spine.md:567` against `agentbox/25-…:31,213`. Both are stamped `e458b308`. | Re-author AB-04.13 |
| AB-30 has no secret-backup panel and starts at AB-30.2 | CONFIRMED (self-flagged) | The header already records it as Debt. Secret-backup is drawn in AB-16.12 (`agentbox/16-…:462`). AB-30 is already in today's GLM re-author queue. | During re-authoring: retitle AB-30 and drop secret-backup from `sources:`, or add AB-30.1 pointing to AB-16.12 |
| Headings that still assert a bug over a RESOLVED body | PARTLY (5 of 6) | Real cases: NF-23.1 (`nostr-rust-forum/23-…:31` vs `:36`), NF-23.2 (`:51` vs `:62`), NF-18.1 (`nostr-rust-forum/18-…:28` vs `:39`), NF-18.4 (`:95` vs `:104-108`), SP-14.1 (`solid-pod-rs/14-…:29` vs `:35`). DW-08.5's heading is neutral, so that one is wrong. | Rewrite the headings during re-authoring (see policy question A) |
| DW-06.6 typo "subset of of" | CONFIRMED | `dreamlab-ai-website/06-…:104` | Fix while re-authoring DW-06, which is already queued |
| Heavy entity escapes in AB-08 and AB-15 | CONFIRMED | AB-08 has 266 `#40;`, `#41;`, `#123;` and `#124;` escapes; AB-15 has 158; AB-36 to AB-43 have none. | Check the Pele render, then normalise when re-authoring |

## 2. Corpus maintenance

| Point | Verdict | Note |
|---|---|---|
| Two repair styles both in use (successor topics plus in-place RESOLVED retrofits) | CONFIRMED | This is the source of the heading mismatches in section 1. Policy question A. |
| Very large files | CONFIRMED | agentbox/02 is 75.8 KB with 21 panels. agentbox/23 is 67.0 KB with 24. agentbox/05 is 66.5 KB with 16. agentbox/08 is 59.0 KB with 13. solid-pod-rs/08 is 48.4 KB with 17. nostr-rust-forum/05 is 43.7 KB with 17. Policy question B. |
| Spread of commit stamps within an area | CONFIRMED | agentbox topics are stamped at `720e0d89`, `3a2c5976`, `e458b308` and `0693c761`. This is the cause of the AB-06/AB-10 skew. The nightly GLM triage already re-queues these. |

## 3. Code defects, ranked

### Security and authority seams

| ID | Claim | Verdict at HEAD | Evidence | Owner |
|---|---|---|---|---|
| AB-36.1 | `role_isolation` locks out nip98-proxy | CONFIRMED (dormant while the flag is off) | `config/lib/nip98-config.sh:33` sets `1000:1000` with mode 0600. `role-accounts.json:47-48` gives ab-ingress uid 962. `proxy.mjs:274-276` maps EACCES to pending because `flake.nix:3170` sets `NIP98_PROXY_REQUIRE_CONFIG=1`. Already recorded in ADR-2122:920 and ADR-2210. The fixture `tests/config/fixtures/role-isolation/supervisord.conf:366` is stale. | agentbox: blocks the flag-on rehearsal |
| AB-13.7 | JS RelayConsumer admits anyone when the allowlist is empty | **FIXED** in `2c1f85706` (2026-11-15, after AB-13's `e458b308` stamp, so stale panel rather than gate miss) | `mcp/nostr-bridge/relay-consumer.js:775` returns false on an empty set. Also, the env var *is* projected, into management-api at `flake.nix:2896`, contrary to the review. It does subscribe to damus and primal (`relay-consumer.js:254-262`), even under `publish-only`. | Corpus: re-author AB-13.7 |
| SP-07.9 and SP-07.17 | A one-string MRC20 deposit proof is accepted | CONFIRMED, with a correction | Only an empty proof is rejected (`mrc20.rs:880`). The genesis input is skipped (`blocktrail.rs:1114`). The amount is whatever the caller asserts (`mrc20.rs:928` to `pay.rs:711`) and is never compared with the UTXO value. **Correction:** the attacker must fund the derived address with real sats, so this is cheap self-minting of unbounded credits, not a free forgery. No route test rejects k=1. SP-15 does not cover it. | solid-pod-rs: fix first |
| AB-03.1 | `POST /v1/exec/record` is described as operator-gated but has no `requireOperator` | CONFIRMED | `management-api/routes/exec-record.js:63` has no preHandler. Only the global `authMiddleware` applies (any valid NIP-98 signer). The comments at `:4-6` and `server.js:431-433` are wrong. | agentbox |
| AB-03.15 | NIP-98 through `/mgmt/` fails to match the URL | CONFIRMED (by code reading; not run) | The `/mgmt/` route is on in the shipped config (`flake.nix:3170`, `strip: true`, `proxy.mjs:324-329`). The header is re-attached unchanged (`proxy.mjs:907`). The upstream rebuilds the URL from the stripped path (`middleware/auth.js:73`). `nostr-bridge.js:549` uses `endsWith`, so `…/mgmt/v1/x` does not match `…/v1/x` and the request gets a 401. URLs with a query string also mismatch. `selftest.mjs:418` tests `/mgmt` with a Bearer only. | agentbox: write the one-line test first |
| AB-11.7, AB-11.8, AB-11.13 | Mandate revocation does not reach pods | CONFIRMED, with a correction | `routes/mandate.js:238-280` only rewrites the registry and signs; there is no `.acl` write and no publish, and the warning at `:267` mentions a publish that never happens. **Correction:** revoke is signed by the operator stack (`_loadSigner`, `:97`) and is unsigned when no stack is configured. pod-signer's container-identity default (`lib/pod-signer.js:14-19`) is real but is not on this path. | agentbox |

### Defects that fail silently

| ID | Verdict at HEAD | Evidence |
|---|---|---|
| NF-25.4 invite revoke by code returns 200 and changes 0 rows | CONFIRMED | `crates/nostr-bbs-auth-worker/src/invites.rs:734` (`id = ?1 OR code = ?1`) and `:761` (`WHERE id = ?3` bound to the raw input). It still holds at `fd31193`. The regression came from `7852fe8`; before that commit, revoking by code returned 404. |
| NF-02.2 / NF-08.13 auth `GET /api/moderation/reports` reads a table nothing writes | CONFIRMED | `moderation.rs:650-651` reads `nip1984_reports`. The only reference elsewhere is the CREATE at `schema.rs:51`. The relay writes to `reports` in a separate D1 (`relay-worker/src/moderation.rs:75`). The doc comment at `moderation.rs:631` is wrong. The auth worker's own `mod_reports` table may be what was intended. |
| AB-26.7 `AGENTBOX_BEADS_DB=""` falls back to `:memory:` on every call | CONFIRMED | `mcp/servers/lib/typed-spawn.js:34` and `mcp/mcp.json:224`. Introduced in `1d15a00f0`. |
| AB-26.3 `headroom-mcp-tools.js` has no caller | CONFIRMED | The only references are its own header and an archived ADR. |
| AB-04.15 schema rejects `[adapters].connect_timeout_ms` | CONFIRMED (dormant; the key is unset) | `schema/agentbox.toml.schema.json:50` has `additionalProperties:false`. The key is read at `adapters/lifecycle.js:121`. |
| AB-05.5 `agentbox.sh config` is not in the allowlist | CONFIRMED | The allowlist is at `agentbox.sh:2054`. The handler is at `:2447` and the dispatch at `:2899`, so the command exits with "Unknown command". |
| SP-02.13 `account delete` uses an empty in-memory store | CONFIRMED | `crates/solid-pod-rs-server/src/cli/mod.rs:380`. The help text at `:105` also promises pod and WebID removal, which never happens. |
| DW-05.1 deploy skips the TYPESAFE_API_KEY check on auth-worker | CONFIRMED | The deploy check is in `workers-deploy.yml:288-289` (search and relay only). `set-worker-secrets.yml:157` provisions auth-worker. |
| DW-10.2 to DW-10.4 poker | CONFIRMED (by code reading) | Buy-ins are not reserved (`poker-citizen/src/table.rs:521-527,560-611`); the cap stops payout but not acceptance (`:534`, `:939`). Hands are in memory only and lost on restart (`table.rs:149`; only the ledger is saved, `run.rs:276-282`). Auto-pay amount and destination are not bound to the verified replay (`forum-client/src/poker/live.rs:968-981,1036`). |

### Release backlog and pin skew

| Claim | Verdict |
|---|---|
| ADR-2151 voicelink is proposed and inactive but `enabled = true` | CONFIRMED. The ADR acknowledges it at line 21. Line 43 ("implementation_status stays none") is stale. |
| agentbox and VisionClaw at solid-pod-rs alpha.22 | **FIXED, corpus stale** (ES-01.1 still says alpha.22; the reviewer read it correctly): `lib/solid-pod-rs.nix:66` is alpha.24 at rev `eb22d216`. Leftover lockfiles are still behind: `crates/nostr-receipt` is on alpha.21 and the vendored forum lock on alpha.20. VisionClaw was not re-checked. |
| Website KIT_REF `3c48e26` against poker-citizen `6064622` | **FIXED, corpus stale**: website KIT_REF is now `fd31193` at all three sites. The poker-citizen pin is still `6064622` (`lib/poker-citizen.nix:31`), so the skew is real and larger than the review says. |
| vowl-wasm npm E404 | Not re-checked. |

## Policy questions for the owner

- **A. Repair convention.** Choose one: (1) successor topics only, with old panels frozen behind a one-line banner pointing to the successor; or (2) in-place RESOLVED retrofits that must also rewrite the heading. The corpus currently uses both.
- **B. Large files.** Decide whether to split agentbox/02, 05, 08 and 23, solid-pod-rs/08 and nostr-rust-forum/05 into focused topics. Doing so renumbers panels and breaks external references to IDs such as AB-02.17.

## Suggested order

1. solid-pod-rs: reject proofs shorter than the full trail, and bind the credited amount to the UTXO value, with a k=1 route test (SP-07.17).
2. agentbox: AB-03.15 test then fix, AB-03.1 `requireOperator`, AB-36.1 config ownership before the role-isolation rehearsal.
3. nostr-rust-forum: NF-25.4 (update by `row.id`) and NF-02.2 (point at `mod_reports` or remove the endpoint).
4. Corpus. **New** re-author items: AB-16, AB-06, AB-04, AB-13 (panel .7), NF-23 and NF-18 headings, AB-08/AB-15 escapes. **Already in today's GLM queue** (fold the fix in, do not double-book): AB-30, AB-26, DW-06 (typo), DW-10, SP-14 (via the solid-pod-rs batch), ES-01 (pin).

Reviewer wrong or overstated: DW-08.5 heading (neutral); AB-13.9 attribution (the note is in AB-13.17); SP-07.17 "no secret" framing (real sats needed); AB-11 signer (operator stack, not pod-signer); AB-13.7 (already fixed in code).
