---
title: Forum clients, agent control surface and search workers
date: 2026-11-02
status: assessed
type: explanation
---

# Forum clients, agent control surface and search workers

I assessed all 43 panels in three nostr-rust-forum topics against nostr-rust-forum `c39f41f` (and dreamlab-ai-website `0e15728` for NF-06's kit-record citations). The topics are NF-05 (Leptos clients, 17 panels), NF-06 (agent control surface, 17 panels) and NF-07 (search and preview workers, 9 panels). Result: **18 accurate, 19 corrected, 6 accurate with limits, 0 inaccurate-unfixed**. All three topics were restamped and every citation was re-resolved. Each topic passes the strict citation and render gate.

The most important finding is a diagram error. NF-06.6 drew an appeal as reopening an *open* case. Core rejects that transition: `reopen` admits only Decided, Delegated, Promoted, Precedent or Superseded. The diagram also left out the path back from Reopened to Decided. The most important finding in the product itself is a stale operator comment. The relay template says the relay has no `ADMIN_PUBKEYS` reader, but the relay reads it at `crates/nostr-bbs-relay-worker/src/auth.rs:185`. An operator who trusts the comment leaves a static admin unrecognised by the relay.

## Findings

**F1. The relay template comment contradicts the relay's admin resolution (Drift, operator-facing).**
- **Trigger:** an operator configures static admins by reading the templates.
- **Consequence:** `crates/nostr-bbs-relay-worker/wrangler.toml:19` says "there is no ADMIN_PUBKEYS reader in src/". In fact the relay resolves admins as static ∪ D1 and reads the variable at `crates/nostr-bbs-relay-worker/src/auth.rs:185`. An operator who follows the comment never sets it on the relay, so a static-only admin is an admin to the search and auth workers but not to the relay.
- **To close:** correct the comment, or declare the variable in the relay template.
- Recorded as **Drift** in NF-07.2 (N4 and prose).

**F2. The NF-06.6 state diagram drew an impossible transition (diagram defect, corrected).**
- **Trigger:** a reader takes `Open --> Reopened` as a valid appeal path.
- **Consequence:** `CaseState::reopen` accepts only terminal states (`crates/nostr-bbs-core/src/governance.rs:1847-1854`), and the relay's appeal SQL requires the same (`crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs:283`). The diagram also lacked `Reopened --> Decided`. The projection admits a decision on `open`, `under_review` or `reopened` (`receipts.rs:188`).
- **Fix:** redrawn with those citations.

**F3. NF-06.8's table anchors had drifted (diagram defect, corrected).**
- Every `relay_lib_NNN` inline-bootstrap anchor pointed at unrelated code. For example, `relay_lib_813` landed on `env.d1("DB")`, and `relay_lib_906` on an `ALTER TABLE` for a different column.
- The two `nip_handlers_NNN` anchors (137, 151) pointed at doc comments.
- **Fix:** re-resolved to `CREATE TABLE` sites `crates/nostr-bbs-relay-worker/src/lib.rs:741, 765, 783, 1000, 1009, 1035, 1047`, the case upsert `relay_do/nip_handlers.rs:1001`, the decision insert `relay_do/receipts.rs:185`, and migration 0006 lines 65, 79 and 89.

**F4. NF-06.10 N3 made a claim the cited README does not support (corrected).**
- It said the one consumer is "a case queue capped at five concurrent" (`README.md:120`).
- The line says only that the documented estate consumer is VisionClaw's ontology-review workflow and that universal production use is not established.
- **Fix:** reworded to match.

**F5. NF-07.8 drew a wrong dependency edge (corrected).**
- It showed `Ascii --> BbsClient`. The BBS client does not depend on `nostr-bbs-ascii`; its `Cargo.toml` has no such dependency at the pin. The pod worker does depend on it.
- **Fix:** the edge now reads `Ascii --> PodWorker`, plus `BbsClient ..> PreviewWorker : fetches /ascii`, which is what the panel's own note already said (`crates/nostr-bbs-bbs-client/src/ascii_img.rs:203`).

**F6. NF-05.6 cited the wrong line and overstated a writer invariant (corrected).**
- `unread_count` was cited at `notifications.rs:223`, which is a field doc; the function is at `:249`.
- N2 called `insert_message` "the ONLY writer of channel_messages". It is the only path that *adds* events. `fold_deletions` (`stores/channels.rs:795`, behind `remove_message` at `:129`) and the empty-EOSE reset (`:320`) remove events. `redecrypt` rewrites them in place, behind a deletion check (`zone_crypto/store.rs:164`).
- The derived-count invariant still holds.

**F7. Anchors that pointed at an adjacent line instead of the mechanism (corrected).**

| Panel | Was | Now |
| --- | --- | --- |
| NF-06.4 G1 | `nip_handlers.rs:1291` (a TL3 deletion message) | `:1324-1327` (the registry gate) |
| NF-06.10 decision-card publish | `governance.rs:848` (`has_agent_reasoning`) | `:918` |
| NF-06.10 panel-action publish | `:501` | `:522` |
| NF-06.10 N1 | `:35` (a blank doc line) | `:36-40` |
| NF-06.15 `supersedes_held` | `panel_registry.rs:461` (an unrelated doc line) | `:496` |
| NF-06.3 | `governance.rs:919`, `:923` | `:918` (kind), `:921` (the unmarked `e` tag) |
| NF-07.5 allowlist install | `preview-worker/src/lib.rs:511` (a comment) | `:514` |
| NF-07.5 cache lookup | `:284` | `:287` |
| NF-07.6 port check | `ssrf.rs:311` | `:313` |
| NF-07.6 allowlist check | `ssrf.rs:323` | `:326` |
| NF-07.6 6to4 | `ssrf.rs:400` (the NAT64 return) | `:395` |
| NF-07.6 NAT64 | `ssrf.rs:404` | `:399` |
| NF-07.9 confidence floor | `rerank.rs:92` (the `head < 2` early return) | `:111` |
| NF-05.4 | `nip98.rs:148` (a doc line) | `:149` |
| NF-05.7 relay fallbacks | `relay.rs:22`, `relay_url.rs:9` | `relay.rs:23`, `relay_url.rs:13` (the two divergent literals) |
| NF-05.10 | `user_table.rs:266` | `:269` |
| NF-05.12 | menu label lines 29, 32, 34, 38 | 28, 31, 33, 37 |
| NF-05.17 prose, ADR-2023 | `:6` | `:7` (`activation_status`) |
| NF-05.9 prose, ADR-2028 | `:6` | `:5`, `:7` |

**F8. In-code documentation drift found while tracing (product, not fixed: product code is out of scope).**
- `crates/nostr-bbs-auth-worker/src/governance_api.rs:85` still says "register stores the pubkey as supplied", but register now canonicalises (`:372`). Recorded in NF-06.15 N4.
- `crates/nostr-bbs-preview-worker/src/ssrf.rs:129` says `is_private_url` is called at `lib.rs:216`; the call is at `lib.rs:274`.
- `crates/nostr-bbs-search-worker/src/store.rs:8` describes the Meta segment as the "JSON id↔label mapping". `to_rvf_bytes` writes only format, dimension, count and metric (`:91-96`), and the mapping lives in KV. NF-07.4 N1 already states the behaviour correctly.
- **To close:** a doc-comment pass in each crate.

**F9. Hardcoded governance kind numbers in the forum client (Debt, recorded in NF-06.1).**
- The core re-exports every kind so that consumers need not hardcode numbers (`crates/nostr-bbs-core/src/lib.rs:155`).
- The board declares its own `31402` and `31403` (`crates/nostr-bbs-forum-client/src/pages/board.rs:46-47`), and the app subscribes with a literal list (`src/app.rs:852`).
- A kind renumbering in core would silently miss these.
- **To close:** import the core constants.

**F10. Duplicate `FORUM_BASE` prefix stripping (Debt, recorded in NF-05.1).**
- `crates/nostr-bbs-forum-client/src/components/mobile_bottom_nav.rs:30` re-reads `option_env!("FORUM_BASE")` and strips the prefix by hand instead of calling `current_app_path` (`src/app.rs:82`).
- It is not a third *application* of the prefix, so the ADR-090 invariant holds. It is a second copy of the stripping rule that can drift.

**F11. The citation gate reported a phantom citation (tooling).**
- With NF-07.6 describing IPv4 ranges as "10/8 :448, 127/8 :451, 172.16/12 :454", the gate counted a citation of `ssrf.rs:127`, and the Jev advisory flagged it as a defect. No such citation is written in the topic.
- **Fix in the topic:** the ranges are now named in words. The count fell from 163 to 162 and the advisory disappeared.
- **Not traced:** I did not trace which parsing rule in `scripts/diagram-index-gen.cjs` produced it. `CITE_RE` alone requires a file extension before `:N`, so a separate rule for bare `:N` continuations is the likely source.
- **To close:** find that rule and stop it reading a number inside CIDR notation as a line.

## Panel dispositions

### NF-05 (`nostr-rust-forum/05-leptos-clients-onboarding-messaging-governance.md`)

| Panel | Disposition | Evidence |
| --- | --- | --- |
| NF-05.1 | accurate | Boot order: `main.rs:24,47,55-57,64,195`; providers: `app.rs:319-451`. The two prefix applications (`app.rs:44,902`, `base_href` `:53`) hold. The other `FORUM_BASE` reads (`:83,96,1535`, `mobile_bottom_nav.rs:30`) strip or sanitise and do not apply the prefix. **Debt** F10 added. |
| NF-05.2 | accurate | Every route verified at `app.rs:920-997`; `/chat` redirect at `:950`; zone slug resolution at `pages/category.rs:102`. |
| NF-05.3 | accurate-with-limits | Every cited call matches: `auth/passkey.rs:115,153,173,237,264,290`, `auth/webauthn.rs:123,169`, `auth/mod.rs:105,107,232,299,328,770`, `auth/session.rs:428,442`. Re-checked independently: the client posts `/auth/register/verify` unsigned (`passkey.rs:173-174`), but the worker requires NIP-98 (`nostr-bbs-auth-worker/src/webauthn.rs:838`). The worker also ignores the client salt and derives its own from the server secret and credential id (`webauthn.rs:1018-1022`), unlike the random creation salt (`:739-742`). Authenticator, PRF and cross-browser behaviour are runtime-only. |
| NF-05.4 | corrected | `nip98.rs:148` → `:149`. Other anchors hold (`auth/mod.rs:202,232,438,669,676,695,714`; `nip07.rs:85,120,165`; `session.rs:156,163,187`). |
| NF-05.5 | accurate | Frame handling at `relay.rs:363,412,528,549,569,587,601,640,715,749,769,797,817,866,889,903`; strict re-verification before dispatch at `:769`. |
| NF-05.6 | corrected | F6. Double tombstone gate at `channels.rs:405,446`; `count_for` at `:159-161`. |
| NF-05.7 | corrected | O8 fallback divergence confirmed: `relay.rs:23` `wss://relay.example.com` vs `relay_url.rs:13` `members-nostr-relay…`. O7 closure at `settings.rs:35`, `onboarding_modal.rs:59`. |
| NF-05.8 | accurate-with-limits | `signup.rs:259,261,274,283,290,292,297,309,641,647,674`, `recovery_sheet.rs:290,294,303`, `settings.rs:1847,1904,1985`. Re-checked: the downloaded sheet interpolates `display_name` raw into HTML (`recovery_sheet.rs:144,159,188`, with no escaping in the file). Print, download and fragment custody are browser behaviour. |
| NF-05.9 | corrected | ADR-2028 prose anchor `:6` → `:5`, `:7`. Mechanism holds: two wraps at `dm/mod.rs:510,527-528`, re-key to the self copy at `:512-519`, lookback at `:713`. ADR-2027 status is partial/inactive and ADR-2028 proposed/inactive, both confirmed. |
| NF-05.10 | corrected | `user_table.rs:266` → `:269`. Git panel routes at `git_panel.rs:213,272,310,359,385,434,466`; manifest at `:893,946`, with no `.well-known` path in the crate. Search fallback at `global_search.rs:336-361`. |
| NF-05.11 | accurate | `bbs-client/src/app.rs:26,45,134`, `signer.rs:106,126,229,332`, `pwa.rs:233`, `core/src/keys.rs:40`, `relay.rs:79`. Re-checked: `get_signer` hands out `Rc` clones (`signer.rs:106-108`), while `logout` only clears the stored cell (`:229-233`), so a retained clone still signs. Preamble updated to record the restamp. |
| NF-05.12 | corrected | Screen variant line labels shifted by one (F7). Other anchors hold: `chrome.rs:63,82,97`, `menu.rs:56,87,138,186`, `config.rs:68,148,162,186,267`, `passkey.rs:179,181`, `screens.rs:906-918`, `relay.rs:88-98`. |
| NF-05.13 | accurate | Verdict order at `notifications.rs:1113-1154`; burn rule at `:1099,1229`; FIFO bound `MAX_SEEN_IDS = 2_000` at `:58`, used at `:515-518`. |
| NF-05.14 | accurate | Relay text search at `global_search.rs:725-760`; zk discard at `:735-742`; runtime base at `search_client.rs:73-81,109-110`; `k` contract at `:119-126`. |
| NF-05.15 | accurate | `message_jump.rs:32,85-98,147-153,157-162,177,180,189,194-208`; `thread.rs:206,266-282,297,667`. |
| NF-05.16 | accurate-with-limits | `app.rs:795-796`, `zone_crypto/mod.rs:568,676`, `zone_crypto/store.rs:163,279,458`, `channels.rs:405,446`. Re-checked: `insert_persisted` replaces the entry for a zone and epoch outright (`zone_crypto/store.rs:290-293`), and `write_plan` refuses an encrypted zone with no key (`zone_crypto/mod.rs:563-572`). Draft retention is UI behaviour, and key lifecycle across reloads and devices is browser-only. |
| NF-05.17 | corrected | ADR-2023 anchor `:6` → `:7`. Push surfaces at `utils/push.rs:163,245,262,302,364,368,413,424,442` and `sw.js:241,335`; `PUSH_ENABLED = "false"` at `relay-worker/wrangler.toml:86`. Re-checked: `CFG_KEY` is read and deleted (`push.rs:339,356,368,374`) but never written anywhere in the file. A live push round trip is not established; the ADR says it is staged. |

### NF-06 (`nostr-rust-forum/06-agent-control-surface-31400-31405.md`)

| Panel | Disposition | Evidence |
| --- | --- | --- |
| NF-06.1 | corrected | Kinds at `governance.rs:27-34`. Only three payload structs exist (`:102,1109,1135`); there is no PanelState, PanelUpdate or PanelRetired struct at the pin. The "no consumer hardcodes" claim was qualified with **Debt** F9. |
| NF-06.2 | accurate | `governance.rs:1142-1227`. Audit alias and `DuplicateAuditEntry` are absent crate-wide; distinctness test at `:2079`. |
| NF-06.3 | corrected | Client anchors F7. Relay gates at `nip_handlers.rs:1325-1357`; correlation fallback at `receipts.rs:128-130`. |
| NF-06.4 | corrected | G1 anchor F7. Supersession gate at `:1389-1396`; kanban exception at `:1319-1322`. |
| NF-06.6 | corrected | F2 transitions redrawn. Outcome mapping at `governance.rs:1747-1750`; self-review at `:1257,1709,1806`; Promote/Demote at `:1465-1476`. |
| NF-06.8 | corrected | F3 anchors. Migrations: 0002 lines 5, 17, 39, 52; 0005 line 12; 0006 lines 65, 79, 89; 0007 line 16. |
| NF-06.9 | accurate | `governance_api.rs:355,381,420-422,429-431,434,464-466,476,490,503,532`. `require_admin` on register, provision, revoke and role grant/revoke; `require_authed` on lists. |
| NF-06.10 | corrected | F4 and F7. Member view composes read-only components (`governance.rs:36-40`); board subscribes to 31402/31403 (`board.rs:393`). |
| NF-06.11 | accurate | `receipts.rs:115,219,234,242,244,283,308,337,452`. The ADR-2024 record is absent from `docs/adr/` at the pin; KF-12 is partial (`known-findings.md:47`). |
| NF-06.12 | accurate-with-limits | `governance.rs:186,204,217,228,331-335,390,415,451,463,508-534,3092`; `nip_handlers.rs:504,2790,2827`; `wrangler.toml:61`. "Activation live" rests on ADR-2011's run record, not on this review. |
| NF-06.13 | accurate | `nip_handlers.rs:275-289,1344-1372,2839,2863`; `governance.rs:710,726-764`. |
| NF-06.14 | accurate | `governance.rs:122,357-364,368,541,560,595,695`; `nip_handlers.rs:386,401-402`; `relay lib.rs:684-711,1063`; `wrangler.toml:64-79`. |
| NF-06.15 | corrected | `supersedes_held` anchor F7. **Drift** N4 added for the stale doc at `governance_api.rs:85`. |
| NF-06.16 | accurate | `cron.rs:731,748,762-766,794-802,834-840` (closes to `CaseState::Closed`); `nip_handlers.rs:352,412-417`; `ontology_governance.rs:251,265`; migration 0007 lines 16 and 19-22. |
| NF-06.17 | accurate | `case_projection.rs:22-28,47-48,57-60,126-136`; `governance_view.rs:276-280`; `governance.rs:145-148,460,501,919-923,969-973`; `receipts.rs:124-130,159`. |
| NF-06.18 | accurate-with-limits | Probe crate `lib.rs:1-8,12-19,23-35,43-45,58`, `main.rs:10-11`. The 11/11 edge run and the activation statuses are ADR-recorded claims (ADR-2011 `:187,195,198`; ADR-2010 `:7,153,155,157`), not re-run here. The kit record pins `9c36b60` at dreamlab `0e15728` (`:26,30,38`). |
| NF-06.19 | accurate-with-limits | `rationale_check.rs:1,4-5,38,44,53,212,230,257,269,279,303`; route at `auth lib.rs:542`; flag on at `auth wrangler.toml:53-58`. Whether the deployed Worker has `TYPESAFE_API_KEY` set is not checkable from source. |

### NF-07 (`nostr-rust-forum/07-search-preview-and-shared-worker-utilities.md`)

| Panel | Disposition | Evidence |
| --- | --- | --- |
| NF-07.1 | accurate | Routes at `search-worker/src/lib.rs:917-1008`. The cron (`wrangler.toml:44`) only calls `load_store` (`:1008`). |
| NF-07.2 | corrected | Added the relay reader citation (`relay-worker/src/auth.rs:185`) and **Drift** F1. The search worker reads only `ADMIN_PUBKEYS` (`auth.rs:52-56`). It is the only template that declares the variable and the only one that ships a real 64-hex key (`wrangler.toml:41`). |
| NF-07.3 | accurate | `embed.rs:15,18,69,80,86,104,113-129`; `store.rs:17,63-80`; `lib.rs:139-160,182,420,480,495-507`. Model drift returns 409 (`lib.rs:499`). |
| NF-07.4 | accurate | `store.rs:7-8,88-97,167,185-199`; `lib.rs:127-128,205,254,297,345-351,787`; README line 81 is as stated. |
| NF-07.5 | corrected | F7 anchors. SSRF check runs before cache (`lib.rs:274` then `:287`). |
| NF-07.6 | corrected | F7 anchors and F11. No wall-clock timeout in the preview crate (no match for timeout or abort). The template sets only `ALLOWED_ORIGIN`. |
| NF-07.7 | accurate | `parse.rs:31,35,40,50,172,178,186,203,213,224`; `oembed.rs:11,38,49,65,90`. |
| NF-07.8 | corrected | F5 edge. Budgets at `auth lib.rs:175`, `search lib.rs:929`, `preview lib.rs:519`; shared replay at `search auth.rs:31,40`; `INSERT OR IGNORE` at `rate-limit/src/replay.rs:27`. |
| NF-07.9 | corrected | Confidence-floor anchor F7. Rerank runs after the visibility filter (`lib.rs:507,511`); manifest flag at `wrangler.toml:32-36`. Remote Jev behaviour and the deployed secret are not checkable from source. |

## Verification

**Evidence class:** source tracing at the pinned revision. Every cited `path:line` was read with `git show <sha>:<path>`. Line numbers were derived with `git grep -n` at the pin. No product tests, runtime probes or deployments were run.

**Pinned revisions:**
- nostr-rust-forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`
- dreamlab-ai-website `0e1572801d03a1af6d8bc8861f928da087d69eed` (NF-06 only)

**Byte identity:** the topics were previously stamped at nostr-rust-forum `1b27f79` and dreamlab `a67cb2a8`. `git diff --quiet 1b27f79 c39f41f -- <every sources: path>` succeeded for each of NF-05, NF-06 and NF-07. `git diff --quiet a67cb2a8 0e1572801d03 -- docs/architecture/kit-compatibility-record.md` succeeded. The intervening commits touch only ADRs, ADR tooling and the ontology-promotion e2e script, none of which these topics cite. So every correction above fixes an error that was already present at the old stamp, not drift between the two stamps. ADR-2013's close-out in that interval changes no claim in NF-06: the topic makes no status claim about ADR-2013.

**Restamps:**
- NF-05: `verified_commit` set to `c39f41f…`; added source `mobile_bottom_nav.rs`.
- NF-06: map set to `{nostr-rust-forum: c39f41f…, dreamlab-ai-website: 0e15728…}`.
- NF-07: set to `c39f41f…`; added sources `relay-worker/wrangler.toml` and `relay-worker/src/auth.rs`.

Every panel in each topic was given a disposition, including NF-05.3, .8, .11, .16 and .17, which earlier reports had credited.

**Gate commands**, run from the worktree root. The exit code was captured directly from `node`, with output redirected to a file:

```sh
node scripts/diagram-index-gen.cjs docs/diagrams --only 05-leptos-clients-onboarding-messaging-governance.md --check --cite-check --strict-citations --render   # exit 0; 323 citations verified at declared revision, 0 warnings, 17/17 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 06-agent-control-surface-31400-31405.md --check --cite-check --strict-citations --render              # exit 0; 224 verified, 0 warnings, 17/17 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 07-search-preview-and-shared-worker-utilities.md --check --cite-check --strict-citations --render    # exit 0; 162 verified, 0 warnings, 9/9 rendered
```

**Not verified:**
- Anything only runtime can show: authenticator PRF behaviour, push delivery, live Jev or TypeSafe answers, Worker secrets, edge deployment state, and the ADR-recorded probe runs.
- Prose citations outside Mermaid blocks (ADR status lines, for example) are not gate-checked. They were read by hand.
