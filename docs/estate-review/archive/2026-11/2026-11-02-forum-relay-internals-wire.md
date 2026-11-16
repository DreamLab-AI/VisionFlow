---
title: Forum relay internals and core wire primitives
date: 2026-11-02
status: assessed
type: explanation
---

# Forum relay internals and core wire primitives

I assessed all 26 panels in NF-11 (relay Durable Object internals, 16 panels) and NF-12 (core wire, crypto and federation envelope, 10 panels) against nostr-rust-forum `c39f41f104b1`. Both topics are restamped to that revision. Results: 9 panels accurate, 16 corrected, 1 accurate with limits, none inaccurate-unfixed. The most important finding is in NF-11.1: the relay's admin-status cache expires after 30 seconds, not the 5 minutes the panel showed. The 5-minute figure was copied from a product comment that contradicts the constant it documents. The other substantive corrections:

- NIP-50 search was attributed to a REST endpoint.
- The scheduled-work order was drawn wrongly, and the push sweeps were missing.
- The broadcast read gate's scope was overstated.
- The federation `MockSocket` was described as test-only, but it is public.

Every source file the two topics cite has the same bytes at the old stamp `1b27f79d` and at `c39f41f104b1`, so the errors were already there at the old stamp. None of them comes from code moving since.

## Findings

**F1. The AdminCache TTL is 30 s; the product comment says 5 minutes (Drift, product).**
- **Trigger:** NF-11.1 listed `admin_cache : AdminCache 5 min TTL`.
- **Evidence:** the field comment says "P2-03: 5-minute TTL cache for admin status" (`crates/nostr-bbs-relay-worker/src/relay_do/mod.rs:98`). `AdminCache` checks `ADMIN_CACHE_TTL_SECS`, which is `30` (`crates/nostr-bbs-relay-worker/src/auth.rs:46`, used at `auth.rs:110`).
- **Consequence:** anyone reasoning about how long a revoked admin keeps admin rights on one DO isolate would get a window ten times too long.
- **Status:** the panel is corrected and carries a `**Drift:**` marker. Fixing the field comment would close it.

**F2. NIP-50 was attributed to the profiles REST endpoint (diagram error, corrected).**
- **Trigger:** NF-11.14 N3 named `profiles.rs:251` as the NIP-50 handler, and NF-11.15 N2 said `profiles::search` "backs the advertised NIP-50".
- **Evidence:** `handle_search` serves `GET /api/profiles/search` (`profiles.rs:248`). NIP-50 is the REQ `search` field, applied in SQL (`relay_do/filter.rs:166`) and on the live path (`relay_do/filter.rs:281-282`).
- **Fix:** both notes are corrected, and NF-11.6 now draws the search constraint in the live predicate. Closed by this restamp.

**F3. The broadcast read gate's scope was overstated (corrected).**
- **Moderation reads:** NF-11.5 said a moderation event "needs only AUTH". `protected_read_permitted` requires AUTH for moderation kinds only in `nip42` mode (`relay_do/nip42.rs:103`). In `allowlist` mode they pass with no authentication.
- **Viewer context:** the panel showed `resolve_viewer_context` running for every candidate. It runs only when `needs_gate` holds, which covers kinds 40 and 42, calendar and kanban (`relay_do/broadcast.rs:75`, `:148`).
- **Citation:** `broadcast_event` is at `broadcast.rs:62`, not `:61`.
- **Status:** closed by this restamp.

**F4. The calendar projection's public arms were missing, and the README note was stale (corrected).**
- **Public zone:** NF-11.8 omitted the three arms that serve a public-zone event in full to friends, business and uncohorted viewers (`relay_do/calendar_projection.rs:112`, `:127`, `:137`).
- **README:** N4 claimed `README.md:79` asserts "25 unit tests". At the pinned revision that line states only the three outcomes. The module does carry 25 `#[test]` functions from `:190`.
- **Status:** closed by this restamp.

**F5. Scheduled work runs in a fixed sequence that includes push sweeps (corrected).**
- **What the panel showed:** NF-11.13 drew retention, trust and ageing fanning in to expiry.
- **What the code does:** the scheduled entry runs strictly in this order (`src/lib.rs:1147` onward):
  1. trust sweep
  2. `push::sweep_stale` (`lib.rs:1179`)
  3. `push::flush_pending` (`lib.rs:1181`)
  4. retention
  5. ageing
  6. expiry
- **Also added:** the five-minute cadence is now cited (`crates/nostr-bbs-relay-worker/wrangler.toml:123`).
- **Status:** closed by this restamp.

**F6. Admin audit writes are best-effort (Debt, product).**
- **Evidence:** every `user_admin` and `moderation` mutation calls `let _ = audit::log_admin_action(...)` (`user_admin.rs:229`, `:307`, `:370`, `:465`, `:625`; `moderation.rs:351`).
- **Consequence:** a failed `admin_log` insert never fails the mutation, so the audit log can miss an action that took effect.
- **Panel change:** NF-11.15 N4 had said "every admin mutation routes through log_admin_action". It is narrowed and marked `**Debt:**`.
- **What would close it:** propagate or record the audit failure.

**F7. `MockSocket` is public, not test-only (corrected).**
- **Evidence:** NF-12.8 called it "test-only". `nostr-bbs-mesh/src/mock.rs:11` says it is deliberately not behind `cfg(test)` or a feature gate, and `lib.rs:48` exports it.
- **Unchanged:** it is still the only `MeshSocket` implementation in the tree (`mock.rs:249`).
- **Citation added:** the relay's dead-code wiring is now cited (`crates/nostr-bbs-relay-worker/src/mesh.rs:48`).
- **Status:** closed by this restamp.

**F8. `wasm_bridge` has one JS consumer: a benchmark (corrected).**
- **Evidence:** NF-12.1 said "no JS consumer exists in this repo". `benchmarks/js-vs-wasm/bench.mjs:32-39` imports `schnorr_sign`, `nip44_encrypt` and others through `@dreamlab/nostr-core-wasm`.
- **Panel change:** it now says no *production* consumer. Closed by this restamp.

**F9. `derive_subkey`'s doc comment says "reduce" (Drift, product).**
- **Evidence:** `keys.rs:223` and `keys.rs:262` say the HMAC output is "validated/reduced". `SecretKey::from_bytes` only validates and rejects (`keys.rs:51`).
- **Status:** NF-12.3 already drew the behaviour correctly ("no reduction or retry") and is credited accurate. The topic now carries a `**Drift:**` marker.
- **What would close it:** rewording `keys.rs:223` and `:262` to say validate-and-reject.

**F10. The "every unwrap error is a fixed string" invariant was too broad (corrected).**
- **What holds:** the parse and decrypt errors are fixed strings (`gift_wrap.rs:189-190`).
- **What doesn't:** `InvalidKind` and `AuthorMismatch` interpolate kind numbers and both pubkeys (`gift_wrap.rs:53`, `:89`). These are metadata from the decrypted layers, not plaintext content.
- **Citation fixed:** the author-binding citation moved from the parameter line `:428`, which JEV flagged, to the check at `:494-497`.
- **Status:** closed by this restamp.

**F11. NIP-98 entry points (corrected).**
- **Title:** NF-12.5's title now says "authenticated" REST call. The relay also serves public endpoints such as `agent_disclosure`.
- **N3:** it previously named `verify_nip98` (`nip98.rs:314`) as the path every worker uses. Workers actually call `nostr_bbs_rate_limit::verify_nip98`, which delegates to the core replay path (`nostr-bbs-rate-limit/src/replay.rs:94`). Both reach `verify_token_full` (`nip98.rs:576`).
- **Unchanged:** the ten numbered checks are confirmed as drawn. Only the pubkey and timestamp order within one node was swapped to match the code.
- **Status:** closed by this restamp.

**F12. Citation-only corrections:**
- **Trust sweep:** the demoted counter moves at `trust_sweep.rs:286`, not `:288`.
- **Deletion:** the 64-hex target check is at `deletion.rs:59`, not `:68`.
- **NF-11.1 cross-reference:** it pointed to NF-11.16 for read-side caching, which is NF-11.17. It also said "four 60 s memos"; the code has three memos plus one ledger.
- **New citations:**
  - NF-11.10: atomic commit (`receipts.rs:672`, `:716`, `:558`).
  - NF-12.2: the three callers of `verify_event_strict` (`relay_do/nip_handlers.rs:1077`, `relay_do/nip42.rs:187`, `nostr-bbs-forum-client/src/relay.rs:769`).
  - NF-12.2: `verify_event` is a thin wrapper (`event.rs:197-198`).
  - NF-12.1: the forum client's bech32 encoder sits under `cfg(test)` (`auth/mod.rs:843`).
- **Status:** closed by this restamp.

## Panel dispositions

Paths are in nostr-rust-forum at `c39f41f104b1`. `relay_do/` is `crates/nostr-bbs-relay-worker/src/relay_do/`.

| Panel | Disposition | Evidence |
| --- | --- | --- |
| NF-11.1 | corrected | AdminCache TTL 5 min → 30 s (`auth.rs:46`; Drift `mod.rs:98`); "three memos plus a ledger"; cross-reference → NF-11.17; 426 cited `mod.rs:132`. Fields `mod.rs:86-108`, slot reservation `mod.rs:394`/`:408`, AUTH refused-OK `mod.rs:307` and `close_admitted` `mod.rs:385` confirmed. |
| NF-11.2 | accurate-with-limits | `fetch` recovers before allocating (`mod.rs:156`); recovery on both paths (`session.rs:75`, `:109`, `:159-165`); persisted mark, never below 1 (`session.rs:285-302`, `:533`); challenge reuse (`session.rs:186-218`, `:443`). Limit: hibernation and tag survival are Cloudflare platform behaviour, not visible in source. |
| NF-11.4 | accurate | Treatment ranges `broadcast.rs:28-36`; deletes `storage.rs:100`, `:121`; `ON CONFLICT (channel_id) DO NOTHING` backs "first write wins" (`storage.rs:162-163`); discarded result `storage.rs:171`. |
| NF-11.5 | corrected | `broadcast_event` `:62`; moderation AUTH only in nip42 mode (`nip42.rs:103`); viewer context only when `needs_gate` (`broadcast.rs:75`). 1059 recipient gate `broadcast.rs:129` and stored gated filter `nip_handlers.rs:1818-1825` confirmed. |
| NF-11.6 | corrected | Added the live NIP-50 constraint (`filter.rs:281-282`); `tag_filter_plan` is shared by the SQL builder (`filter.rs:129`) and the live matcher (`filter.rs:292`), not SQL-only. Other cited fields and constraints confirmed. |
| NF-11.7 | accurate | `is_blocked` maps Unknown→true and MutedUntil→`t > now` (`mod_cache.rs:67`); Unknown never cached (`mod_cache.rs:85`); `invalidate` is called on admin ban, mute, unban or unmute mirroring (`nip_handlers.rs:1590`). |
| NF-11.8 | corrected | Public-zone arms added (`calendar_projection.rs:112`, `:127`, `:137`); stale README test-count claim replaced; 25 tests counted from `:190`. |
| NF-11.9 | accurate | Variants `governance.rs:940-962`; Ord caveat `:919-935`; `ladder_rank` `:999`; re-export `receipts.rs:77`. |
| NF-11.10 | corrected | Atomic batch now cites `receipts.rs:672`, `:716` and `run_batch` `:558`. Read API `receipts.rs:869`, route `lib.rs:340`, flags `:844-850`, correlation `:128-130` confirmed. Limit: N4/N5 restate ADR-2010's live-journey record (`ADR-2010…md:153`, `:157`), not reproduced. |
| NF-11.11 | corrected | Counter citation `:288` → `:286`; cron cadence cited (`wrangler.toml:123`). Cursor advance before commit (`trust_sweep.rs:274-276`), one-batch commit (`:488`), page abort (`:117-121`) and fresh cursor (`:253`) confirmed. |
| NF-11.12 | accurate | Module rationale `trust_sweep.rs:13-38`; snapshot-guarded UPDATE and `WHERE changes() = 1` audit (`:418-427`); TL2→TL0 note `:231-233`. |
| NF-11.13 | corrected | Strict sequence drawn, push sweeps added (`lib.rs:1179`, `:1181`). Backfill, retention, ageing and expiry citations confirmed (`cron.rs:72`, `:345`, `:592`, `:748`, `:419`, `:571-572`, `:603-613`). |
| NF-11.14 | corrected | N3: NIP-50 is the REQ field (`filter.rs:166`, `:282`), not `profiles.rs:251`. Same-parser `auth_mode` (`nip11.rs:155-158`), labels `:164-170` and `supported_nips` `:207` confirmed. |
| NF-11.15 | corrected | N2 NIP-50 misattribution fixed; N4 narrowed to the audited call sites plus a Debt marker for discarded audit results. Alias exclusion (`profiles.rs:307`, `:318`) and cohort merge (`user_admin.rs:602-612`) confirmed. |
| NF-11.16 | accurate | Drift exemption past-only (`nip_handlers.rs:1727-1728`), well-formed vs any-tag split (`:1713-1717`), rejection order (`:1748-1760`, `:1435-1449`), ciphertext shape (`zone_config.rs:208`). |
| NF-11.17 | corrected | WRITE label now says every event is recorded and only the stamp is throttled (`nip_handlers.rs:2237-2238`). Memo behaviour (`:2189`, `:2203`, `:2216`), thresholds (`read_cache.rs:44-48`) and flush body (`:2245`) confirmed. Limit: the ~110 D1 queries and eviction-loss figures come from the module doc (`read_cache.rs:5-15`, `:31-33`) and were not measured. |
| NF-12.1 | corrected | O10 reworded to "no production consumer" (`bench.mjs:32-39`); nip19 universal narrowed (`auth/mod.rs:843`). Module lines `lib.rs:29-57`, licence and feature (`Cargo.toml:6`, `:23-28`), pod-worker `did` (`pod-worker/Cargo.toml:29`), did parity (`did.rs:15`, `:74`, `:109`, `:121`) confirmed. |
| NF-12.2 | corrected | Caller claims now cited (`nip_handlers.rs:1077`, `nip42.rs:187`, `forum-client/src/relay.rs:769`), plus the `verify_event` wrapper (`event.rs:197-198`). Signing path `event.rs:78-141` and verify `:205-241` confirmed. |
| NF-12.3 | accurate | HKDF Extract/Expand with counter (`keys.rs:198-202`); HMAC child validated, not reduced (`keys.rs:254-263`, `:51`); vector test `identity_subkey_vectors.rs:151`; JS substring check `identity-vector-parity.mjs:204`. Drift on the product comment recorded separately (F9). |
| NF-12.4 | corrected | Author-binding citation → `gift_wrap.rs:494-497`; fixed-string invariant narrowed (`:53`, `:89`). Outer verify before decrypt `:452`, seal verify `:475`, past-only jitter `:161`, Zeroizing `:335` confirmed; Open marker statuses confirmed (ADR-2027 `:5-7` accepted, partial, inactive; ADR-2028 `:5-7` proposed, none). |
| NF-12.5 | corrected | Ten numbered checks confirmed (`nip98.rs:417-498`); title, node order and N3 entry-point account corrected (`nip98.rs:576`, `rate-limit/src/replay.rs:94`). |
| NF-12.6 | corrected | Deletion check `deletion.rs:68` → `:59`. Kinds and ADMIN_ONLY list without REPORT (`moderation_events.rs:67-74`), legacy `h` builder vs `d` builder (`groups.rs:146`, `:254-268`) confirmed. |
| NF-12.7 | accurate | `from`/`to` doc lines `envelope.rs:162-164`; six kinds `:81`; AS2 mapping `:97-106`; validation of version, DIDs, the `via` chain cap, body size and delegation structure `:255-294`, `:458`, `:463`; delegation types `delegation.rs:49`, `:187`. |
| NF-12.8 | corrected | `MockSocket` is public, not test-only (`mock.rs:11`); dead-code wiring cited (`relay-worker/src/mesh.rs:48`). JCS rationale `jcs.rs:4-12`, transport traits `transport.rs:238`, `:251`, `:325` confirmed. |
| NF-12.10 | accurate | One rumor, two seals (`gift_wrap.rs:667-678`); doc rationale `:631-633`, `:648-650`, `:654-656`. |
| NF-12.11 | accurate | Seal preconditions (`sealed.rs:312-330`), outer tags (`:462-485`), six reader checks and strict inner JSON (`sealed.rs:375-416`, `deny_unknown_fields` at `:424`), caller duty (`:71-76`). Migrator verifies the outer signature (`engine.rs:550`). `open_and_check` never compares the outer `zk` tag to the key it is handed (`engine.rs:544-560`, key chosen from state at `:665`), so the NF-26.2 gap remains in source. |

## Verification

- **Evidence class:** source tracing at the pinned revision. I read every cited line with `git -C /home/devuser/workspace/nostr-rust-forum show c39f41f104b1:<path>`, plus the surrounding function wherever the claim depended on behaviour. No product tests were run. Nothing here is runtime, deployment or test evidence.
- **Pinned revision:** nostr-rust-forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`. Both topics' `verified_commit` was changed from `1b27f79d08537b61b3d5a5d02cac9ee5999b79ed` to that SHA. `git diff --stat 1b27f79d c39f41f104b1 -- <all 71 source paths of both topics>` is empty. The same diff is also empty for the two ADRs that NF-12.4 cites outside `sources:` (ADR-2027, ADR-2028), and both were read directly at the pin.
- **Citation discipline:** new and moved line numbers were taken from `git grep -n` at the pinned SHA, never by shifting.
- **Gate commands,** run from the worktree root after the final edit:
  - `node scripts/diagram-index-gen.cjs docs/diagrams --only 11-relay-durable-object-internals.md --check --cite-check --strict-citations --render`: exit 0; 305 citations verified at the declared revision, 0 unresolvable, 16/16 rendered.
  - `node scripts/diagram-index-gen.cjs docs/diagrams --only 12-core-wire-crypto-and-federation-envelope.md --check --cite-check --strict-citations --render`: exit 0; 194 citations verified, 0 unresolvable, 10/10 rendered.
  - An earlier topic 12 run failed only on NF-12.4's width (4677 px), which was fixed by wrapping the note.
  - No JEV advisories remain. The pre-edit advisory on `gift_wrap.rs:428` is cleared.
- **Not verified:**
  - Cloudflare DO hibernation semantics (NF-11.2).
  - The live receipt journeys recorded in ADR-2010 (NF-11.10).
  - The D1 query-cost figures in the read-cache module doc (NF-11.17).
  - Cross-references to other topics (NF-03, NF-05, NF-06, NF-08, NF-24, NF-26, AB-*, ES-*), beyond confirming that the code they describe in this topic exists.
  - External-repository claims marked EXTERNAL.
