---
title: Forum membership lifecycle and administrative consistency
date: 2026-11-01
status: in-progress
type: explanation
---

# Forum membership lifecycle and administrative consistency

Shared cohort SQL repairs four destructive writers, but the forum still has separate definitions of membership and administrator authority. The client can call a live member pending, demote an administrator in only one store, or report deletion while auth and pod cleanup remain incomplete. ADR-2014 proposes the right consolidation boundary; most of that boundary is still unbuilt.

This pass assesses every section of ADR-2014, adds six NF-25 panels and corrects the normalisation and failure claims in NF-08.7. Other NF-08 panels retain their earlier review scope. It does not implement the proposed membership service or certify a deployed onboarding journey.

## Evidence and limits

The [manifest](evidence/2026-11-01/nf-membership/source-manifest.json) pins 25 source/document exports to forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`. All match the old diagram revision `1b27f79` and website-consumed revision `9c36b6032640c89879ae52fe25da99493979a0b6`. These comparisons establish file parity, not deployed database identity.

Thirty-seven unique native cases pass: 15 shared-SQL cases, 12 client membership cases, six relay cohort-delta cases and four source/migration guards. Six are added audit cases; baseline reruns are not counted twice. Seven additional scenarios execute SQL extracted from the retained Rust against disposable SQLite, including positive controls. The client cases call the actual pager and pending derivation with the existing fake relay; the 50,000-row ceiling case uses the actual default cap. No live Worker, D1, browser, administrator account or member state was exercised. [Reproduction and receipts](evidence/2026-11-01/nf-membership/README.md) preserve these distinctions.

## ADR-2014 section assessment

| Section | Current assessment |
| --- | --- |
| Metadata and context | Proposed/partial/staged accurately describes consolidation. `verified_commit: 1a26e51` identifies the earlier SQL/DDL slice. The original first-page defect is repaired, but list completeness is still conditional (M1). |
| D1: stores | Auth reservations, auth members and relay whitelist remain distinct. The checked-in whitelist DDL includes `expires_at`; the older “none in code” map is explicitly historical. This pass traces the membership stores and principal writers; device, alias and trust enforcement retain the limits of their separate reviews. |
| D1: writers | Admin add, cohort delta, agent provision and alias inheritance use shared grant SQL. The inventory records current call sites. Invite redemption still performs a read/merge/write outside that statement. Username auto-whitelisting uses `DO NOTHING`; it does not create auth membership. |
| D1: administrator resolution | The auth worker checks static authority, relay whitelist and auth members. The relay now checks only static authority and whitelist: the map's relay-members query is historical. UI demotion still leaves auth authority behind (M3). |
| D1: UI surfaces and cohort derivation | Pending subtracts whitelist rows from active reservations. Bell alerts instead select non-admin/non-agent users lacking a zone-gating cohort. The approval default, editor and alert calculation remain separate functions. Trust and silence are still absent from the list contract (M5). The Access view still subscribes to 9021 and emits separate confirmations; the proposed deletion has not occurred. This pass does not re-prove the historical whole-tree claim that no producer exists. |
| D2: states and handle rules | There is no authoritative requested/active/declined/suspended/removed state machine. Dismissal changes the reservation status and retains its uniqueness slots; it does not transition the relay row. Relay deletion removes the row, with no tombstone (M6). |
| D2: cohort set | The shared merge protects existing valid string cohorts from an ordinary grant. Duplicate incoming values remain possible, and a legacy null can suppress additions (M2). Invite writes can still lose concurrent grants. |
| D2: trust, suspension and agents | Trust computation remains separate, as the [trust review](2026-11-01-forum-trust-transitions.md) describes. Agent provision batches whitelist and registry writes, but revoke updates registry activity without suspending the whitelist member. That is a lifecycle gap; it does not by itself prove an agent-specific admission bypass. |
| D2: zone access | The proposed filter would give operators a way to find the people named by alerts. Current Pending and awaiting-access predicates answer different questions. Keep both states visible without presenting them as interchangeable. |
| D3: authoritative store/module | No relay members table, relay membership module or unified transition route is present in the bounded source/migration inventory. Shared SQL is useful groundwork, not the full module. Administrative audit writes remain separate and sometimes ignored; one successful state update is not proof of an audit commit. |
| D3: data ownership | Retaining real names in auth and joining by public key is an explicit privacy boundary. The future joined admin endpoint needs authorisation, failure and partial-result contracts across both stores. Consolidation must not quietly move real names into relay events. |
| D4: endpoint and UI | Server-side state/cohort/query/cursor filtering and bulk transitions are proposed. The current client still walks OFFSET pages and derives Pending locally. Specify whether bulk requests are all-or-nothing or independently committed per person: the record currently promises both an atomic batch and per-person failures without resolving their relationship. |
| D5: copy and expired rows | Copying an expired `expires_at` into a past `suspended_until` while setting state active is ambiguous. Current admission denies past expiry (`storage.rs:372`); current suspension checks only future timestamps (`trust.rs:560–568`). Define an explicit expired/indefinite-denial mapping before claiming admission equivalence. No proposed migration was executed. |
| D5: admin OR and collisions | OR-ing auth and relay flags preserves any residual grant, including M3. Require an explicit authority-reconciliation policy. Lowercase collisions need one merged identity and a retained provenance record; the fixture's “exactly N active” must use unique canonical keys, not raw whitelist-row count. |
| D5: dual-write, fallback and reconciliation | No membership reconciliation or cut-over evidence exists. Seven zero-difference days must mean successful complete scans with pinned schemas and a watermark, not seven empty/error reports. Fallback must apply only to a genuine missing row, never a denied/tombstoned member or a failed lookup. These are design acceptance requirements. |
| D6 phase 0 | Static pagination beyond the original 20-row page works. Calling this “every page” is too strong under cap exhaustion and mutation between pages (M1). Preserve the repaired baseline and add completeness criteria. |
| D6 phase 1 | DDL, four shared writers and cohort deltas exist. Lowercase enforcement, revoke-by-code and moderation fields in the list remain incomplete. Existing source guards test literal SQL idioms and exempt the invite file; passing guards are not exhaustive writer correctness. |
| D6 phase 2 | New schema, migration, transition module and atomic audit remain unbuilt. Required fixtures must cover expiry, case collisions, conflicting admin flags, removed identities and failed audit writes. |
| D6 phase 3 | Dual-write, fallback and reconciliation remain unbuilt. Add interrupted writes, scan failure, denied fallback and suspended device-owner cases to the proposed parity check. |
| D6 phase 4 | Unified table and bulk transition contract remain unbuilt. The proposed 50-member approve/refetch and signup-to-post tests are still required; this pass ran neither the WASM check nor that journey. |
| D6 phase 5 | Auth members, whitelist and reservation status still have active readers. Shared SQL constants do not make the two resolver implementations identical. Retirement cannot precede migration and consumer acceptance. |
| Consequences | No Pending bounce, atomic bulk audit, lossless grants and shared agent/human revocation are future outcomes. M1–M6 prevent treating them as current guarantees. |
| Costs and follow-on work | Migration, an auth-to-relay dependency and legacy readers remain real costs. The identity document update is conditional on phase 2. The record's stale-comment cleanup is still a separate documentation obligation; this pass does not claim every named prose site is repaired. |
| Forbidden after acceptance | The decision is proposed, so these rules describe a future boundary. Turn them into executable write-path and state-derivation checks when accepting it, with an explicit migration exception and expiry. |
| Verification | Historical suite totals and WASM/clippy results are not today's results. The sentence saying no repository DDL creates `expires_at` is stale beside the later phase-1 evidence. Production schema remains unverified here. |
| Disposition and 2 October update | The initial replace/no-DDL assessment is superseded by the same-day update. P2 is the recorded owner priority, not a conclusion this audit silently changes. M3 merits prompt authority testing in any deployed admin workflow. The inspected files match the consumed kit revision; delivery identity remains a separate question. |

## Reproduced gaps and operational consequences

### M1 — An incomplete list can become a false Pending decision

`admin/membership.rs:82–99` returns collected rows after any stop condition, including 500 pages. A synthetic 50,001-member response set leaves one active registration pending after 50,000 rows. In a smaller 201-member fixture, deleting the newest row after page one shifts the next OFFSET and skips a still-live member. Deduplication handles repeats, not omissions. The source query orders by `added_at DESC` without a key tie-break and counts separately from its page read (`whitelist.rs:235–280`).

The SQL expiry case adds a second route to repeated Pending: granting another cohort does not clear an expired row's `expires_at`. It remains absent from the active list and fails the storage admission predicate. Fix the state transition rather than repeatedly approving the same key. Short-term, return explicit completeness and refuse to infer Pending from a truncated scan; long-term, answer requested membership on the server with stable paging.

Case-folding only in the client creates another mismatch: its pager keeps the first case-variant row and discards the second row's cohorts. An upper-case-only whitelist row can disappear from Pending while the canonical lower-case key misses the database admission query. The native and SQL cases reproduce these two boundaries separately; they do not execute an end-to-end signed admission.

### M2 — Cohort preservation has input and concurrency limits

Executing the shared SQL retains duplicate values on a new row and duplicate newly granted values on an existing row. With a legacy `["home", null]` array, `NOT IN` suppresses the new grant and the result remains unparseable as `Vec<String>`. Ordinary typed writer inputs cannot introduce that null; the case tests legacy or externally malformed storage.

Invite redemption also remains vulnerable to concurrent grants. After invite code reads `[members]`, a shared-SQL grant adds `zoneB`; the invite's later replacement writes `[members, zoneA]`, dropping `zoneB`. A serial control preserves both. These are chosen SQLite statement schedules, not an observed D1 race. Reuse the atomic shared grant for invites, validate stored string arrays and make set uniqueness explicit. Keep the editor's add/remove delta: it already avoids revoking grants the editor never saw.

### M3 — Relay demotion can retain auth administrator authority

`admin/mod.rs:426–452` calls only `/api/whitelist/set-admin`. The relay updates its row (`whitelist.rs:451`), while auth resolution still accepts `members.is_admin=1` (`admin.rs:90–103`). The SQL fixture ends with relay flag zero and auth flag one. With no matching static grant, the source resolvers therefore disagree. The auth-side remove handler attempts both stores, but the UI does not call it and its writes are not one cross-database transaction.

NF-08.7 previously claimed every lookup lowercased keys and any D1 error returned false. Auth normalises at entry; relay's local query binds its supplied key. Auth can continue after a failed relay lookup and succeed through members. These are union-of-authority semantics, not a global denial on any error. A revocation acceptance test must exercise every authority source, and explicitly distinguish an immutable configured admin from a revocable dynamic grant.

### M4 — Invite revocation sends a code to an ID lookup

Revocation embeds `row.code` in the revoke URL (`admin/invites.rs:175`); the handler queries `WHERE id = ?1` (`invites.rs:733`). Creation generates distinct eight-character IDs and sixteen-character codes. The fixture's code misses while its ID succeeds. The source maps the miss to 404, and the UI displays failure. Align the resource identifier and add a client/handler contract test; this is not a silent-success claim.

### M5 — Displayed moderation state is not loaded state

Listing members returns pubkey, cohorts, time, display name and admin flag (`whitelist.rs:315–321`). `WhitelistUser` omits trust/suspension/silence. The table starts silence at false (`user_table.rs:652`) and prints TL0 (`:789`). An operator can therefore see an unmuted/TL0 presentation for a differently configured member. Return authoritative fields and distinguish unavailable state from a false/zero value. This is a source finding, without a rendered browser reproduction.

### M6 — Removal is not the proposed durable lifecycle transition

Relay deletion removes the whitelist row (`user_admin.rs:199`). The SQL fixture can subsequently re-create it through the existing grant statement; that demonstrates absent tombstone enforcement, not unauthorised re-admission. Client cleanup then attempts primary-pod and auth deletion separately, logs failures and still reports Deleted (`admin/mod.rs:492–544`). A complete removal needs durable per-system outcomes and retry/reconciliation. Dismissal similarly changes only the reservation status (`username.rs:762`), so it must not be described as revoking an already admitted member.

## Comparison and user utility

SCIM provides useful membership protocol precedents: filtered queries, targeted multi-value additions/removals and idempotent addition of an existing value. Its pagination section explicitly warns that stateless paging need not produce consistent results across requests. Adopting a standard-shaped API alone would not solve M1; the implementation needs an explicit consistency/completeness contract. These are design comparisons, not a claim of SCIM compatibility. [RFC 7644 §§3.4.2.4 and 3.5.2](https://www.rfc-editor.org/rfc/rfc7644.html#section-3.4.2.4).

Keycloak separates updating a user from a logout operation that removes the user's sessions and notifies clients with an admin URL. That is a useful comparison for explicit propagation outcomes, rather than assuming a changed row reaches every consumer. The documentation does not establish universal immediate revocation. [Keycloak 26.4.7 Admin REST API](https://www.keycloak.org/docs-api/26.4.7/rest-api/).

A potential advantage for the forum is one understandable path from a Nostr identity to community, zone and agent participation. Operators should be able to answer “why can this person enter?” and “did removal take effect everywhere?” without comparing stores. Prioritise a canonical authority decision and observable transition receipts before adding another admin tab. Measure approval-to-first-post success, repeated Pending after approval, inconsistent admin decisions and incomplete removal age. A useful acceptance run includes concurrent grants, an expired member, a case collision, partial cleanup and retries; no market adoption or usability benefit was measured in this pass.

This review completes the bounded ADR-2014 section assessment, while implementation, deployment acceptance and remaining forum decisions/topics stay open.
