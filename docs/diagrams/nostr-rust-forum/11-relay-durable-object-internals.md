---
id: NF-11
title: Inside the relay Durable Object — sessions, storage, filters, broadcast, projection, receipts and the scheduled sweeps
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2005, ADR-2006, ADR-2010]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/session.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/broadcast.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/filter.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/mod_cache.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/calendar_projection.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip42.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/cron.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/nip11.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/audit.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/user_admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/profiles.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/moderation.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/migrations/0005_governance_receipts.sql
  - ../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
  - ../nostr-rust-forum/README.md
verified_commit: 380a595f150dd96bfe27ff278fff9ded1be7fbd0
---

## NF-11.1 The Durable Object and its in-memory state

```mermaid
classDiagram
    class NostrRelayDO {
        state : State — relay_do/mod.rs:79
        env : Env — relay_do/mod.rs:80
        sessions : RefCell HashMap u64 SessionInfo — relay_do/mod.rs:81
        next_session_id : relay_do/mod.rs:82
        rate_limits : RefCell HashMap String Vec f64 — relay_do/mod.rs:83
        rate_limit_per_sec : Cell Option usize — relay_do/mod.rs:87
        connection_counts : relay_do/mod.rs:88
        mod_cache : ModCache 60 s TTL — relay_do/mod.rs:90
        admin_cache : AdminCache 5 min TTL — relay_do/mod.rs:93
    }
    class DurableObject {
        new : relay_do/mod.rs:97
        fetch — websocket upgrade only, else 426 : relay_do/mod.rs:111
        websocket_message : relay_do/mod.rs:187
        websocket_close : relay_do/mod.rs:306
    }
    NostrRelayDO ..|> DurableObject

    note for NostrRelayDO "EVERY field except state and env is volatile in-memory cache. Hibernation wipes all of it, which is what NF-11.2 exists to survive."
    note for NostrRelayDO "rate_limit_per_sec is lazily resolved and CACHED because Env::var crosses the JS boundary on every read relay_do/mod.rs:84-86"
    note for DurableObject "The DO is a singleton reached by get_by_name main from the worker fetch nostr-bbs-relay-worker/src/lib.rs:172 - see NF-03.1"
```

## NF-11.2 Hibernation — what survives, and why the challenge must not be re-minted

```mermaid
sequenceDiagram
    autonumber
    participant C as Client socket
    participant DO as NostrRelayDO
    participant TAG as WebSocket tag
    participant ST as DO storage

    DO->>TAG: tag the socket so session data survives hibernation relay_do/mod.rs:141
    Note over DO: DO hibernates - sessions, rate limits and connection counts are gone
    C->>DO: next frame arrives
    DO->>DO: find_session_id in memory relay_do/session.rs:40
    alt not found - woke from hibernation
        DO->>ST: recover_session relay_do/session.rs:75
        ST-->>DO: subscriptions + load_auth authed_pubkey relay_do/session.rs:158
        DO->>DO: log recovered sessions, subs and authed count relay_do/session.rs:216
    end

    Note over DO: INVARIANT: the challenge issued BEFORE hibernation is preserved, so a client that connected earlier can still answer its ORIGINAL challenge relay_do/session.rs:106-109
    Note over DO: Minting a fresh challenge on recovery would silently invalidate every in-flight AUTH relay_do/session.rs:109
    Note over ST: authed_pubkey is persisted, so authenticated operations continue across the hibernation boundary relay_do/session.rs:54-59
    Note over DO: recovered_challenge is a PURE decision function, unit-testable without a DO relay_do/session.rs:381
```

## NF-11.3 Frame dispatch

```mermaid
flowchart TB
    WM["websocket_message<br/>relay_do/mod.rs:187"]
    STR["String or Binary to UTF-8<br/>relay_do/mod.rs:192"]
    SESS["find_session_id else recover_session<br/>relay_do/mod.rs:199"]
    JSON["parse as a JSON array of length >= 2<br/>relay_do/mod.rs:211 relay_do/mod.rs:219"]
    EV["EVENT to handle_event<br/>relay_do/mod.rs:236 - see NF-03.4"]
    RQ["REQ to handle_req<br/>relay_do/mod.rs:246, handler relay_do/nip_handlers.rs:1002"]
    CL["CLOSE relay_do/mod.rs:264"]
    AU["AUTH relay_do/mod.rs:269 - see NF-03.2"]
    CO["COUNT relay_do/mod.rs:279, handler relay_do/nip_handlers.rs:1502"]
    UN["unknown frame relay_do/mod.rs:299"]
    WC["websocket_close relay_do/mod.rs:306"]

    WM --> STR --> SESS --> JSON
    JSON --> EV & RQ & CL & AU & CO & UN

    N1["A malformed frame is answered with a NOTICE rather than a socket close - the relay never drops a<br/>connection for one bad message relay_do/mod.rs:211-219"]
    N2["Subscriptions are capped at MAX_SUBSCRIPTIONS = 20 per session, enforced on REQ<br/>relay_do/nip_handlers.rs:44 relay_do/nip_handlers.rs:1028"]
```

## NF-11.4 Storage — NIP-16 treatment decides what a save deletes

```mermaid
flowchart TB
    T["event_treatment<br/>relay_do/broadcast.rs:27"]
    EPH["Ephemeral 20000-29999<br/>relay_do/broadcast.rs:28 - never saved, OK then broadcast"]
    REP["Replaceable 10000-19999 plus kinds 0 and 3<br/>relay_do/broadcast.rs:30"]
    PAR["ParameterizedReplaceable 30000-39999<br/>relay_do/broadcast.rs:32"]
    REG["Regular - everything else<br/>relay_do/broadcast.rs:34"]
    SAVE["save_event<br/>relay_do/storage.rs:66"]
    INS["INSERT INTO events with d_tag and received_at<br/>relay_do/storage.rs:80"]
    D1D["Replaceable: DELETE older rows for (pubkey, kind)<br/>relay_do/storage.rs:100"]
    D2D["Parameterized: DELETE older rows for (pubkey, kind, d_tag)<br/>relay_do/storage.rs:121"]

    T --> EPH & REP & PAR & REG
    REP --> SAVE --> INS --> D1D
    PAR --> SAVE
    INS --> D2D
    REG --> SAVE

    N1["The d_tag column is what makes parameterised replacement a single indexed DELETE rather than a scan -<br/>it is written at INSERT time relay_do/storage.rs:80, extracted by relay_do/filter.rs:271"]
    N2["kind-1059 gift wraps are REGULAR events - no replacement semantics - asserted<br/>relay_do/broadcast.rs:434"]
    N3["query_events serves REQ from the same table relay_do/storage.rs:249; is_whitelisted is the admission<br/>lookup NF-03.4 step 6 calls relay_do/storage.rs:345"]
```

## NF-11.5 Broadcast — the kind-1059 delivery gate

```mermaid
sequenceDiagram
    autonumber
    participant H as handle_event after save
    participant B as broadcast_event<br/>relay_do/broadcast.rs:47
    participant S as Session candidate
    participant W as WebSocket

    H->>B: event
    B->>B: if kind 1059, extract the recipient p tag relay_do/broadcast.rs:51
    B->>B: snapshot candidates - authed pubkey, socket, subscriptions relay_do/broadcast.rs:74
    loop each candidate relay_do/broadcast.rs:91
        alt event is kind 1059
            B->>B: skip unless the session is AUTHENTICATED as that recipient relay_do/broadcast.rs:94
        end
        B->>B: resolve_viewer_context relay_do/broadcast.rs:113
        B->>W: deliver if the event matches that subscription's filters
    end

    Note over B: INVARIANT NIP-59: a sealed DM is delivered ONLY to the session whose AUTHENTICATED pubkey matches the p tag - subscribing to kind 1059 is not enough relay_do/broadcast.rs:49-51
    Note over B: This is the read-side twin of the write-side recipient gate in NF-03.5. Admission bounds who may PUBLISH a wrap, this bounds who may RECEIVE one.
    Note over B: A separate filter-level gate rewrites kind-1059 REQ filters to a mandatory #p in BOTH auth modes relay_do/nip_handlers.rs:1335 - so DM privacy never depends on AUTH_MODE relay_do/nip42.rs:183-186
```

## NF-11.6 Subscription matching — the REQ filter predicate

```mermaid
flowchart TB
    F["NostrFilter<br/>relay_do/filter.rs:20"]
    FLDS["ids relay_do/filter.rs:22 | authors relay_do/filter.rs:24 | kinds relay_do/filter.rs:26<br/>since relay_do/filter.rs:28 | until relay_do/filter.rs:30 | limit relay_do/filter.rs:32"]
    M["event_matches_filters<br/>relay_do/filter.rs:199"]
    C1["ids: event.id must be in the set relay_do/filter.rs:202"]
    C2["authors: event.pubkey must be in the set relay_do/filter.rs:207"]
    C3["kinds: event.kind must be in the set relay_do/filter.rs:212"]
    C4["since / until bound created_at relay_do/filter.rs:217 relay_do/filter.rs:222"]
    C5["tag filters: at least one tag must match relay_do/filter.rs:243"]
    OK["all constraints satisfied relay_do/filter.rs:251"]
    HELP["tag_value relay_do/filter.rs:262 | d_tag_value relay_do/filter.rs:271"]

    F --> FLDS --> M
    M --> C1 --> C2 --> C3 --> C4 --> C5 --> OK
    M -.-> HELP

    N1["Semantics are AND across fields, OR within a field - an absent field is UNCONSTRAINED, which is why<br/>a filter naming no kinds requests everything and is still not blocked by the protected-read gate<br/>(asserted relay_do/nip42.rs:307). See NF-03.3."]
    N2["The SAME predicate serves both directions: query_events replays history relay_do/storage.rs:249 and<br/>broadcast_event tests each live event against every session subscription - see NF-11.5"]
    N3["d_tag_value is what storage stamps at INSERT so parameterised replacement is an indexed DELETE -<br/>see NF-11.4 N1"]
```

## NF-11.7 ModCache — a 60-second ban gate that fails CLOSED

```mermaid
stateDiagram-v2
    [*] --> Miss: is_blocked<br/>relay_do/mod_cache.rs:67
    Miss --> Load: entry absent or older than 60 s<br/>relay_do/mod_cache.rs:80, TTL relay_do/mod_cache.rs:21
    Load --> None: no active action - ADMITTED<br/>relay_do/mod_cache.rs:27
    Load --> Banned: BLOCKED<br/>relay_do/mod_cache.rs:29
    Load --> MutedUntil: blocked while the timestamp is in the future<br/>relay_do/mod_cache.rs:31
    Load --> Unknown: D1 fault - BLOCKED anyway<br/>relay_do/mod_cache.rs:36
    None --> [*]
    Banned --> [*]
    MutedUntil --> [*]
    Unknown --> [*]

    note right of Unknown
        INVARIANT fail-closed: a transient D1 fault returns Unknown, which
        is_blocked maps to TRUE relay_do/mod_cache.rs:67 - a database blip must
        not let a banned author publish
    end note
    note right of Load
        INVARIANT: an Unknown is NEVER cached relay_do/mod_cache.rs:85 - caching a
        transient fault would extend one blip into a 60-second outage for every
        innocent author. Only a real verdict is stored.
        Resolution from the action rows is pure: resolve_block relay_do/mod_cache.rs:145
    end note
    note right of Miss
        invalidate relay_do/mod_cache.rs:59 is called the moment an admin mirrors a
        ban or unban, so a lifted ban stops being enforced without waiting out the
        TTL - see NF-03.10
    end note
```

## NF-11.8 The tiered NIP-52 calendar projection — the README's "tiered calendar"

```mermaid
flowchart TB
    PT["project_tier - pure, unit-testable<br/>relay_do/calendar_projection.rs:72"]
    OWN["is_owner or is_admin to Full<br/>relay_do/calendar_projection.rs:80"]
    CADM["admin cohort to Full<br/>relay_do/calendar_projection.rs:87"]
    VEN["venue_is_shared = is_known_venue<br/>relay_do/calendar_projection.rs:91"]
    FAM["family cohort to Full<br/>relay_do/calendar_projection.rs:94"]
    FRI["friends cohort<br/>relay_do/calendar_projection.rs:99"]
    FRIV["family or business zone at a SHARED venue to FreeBusy<br/>relay_do/calendar_projection.rs:105"]
    FRIO["off-site to Omit<br/>relay_do/calendar_projection.rs:116"]
    BUS["business cohort - own zone Full, family and friends Omit<br/>relay_do/calendar_projection.rs:121 relay_do/calendar_projection.rs:125"]
    DEF["anything unmatched to Omit<br/>relay_do/calendar_projection.rs:138"]

    PT --> OWN --> CADM --> VEN
    VEN --> FAM & FRI & BUS
    FRI --> FRIV & FRIO
    PT --> DEF

    N1["Three outcomes only: Full serves the event as-is, FreeBusy serves to_free_busy, Omit drops it<br/>relay_do/calendar_projection.rs:53 relay_do/calendar_projection.rs:66-68"]
    N2["INVARIANT deny-by-default: the terminal arm is Omit relay_do/calendar_projection.rs:138 - a viewer with<br/>no recognised cohort must remain UNAWARE the event exists relay_do/calendar_projection.rs:58"]
    N3["Free/busy keeps start, end, venue and a busy flag ONLY. An event NOT at a recognised venue is omitted<br/>rather than shown as free/busy, so friends see venue blocking and never off-site activity<br/>relay_do/calendar_projection.rs:25-29 - this is what makes 'you learn the room is booked, not whose party<br/>it is' true rather than aspirational"]
    N4["DOC-DRIFT CLOSED: README.md:365 asserts the tiered calendar and its 25 unit tests. The tests are here -<br/>the module carries its own suite from relay_do/calendar_projection.rs:190 - and the write side is gated<br/>by the SAME function, see NF-03.12 N3"]
```

## NF-11.9 Governance receipts — ADR-2010 is implemented, not merely proposed

```mermaid
stateDiagram-v2
    [*] --> Signed: valid signature, correlates to a case<br/>relay_do/receipts.rs:80
    Signed --> RelayAccepted: durably stored - what an OK actually certifies<br/>relay_do/receipts.rs:83
    RelayAccepted --> ProjectionCommitted: decision row, case state and receipt commit TOGETHER<br/>relay_do/receipts.rs:85
    RelayAccepted --> ProjectionFailed: attempted and did not commit<br/>relay_do/receipts.rs:88
    ProjectionCommitted --> [*]
    ProjectionFailed --> [*]

    note right of Signed
        INVARIANT monotonic: a stage may only ADVANCE, never regress, which is
        what stops a late duplicate from downgrading a committed receipt back
        to accepted relay_do/receipts.rs:71-73
    end note
    note right of ProjectionFailed
        is_applied is the distinction a downstream operator needs: a DENIED
        action and an APPROVED action whose write FAILED must never look the
        same relay_do/receipts.rs:112-115
        Terminal until a reconciliation retry supersedes it relay_do/receipts.rs:86
    end note
    note right of ProjectionCommitted
        correlate maps an event to its case relay_do/receipts.rs:164
        apply_with_receipt drives the transition relay_do/receipts.rs:334
        ReceiptStore is the seam relay_do/receipts.rs:302, D1ReceiptStore the
        implementation relay_do/receipts.rs:420, table from migration
        0005_governance_receipts.sql:12
    end note
```

## NF-11.10 What ADR-2010 still leaves open

```mermaid
flowchart LR
    IMPL["IMPLEMENTED in-relay<br/>signed, relay-accepted, projection-committed, projection-failed<br/>relay_do/receipts.rs:76"]
    OPEN["SEPARATE consumer implementations<br/>Agentbox durable received/outcome ledger<br/>VisionClaw dispatch journal and conditional PR claim"]
    LEDGER["ADR-2010 ledger row: proposed / partial / inactive<br/>docs/adr/ADR-2010-durable-governance-outcome-receipts.md:1"]

    IMPL --> OPEN
    LEDGER -.-> IMPL

    N1["DOC-DRIFT: BASELINE-architecture's closing section calls the receipt contract 'proposed and inactive'<br/>and says 'current relay OK establishes acceptance only'. The relay-side stage machine is REAL and wired -<br/>relay_do/receipts.rs:76 defines the stages, relay_do/receipts.rs:334 applies them, and NF-03.10 shows<br/>handle_event logging 'accepted but not applied' from the returned receipt. Consumer stages now exist in Agentbox and VisionClaw, with explicit uncertain-outcome<br/>reconciliation. A relay receipt still cannot prove external application."]
    N2["This refines NF-06.7 and NF-10.8: the remaining acceptance needs deployed correlation and witnessed external outcomes.<br/>EXTERNAL: consumer-received and applied belong to VC-24 and AB-14, estate loop ES-05"]
    N3["INVARIANT: the projection commit is ATOMIC - decision row, case state and receipt in one batch<br/>relay_do/receipts.rs:84-85. A receipt that says committed cannot outlive a decision that did not land."]
```

## NF-11.11 The trust demotion sweep — keyset paging, explicit outcomes

```mermaid
sequenceDiagram
    autonumber
    participant CR as cron trigger every 5 min
    participant SW as sweep_inactive_demotions<br/>trust_sweep.rs:521
    participant RUN as run_demotion_sweep<br/>trust_sweep.rs:239
    participant POL as trust::decide_demotion<br/>trust.rs:326
    participant D1 as whitelist + admin_log

    CR->>SW: scheduled entry, see NF-03.1
    SW->>RUN: page candidates by keyset cursor
    RUN->>D1: page query ordered by (last_active_at, pubkey) trust_sweep.rs:363
    loop each row
        RUN->>POL: decide Hold or Demote - the SHARED pure policy
        alt Demote
            RUN->>D1: trust UPDATE and audit INSERT in ONE batch trust_sweep.rs:207
            RUN->>RUN: counters move only on a CONFIRMED commit trust_sweep.rs:287
        end
        RUN->>RUN: advance the cursor for EVERY consumed row trust_sweep.rs:235
    end
    RUN-->>SW: DemotionSweepResult trust_sweep.rs:142

    Note over RUN: INVARIANT auditable: scanned == demoted + held + failed always holds, checked by is_balanced trust_sweep.rs:167-168 - no row is silently unaccounted for
    Note over RUN: A failed page query stops the sweep early and is reported DISTINCTLY from a failed row commit trust_sweep.rs:117-121 trust_sweep.rs:154
    Note over POL: The cursor advances for held AND failed rows too, so a permanently failing row cannot wedge the sweep trust_sweep.rs:235
```

## NF-11.12 Why the sweep is keyset — and why the closeout qualification is now stale

```mermaid
flowchart TB
    PROB["The sweep MUTATES the very column its candidate predicate filters on - trust_level<br/>trust_sweep.rs:13-14"]
    OFF["With LIMIT/OFFSET each demoted row shrinks the result set UNDERNEATH the offset<br/>trust_sweep.rs:14-16"]
    CONC["Concretely: 400 eligible rows, batch 200, all of page one demoted to TL0 - page two asks<br/>OFFSET 200 over a set now holding 200 rows, gets nothing, and the sweep stops having done half<br/>the work while reporting clean completion trust_sweep.rs:17-20"]
    FIX["A keyset cursor over (last_active_at, pubkey) is immune: neither column is written by a demotion,<br/>so every row's ordering key is stable across the mutation boundary trust_sweep.rs:22-25"]
    TIE["The pubkey tiebreak makes the order TOTAL, so a cohort sharing one last_active_at - the common<br/>case for bulk-seeded rows - pages deterministically instead of looping or skipping trust_sweep.rs:26-29"]
    OUT["Outcomes made explicit: the previous implementation discarded both the UPDATE and the audit INSERT<br/>result and returned the PLANNED level, so a failed write was indistinguishable from a committed one<br/>and the demoted count included demotions that never happened trust_sweep.rs:32-38"]

    PROB --> OFF --> CONC --> FIX --> TIE
    PROB --> OUT

    N1["DOC-DRIFT: the IDENTITY-keys-and-trust closeout says the sweep can skip rows because OFFSET pages over<br/>a shrinking set, and that UPDATE and audit-INSERT errors are ignored before returning the planned level -<br/>concluding ADR-2006 is PARTIAL. BOTH defects are fixed in code: keyset paging trust_sweep.rs:22 and<br/>confirmed-commit-only counters trust_sweep.rs:36. The qualification is stale; ADR-2006's remaining ask -<br/>committed outcome reporting, stable pagination, recoverable audit/state consistency - is MET."]
    N2["The one closeout clause that still holds: TL2 CAN land directly on TL0 - but that is DELIBERATE,<br/>ADR-2006 permits one committed transition per sweep rather than one rung per sweep trust_sweep.rs:230-232"]
    N3["This supersedes the note in NF-03.9 and the ADR-2006 row in NF-10.8"]
```

## NF-11.13 The other scheduled work

```mermaid
flowchart LR
    CRON["scheduled entry<br/>nostr-bbs-relay-worker/src/lib.rs:853"]
    BF["backfill_profiles - ONE-SHOT, manual only<br/>nostr-bbs-relay-worker/src/cron.rs:72"]
    CAP["BACKFILL_MAX_ROWS ceiling per run<br/>nostr-bbs-relay-worker/src/cron.rs:45, stop at cron.rs:128"]
    RES["BackfillResult<br/>nostr-bbs-relay-worker/src/cron.rs:157"]
    RET["retention / NIP-40 expiry sweep<br/>nostr-bbs-relay-worker/src/cron.rs:284"]
    SW["trust demotion sweep - moved OUT to trust_sweep<br/>nostr-bbs-relay-worker/src/cron.rs:269"]

    CRON --> RET & SW
    BF --> CAP --> RES

    N1["The profiles backfill is triggered manually via POST /api/admin/profiles/backfill and NOT from the cron,<br/>because it is a one-shot operation and the live ingest hook keeps rows fresh thereafter<br/>nostr-bbs-relay-worker/src/cron.rs:24-26"]
    N2["It is idempotent behind a freshness guard, so a re-run never overwrites a newer row<br/>nostr-bbs-relay-worker/src/cron.rs:11"]
    N3["The sweep was MOVED out of cron.rs because the inline form was unsound - it mutates trust_level, the<br/>very column its own candidate predicate filters on nostr-bbs-relay-worker/src/cron.rs:269-271. See NF-11.12."]
    N4["The advertised retention windows are built from the SAME RETENTION_POLICY the sweep uses, so NIP-11 and<br/>the cron can never diverge nostr-bbs-relay-worker/src/nip11.rs:173-175"]
```

## NF-11.14 NIP-11 — the relay information document cannot lie about its own gate

```mermaid
flowchart TB
    RI["relay_info<br/>nostr-bbs-relay-worker/src/nip11.rs:134"]
    AM["auth_mode from the SAME parse_auth_mode the DO uses<br/>nostr-bbs-relay-worker/src/nip11.rs:158"]
    LBL["nip42 to write_model nip42+allowlist, rejection auth-required<br/>nostr-bbs-relay-worker/src/nip11.rs:164-167"]
    ALT["allowlist to write_model whitelist, rejection blocked<br/>nostr-bbs-relay-worker/src/nip11.rs:170"]
    NIPS["supported_nips 1 9 11 16 29 33 40 42 45 50 56 59 65 98<br/>nostr-bbs-relay-worker/src/nip11.rs:207"]
    ESC["escalation_defaults block<br/>nostr-bbs-relay-worker/src/nip11.rs:35, tier read at nip11.rs:146, posture nip11.rs:150"]

    RI --> AM --> LBL
    AM --> ALT
    RI --> NIPS & ESC

    N1["INVARIANT: the advertised auth_required must reflect the mode the handlers ACTUALLY enforce - it is<br/>sourced from the same parser, so the NIP-11 claim can never drift from behaviour<br/>nostr-bbs-relay-worker/src/nip11.rs:155-157. This is the strongest possible refutation of the stale<br/>README status row in NF-03.3."]
    N2["The escalation block is explicitly a SCAFFOLD whose authoritative schema is owned by agentbox -<br/>said in the served document itself nostr-bbs-relay-worker/src/nip11.rs:55.<br/>EXTERNAL: see AB-15, and NF-06.5"]
    N3["NIP-45 COUNT and NIP-50 SEARCH are advertised - handlers at relay_do/nip_handlers.rs:1502 and<br/>nostr-bbs-relay-worker/src/profiles.rs:249"]
```

## NF-11.15 Admin and moderation surfaces on the worker

```mermaid
flowchart TB
    UA["user_admin<br/>delete_user user_admin.rs:147 | suspend :257 | silence :335<br/>notes get :397 set :425 | aliases list :483 set :523"]
    MOD["moderation<br/>insert_report moderation.rs:62 | list moderation.rs:142 | resolve moderation.rs:234"]
    AUD["audit<br/>log_admin_action audit.rs:25 | list audit.rs:82"]
    PRO["profiles<br/>batch profiles.rs:94 | search profiles.rs:249"]
    AGD["agent_disclosure<br/>handle_agent_disclosure agent_disclosure.rs:65"]

    UA --> AUD
    MOD --> AUD

    N1["suspend and silence are the two states the admission pipeline reads at NF-03.4 step 8 -<br/>written here, enforced there"]
    N2["profiles::batch is what the forum client's ProfileCache fetches over HTTP rather than by relay REQ -<br/>see NF-05.6; profiles::search backs the advertised NIP-50"]
    N3["agent_disclosure is a PUBLIC endpoint - the client's agent badge reads it without auth, see NF-05.10.<br/>It is how a reader can tell a human post from an agent post."]
    N4["Every admin mutation routes through log_admin_action into admin_log, the same table the trust sweep<br/>writes its audit rows to nostr-bbs-relay-worker/src/audit.rs:25 - see NF-08.5"]
```
