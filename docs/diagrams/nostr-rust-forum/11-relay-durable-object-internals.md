---
id: NF-11
title: Inside the relay Durable Object — sessions, storage, filters, broadcast, projection, receipts and the scheduled sweeps
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
adrs: [ADR-2005, ADR-2006, ADR-2010, ADR-2011, ADR-2013, ADR-2014, ADR-2017, ADR-2018]
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
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/read_cache.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/zone_config.rs
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
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/governance.rs
  - ../nostr-rust-forum/docs/adr/ADR-2010-durable-governance-outcome-receipts.md
  - ../nostr-rust-forum/README.md
verified_commit: 13cbe6cbad7ee7ff3b609233a8bee3dd8eae1f3e
---

## NF-11.1 The Durable Object and its in-memory state

```mermaid
classDiagram
    class NostrRelayDO {
        state : State — relay_do/mod.rs:81
        env : Env — relay_do/mod.rs:82
        sessions : RefCell HashMap u64 SessionInfo — relay_do/mod.rs:83
        next_session_id : relay_do/mod.rs:84
        rate_limits : RefCell HashMap String Vec f64 — relay_do/mod.rs:85
        rate_limit_per_sec : Cell Option usize — relay_do/mod.rs:89
        connection_counts : relay_do/mod.rs:90
        mod_cache : ModCache 60 s TTL — relay_do/mod.rs:92
        admin_cache : AdminCache 5 min TTL — relay_do/mod.rs:95
        channel_zone_cache : TtlCache String, 60 s TTL — relay_do/mod.rs:97
        cohort_cache : TtlCache Vec String, bool, 60 s TTL — relay_do/mod.rs:99
        device_owner_cache : TtlCache Option String, 60 s TTL — relay_do/mod.rs:101
        activity : ActivityLedger — relay_do/mod.rs:103
    }
    class DurableObject {
        new : relay_do/mod.rs:107
        fetch — websocket upgrade only, else 426 : relay_do/mod.rs:125
        websocket_message : relay_do/mod.rs:208
        websocket_close : relay_do/mod.rs:327
    }
    NostrRelayDO ..|> DurableObject

    note for NostrRelayDO "EVERY field except state and env is volatile in-memory cache. Hibernation wipes all of it, which is what NF-11.2 exists to survive."
    note for NostrRelayDO "rate_limit_per_sec is lazily resolved and CACHED because Env::var crosses the JS boundary on every read relay_do/mod.rs:86-88"
    note for NostrRelayDO "ADR-2018: the four newest fields are 60s-TTL memos plus a throttled activity ledger, added because authorize_event and resolve_viewer_context were re-reading D1 per frame - see NF-11.16"
    note for DurableObject "The DO is a singleton reached by get_by_name main from the worker fetch nostr-bbs-relay-worker/src/lib.rs:207 - see NF-03.1"
```

## NF-11.2 Hibernation — recover_untracked_sessions runs on EVERY wake path

```mermaid
sequenceDiagram
    autonumber
    participant C1 as Sleeping socket A
    participant C2 as Waking socket B
    participant DO as NostrRelayDO
    participant ST as DO storage

    Note over DO: DO hibernates - sessions map, next_session_id and rate limits are gone, but sockets and their sid/ip tags survive relay_do/mod.rs:164
    alt a NEW connection wakes the DO
        C2->>DO: fetch - Upgrade websocket relay_do/mod.rs:125
        DO->>DO: recover_untracked_sessions FIRST, before a new id is allocated relay_do/mod.rs:149
        DO->>ST: for every tagged socket NOT yet tracked - load challenge, subscriptions, authed_pubkey relay_do/session.rs:186-218
        ST-->>DO: A rejoins sessions with its restored state
    else a FRAME on an old socket wakes the DO
        C1->>DO: websocket_message relay_do/mod.rs:208
        DO->>DO: find_session_id in memory relay_do/session.rs:40
        alt not found - woke from hibernation
            DO->>ST: recover_session for THIS socket relay_do/session.rs:75
            DO->>DO: recover_untracked_sessions for every OTHER tagged socket relay_do/session.rs:109
        end
    end

    Note over DO: INVARIANT: recover_untracked_sessions runs on BOTH wake paths relay_do/session.rs:159-161 - a new connection that skipped it would publish to an empty sessions map, so sleeping subscribers (zone-key grants included) never see it relay_do/session.rs:164-165
    Note over DO: INVARIANT: the challenge issued BEFORE hibernation is preserved per socket, so a client that connected earlier can still answer its ORIGINAL challenge relay_do/session.rs:404-409
    Note over ST: authed_pubkey is persisted per session id, so authenticated operations continue across the hibernation boundary for every recovered socket, not only the one that woke the DO relay_do/session.rs:199
    Note over DO: recovered_challenge is a PURE decision function, unit-testable without a DO relay_do/session.rs:413
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
    Note over B: INVARIANT what broadcast matches against is the GATED filter, because handle_req now stores the authorised filter rather than the client's raw one relay_do/nip_handlers.rs:1485-1487 relay_do/nip_handlers.rs:1491-1493 - see NF-03.14
    Note over B: A separate filter-level gate rewrites kind-1059 REQ filters to a mandatory #p in BOTH auth modes relay_do/nip_handlers.rs:1760 - so DM privacy never depends on AUTH_MODE relay_do/nip42.rs:236-239
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

    N1["Semantics are AND across fields, OR within a field - an absent field is UNCONSTRAINED, which is why<br/>a filter naming no kinds requests everything and is still not blocked by the protected-read gate<br/>(asserted relay_do/nip42.rs:360). See NF-03.3."]
    N2["The SAME predicate serves both directions: query_events replays history relay_do/storage.rs:249 and<br/>broadcast_event tests each live event against every session subscription - see NF-11.5"]
    N3["d_tag_value is what storage stamps at INSERT so parameterised replacement is an indexed DELETE -<br/>see NF-11.4 N1"]
    N4["Before a filter is even evaluated: a malformed websocket frame is answered with a NOTICE rather than a<br/>socket close - the relay never drops a connection for one bad message relay_do/mod.rs:232-240"]
    N5["Subscriptions that install a filter are capped at MAX_SUBSCRIPTIONS = 20 per session, enforced on REQ<br/>relay_do/nip_handlers.rs:45 relay_do/nip_handlers.rs:1432"]
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
    N4["DOC-DRIFT CLOSED: README.md:368 asserts the tiered calendar and its 25 unit tests. The tests are here -<br/>the module carries its own suite from relay_do/calendar_projection.rs:190 - and the write side is gated<br/>by the SAME function, see NF-03.12 N3"]
```

## NF-11.9 Governance receipts — the ten-stage ladder, owned by core

```mermaid
stateDiagram-v2
    [*] --> Signed: valid signature, correlates to a case<br/>nostr-bbs-core/src/governance.rs:916
    Signed --> RelayAccepted: durably stored - what an OK actually certifies<br/>nostr-bbs-core/src/governance.rs:919
    RelayAccepted --> ProjectionCommitted: decision row, case state and receipt commit TOGETHER<br/>nostr-bbs-core/src/governance.rs:921
    RelayAccepted --> ProjectionFailed: attempted and did not commit<br/>nostr-bbs-core/src/governance.rs:924
    ProjectionCommitted --> ConsumerReceived: the mutation owner has READ it, not acted<br/>nostr-bbs-core/src/governance.rs:926
    ConsumerReceived --> Applied: the act was performed and took effect<br/>nostr-bbs-core/src/governance.rs:928
    ConsumerReceived --> NotApplied: the owner did not perform it, and says so<br/>nostr-bbs-core/src/governance.rs:931
    ConsumerReceived --> AppliedManually: an operator did it by hand during an outage<br/>nostr-bbs-core/src/governance.rs:933
    Applied --> [*]
    NotApplied --> [*]
    AppliedManually --> [*]
    ProjectionFailed --> [*]

    note right of Signed
        The ladder MOVED out of the relay into nostr-bbs-core when FR4.1 added
        the application stages, because the auth worker's receipts endpoint and
        this projection path must agree on it and share no other code
        relay_do/receipts.rs:69-77
        Re-exported here so every existing path keeps working
        relay_do/receipts.rs:77
    end note
    note right of ProjectionFailed
        is_applied is the distinction a downstream operator needs - a DENIED
        action and an APPROVED action whose write FAILED must never look the
        same nostr-bbs-core/src/governance.rs:929-930
        Terminal until a reconciliation retry supersedes it
        nostr-bbs-core/src/governance.rs:922-923
    end note
    note right of Applied
        INVARIANT the three application outcomes are a SET, not an order.
        The derived Ord ranks declaration order, which is meaningful for the
        rungs up to consumer-received and NOT for these three - they are
        mutually exclusive claims about the world, and a consumer that reduces
        them with max displays a success as a failure
        nostr-bbs-core/src/governance.rs:892-902
        Compare rungs by ladder_rank, treat outcomes as a set
        nostr-bbs-core/src/governance.rs:975
    end note
```

**Two side receipts** record something that happened to a case without advancing it
toward application, and so never overwrite a ladder stage
(`nostr-bbs-core/src/governance.rs:988`).

## NF-11.10 Side receipts, the read API, and what ADR-2010 still leaves open

```mermaid
flowchart TB
    SIDE["Side receipts - never on the ladder<br/>escalated-on-age nostr-bbs-core/src/governance.rs:936<br/>expired nostr-bbs-core/src/governance.rs:938<br/>ladder_rank returns None for both nostr-bbs-core/src/governance.rs:982"]
    API["GET /api/governance/receipts - NIP-98 ADMIN<br/>relay_do/receipts.rs:639, routed at nostr-bbs-relay-worker/src/lib.rs:312"]
    JSON["receipt_json derives the flags rather than making every client<br/>re-implement stage semantics relay_do/receipts.rs:594"]
    FLAGS["applied relay_do/receipts.rs:614<br/>awaitsProjection relay_do/receipts.rs:615<br/>isApplicationStage relay_do/receipts.rs:619<br/>appliedAt appliedBy acknowledgement relay_do/receipts.rs:620"]
    LEDGER["ADR-2010 ledger row: proposed / partial / staged<br/>docs/adr/ADR-2010-durable-governance-outcome-receipts.md:5-7"]
    CORR["correlate now reads the first UNMARKED e tag as the request, the shape the forum client signs -<br/>before a5b809e every UI decision came back Uncorrelated relay_do/receipts.rs:128-130"]
    OPEN["Still SEPARATE consumer implementations<br/>agentbox durable received/outcome ledger<br/>VisionClaw dispatch journal and conditional PR claim"]

    SIDE --> JSON
    API --> JSON --> FLAGS
    LEDGER -.-> API
    CORR --> API
    FLAGS --> OPEN

    N1["The relay now serves the APPLICATION stages on the read API, so the human who approved something<br/>can learn whether it actually happened - the loop ADR-2010 left open relay_do/receipts.rs:616-619"]
    N6["INVARIANT: migration 0006 is mirrored into ensure_schema, the live schema path, so case_side_receipts<br/>and case_delegations exist on a cold start without the migration runner<br/>nostr-bbs-relay-worker/src/lib.rs:917 nostr-bbs-relay-worker/src/lib.rs:929 - see NF-08.5"]
    N2["INVARIANT: the projection commit is ATOMIC - decision row, case state and receipt in one batch<br/>relay_do/receipts.rs:301. A receipt that says committed cannot outlive a decision that did not land."]
    N3["DIVERGENCE: the read stays scoped to the relay's existing ADMIN authority because ADR-2010's<br/>history-consumer extension leaves cross-case read authority to ratify relay_do/receipts.rs:635-638"]
    N4["RESOLVED at 341c5d2: the ledger row moved from inactive to staged on a live signed journey - the M4 run's<br/>system-decider 31403 read back projection-failed and applied false on the edge<br/>docs/adr/ADR-2010-durable-governance-outcome-receipts.md:153. A relay receipt still cannot prove external<br/>application - EXTERNAL: see VC-24 and AB-14, estate loop ES-05"]
    N5["OPEN: the commit path is not yet witnessed live - acceptance needs the owner's own high-tier 31403 at<br/>projection-committed with its broker_decisions row docs/adr/ADR-2010-durable-governance-outcome-receipts.md:157.<br/>Refines NF-10.8; the decision card and correlation half is NF-06.17"]
```

## NF-11.11 The trust demotion sweep — keyset paging, explicit outcomes

```mermaid
sequenceDiagram
    autonumber
    participant CR as cron trigger every 5 min
    participant SW as sweep_inactive_demotions<br/>trust_sweep.rs:522
    participant RUN as run_demotion_sweep<br/>trust_sweep.rs:240
    participant POL as trust::decide_demotion<br/>trust.rs:326
    participant D1 as whitelist + admin_log

    CR->>SW: scheduled entry, see NF-03.1
    SW->>RUN: page candidates by keyset cursor
    RUN->>D1: page query ordered by (last_active_at, pubkey) trust_sweep.rs:363
    loop each row
        RUN->>POL: decide Hold or Demote - the SHARED pure policy
        alt Demote
            RUN->>D1: trust UPDATE and audit INSERT in ONE batch trust_sweep.rs:208
            RUN->>RUN: counters move only on a CONFIRMED commit trust_sweep.rs:288
        end
        RUN->>RUN: advance the cursor for EVERY consumed row trust_sweep.rs:236
    end
    RUN-->>SW: DemotionSweepResult trust_sweep.rs:142

    Note over RUN: INVARIANT auditable: scanned == demoted + held + failed always holds, checked by is_balanced trust_sweep.rs:167-168 - no row is silently unaccounted for
    Note over RUN: A failed page query stops the sweep early and is reported DISTINCTLY from a failed row commit trust_sweep.rs:117-121 trust_sweep.rs:154
    Note over POL: The cursor advances for held AND failed rows too, so a permanently failing row cannot wedge the sweep trust_sweep.rs:236
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
    N2["The one closeout clause that still holds: TL2 CAN land directly on TL0 - but that is DELIBERATE,<br/>ADR-2006 permits one committed transition per sweep rather than one rung per sweep trust_sweep.rs:231-233"]
    N3["This supersedes the note in NF-03.9 and the ADR-2006 row in NF-10.8"]
```

## NF-11.13 The other scheduled work

```mermaid
flowchart LR
    CRON["scheduled entry<br/>nostr-bbs-relay-worker/src/lib.rs:1022"]
    BF["backfill_profiles - ONE-SHOT, manual only<br/>nostr-bbs-relay-worker/src/cron.rs:72"]
    CAP["BACKFILL_MAX_ROWS ceiling per run<br/>nostr-bbs-relay-worker/src/cron.rs:49, stop at cron.rs:126"]
    RES["BackfillResult<br/>nostr-bbs-relay-worker/src/cron.rs:159"]
    RET["retention / NIP-40 expiry sweep<br/>nostr-bbs-relay-worker/src/cron.rs:345"]
    AGE["ageing sweep - escalate_stale_cases<br/>nostr-bbs-relay-worker/src/cron.rs:591"]
    SW["trust demotion sweep - moved OUT to trust_sweep<br/>nostr-bbs-relay-worker/src/cron.rs:269"]
    EXP["ADR-2013: expire_stale_proposals - closes ontology proposals past stale_after<br/>nostr-bbs-relay-worker/src/cron.rs:747, runs AFTER the ageing sweep<br/>nostr-bbs-relay-worker/src/lib.rs:1094"]

    CRON --> RET & SW & AGE --> EXP
    BF --> CAP --> RES

    N1["The profiles backfill is triggered manually via POST /api/admin/profiles/backfill and NOT from the cron,<br/>because it is a one-shot operation and the live ingest hook keeps rows fresh thereafter<br/>nostr-bbs-relay-worker/src/cron.rs:24-26"]
    N2["It is idempotent behind a freshness guard, so a re-run never overwrites a newer row<br/>nostr-bbs-relay-worker/src/cron.rs:11"]
    N3["The sweep was MOVED out of cron.rs because the inline form was unsound - it mutates trust_level, the<br/>very column its own candidate predicate filters on nostr-bbs-relay-worker/src/cron.rs:269-271. See NF-11.12."]
    N4["The advertised retention windows are built from the SAME RETENTION_POLICY the sweep uses, so NIP-11 and<br/>the cron can never diverge nostr-bbs-relay-worker/src/nip11.rs:173-175"]
    N5["INVARIANT: the ageing sweep is ordered and filtered by each case's OWN deadline, not by created_at -<br/>panels declare different deadlines, so oldest first is not most overdue first, and ordering by<br/>created_at made the page ceiling cut the LEAST overdue nostr-bbs-relay-worker/src/cron.rs:602-609,<br/>the SQL at nostr-bbs-relay-worker/src/cron.rs:612"]
    N6["A case exactly AT its deadline has not yet exceeded it nostr-bbs-relay-worker/src/cron.rs:570,<br/>the predicate at nostr-bbs-relay-worker/src/cron.rs:571, asserted nostr-bbs-relay-worker/src/cron.rs:1172"]
    N7["An ageing escalation is a SIDE receipt - it records what happened to a case without advancing it<br/>toward application nostr-bbs-core/src/governance.rs:936 - see NF-11.10"]
    N8["ADR-2013: a proposal expires when the corpus has moved past its digest - the ontology page it named no<br/>longer describes the same content, so applying a stale decision would silently promote or demote the<br/>WRONG state. Closed WITHOUT a decision, receipted expired rather than left pending forever<br/>nostr-bbs-relay-worker/src/cron.rs:721-746. Runs after ageing so a case that is both overdue and<br/>expired accrues both receipts in the order they became true, per the scheduled-entry comment<br/>nostr-bbs-relay-worker/src/lib.rs:1089-1093"]
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
    N2["The escalation block is explicitly a SCAFFOLD whose authoritative schema is owned by agentbox -<br/>said in the served document itself nostr-bbs-relay-worker/src/nip11.rs:55.<br/>EXTERNAL: see AB-15, and NF-06.12"]
    N3["NIP-45 COUNT and NIP-50 SEARCH are advertised - handlers at relay_do/nip_handlers.rs:2023 and<br/>nostr-bbs-relay-worker/src/profiles.rs:251"]
```

## NF-11.15 Admin and moderation surfaces on the worker

```mermaid
flowchart TB
    UA["user_admin<br/>delete_user user_admin.rs:149 | suspend user_admin.rs:259 | silence user_admin.rs:337<br/>notes get user_admin.rs:399 set user_admin.rs:427 | aliases list user_admin.rs:485 set user_admin.rs:525"]
    MOD["moderation<br/>insert_report moderation.rs:62 | list moderation.rs:142 | resolve moderation.rs:234"]
    AUD["audit<br/>log_admin_action audit.rs:25 | list audit.rs:82"]
    PRO["profiles<br/>batch profiles.rs:96 | search profiles.rs:251 - search excludes pubkey_aliases old_pubkey rows"]
    AGD["agent_disclosure<br/>handle_agent_disclosure agent_disclosure.rs:65"]

    UA --> AUD
    MOD --> AUD

    N1["suspend and silence are the two states the admission pipeline reads at NF-03.4 step 8 -<br/>written here, enforced there"]
    N2["profiles::batch is what the forum client's ProfileCache fetches over HTTP rather than by relay REQ -<br/>see NF-05.6; profiles::search backs the advertised NIP-50"]
    N3["agent_disclosure is a PUBLIC endpoint - the client's agent badge reads it without auth, see NF-05.10.<br/>It is how a reader can tell a human post from an agent post."]
    N4["Every admin mutation routes through log_admin_action into admin_log, the same table the trust sweep<br/>writes its audit rows to nostr-bbs-relay-worker/src/audit.rs:25 - see NF-08.5"]
    N5["A key REPLACED via pubkey_aliases is excluded from both search shapes - roster mode profiles.rs:307<br/>and query mode profiles.rs:318 - so a renamed pubkey never shows twice under two identities"]
    N6["INVARIANT since 1a26e51: alias_set with inherit_cohorts grants the old key's cohorts to the new one through the<br/>shared merge statement, so the successor keeps any cohort it already held user_admin.rs:602-605 user_admin.rs:612"]
```

## NF-11.16 ADR-2017 sealed originals — migrating history into an encrypted zone without a fresh-write hole

```mermaid
flowchart TB
    VAL["validate_event<br/>relay_do/nip_handlers.rs:1298"]
    DRIFT["timestamp_drift_ok<br/>relay_do/nip_handlers.rs:1363"]
    EXEMPT["sealed_drift_exempt - kind 42 AND a well-formed sealed tag<br/>relay_do/nip_handlers.rs:1353"]
    PAST["exempt ONLY from the past bound - created_at must still not be in the future<br/>relay_do/nip_handlers.rs:1364"]
    HAS["has_sealed_tag - ANY sealed tag, well formed or not<br/>relay_do/nip_handlers.rs:1105"]
    REJ["sealed_write_rejection<br/>relay_do/nip_handlers.rs:1384"]
    ADMIN["not admin -> SEALED_ADMIN_ONLY<br/>relay_do/nip_handlers.rs:1393"]
    ENCZ["zone not encrypted -> SEALED_ENCRYPTED_ZONES_ONLY<br/>relay_do/nip_handlers.rs:1396"]
    CIPHER["is_zone_ciphertext shape check - NIP-44 v2 ciphertext, no readable text even for admins<br/>zone_config.rs:202, gate at relay_do/nip_handlers.rs:1124"]
    OK["event accepted, saved with its ORIGINAL created_at"]

    VAL --> DRIFT --> EXEMPT --> PAST
    HAS --> REJ --> ADMIN
    REJ --> ENCZ
    ENCZ -.-> CIPHER --> OK

    N1["INVARIANT: the drift exemption and the write rejection are SEPARATE gates that must both hold - a<br/>malformed sealed marker fails parse_sealed so EXEMPT never fires, but HAS still fires on the raw tag, so<br/>the write is refused rather than silently accepted as a fresh non-exempt event<br/>relay_do/nip_handlers.rs:1353-1360"]
    N2["INVARIANT: order matters. An UNSCOPED channel otherwise lets any whitelisted member post, so the<br/>encrypted-zone check must reject a sealed envelope there even from an admin - checked BEFORE the ordinary<br/>zone-write gate, not folded into it relay_do/nip_handlers.rs:1101-1113"]
    N3["The relay holds no zone key, so CIPHER is a SHAPE check only - it cannot prove the ciphertext decrypts<br/>to anything, only that it is not plaintext masquerading as migrated history relay_do/nip_handlers.rs:1121-1129"]
    N4["Why this exists: an envelope keeps its original created_at so pagination, ordering and client unread<br/>logic stay correct across a migration that is usually far older than MAX_TIMESTAMP_DRIFT (7 days)<br/>relay_do/nip_handlers.rs:1342-1344"]
```

## NF-11.17 ADR-2018 read-side caching — per-DO memos and a throttled activity ledger

```mermaid
flowchart TB
    ZONE["cached_channel_zone - 60s memo, POSITIVE results only<br/>relay_do/nip_handlers.rs:1821"]
    COH["cached_viewer_cohorts - 60s memo of cohorts, is_admin, BOTH outcomes cached<br/>relay_do/nip_handlers.rs:1835"]
    DEV["cached_device_owner - 60s memo, BOTH outcomes cached<br/>relay_do/nip_handlers.rs:1848"]
    TTL["TtlCache generic memo, entries expire ttl_secs after being stored<br/>read_cache.rs:56"]
    READ["note_read_activity - accumulates delivered reads<br/>relay_do/nip_handlers.rs:1860"]
    WRITE["note_write_activity - stamps EVERY accepted EVENT<br/>relay_do/nip_handlers.rs:1870"]
    LEDGER["ActivityLedger - flush at most once per pubkey per 300s, or at 50 pending reads<br/>read_cache.rs:98, thresholds read_cache.rs:44-48"]
    FLUSH["flush_activity - increment_posts_read_by, update_last_active, check_promotion<br/>relay_do/nip_handlers.rs:1877"]

    ZONE & COH & DEV --> TTL
    READ & WRITE --> LEDGER --> FLUSH

    N1["Why: authorize_event ran TWO D1 queries per kind-40/42 event delivered, and resolve_viewer_context<br/>re-read device-owner and cohort rows on every frame - a 50-message channel cost ~110 D1 queries, and one<br/>busy evening exhausted the D1 free-tier row-read budget before a human read anything read_cache.rs:5-15"]
    N2["INVARIANT: a NEGATIVE channel-zone result is never cached - a channel not yet bound to a zone must not<br/>be remembered as unscoped for a minute after an admin binds it read_cache.rs:24-26"]
    N3["The 60s TTL matches the existing ModCache (NF-11.7): a cohort revocation or zone re-binding self-heals<br/>within the same window the moderation cache already accepts read_cache.rs:22-23"]
    N4["Stamps are activity SIGNALS for the six-month inactivity sweep (NF-11.11) and TL0-to-TL1 promotion, so a<br/>five-minute coalescing delay is invisible; a DO eviction can drop at most one window of pending reads<br/>read_cache.rs:31-33"]
    N5["Everything here is PURE over an explicit now, so the TTL and flush-threshold decisions are unit-tested<br/>natively without the Workers runtime read_cache.rs:35-36"]
```
