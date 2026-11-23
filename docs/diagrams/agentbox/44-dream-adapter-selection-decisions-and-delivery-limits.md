---
id: AB-44
title: Dream adapter — finding selection, providers, forum decisions, signing and delivery limits
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2053, ADR-2071, ADR-2081, ADR-2122, ADR-2143]
sources:
  - ../project/agentbox/services/dream-engine/src/zone_crypto.rs
  - ../project/agentbox/scripts/activation/adr-2087-check.sh
  - ../project/agentbox/management-api/lib/execution-journal.js
  - ../project/agentbox/management-api/routes/exec-record.js
  - ../project/agentbox/services/dream-engine/src/engine.rs
  - ../project/agentbox/services/dream-engine/src/config.rs
  - ../project/agentbox/services/dream-engine/src/dispatch.rs
  - ../project/agentbox/services/dream-engine/src/llm.rs
  - ../project/agentbox/services/dream-engine/src/main.rs
  - ../project/agentbox/services/dream-engine/src/journal.rs
  - ../project/agentbox/services/dream-engine/src/governance.rs
  - ../project/agentbox/services/dream-engine/src/relay.rs
  - ../project/agentbox/services/dream-engine/src/inbox.rs
  - ../project/agentbox/services/dream-engine/src/digest.rs
  - ../project/agentbox/services/dream-engine/src/records.rs
  - ../project/agentbox/services/dream-engine/src/estate.rs
  - ../project/agentbox/services/dream-engine/src/wanted.rs
  - ../project/agentbox/services/dream-engine/src/jev_triage.rs
  - ../project/agentbox/config/custody/identity-port-acl.json
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/acl.rs
  - ../project/agentbox/scripts/activation/adr-2071-api-down-night.sh
  - ../project/agentbox/docs/adr/ADR-2071-journal-the-nightly-dream-cycle.md
  - ../dream-engine/crates/dream-core/src/screen.rs
  - ../dream-engine/crates/dream-core/src/source.rs
  - ../dream-engine/crates/dream-core/src/ledger.rs
  - ../dream-engine/crates/dream-core/src/streak.rs
  - ../dream-engine/crates/dream-gate/src/gate.rs
  - ../dream-engine/crates/dream-gate/src/verdict.rs
  - ../dream-engine/crates/dream-gate/src/patch.rs
  - ../dream-engine/crates/dream-gate/src/config.rs
verified_commit: {agentbox: 59b1aca1cc3541b767872ecbc161c89e65d6ebca, dream-engine: 881075377625a9e20e69e9137f679506e93f4b68}
aliases:
  AB-23.13: AB-44.1
  AB-23.14: AB-44.2
  AB-23.15: AB-44.3
  AB-23.16: AB-44.4
  AB-23.17: AB-44.5
  AB-23.18: AB-44.6
  AB-23.19: AB-44.7
  AB-23.20: AB-44.8
  AB-23.21: AB-44.9
  AB-23.22: AB-44.10
  AB-23.23: AB-44.11
  AB-23.24: AB-44.12
---
Split from AB-23 on 2026-11-23: the adapter's selection, provider, governance, signing, source-supply and delivery boundaries (formerly AB-23.13 to AB-23.24). The night mechanics, module map, persistence, journal and core types stay in AB-23. Old panel IDs resolve through the `aliases:` map above; the stamp is AB-23's, and the panels were not re-read.

## AB-44.1 No-patch ACCEPT is unproven, but the adapter still alerts

```mermaid
flowchart TB
    ACCEPT["Strict model verdict is ACCEPT<br/>dream-gate/src/gate.rs:406"]
    ACCEPT --> STATE{"candidate state?<br/>dream-gate/src/gate.rs:419"}
    STATE -->|NoPatch / Refused / NotAttempted| UNPROVEN["Unproven veto; no trusted candidate tree<br/>dream-gate/src/gate.rs:432-451"]
    STATE -->|DidNotApply| HARNESS["Harness veto<br/>dream-gate/src/gate.rs:441"]
    STATE -->|Applied| GUARD["Check protected paths, then candidate receipts if tree trusted<br/>dream-gate/src/gate.rs:421,461"]
    UNPROVEN --> VERDICT["Only unproven vetoes => INCONCLUSIVE<br/>dream-gate/src/gate.rs:506"]
    HARNESS --> BLOCK["Harness veto => BLOCKED-ENV<br/>dream-gate/src/gate.rs:500"]
    VERDICT --> STREAK["INCONCLUSIVE advances dry streak<br/>engine.rs:1697; dream-core/src/streak.rs:151"]
    VERDICT --> ALERT["Any nonempty veto list adds operator inbox alert<br/>engine.rs:1784-1800"]
    BLOCK --> ALERT
    GUARD --> RESULT["Gate result and typed vetoes<br/>dream-gate/src/gate.rs:489"]
    RESULT -->|nonempty veto list| ALERT
    ALERT -.-> GAP["ADR-2081 no-operator-alert clause is not implemented here<br/>engine.rs:1784"]
```

**Verified distinction.** The no-patch fixture yields INCONCLUSIVE and no Harness veto, but does add an alert. Candidate receipt grading and notification policy are separate; a verdict class alone does not suppress operator messages.

## AB-44.2 Finding selection, validation and append-time repair

```mermaid
flowchart TB
    REPORT["Model report text"] --> TABLE["First pipe row with enough cells, matching date and valid finding<br/>Quoted/backticked rows included; no Step-19 or deep check<br/>dream-gate/src/verdict.rs:488"]
    TABLE -->|found| SELECTED["Selected finding<br/>dream-gate/src/verdict.rs:362"]
    TABLE -->|none| LINE["First Finding: line passing cell contract<br/>dream-gate/src/verdict.rs:367"]
    LINE -->|found| SELECTED
    LINE -->|none| FALL["select_finding then take 80 Rust chars<br/>May retain Given prefix or exceed 80 UTF-16 units<br/>dream-gate/src/verdict.rs:374"]
    FALL --> SELECTED
    SELECTED --> BACK["Adapter checks entire row and repairs violations before append<br/>records.rs:53; dream-core/src/ledger.rs:254"]
    BACK --> LEDGER["Append repaired or original row; commit is fail-open<br/>records.rs:66-111"]
    RULES["Cell rules: nonempty, <=80 UTF-16 units, no hypothesis/pointer prefix<br/>Checks through repeated VETOED prefixes<br/>dream-gate/src/verdict.rs:394"] -.-> TABLE
    RULES -.-> LINE
    REPORT --> FULL["Full finding uses older hypothesis-first selection with a separate bound<br/>dream-gate/src/verdict.rs:514,523"]
    FULL --> OTHER["Memory/PR presentation path; not the ledger cell selector<br/>dream-gate/src/verdict.rs:510"]
```

**Limits.** Selection establishes syntax and date, not provenance from a particular report section or evaluated deep. The fallback needs the adapter's repair backstop; UTF-16 validation does not make its character truncation equivalent. Unicode parity differences remain recorded in the migration audit.

## AB-44.3 Provider selection, retries and cross-provider fallback

```mermaid
flowchart TB
    CFG["DREAM_LLM_PROVIDER overrides runtime config; unknown name becomes Zai<br/>estate.rs:165<br/>llm.rs:35"]
    CFG --> PRIMARY["Primary LlmAdapter; request token budget honoured<br/>llm.rs:278"]
    PRIMARY --> ZAI["Zai: POST /v1/messages; one transient retry after 20s<br/>llm.rs:49,120"]
    PRIMARY --> LOOM["Loom: published client, 600s timeout, 20s backoff<br/>declining_verbatim; pinned sampling<br/>llm.rs:205"]
    ZAI --> RESULT{"primary verdict request succeeds?<br/>engine.rs:1421"}
    LOOM --> RESULT
    RESULT -->|yes| REPORT["Use report text<br/>engine.rs:1422"]
    RESULT -->|no| FB{"fallback configured?<br/>estate.rs:205"}
    FB -->|no| DEG["Record degraded INCONCLUSIVE report<br/>engine.rs:1467"]
    FB -->|yes| SEND["Send the same prompt to fallback provider<br/>engine.rs:1439"]
    SEND --> OUT["Use fallback report, or degraded INCONCLUSIVE if both fail<br/>engine.rs:1450"]
    CFG -.-> POLICY["Zai primary: Loom fallback; Loom primary: Zai if key exists<br/>DREAM_LLM_FALLBACK=off disables fallback<br/>estate.rs:205-266"]
    POLICY -.-> FB
    POLICY -.-> EGRESS["Selecting Loom alone does not establish LAN confinement<br/>ADR-2053's one-setting claim exceeds implementation<br/>estate.rs:205"]
```

**Egress.** Retaining Z.AI as the default is the recorded operator decision. LAN confinement requires the effective primary, fallback and endpoint policy to agree; a provider label alone does not enforce it. Client-level guarantees remain owned by AB-28.11.

## AB-44.4 Placeholder and dispatch guards have limited coverage

```mermaid
flowchart TB
    INPUT["Parsed RuntimeConfig<br/>main.rs:293"]
    INPUT --> WHOLE["hp_host, loom_url, zai_url: whole-value placeholder resolution<br/>dream-engine/src/config.rs:150"]
    INPUT --> INFIX["hp_annexe_dir: expand embedded placeholders<br/>dream-engine/src/config.rs:169"]
    WHOLE --> DEFAULT["Unset/empty whole placeholder uses field default<br/>dream-engine/src/config.rs:151"]
    INFIX --> DEFAULT2["Missing/empty referenced variable uses annexe default<br/>dream-engine/src/config.rs:158"]
    WHOLE -.-> REMAIN["Embedded host/URL placeholders and nested replacement values survive<br/>dream-engine/src/config.rs:127"]
    INFIX -.-> NESTED["Replacement values are not expanded recursively<br/>dream-engine/src/config.rs:181"]
    DEFAULT --> GUARD["ssh, ssh_capture and scp_to reject trimmed-empty host<br/>dispatch.rs:65,92,124"]
    DEFAULT2 --> GUARD
    GUARD --> EXEC["Nonempty host: BatchMode and 10s connection timeout<br/>remote command completion has no overall deadline<br/>dispatch.rs:38"]
    FROM["scp_from: no empty-host guard; invokes scp with host:path<br/>dispatch.rs:147"]
    BAD["Explicit runtime file unreadable or invalid?<br/>main.rs:293"]
    BAD --> FALL["Log then use defaults, including enabled=true and provider=Zai<br/>main.rs:305<br/>dream-engine/src/config.rs:52,206"]
```

**Limit.** Existing tests cover the original simple-placeholder failures, not a universal ban on unresolved placeholders. The isolated controls probe confirms nested values survive and empty-host scp_from launches a fake executable.

## AB-44.5 Forum decisions — local application and delivery boundaries

```mermaid
flowchart TB
    ING["Start of night: ingest published open items<br/>governance.rs:1060-1071"] --> FETCH["Fetch request times and responses<br/>Query errors become empty results<br/>governance.rs:1088-1116"]
    FETCH --> VALID["Keep kind 31403, valid signature, signer not this agent<br/>No local admin or delegated-role lookup<br/>governance.rs:859-864"]
    VALID --> NEW["Choose newest timestamp and event ID per d tag FIRST<br/>governance.rs:866-877"]
    NEW --> BIND{"Request e tag absent, or equals stored request?<br/>governance.rs:881-885"}
    BIND -->|yes| MAP["Approve, reject or nonempty amend maps to resolution<br/>Delegate and unsupported responses stay open<br/>governance.rs:785-815"]
    BIND -->|no| NONE["No older valid decision is reconsidered"]
    NONE -->|"alert only"| ACK["Alert fallback: panel action plus publication time<br/>No outcome or local authority check<br/>governance.rs:897-916"]
    MAP -->|"supported resolution"| SAVE["Resolve OPEN local item; later decisions do not revise it<br/>inbox.rs:157-170"]
    MAP -->|"delegate or unsupported"| NONE
    ACK --> SAVE
    SAVE --> WITHDRAW["Sign kind-5 request; ACK increments withdrawn<br/>Local receipt write can fail, allowing resend<br/>governance.rs:663-709"]
    PUB["End of night: publish panel if content or tags differ<br/>Comparison does not verify returned event signature<br/>governance.rs:991-1014"] --> SWEEP["Sweep settled items, then publish unmarked open items<br/>governance.rs:1017-1046"]
    SWEEP --> WITHDRAW
    SWEEP --> CASE["Sign case 31402; relay ACK triggers local mark<br/>No request read-back here<br/>governance.rs:1023-1037"]
    CASE --> REOPEN["mark_published selects FIRST item with matching ID<br/>A reopened duplicate can update its closed predecessor<br/>inbox.rs:114-130"]
    REOPEN --> RISK["Old record can then target the new request for deletion<br/>Coordinate suppression alone does not protect its e tag<br/>governance.rs:589-603; governance.rs:622-635"]
```

**Authority.** Signature validity and rejection of this agent's key do not establish a human decision. The configured relay supplies admission policy; current relay source allows admins and case-delegated reviewers. The adapter also accepts missing request binding and ignores supersession after local resolution; these boundaries are tested in the [governance audit](../../estate-review/2026-11-01-dream-governance-and-delivery.md).
**Delivery.** An accepted deletion is a request acknowledgement, not proof of remote removal, local persistence or live card removal. The relay deletes rows but ignores database errors; the current panel subscription excludes kind 5 and its registry ignores it. Those cross-repository source findings are recorded separately in the audit, without a live UI claim.
**Compatibility.** Shared wire types reduce duplication but do not prevent version drift: the adapter consumes core beta.16 while the inspected forum checkout is beta.19. Missing keys, transport failures and local write errors remain fail-open; a zero-count report is not proof that no work was pending.

## AB-44.6 Annexe health and retention limits

```mermaid
flowchart TB
    MAN["Manifest frozen; attempt not yet counted<br/>engine.rs:858"] --> SWEEP["SSH retention: find date-like dirs by parent mtime, then rm<br/>engine.rs:901-905"]
    SWEEP --> MASK["Shell ends with true; find and deletion errors hidden<br/>engine.rs:904"]
    MASK --> SSH["annexe_health via synchronous SSH<br/>engine.rs:918; dispatch.rs:347"]
    SSH --> WRITE{"mkdir, write probe, remove probe succeed?<br/>dispatch.rs:327-332"}
    WRITE -->|no| FAIL["Health error<br/>dispatch.rs:348"]
    WRITE -->|yes| DF["df -Pk emits AVAIL-KB<br/>dispatch.rs:331"]
    DF --> PARSE{"Parse KiB and floor to GiB<br/>dispatch.rs:336,351"}
    PARSE -->|missing or below floor| FAIL
    PARSE -->|sufficient| BEGIN["begin counts attempt<br/>engine.rs:962"]
    FAIL --> REC["Alert plus BLOCKED-ENV record; no attempt count<br/>engine.rs:931,850"]
    BEGIN --> CLONE["Clone then check directory exists and is non-empty<br/>engine.rs:1049,1074"]
    CLONE --> RETRY["One reprovision attempt if checkout probe fails<br/>engine.rs:1088"]
```

The write test catches failures that a free-space figure alone misses; local fixtures exercise the actual public health function through a substitute SSH executable. The minimum defaults to 10 GiB; an explicit zero disables the capacity floor, not the write test (`../project/agentbox/services/dream-engine/src/engine.rs:2392`). Health checks are observations at that moment, not reservations of space or proof of checkout identity.

The exact retention command deleted a disposable old parent directory containing a freshly written child. It checks neither active jobs nor night locks, and a failing find command still returned success with no stderr. This is fixture evidence, not a claim that a live run was deleted. SSH has a 10-second connection timeout but no overall command deadline (`../project/agentbox/services/dream-engine/src/dispatch.rs:39-60`); a connected hung sweep or probe can therefore hold up later serial work. See the [execution review](../../estate-review/2026-11-01-dream-execution-and-triage.md).

## AB-44.7 ADR-2071 clause (c) — historical acceptance and retained tick

```mermaid
flowchart TB
    %% ADR-2071-journal-the-nightly-dream-cycle.md:204-212
    %% adr-2071-api-down-night.sh:29-31
    %% adr-2071-api-down-night.sh:83-95
    %% adr-2071-api-down-night.sh:94
    %% adr-2071-api-down-night.sh:100-105
    %% adr-2071-api-down-night.sh:102-103
    %% adr-2071-api-down-night.sh:106-108
    %% adr-2071-api-down-night.sh:108
    %% adr-2071-api-down-night.sh:109-111
    %% adr-2071-api-down-night.sh:113-118
    %% adr-2071-api-down-night.sh:129
    %% adr-2071-api-down-night.sh:137-142
    %% adr-2071-api-down-night.sh:132-138
    %% adr-2071-api-down-night.sh:139-140
    %% adr-2071-api-down-night.sh:141-142
    %% adr-2071-api-down-night.sh:145-148
    %% adr-2087-check.sh:401-423
    %% adr-2087-check.sh:414-421
    %% adr-2071-api-down-night.sh:121-124
    HIST["Historical acceptance: manual wrapper on 4 October<br/>Cron experiment was NOT used; marker and cron block retired<br/>ADR-2071 lines 195-203"]
    HIST -.-> TICK["Retained script for future reuse<br/>Requires external scheduling plus marker and state<br/>tick lines 29-31"]
    TICK --> PH{"Marker present and syntactically dated;<br/>read existing phase<br/>tick lines 83-95"}
    PH -->|"done or missed, even for older night"| RET["Retire new marker without stopping<br/>tick lines 94"]
    PH -->|"no phase, matching date 00:30-00:59"| PAUSE{"Dream paused?<br/>tick lines 100-105"}
    PAUSE -->|"yes"| MISS["Mark missed and retire<br/>tick lines 102-103"]
    PAUSE -->|"no"| TRY["Arm EXIT trap; attempt stop<br/>tick lines 106-108"]
    TRY --> OK{"Stop succeeded or status STOPPED?<br/>tick lines 108"}
    OK -->|"yes"| STOP["Write stopped state, then disarm trap<br/>tick lines 109-111"]
    OK -->|"no"| STOPFAIL["Attempt restart; mark missed and retire<br/>tick lines 113-118"]
    PH -->|"no phase, matching date at or after 01:00"| MISS
    STOP --> LATER["Later tick with phase stopped<br/>No independent watchdog in this script<br/>tick lines 129"]
    PH -->|"stopped"| LATER
    LATER --> WHY{"Ordered checks<br/>tick lines 137-142"}
    WHY -->|"first: matching date text in health file"| REC["night-record; no JSON or run-identity validation<br/>tick lines 132-138"]
    WHY -->|"else: RUNNING or STARTING"| INT["interrupted<br/>tick lines 139-140"]
    WHY -->|"else: deadline passed"| DEAD["deadline<br/>tick lines 141-142"]
    REC --> START["Attempt start; ignore failure<br/>Write phase done and started_at; retire marker<br/>tick lines 145-148"]
    INT --> START
    DEAD --> START
    START --> C3["C3: dated health, at least one verdict and failed/skipped post;<br/>matching done state, nonempty stopped_at, allowed reason<br/>checker lines 397-419"]
    C3 --> LIMIT["No restart-success or engine-exit validation;<br/>no proof of continuous outage or ledger append<br/>checker lines 410-417"]
```

**History.** ADR-2071 records a manual 4 October acceptance run, three newly dreamed repositories and one previously completed skip. This is historical evidence, not a fresh deployment check (`../project/agentbox/docs/adr/ADR-2071-journal-the-nightly-dream-cycle.md:204-212`).
**Current boundary.** The retained tick can consume a fresh marker using old done state, restart before the window from matching date text, mask an interruption and retire after a failed start. C3 trusts aggregate files rather than binding an outage to a specific completed run; see the [journal audit](../../estate-review/2026-11-01-dream-journal-and-api-down.md).
**Recovery limit.** The EXIT trap is removed after writing stopped state; later recovery depends on another tick and a successful supervisor start. It is not an unconditional guarantee that the API returns (`../project/agentbox/scripts/activation/adr-2071-api-down-night.sh:106-111`, `../project/agentbox/scripts/activation/adr-2071-api-down-night.sh:145-148`).

## AB-44.8 Signing keys — the identity-port capability and the local fallback

```mermaid
flowchart TB
    GOV["governance load_key, purpose agent_governance<br/>governance.rs:922-923"] --> ISO{"AGENTBOX_ROLE_ISOLATION equals 1?<br/>relay.rs:155"}
    ISO -->|"no"| LOCAL["Local key: FILE variable, bare variable or repo env line<br/>load_signing_key relay.rs:242"]
    ISO -->|"yes"| PUB["port pubkey for the purpose, no key file opened<br/>relay.rs:158"]
    PUB --> PORTK["PublishingKey.Port, public identity and alias only<br/>relay.rs:49-53"]
    LOCAL --> SIGNL["sign in engine memory<br/>relay.rs:104-106"]
    PORTK --> REQ["port_sign_request: kind 22242 goes to nip42_auth<br/>agent_governance goes to governance_event<br/>relay.rs:125, relay.rs:134, relay.rs:138-139"]
    REQ --> SOCK["one JSON line on the identity socket, 2 s timeouts<br/>port_request relay.rs:56"]
    SOCK --> ACL["shipped ACL: devuser may governance_event kinds 31400, 31402, 5<br/>with the optional agent_governance key<br/>identity-port-acl.json:39-42, identity-port-acl.json:54"]
    ACL --> GUARD["agent key: only AGENT_GOVERNANCE_KINDS, else refused at load<br/>identity_port/acl.rs:102, identity_port/acl.rs:342-343"]
    GUARD --> HUMAN["Human-decision kinds 31401, 31403-31405 stay never-grantable<br/>identity_port/acl.rs:89-91, identity_port/acl.rs:348-351"]
    PUB -->|"error"| SKIP["Governance skipped fail-open<br/>governance.rs:924-928"]
    LOCAL -->|"error"| SKIP
```

**Correction.** RESOLVED at agentbox 720e0d8 — the dream engine no longer reads its governance key under isolation: it takes a purpose-bound port capability (`../project/agentbox/services/dream-engine/src/relay.rs:150-158`) and the shipped ACL grants `governance_event` for kinds 31400, 31402 and 5 to the optional `agent_governance` key (`../project/agentbox/config/custody/identity-port-acl.json:54`); the key is `required: false` and must be provisioned by the operator before use (`../project/agentbox/config/custody/identity-port-acl.json:41`).
**Activation.** Flag-on behaviour depends on effective boot configuration, ownership and running identity. This pass reads those contracts but does not enable isolation, inspect keys or certify a running image. Key-file precedence alone does not imply the flag is on.
**Scope.** ADR-2122's ER-6 decision is assessed here, not its entire custody programme. Complete the signing, authentication and withdrawal design together, including old-key cases; see the [governance audit](../../estate-review/2026-11-01-dream-governance-and-delivery.md).

## AB-44.9 ADR-2143 — source supply is plumbing, not a decision

```mermaid
flowchart TB
    REPORT["Extract report requests and classify human actions<br/>engine.rs:1743<br/>wanted.rs:191"] --> STORE["Record pending keys or report a repeated inactive key<br/>wanted.rs:231"]
    STORE --> ALERT["Loop alert per record call<br/>engine.rs:1764"]
    STORE --> TAKE["Take active requests and increment each offer count<br/>wanted.rs:276"]
    TAKE --> SCAN["Append path-shaped scan entries<br/>engine.rs:1361"]
    SCAN --> CORE["Validate and select requested entries first<br/>source.rs:1099"]
    CORE --> RENDER["Clip then add markers and redact<br/>source.rs:858"]
    RENDER --> RECEIPT["Return receipt only if a file is marked inlined<br/>source.rs:1192"]
    RECEIPT --> SETTLE["Settle by path or directory prefix<br/>ranges are not compared<br/>wanted.rs:328"]
```


**What it shows.** Requests are offered before planner picks, with scan paths appended afterwards. Reservation counts candidate entries before validation, clipping markers can exceed the nominal byte allocation, and an empty line window can count as honoured. `take_for` counts invocations, not distinct night identifiers; `settle` treats any range from the same path as satisfying all its ranges (`../project/agentbox/services/dream-engine/src/wanted.rs:284`, `../project/agentbox/services/dream-engine/src/wanted.rs:328`). **Why it is this way.** ADR-2143 moves repetitive file requests out of the operator panel, but the [source-supply audit](../../estate-review/2026-11-01-dream-recovery-and-source-supply.md) finds that its completion markers do not prove the requested content was supplied. A requested directory is satisfied by any file under it. An all-refused read returns no source record (`../dream-engine/crates/dream-core/src/source.rs:1192`).

**Drift (question shape):** four fields are the prompt's desired format, not an enforced invariant. Parsing requires only Decision and If approved; unstructured questions pass through unchanged (`../project/agentbox/services/dream-engine/src/inbox.rs:278`, `../project/agentbox/services/dream-engine/src/inbox.rs:289`).

**Tension (expiry and evidence):** source receipt writes and settlement writes are best effort, and the shared request store has no transaction around read-modify-write (`../project/agentbox/services/dream-engine/src/engine.rs:1390`, `../project/agentbox/services/dream-engine/src/wanted.rs:91`). Current source does not establish three distinct nights, exact range delivery or one lifetime loop alert per request.

## AB-44.10 Optional triage can defer without a durable receipt

```mermaid
flowchart TB
    CLAIM{"Report claims ACCEPT and triage enabled?<br/>engine.rs:1534-1536"}
    CLAIM -->|no| BUILD["Proceed to ordinary candidate handling<br/>engine.rs:1560"]
    CLAIM -->|yes| ROUTE{"Endpoint available under configured locality?<br/>jev_triage.rs:147-188"}
    ROUTE -->|no| BUILD
    ROUTE -->|yes| ASK["First request: report head, path hints and evaluator names<br/>jev_triage.rs:320"]
    ASK --> SIGNAL{"Valid typed usefulness and support values?<br/>jev_triage.rs:347"}
    SIGNAL -->|fallback| BUILD
    SIGNAL -->|yes| DEFER["Compute defer from support and explicit threshold<br/>jev_triage.rs:491"]
    DEFER --> SCREEN["Second request: library discipline screen, capped inputs<br/>jev_triage.rs:522; screen.rs:222"]
    SCREEN --> RECORD["Write annotation or screen-fallback record<br/>jev_triage.rs:522-550"]
    RECORD --> CHECK{"Defer computed true?<br/>jev_triage.rs:558"}
    CHECK -->|no| BUILD
    CHECK -->|yes| SKIP["Skip build even if annotation write failed<br/>engine.rs:1560"]
    SKIP --> GATE["NotAttempted: ACCEPT stays unproven<br/>gate.rs:445"]
    RECORD -.-> AUTH["Each HTTP call has 15-second timeout and no redirects<br/>jev_triage.rs:275-279"]
```

Default threshold is unset, so valid signals annotate without deferral. A configured threshold must lie in 0 to 0.5. Neither usefulness nor the discipline score can approve a candidate. The two requests are sequential; the first failing means the second is not sent. An initial fallback writes nothing and leaves any prior annotation in place (`../project/agentbox/services/dream-engine/src/jev_triage.rs:506-512`).

The first request clips the report to 4,000 characters but does not cap the complete request. Its path hints omit deletions and pure renames. The second request retains at most 16 disciplines, while the saved annotation lists the full input set (`../project/agentbox/services/dream-engine/src/jev_triage.rs:306,320,528`; `../dream-engine/crates/dream-core/src/screen.rs:230`). These signals cannot replace evaluator evidence. The [execution review](../../estate-review/2026-11-01-dream-execution-and-triage.md) records transport-free probes, current configuration and the limits of ADR-2155's staged activation record. A staged record does not prove that no other running image has the feature.

## AB-44.11 Journal delivery — local counters and remote evidence

```mermaid
flowchart TB
    SESSION["Night ID plus whole seconds, normalised and truncated to 160<br/>Not unique across all attempts or names<br/>journal.rs:300-315"] --> CALL["called: post tool.called, then increment open_calls<br/>journal.rs:165-187"]
    CALL --> EFFECT["Caller performs effect independently of delivery success<br/>Phase 1 is fail-open, not an approval gate<br/>journal.rs:11-17"]
    EFFECT --> DONE["completed: decrement open_calls BEFORE posting<br/>causation present only if called returned an event ID<br/>journal.rs:199-215"]
    CALL --> HTTP
    DONE --> HTTP
    HTTP{"Client enabled and fewer than 3 consecutive failures?<br/>journal.rs:236-244"}
    HTTP -->|"no client"| OFF["No delivery counters; open_calls still changes<br/>journal.rs:236"]
    HTTP -->|"breaker open"| SKIP["skipped++; no recovery probe this session<br/>journal.rs:237-243"]
    HTTP -->|"yes"| POST["POST /v1/exec/record<br/>same API journal singleton<br/>exec-record.js:59-62"]
    POST --> KEY["Caller key becomes deterministic event ID<br/>Duplicate ID returns prior envelope, without payload comparison<br/>exec-record.js:52-54; execution-journal.js:144-151"]
    KEY --> APPEND["Assign seq, await dispatch, then update in-memory state<br/>Concurrent appends can share seq; restart needs hydration<br/>execution-journal.js:152-185"]
    APPEND --> ACK{"HTTP success and string event_id?<br/>journal.rs:257-270"}
    ACK -->|"yes, including duplicate or empty ID"| REC["recorded++; reset failure streak<br/>journal.rs:267-270"]
    ACK -->|"no"| FAIL["failed++; increment streak; no retry or local outbox<br/>journal.rs:272-276"]
    DONE --> STATS["unpaired = current open_calls, not remote pair reconciliation<br/>journal.rs:224-229"]
    STATS --> HEALTH["Normal path records repo stats, then night-level stats<br/>Early returns and crashes have narrower coverage<br/>engine.rs:359-368; engine.rs:461-462; records.rs:411-428"]
```

**Evidence.** Loopback probes reproduce unmatched remote records with unpaired=0; route probes reproduce changed-payload deduplication, restart sequence reuse and concurrent duplicate sequence numbers while the on-disk hash chain verifies. These are separate integrity properties, detailed in the [journal audit](../../estate-review/2026-11-01-dream-journal-and-api-down.md).
**Coverage.** Several inbox writes have no tool pair, including shared-remote warnings, annexe-health failure and damaged-run alerts; inbox::add does not journal internally (`../project/agentbox/services/dream-engine/src/engine.rs:251`, `../project/agentbox/services/dream-engine/src/engine.rs:933`, `../project/agentbox/services/dream-engine/src/engine.rs:2160`, `../project/agentbox/services/dream-engine/src/inbox.rs:174-215`).
**Skip semantics.** The repo wrapper posts turn.started before calling the body and turn.completed afterwards, including locked or already-finished returns. No tool effect is required for those lifecycle events (`../project/agentbox/services/dream-engine/src/engine.rs:603-624`).


## AB-44.12 Digest — evidence limits and encryption configuration

```mermaid
flowchart TB
    DISK["Read health, configured ledgers and open-item count<br/>digest.rs:386-399"] --> DATE{"Matching health date?<br/>digest.rs:186"}
    DATE -->|no| ABSENT["Text says nothing was tested<br/>Missing evidence does not establish that conclusion<br/>digest.rs:187-191"]
    DATE -->|yes| ANY{"Any scheduled outcomes?<br/>digest.rs:193-217"}
    ANY -->|no| ZERO["No-eligible text and available standby reasons<br/>digest.rs:193-215"]
    ZERO --> LOCAL
    ANY -->|yes| WORDS["Compose from outcome labels and last ledger row<br/>ACCEPT says confirmed; REJECT says disproved<br/>digest.rs:242-275"]
    WORDS --> LOCAL["Append local open count as panel backlog<br/>digest.rs:220-227"]
    ABSENT --> LOCAL
    LOCAL --> ZONE["Resolve section and zone metadata; select encryption plan<br/>digest.rs:421-430"]
    ZONE --> PLAIN["Malformed or absent zones become empty;<br/>unknown section can select first configured zone<br/>zone_crypto.rs:86-105"]
    ZONE --> SECRET["Known encrypted zone with gate on: latest held epoch or refuse<br/>zone_crypto.rs:109-115; zone_crypto.rs:221-233"]
    PLAIN --> FALL["No encrypted match means plaintext plan<br/>zone_crypto.rs:222-224"]
    SECRET -->|"no held key"| STOP["Withhold digest<br/>digest.rs:431-432"]
    SECRET -->|"held key"| POST["Sign and publish; retry once after missing read-back<br/>digest.rs:453-479"]
    FALL --> POST
    POST --> VERIFY["Read-back match checks ID only<br/>Relay query does not verify signature or filter membership<br/>digest.rs:467-476; relay.rs:483-489"]
```

**Presentation.** A missing health file, REJECT label or unpublished inbox count is insufficient for the digest's stronger claims. Standby repositories disappear from the ordinary outcomes branch. Case context now reads the configured ledgerPath, RESOLVED at agentbox 720e0d8 (`../project/agentbox/services/dream-engine/src/governance.rs:279`).
**Privacy.** The no-key refusal applies only after zone policy identifies an encrypted zone. Malformed metadata produces plaintext selection even with the gate on; the probe does not publish plaintext or claim the live relay would admit it. No current zone-key epoch is fetched, so latest means latest locally held.
**Evidence.** The [governance audit](../../estate-review/2026-11-01-dream-governance-and-delivery.md) records isolated probes and the source boundary. Neither a publish ACK nor an ID-only read-back establishes delivery to the operator or a verified current deployment.
