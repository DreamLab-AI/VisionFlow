---
id: SP-10
title: Failure modes, fail-closed boundaries and the invariant register
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2002, ADR-2003, ADR-2004, ADR-2005, ADR-2006, ADR-2007]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/nip98.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/replay.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/replay_store.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/resolver.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/evaluator.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/parser.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/multitenant.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/security/dotfile.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/storage/fs.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/provenance.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/ldp.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/oidc/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/payments.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/error.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mempool.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/handlers/pay.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/main.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs/Cargo.toml
  - ../solid-pod-rs/Cargo.toml
  - ../solid-pod-rs/crates/solid-pod-rs/src/config/schema.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/handlers/prov.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/notifications/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/metrics.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/security/ssrf.rs
  - ../solid-pod-rs/.github/workflows/ci.yml
  - ../solid-pod-rs/scripts/parity-check.sh
verified_commit: 1d9da5270
---

## SP-10.1 The seven baseline invariants and where each lives in code

```mermaid
flowchart TD
    I1["1. NIP-98 is SINGLE-SOURCED — one verifier, siblings delegate<br/>solid-pod-rs/src/auth/nip98.rs:116"]
    I2["2. Replay protection keeps BOTH layers — freshness window and a<br/>single-use claim — and never evicts an unexpired entry<br/>solid-pod-rs/src/auth/replay.rs:275"]
    I3["3. Pod-label '..' scrubbing stays ITERATIVE<br/>solid-pod-rs/src/multitenant.rs:184"]
    I4["4. ACL parsing stays SIZE-CAPPED at the parse boundary<br/>solid-pod-rs/src/wac/parser.rs:40"]
    I4b["4b. ONLY an absent policy inherits<br/>solid-pod-rs/src/wac/resolver.rs:153"]
    I5["5. Provenance never changes a write's HTTP status —<br/>but the failure is no longer swallowed<br/>solid-pod-rs/src/provenance.rs:567"]
    I6["6. Discovery metadata must MATCH the crypto path<br/>solid-pod-rs/src/oidc/mod.rs:184"]
    I7["7. A private response is never advertised as publicly cacheable<br/>solid-pod-rs/src/ldp.rs:1915"]

    I1 --> SUR["the compliance surface"]
    I2 --> SUR
    I3 --> SUR
    I4 --> SUR
    I4b --> SUR
    I5 --> SUR
    I6 --> SUR
    I7 --> SUR

    N["Invariant 6 is the one currently BROKEN in the shipped direction: the verifier<br/>accepts EdDSA (solid-pod-rs/src/oidc/mod.rs:549) and discovery does not<br/>advertise it. Closing the gap must move both halves together. See SP-05.9."]
    I6 -.-> N
    N2["Each invariant names a specific regression, not a general principle — reverting<br/>any one of them re-opens a named, previously-fixed bug."]
    SUR -.-> N2
```

## SP-10.2 Fail-closed boundaries, in request order

```mermaid
flowchart TD
    A["path traversal — parsed components, not substrings<br/>solid-pod-rs/src/storage/fs.rs:81"]
    B["dotfile allowlist — a closed set of three<br/>solid-pod-rs/src/security/dotfile.rs:24"]
    C["NIP-98 — Schnorr verification is UNCONDITIONAL; without the feature<br/>the verifier is a fail-closed stub<br/>solid-pod-rs/src/auth/nip98.rs:376"]
    D["replay — a store at capacity REFUSES the credential<br/>solid-pod-rs/src/auth/replay_store.rs:75"]
    E["policy resolution — invalid or unreadable DENIES, never inherits<br/>solid-pod-rs/src/wac/resolver.rs:275"]
    F["evaluation — no ACL means no access; an unmatched condition skips the rule<br/>solid-pod-rs/src/wac/evaluator.rs:264"]
    G["sidecar elevation — reading or writing an ACL demands Control<br/>solid-pod-rs/src/wac/mod.rs:252"]
    H["lockout guard — an ACL that drops the caller's Control is REFUSED<br/>solid-pod-rs-server/src/lib.rs:721"]
    I["payment — a raced debit failure denies rather than serving unpaid<br/>solid-pod-rs-server/src/lib.rs:928"]
    J["PATCH — an unparseable stored body is refused, not overwritten<br/>solid-pod-rs-server/src/lib.rs:1673"]
    K["method mapping — an unknown verb maps to Read, the least privilege<br/>solid-pod-rs/src/wac/mod.rs:185"]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J
    K --> F

    N["Every one of these turns an UNKNOWN into a denial. The pattern is uniform:<br/>the code never treats 'I could not determine' as 'therefore allow'."]
    E -.-> N
```

## SP-10.3 Where a partial failure is visible rather than swallowed

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant H as write handler
    participant S as Storage
    participant P as provenance

    C->>H: PUT a resource
    H->>S: storage.put
    S-->>H: Ok — the LDP write is now COMMITTED
    H->>P: git_mark_write
    alt mark succeeded
        P-->>H: stage local-mark-committed
    else mark failed
        P-->>H: stage resource-stored plus mark_error<br/>solid-pod-rs/src/provenance.rs:642
    else anchor failed
        P-->>H: the mark stands, anchor_error carried alongside<br/>solid-pod-rs/src/provenance.rs:565
    else sidecar write failed
        P-->>H: stage unchanged, mark_error set<br/>solid-pod-rs-server/src/lib.rs:3650
    end
    H-->>C: 201 in EVERY branch, plus X-Provenance naming the tier reached<br/>solid-pod-rs-server/src/lib.rs:1162

    Note over H: INVARIANT (ADR-2004): the status never changes, because the bytes ARE stored —<br/>but "stored" and "stored and provably marked" are now distinguishable to the<br/>caller. The receipt makes the write-then-mark window observable — it does not<br/>close it. The three steps are still not atomic.
```

## SP-10.4 Windows that are open by construction

```mermaid
flowchart TD
    W1["write-then-mark is NOT atomic — a crash between the storage put and the<br/>git commit leaves an unmarked resource<br/>solid-pod-rs-server/src/lib.rs:1493"]
    W2["body and .meta sidecar are two separate atomic renames — a crash between<br/>them leaves a mismatched pair<br/>solid-pod-rs/src/storage/fs.rs:111"]
    W3["replay protection is process-local — a second replica shares nothing, and a<br/>restart reopens the window for one TTL<br/>solid-pod-rs-server/src/lib.rs:192"]
    W4["PAYMENT_STATE_LOCK is process-wide, not storage-wide — two replicas over one<br/>pod store share no lock<br/>solid-pod-rs-server/src/lib.rs:200"]
    W5["the NIP-98 freshness window is two-sided and 60 s wide; only the single-use<br/>store closes it<br/>solid-pod-rs/src/auth/nip98.rs:28"]
    W6["COPY authorises Write on the destination and never checks Read on the source<br/>solid-pod-rs-server/src/lib.rs:2578"]
    W7["glob GET gates on the FOLDER; a child with a stricter ACL is still merged<br/>solid-pod-rs-server/src/lib.rs:2655"]

    W1 --> R["recorded, not closed"]
    W2 --> R
    W3 --> R
    W4 --> R
    W5 --> R
    W6 --> R
    W7 --> R

    N["Each of these is a KNOWN property with a named mitigation, not an oversight:<br/>W1 and W2 are made observable by the receipt and the ETag, W3 and W4 need shared<br/>state before a multi-replica deployment, W5 is closed by SP-05.4."]
    R -.-> N
```

## SP-10.5 Default-off switches — the security posture of a bare binary

```mermaid
flowchart LR
    subgraph OFF["OFF unless explicitly enabled"]
        MCP["MCP tool surface — --mcp / JSS_MCP, and --no-mcp always wins<br/>solid-pod-rs-server/src/main.rs:301"]
        REG["open registration — --open-registration<br/>solid-pod-rs-server/src/main.rs:102"]
        ADM["admin provisioning — 403 unconditionally with no PSK<br/>solid-pod-rs-server/src/main.rs:96"]
        TXO["unverified TXO deposit stand-in<br/>solid-pod-rs-server/src/main.rs:124"]
        GIT["the git feature, and therefore all provenance<br/>crates/solid-pod-rs-server/Cargo.toml:123"]
        FORGE["the forge<br/>crates/solid-pod-rs-server/Cargo.toml:155"]
        TLS["TLS<br/>crates/solid-pod-rs-server/Cargo.toml:126"]
        ORG["the acl:origin gate<br/>crates/solid-pod-rs/Cargo.toml:159"]
        OIDC["OIDC routes — auth.oidc_enabled defaults false<br/>solid-pod-rs/src/config/schema.rs:190"]
        QUOTA["quota — default_quota_bytes 0 means off<br/>solid-pod-rs/src/config/schema.rs:289"]
    end
    ON["ON in a default library build: fs and memory backends,<br/>the tokio runtime, the notifications stack<br/>crates/solid-pod-rs/Cargo.toml:98"]

    OFF --> POSTURE["a bare solid-pod-rs-server serves LDP under WAC and NIP-98,<br/>and opens nothing else"]
    ON --> POSTURE

    N["Two of these defaults cut the other way. Provenance being off means the<br/>README's 'every write is a git-mark commit' needs --features git to be true<br/>(SP-07.2), and the origin gate being off means a default build ignores the<br/>Origin header entirely (SP-04.8)."]
    GIT -.-> N
    N2["The loud ones announce themselves: the binary warns when the TXO stand-in is<br/>enabled (solid-pod-rs-server/src/main.rs:307) and when OIDC is disabled<br/>(solid-pod-rs-server/src/main.rs:339)."]
    TXO -.-> N2
```

## SP-10.6 The DIVERGENCE register — governing-doc open items

```mermaid
flowchart TD
    D1["ADR-2003: access-token aud is NOT validated — only claim presence<br/>solid-pod-rs/src/oidc/mod.rs:812"]
    D2["ADR-2003: extract_webid has no cnf.webid branch; the LWS10 delta is unshipped<br/>solid-pod-rs/src/oidc/mod.rs:864"]
    D3["ADR-2003: discovery does not advertise EdDSA though the verifier accepts it<br/>solid-pod-rs/src/oidc/mod.rs:184"]
    D4["ADR-2004: provenance is off in a default build<br/>crates/solid-pod-rs-server/Cargo.toml:123"]
    D5["ADR-2005: the edge ACL resolver is out of repo and still treats a failed or<br/>invalid read as a miss<br/>solid-pod-rs/src/wac/resolver.rs:265"]
    D6["ADR-2006: the ReplayStore seam has exactly ONE implementor<br/>solid-pod-rs/src/auth/replay_store.rs:90"]
    D7["ADR-2007: one mempool URL, no fallback chain, public testnet4 default<br/>solid-pod-rs-server/src/mempool.rs:56"]
    D8["REC-11: no pod-wide _prov enumeration — point lookup only<br/>solid-pod-rs-server/src/handlers/prov.rs:529"]

    D1 --> LEDGER["the ADR ledger amending the baseline"]
    D2 --> LEDGER
    D3 --> LEDGER
    D4 --> LEDGER
    D5 --> LEDGER
    D6 --> LEDGER
    D7 --> LEDGER
    D8 --> LEDGER

    N["D1, D2 and D3 are DOCUMENTED AND PINNED by tests, not fixed — the compatibility<br/>matrix records them as known deltas rather than claiming Solid-OIDC conformance<br/>the code does not have."]
    D3 -.-> N
    N2["D5 and D6 are the same shape: an interface exists in this repo, its second<br/>implementor lives in nostr-rust-forum, and neither can be discharged here.<br/>See SP-09.7 for the pin that blocks adoption."]
    D6 -.-> N2
```

## SP-10.7 The DOC-DRIFT register

```mermaid
flowchart LR
    R1["README: 'every write to a pod is a git-mark commit'<br/>— true only with --features git<br/>solid-pod-rs-server/src/lib.rs:3668"]
    R2["README: 'the exit right sits in the floor' — GET /api/exports/all is behind<br/>the default-off export-jsonld feature<br/>crates/solid-pod-rs-server/Cargo.toml:145"]
    R3["ecosystem-integration: 'S3 is configuration/dependency scaffolding only'<br/>— the S3 variant no longer exists in StorageBackendConfig<br/>solid-pod-rs-server/src/main.rs:139"]
    R4["ecosystem-integration: consumers pin solid-pod-rs 0.4 — the workspace is at<br/>0.5.0-alpha.9<br/>../solid-pod-rs/Cargo.toml:15"]
    R5["README: forge 'Phases 0-3 shipped' — Phases 4-7 are feature scaffolds that<br/>compile, not implementations<br/>crates/solid-pod-rs-server/Cargo.toml:156"]
    R6["README: '97.6% strict JSS parity' — computed by a script over a curated<br/>checklist, not by an executable conformance suite<br/>../solid-pod-rs/scripts/parity-check.sh:7"]

    R1 --> REG["doc drift against the code at this commit"]
    R2 --> REG
    R3 --> REG
    R4 --> REG
    R5 --> REG
    R6 --> REG

    N["None of these is a code defect. Each is a claim the living docs make that the<br/>code at 1d9da5270 qualifies or contradicts — which is exactly what the baseline's<br/>'live code beats doc prose' lookup order exists to catch."]
    REG -.-> N
```

## SP-10.8 The audit findings the README itself records

```mermaid
flowchart TD
    A["dated security and quality audit, 2026-08-19"]
    F1["filesystem symlink root escape — closed by the cap-std capability handle<br/>solid-pod-rs/src/storage/fs.rs:53"]
    F2["anonymous MCP reads / WAC sidecar bypass — mitigated by MCP being off<br/>solid-pod-rs-server/src/lib.rs:386"]
    F3["forged IdP identity — mitigated by not exposing the optional IdP router"]
    F4["non-atomic payment state — the process lock narrows it, does not close it<br/>solid-pod-rs-server/src/lib.rs:200"]
    F5["filesystem writes violate the advertised atomic storage contract<br/>solid-pod-rs/src/storage/fs.rs:111"]
    F6["cargo audit --deny warnings fails on RUSTSEC-2026-0258 in both HTTP/2 stacks<br/>../solid-pod-rs/.github/workflows/ci.yml:185"]

    A --> F1
    A --> F2
    A --> F3
    A --> F4
    A --> F5
    A --> F6

    N["The README's own guidance follows from these: keep MCP disabled, do not expose<br/>the optional IdP router, and do not carry value through the payment routes until<br/>the findings are fixed. The security controls the crate ships do NOT make this<br/>checkout production-safe, and the README says so."]
    A -.-> N
    N2["This is a pre-1.0 posture stated honestly rather than a passing gate — SP-09.4<br/>shows the audit job is genuinely red, not skipped."]
    F6 -.-> N2
```

## SP-10.9 Error taxonomy and how a failure reaches the caller

```mermaid
flowchart TD
    POD["PodError — the library's error type<br/>solid-pod-rs/src/error.rs:9"]
    TA["to_actix maps it to a status<br/>solid-pod-rs-server/src/lib.rs:499"]
    PF["policy_failure_to_actix — 403 for Invalid, 503 for Unavailable<br/>solid-pod-rs-server/src/lib.rs:1957"]
    AD["acl_denial — 401 with a WWW-Authenticate challenge, or 403<br/>solid-pod-rs-server/src/lib.rs:939"]
    PE["PaymentError<br/>solid-pod-rs/src/payments.rs:465"]
    PR["payment_error_response<br/>solid-pod-rs-server/src/handlers/pay.rs:343"]
    LOG["ErrorLoggingMiddleware — log_5xx and format_error_chain<br/>solid-pod-rs-server/src/lib.rs:3297"]
    C["the caller"]

    POD --> TA --> C
    PF --> C
    AD --> C
    PE --> PR --> C
    TA --> LOG
    PF --> LOG
    AD --> LOG

    N["The distinction between 403 and 503 on a policy failure is deliberate: Invalid<br/>is the operator's problem and will not fix itself, Unavailable may be transient.<br/>Collapsing both to 403 would tell a client to stop retrying a recoverable fault."]
    PF -.-> N
    N2["log_5xx runs in the OUTERMOST middleware so it observes responses that<br/>short-circuited inside an inner guard (SP-02.8), and format_error_chain<br/>(solid-pod-rs-server/src/lib.rs:3333) walks the cause chain rather than logging<br/>only the outer message."]
    LOG -.-> N2
```

## SP-10.10 Denial-of-service bounds

```mermaid
flowchart LR
    B1["request body cap — JSS_MAX_REQUEST_BODY, default 50 MiB<br/>solid-pod-rs-server/src/lib.rs:353"]
    B2["ACL byte cap — MAX_ACL_BYTES 1 MiB<br/>solid-pod-rs/src/wac/mod.rs:28"]
    B3["ACL JSON depth cap — 32, checked WITHOUT parsing<br/>solid-pod-rs/src/wac/mod.rs:33"]
    B4["NIP-98 token cap — MAX_EVENT_SIZE 64 KiB, before and after base64<br/>solid-pod-rs/src/auth/nip98.rs:29"]
    B5["replay store cap — DEFAULT_MAX_SIZE 10 000, refuses rather than evicting<br/>solid-pod-rs/src/auth/replay.rs:82"]
    B6["pod creation — one POST /.pods per IP per day<br/>solid-pod-rs-server/src/lib.rs:466"]
    B7["proxy byte cap — DEFAULT_PROXY_BYTE_CAP 50 MiB<br/>solid-pod-rs-server/src/lib.rs:2765"]
    B8["notification subscriptions — 10 000 overall, 100 per legacy connection<br/>solid-pod-rs/src/notifications/mod.rs:125"]
    B9["Slug length — MAX_SLUG_BYTES 255<br/>solid-pod-rs/src/ldp.rs:144"]
    B10["POST unique-name probing is bounded, with a hash fallback<br/>solid-pod-rs-server/src/lib.rs:1509"]

    B1 --> D["bounded before allocation"]
    B2 --> D
    B3 --> D
    B4 --> D
    B5 --> D
    B6 --> D
    B7 --> D
    B8 --> D
    B9 --> D
    B10 --> D

    N["B3 is the sharpest of these: the depth is counted over raw bytes because<br/>serde_json allocates stack proportional to nesting, so the check MUST happen<br/>before the parser is reached.<br/>solid-pod-rs/src/wac/mod.rs:39"]
    B3 -.-> N
    N2["B5 is the only bound that fails the REQUEST rather than truncating the DATA —<br/>see the ADR-2006 reasoning in SP-05.4."]
    B5 -.-> N2
```

## SP-10.11 What a reviewer should re-check first after any change

```mermaid
flowchart TD
    START["a change lands"]
    Q1{"did it touch auth/nip98.rs or auth/replay*?"}
    Q2{"did it touch wac/resolver.rs or wac/evaluator.rs?"}
    Q3{"did it touch provenance or git_mark_write?"}
    Q4{"did it touch oidc/mod.rs?"}
    Q5{"did it touch multitenant.rs or storage/fs.rs?"}
    Q6{"did it change a crate version or feature?"}

    C1["re-read invariants 1 and 2; confirm the store still REFUSES at capacity<br/>solid-pod-rs/src/auth/replay.rs:275"]
    C2["re-read invariants 4 and 4b; confirm Invalid and Unavailable still deny<br/>solid-pod-rs/src/wac/resolver.rs:143"]
    C3["re-read invariant 5; confirm the status is unchanged AND the receipt still<br/>reaches the caller<br/>solid-pod-rs-server/src/lib.rs:1162"]
    C4["re-read invariant 6; discovery and the verifier move TOGETHER<br/>solid-pod-rs/src/oidc/mod.rs:184"]
    C5["re-read invariant 3; confirm scrub_dotdot is still a loop<br/>solid-pod-rs/src/multitenant.rs:184"]
    C6["update the pin matrix and check whether the edge consumer can still compile<br/>the core surface — SP-09.7"]
    BL["update BASELINE-solid-pod-rs.md in the SAME commit, bump its version, and<br/>re-record its verified_commit"]

    START --> Q1 --> C1 --> BL
    START --> Q2 --> C2 --> BL
    START --> Q3 --> C3 --> BL
    START --> Q4 --> C4 --> BL
    START --> Q5 --> C5 --> BL
    START --> Q6 --> C6 --> BL

    N["The baseline's own change process requires this: revise the affected section<br/>with the new file:line, confirm the relevant invariant still holds, bump the<br/>version, and re-record verified_commit — in the same commit as the code."]
    BL -.-> N
    N2["A new decision is a thin ADR in the repo's docs/adr, citing the baseline.<br/>Archived ADRs are frozen evidence and are never authority."]
    BL -.-> N2
```

## SP-10.12 Observability — everything the pod emits

```mermaid
flowchart TD
    subgraph COUNT["Counters — SecurityMetrics"]
        SM["SecurityMetrics, a cheap-to-clone Arc bundle<br/>solid-pod-rs/src/metrics.rs:26"]
        SS["record_ssrf_block, labelled by IpClass<br/>solid-pod-rs/src/metrics.rs:63"]
        ST["ssrf_blocked_total<br/>solid-pod-rs/src/metrics.rs:76"]
        DF["record_dotfile_deny<br/>solid-pod-rs/src/metrics.rs:88"]
        DT["dotfile_denied_total<br/>solid-pod-rs/src/metrics.rs:93"]
    end
    subgraph WIRE["Wired in via with_metrics builders"]
        W1["SsrfPolicy::with_metrics<br/>solid-pod-rs/src/security/ssrf.rs:150"]
        W2["DotfileAllowlist::with_metrics<br/>solid-pod-rs/src/security/dotfile.rs:87"]
    end
    subgraph HDR["Per-response signals a client can read"]
        H1["X-Provenance and X-Provenance-Commit<br/>solid-pod-rs-server/src/lib.rs:1162"]
        H2["WAC-Allow, advisory on grant AND denial<br/>solid-pod-rs-server/src/lib.rs:1134"]
        H3["Updates-via — the notification endpoint<br/>solid-pod-rs-server/src/lib.rs:1141"]
    end
    subgraph LOG["Structured tracing"]
        L1["log_5xx and format_error_chain<br/>solid-pod-rs-server/src/lib.rs:3297"]
        L2["mempool selection recorded once at startup<br/>solid-pod-rs-server/src/mempool.rs:325"]
        L3["a denied ACL logs the policy path and reason<br/>solid-pod-rs-server/src/lib.rs:1957"]
    end

    SM --> SS
    SM --> ST
    SM --> DF
    SM --> DT
    SM --> WIRE
    HDR --> OP["what an operator or client can actually see"]
    LOG --> OP
    COUNT --> OP

    N["DIVERGENCE: Prometheus export is explicitly OUT OF SCOPE<br/>(solid-pod-rs/src/metrics.rs:4). The crate ships raw atomics and expects the<br/>binder to lift them into gauges — so a bare solid-pod-rs-server exposes NO<br/>metrics endpoint at all. There is no /metrics route in SP-02.9 to SP-02.11."]
    COUNT -.-> N
    N2["EXTERNAL: the doc names the upstream binder that owns the Prometheus registry<br/>(solid-pod-rs/src/metrics.rs:5) — that is VisionClaw's webxr server, which<br/>embeds the pod. See VC-26 and VC-08."]
    WIRE -.-> N2
    N3["Only TWO subsystems are counted — SSRF blocks and dotfile denials. WAC<br/>denials, replay rejections, quota refusals and provenance failures are logged<br/>but never counted, so their rates are not observable without log scraping."]
    SM -.-> N3
    N4["The counters are also OPT-IN: without a with_metrics call the policies hold a<br/>default bundle nobody reads, so a deployment that never wires one records<br/>nothing."]
    WIRE -.-> N4
```
