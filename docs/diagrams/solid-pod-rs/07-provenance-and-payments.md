---
id: SP-07
title: Provenance — git-marks, block-trails, Bitcoin anchoring — and the web ledger
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2004, ADR-2007]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs/src/provenance.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/mrc20.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/bitcoin_tx.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/payments.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/trading.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/anchor.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/mark.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/handlers/prov.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/handlers/pay.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mempool.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/trail_store.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/main.rs
verified_commit: 1d9da5270
---

## SP-07.1 The two provenance tiers

```mermaid
flowchart LR
    W["an LDP write that already succeeded"]
    GM["git-mark — cheap, ALWAYS attempted<br/>solid-pod-rs/src/provenance.rs:526"]
    AN["block-trail anchor — expensive, opt-in<br/>solid-pod-rs/src/provenance.rs:535"]
    SC["PROV-O sidecar at <resource>.prov.ttl<br/>solid-pod-rs/src/provenance.rs:1165"]
    MK["ProvenanceMark<br/>solid-pod-rs/src/provenance.rs:45"]

    W --> GM --> MK
    GM --> AN --> MK
    MK --> SC

    N["INVARIANT: the anchored state_hash IS the git commit SHA, so the Bitcoin UTXO<br/>commits to the git history — the two primitives form one chain rather than two<br/>parallel records.<br/>solid-pod-rs/src/provenance.rs:537"]
    AN -.-> N
    N2["GitMark (solid-pod-rs/src/provenance.rs:62) and BlockTrailAnchor<br/>(solid-pod-rs/src/provenance.rs:158) are the two tier records;<br/>GitMarkEnvelope (:92) and BlocktrailEnvelope (:217) are their wire forms."]
    MK -.-> N2
```

## SP-07.2 git_mark_write — the skip ladder

```mermaid
flowchart TD
    IN["git_mark_write(state, resource_path, agent, message)<br/>solid-pod-rs-server/src/lib.rs:3462"]
    SIDE{"path ends .acl, .meta or .prov.ttl?<br/>solid-pod-rs-server/src/lib.rs:3473"}
    S1["Skipped: ExcludedPath<br/>solid-pod-rs/src/provenance.rs:688"]
    CONT{"path ends with a slash?<br/>solid-pod-rs-server/src/lib.rs:3480"}
    S2["Skipped: Container<br/>solid-pod-rs/src/provenance.rs:690"]
    ROOT{"data_root configured?<br/>solid-pod-rs-server/src/lib.rs:3485"}
    S3["Skipped: NotConfigured<br/>solid-pod-rs/src/provenance.rs:682"]
    SPLIT{"path splits into pod plus rest?<br/>solid-pod-rs-server/src/lib.rs:3494"}
    S4["Skipped: UnresolvablePath<br/>solid-pod-rs/src/provenance.rs:692"]
    GITD{"data_root/{pod}/.git is a directory?<br/>solid-pod-rs-server/src/lib.rs:3502"}
    S5["Skipped: NotGitBacked<br/>solid-pod-rs/src/provenance.rs:685"]
    GO["proceed to policy resolution — SP-07.3"]

    IN --> SIDE
    SIDE -- yes --> S1
    SIDE -- no --> CONT
    CONT -- yes --> S2
    CONT -- no --> ROOT
    ROOT -- no --> S3
    ROOT -- yes --> SPLIT
    SPLIT -- no --> S4
    SPLIT -- yes --> GITD
    GITD -- no --> S5
    GITD -- yes --> GO

    N["INVARIANT: marking a .prov.ttl would RECURSE, and ACL/meta writes are<br/>control-plane traffic, not content — both are excluded by construction."]
    S1 -.-> N
    N2["A skip is not a fault. ProvenanceSkip exists precisely so a correctly<br/>unmarked write is never reported as a provenance failure.<br/>solid-pod-rs/src/provenance.rs:677"]
    S5 -.-> N2
    N3["DIVERGENCE (ADR-2004): with the server's empty default feature set this whole<br/>function is the no-op shim (solid-pod-rs-server/src/lib.rs:3668), so a default<br/>build records ZERO marks. Every provenance claim carries a --features git caveat."]
    IN -.-> N3
```

## SP-07.3 Anchor policy resolution

```mermaid
flowchart TD
    ACL["the resource's effective ACL"]
    COND["acl:ProvenanceAnchor condition<br/>solid-pod-rs/src/wac/conditions.rs:61"]
    MODE["anchor_mode_of -> AnchorMode<br/>solid-pod-rs/src/wac/anchor.rs:119"]
    RES["resolve_anchor_policy<br/>solid-pod-rs-server/src/handlers/prov.rs:82"]
    NEVER["AnchorPolicy::Never — the default for an ordinary write<br/>solid-pod-rs/src/provenance.rs:376"]
    ALWAYS["AnchorPolicy::Always<br/>solid-pod-rs/src/provenance.rs:378"]
    HV["AnchorPolicy::HighValue<br/>solid-pod-rs/src/provenance.rs:381"]
    EP["AnchorPolicy::Epoch<br/>solid-pod-rs/src/provenance.rs:384"]
    INLINE["anchors_inline(high_value)<br/>solid-pod-rs/src/provenance.rs:396"]
    BLD["build_anchorer — only when the policy is not Never<br/>solid-pod-rs-server/src/handlers/prov.rs:120"]

    ACL --> COND --> MODE --> RES
    RES --> NEVER
    RES --> ALWAYS
    RES --> HV
    RES --> EP
    NEVER --> INLINE
    ALWAYS --> INLINE
    HV --> INLINE
    EP --> INLINE
    RES --> BLD

    N["Never and Epoch never anchor INLINE — Epoch defers to the accumulator.<br/>Always is unconditional; HighValue is inline only when the resource is flagged.<br/>solid-pod-rs/src/provenance.rs:398"]
    INLINE -.-> N
    N2["The handler rewrites Epoch to Never before calling record, then batches the SHA<br/>itself, so record's inline path only ever sees Never, Always or HighValue.<br/>solid-pod-rs-server/src/lib.rs:3545"]
    EP -.-> N2
```

## SP-07.4 record_receipt — typed, per-tier outcomes (ADR-2004)

```mermaid
stateDiagram-v2
    [*] --> Attempt: record_receipt(WriteRecord)<br/>solid-pod-rs/src/provenance.rs:569
    Attempt --> MarkFailed: marker.mark_write returned an error<br/>solid-pod-rs/src/provenance.rs:579
    Attempt --> Marked: a commit SHA exists

    MarkFailed --> ResourceStored: stage resource-stored plus mark_error<br/>solid-pod-rs/src/provenance.rs:642
    Marked --> NoAnchor: policy did not want one
    Marked --> AnchorTried: policy wanted one

    NoAnchor --> LocalMarkCommitted: stage local-mark-committed<br/>solid-pod-rs/src/provenance.rs:645
    AnchorTried --> AnchorFailed: anchor_error recorded, the MARK STANDS
    AnchorTried --> Submitted: stage anchor-submitted<br/>solid-pod-rs/src/provenance.rs:648
    Submitted --> Confirmed: stage anchor-confirmed<br/>solid-pod-rs/src/provenance.rs:651
    AnchorFailed --> LocalMarkCommitted

    ResourceStored --> [*]
    LocalMarkCommitted --> [*]
    Confirmed --> [*]

    note right of MarkFailed
      INVARIANT (ADR-2004): the LDP write has already succeeded, so provenance
      NEVER changes the response status — but a failed mark is no longer
      swallowed into a warn! line. Silently discarding it would make a stored write
      and a stored-and-provably-marked write indistinguishable to the caller.
      record_receipt never returns an error at all
      (solid-pod-rs/src/provenance.rs:567).
    end note
    note right of AnchorFailed
      A failed anchor never suppresses a successful git-mark
      (solid-pod-rs/src/provenance.rs:565), and the server logs it separately
      (solid-pod-rs-server/src/lib.rs:3586).
    end note
```

## SP-07.5 The receipt on the wire

```mermaid
sequenceDiagram
    autonumber
    participant H as LDP write handler
    participant G as git_mark_write<br/>solid-pod-rs-server/src/lib.rs:3462
    participant R as ProvenanceReceipt<br/>solid-pod-rs/src/provenance.rs:726
    participant S as Storage
    participant C as Client

    H->>G: mark the write
    G->>R: record_receipt then restore the full pod-relative path<br/>solid-pod-rs-server/src/lib.rs:3571
    opt policy is Epoch
        G->>G: epoch_push_and_maybe_anchor — one tx notarises many commits<br/>solid-pod-rs-server/src/handlers/prov.rs:177
    end
    G->>S: write <resource>.prov.ttl<br/>solid-pod-rs-server/src/lib.rs:3640
    alt sidecar write failed
        G->>R: mark_error set, stage stays local-mark-committed<br/>solid-pod-rs-server/src/lib.rs:3650
    end
    G-->>H: receipt
    H->>C: X-Provenance = receipt.summary()<br/>solid-pod-rs/src/provenance.rs:806
    H->>C: X-Provenance-Commit = receipt.commit_sha()<br/>solid-pod-rs/src/provenance.rs:796
    Note over G: The sidecar write also fires the FS-watch StorageEvent the Updates-via<br/>notification stream relays, so subscribers see the new mark. It ends in<br/>.prov.ttl, so the skip guard in SP-07.2 stops the recursion.
    Note over H: header_safe (solid-pod-rs/src/provenance.rs:827) sanitises the summary before<br/>it becomes a header value.
```

## SP-07.6 The `_prov` query surface

```mermaid
flowchart LR
    R1["GET /{pod}/_prov/{commit_sha} -> handle_resolve<br/>solid-pod-rs-server/src/handlers/prov.rs:223"]
    R2["POST /{pod}/_prov/anchor -> handle_anchor<br/>solid-pod-rs-server/src/handlers/prov.rs:316"]
    REG["handlers::prov::register<br/>solid-pod-rs-server/src/handlers/prov.rs:529"]
    RC["resolve_commit against the pod repo<br/>solid-pod-rs-git/src/api.rs:374"]
    PRICE["anchor_price_sats<br/>solid-pod-rs-server/src/handlers/prov.rs:457"]
    DEB["debit then credit on failure<br/>solid-pod-rs-server/src/handlers/prov.rs:474"]
    SIDE["the .prov.ttl sidecar GET is an ORDINARY LDP read"]

    REG --> R1 --> RC
    REG --> R2 --> PRICE --> DEB
    R1 -.-> SIDE

    N["DIVERGENCE (baseline open item REC-11): the provenance query surface is<br/>POINT-LOOKUP only — one commit SHA, or a per-resource sidecar. There is no<br/>pod-wide _prov enumeration, so 'one queryable trace over a whole pod' is not<br/>delivered by this repo."]
    R1 -.-> N
    N2["The explicit git-mark to Bitcoin-anchor upgrade is payment-gated and refunds on<br/>failure — credit (solid-pod-rs-server/src/handlers/prov.rs:502) is the<br/>compensating half of debit."]
    DEB -.-> N2
    N3["is_sidecar (solid-pod-rs-server/src/handlers/prov.rs:451) keeps the anchor<br/>endpoint off control-plane paths."]
    R2 -.-> N3
```

## SP-07.7 Epoch batching — one transaction, many commits

```mermaid
sequenceDiagram
    autonumber
    participant G as git_mark_write
    participant L as load_epoch_commits<br/>solid-pod-rs-server/src/handlers/prov.rs:148
    participant A as EpochAccumulator<br/>solid-pod-rs/src/provenance.rs:937
    participant M as merkle_root<br/>solid-pod-rs/src/provenance.rs:878
    participant AN as BlockAnchorer
    participant S as save_epoch_commits<br/>solid-pod-rs-server/src/handlers/prov.rs:156

    G->>L: read the pod's current epoch commit list
    G->>A: push the fresh commit SHA<br/>solid-pod-rs/src/provenance.rs:967
    alt the epoch is below JSS_PROV_EPOCH_SIZE
        A-->>S: persist and stop<br/>solid-pod-rs-server/src/handlers/prov.rs:49
    else the epoch fills
        A->>M: merkle_leaf per commit, then the root<br/>solid-pod-rs/src/provenance.rs:906
        M->>AN: anchor the ROOT once
        AN-->>A: ClosedEpoch<br/>solid-pod-rs/src/provenance.rs:948
        A->>S: reset the accumulator
    end
    Note over AN: A failed batch anchor is swallowed — it never fails the write nor the<br/>git-mark (solid-pod-rs-server/src/lib.rs:3626).
    Note over M: MerkleProof (solid-pod-rs/src/provenance.rs:914) lets one commit prove its<br/>membership in the anchored root without revealing the rest of the epoch.<br/>merkle_root is deterministic and order-sensitive<br/>(solid-pod-rs/src/provenance.rs:1757).
```

## SP-07.8 The git marker — shelling out to git

```mermaid
sequenceDiagram
    autonumber
    participant P as ProvenanceLog
    participant SM as ShellGitMarker::mark_write<br/>solid-pod-rs-git/src/mark.rs:147
    participant G as git subprocess<br/>solid-pod-rs-git/src/mark.rs:78
    participant H as head_sha<br/>solid-pod-rs-git/src/mark.rs:126

    P->>SM: (repo, path, agent_did, message)
    SM->>G: stage and commit the single path
    G-->>SM: exit status and output
    SM->>H: read HEAD
    H-->>SM: the new commit SHA
    SM-->>P: GitMark<br/>solid-pod-rs/src/provenance.rs:62
    Note over SM: The committer name is configurable<br/>(solid-pod-rs-git/src/mark.rs:61) — repo_slug<br/>(solid-pod-rs-git/src/mark.rs:118) names the repo in the mark.
    Note over P: GitMarker is a trait (solid-pod-rs/src/provenance.rs:309), so a consumer can<br/>substitute a libgit2 or in-process implementation without touching the<br/>composition in ProvenanceLog.
```

## SP-07.9 MRC20 state chain and Bitcoin anchoring

```mermaid
flowchart TD
    JCS["jcs — RFC 8785 canonical JSON<br/>solid-pod-rs/src/mrc20.rs:32"]
    SHA["sha256_hex over the canonical form<br/>solid-pod-rs/src/mrc20.rs:66"]
    ST["Mrc20State<br/>solid-pod-rs/src/mrc20.rs:77"]
    VAL["validate_mrc20_state<br/>solid-pod-rs/src/mrc20.rs:134"]
    LNK["verify_state_link — prev-state hash chaining<br/>solid-pod-rs/src/mrc20.rs:151"]
    TR["Mrc20Trail<br/>solid-pod-rs/src/mrc20.rs:110"]
    DEP["verify_mrc20_deposit<br/>solid-pod-rs/src/mrc20.rs:196"]
    KEY["bt_derive_chained_pubkey / privkey<br/>solid-pod-rs/src/mrc20.rs:350"]
    ADDR["bt_address — bech32m taproot<br/>solid-pod-rs/src/mrc20.rs:474"]
    VER["verify_mrc20_anchor<br/>solid-pod-rs/src/mrc20.rs:509"]
    ML["MempoolLookup trait<br/>solid-pod-rs/src/mrc20.rs:292"]

    JCS --> SHA --> ST --> VAL --> LNK --> TR
    TR --> DEP
    KEY --> ADDR --> VER --> ML

    N["INVARIANT: the state hash is taken over the RFC 8785 canonical serialisation,<br/>not over the raw bytes — two semantically identical JSON documents must hash<br/>identically or the chain breaks on a formatting change."]
    JCS -.-> N
    N2["MRC20_PROFILE mono.mrc20.v0.1 and TRANSFER_OP name the wire contract<br/>solid-pod-rs/src/mrc20.rs:23"]
    ST -.-> N2
    N3["extract_transfers_to and total_transferred_to are the read side<br/>solid-pod-rs/src/mrc20.rs:174"]
    DEP -.-> N3
```

## SP-07.10 The hand-rolled Bitcoin transaction path

```mermaid
flowchart LR
    BLD["build_transaction<br/>solid-pod-rs/src/bitcoin_tx.rs:293"]
    P2TR["p2tr_script<br/>solid-pod-rs/src/bitcoin_tx.rs:191"]
    TAG["tagged_hash — BIP-340/341 domain separation<br/>solid-pod-rs/src/bitcoin_tx.rs:117"]
    XO["xonly_of<br/>solid-pod-rs/src/bitcoin_tx.rs:206"]
    VS["verify_keypath_signature<br/>solid-pod-rs/src/bitcoin_tx.rs:464"]
    FEE["DEFAULT_FEE_SATS 300, DUST_LIMIT_SATS 546<br/>solid-pod-rs/src/bitcoin_tx.rs:69"]
    CO["checked_output — refuses to mint value<br/>solid-pod-rs/src/bitcoin_tx.rs:929"]
    MINT["mint_token<br/>solid-pod-rs/src/bitcoin_tx.rs:617"]
    XFER["transfer_token_with_key<br/>solid-pod-rs/src/bitcoin_tx.rs:718"]
    ANC["anchor_state<br/>solid-pod-rs/src/bitcoin_tx.rs:824"]
    VCH["build_withdraw_voucher<br/>solid-pod-rs/src/bitcoin_tx.rs:985"]
    TXO["parse_txo_voucher<br/>solid-pod-rs/src/bitcoin_tx.rs:511"]
    BC["MempoolBroadcast trait<br/>solid-pod-rs/src/bitcoin_tx.rs:580"]

    TAG --> P2TR --> BLD --> CO
    XO --> BLD
    BLD --> MINT
    BLD --> XFER
    BLD --> ANC
    BLD --> VCH
    BLD --> BC
    VS --> BLD
    FEE --> BLD
    TXO --> BC

    N["DIVERGENCE: this is a hand-rolled consensus-serialisation and BIP-341 sighash<br/>implementation rather than a maintained Bitcoin crate. The README states the<br/>write side is validated against the OFFICIAL BIP-340/341 test vectors, which is<br/>what makes it auditable — but it remains bespoke crypto plumbing in a security<br/>path, and the workspace's own unsafe_code = deny does not cover correctness."]
    BLD -.-> N
    N2["reverse_txid, write_var_int and the modular helpers are the raw wire layer<br/>solid-pod-rs/src/bitcoin_tx.rs:86"]
    BLD -.-> N2
```

## SP-07.11 Mempool endpoint selection (ADR-2007)

```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>solid-pod-rs-server/src/main.rs:284
    participant SEL as select_mempool_endpoint<br/>solid-pod-rs-server/src/mempool.rs:260
    participant INF as infer_network<br/>solid-pod-rs-server/src/mempool.rs:164
    participant LOG as log_mempool_selection_once<br/>solid-pod-rs-server/src/mempool.rs:325
    participant C as MempoolHttpClient<br/>solid-pod-rs-server/src/mempool.rs:336

    M->>SEL: the configured URL, or None
    SEL->>INF: derive the Bitcoin network from the URL
    INF-->>SEL: BitcoinNetwork<br/>solid-pod-rs-server/src/mempool.rs:109
    SEL-->>M: MempoolSelection with a MempoolConfigSource<br/>solid-pod-rs-server/src/mempool.rs:77
    M->>LOG: record base URL, inferred network and explicit-vs-defaulted, ONCE
    LOG-->>M: a warning when the network is unknown or the endpoint defaulted<br/>solid-pod-rs-server/src/mempool.rs:289
    C->>C: transaction_exists distinguishes 404 from an ambiguous failure<br/>solid-pod-rs-server/src/mempool.rs:406

    Note over SEL: DEFAULT_MEMPOOL_URL is the PUBLIC mempool.space testnet4 explorer<br/>(solid-pod-rs-server/src/mempool.rs:56), read from JSS_PAY_MEMPOOL_URL<br/>(solid-pod-rs-server/src/mempool.rs:52).
    Note over M: DIVERGENCE (ADR-2007): ONE URL, no fallback chain. Without the startup log an<br/>operator cannot tell from the logs which chain anchors are being written to.<br/>The LAN Bitcoin node cutover remains deployment config, not a crate default.
    Note over C: to_manifest_json (solid-pod-rs-server/src/mempool.rs:243) makes the selection<br/>machine-readable for a deployment manifest.
```

## SP-07.12 The block anchorer

```mermaid
sequenceDiagram
    autonumber
    participant P as ProvenanceLog
    participant A as MempoolBlockAnchorer::anchor<br/>solid-pod-rs-server/src/mempool.rs:647
    participant TS as trail_store<br/>solid-pod-rs-server/src/trail_store.rs:89
    participant M as mempool REST
    participant V as verify<br/>solid-pod-rs-server/src/mempool.rs:721

    P->>A: (ticker, commit_sha as the state hash, network)
    A->>TS: load_trail for the ticker<br/>solid-pod-rs-server/src/trail_store.rs:27
    TS-->>A: StoredTrail, private key included<br/>solid-pod-rs-server/src/trail_store.rs:37
    A->>M: build, sign and broadcast the anchoring tx
    M-->>A: txid
    A->>TS: save_trail with the new state<br/>solid-pod-rs-server/src/trail_store.rs:105
    A-->>P: BlockTrailAnchor
    P->>V: later, confirm the anchor's depth
    V-->>P: confirmed or not

    Note over TS: to_public (solid-pod-rs-server/src/trail_store.rs:58) strips the private key<br/>before a trail is served — merge_public (:76) folds a public update back in<br/>WITHOUT losing the secret. A test pins that the key never leaks<br/>(solid-pod-rs-server/src/trail_store.rs:167).
```

## SP-07.13 The web ledger

```mermaid
classDiagram
    class WebLedger {
        +get_balance(did)  solid-pod-rs/src/payments.rs:136
        +credit(did, amount)  solid-pod-rs/src/payments.rs:144
        +debit(did, amount)  solid-pod-rs/src/payments.rs:158
    }
    class LedgerEntry {
        solid-pod-rs/src/payments.rs:35
        +LedgerAmount  solid-pod-rs/src/payments.rs:47
        +chain_balance(chain)  solid-pod-rs/src/payments.rs:83
    }
    class PayConfig {
        solid-pod-rs/src/payments.rs:184
        +ChainConfig bitcoin_mainnet  solid-pod-rs/src/payments.rs:223
        +ChainConfig bitcoin_testnet4  solid-pod-rs/src/payments.rs:241
        +ChainConfig bitcoin_signet  solid-pod-rs/src/payments.rs:250
    }
    class PaymentStore {
        <<trait>>
        solid-pod-rs/src/payments.rs:438
    }
    class Identity {
        +pubkey_to_did  solid-pod-rs/src/payments.rs:450
        +did_to_pubkey  solid-pod-rs/src/payments.rs:455
    }
    WebLedger *-- LedgerEntry
    WebLedger ..> PaymentStore
    PayConfig ..> WebLedger
    WebLedger ..> Identity
    note for WebLedger "debit fails closed on an insufficient or missing balance\n(solid-pod-rs/src/payments.rs:158) — that is what makes the WAC PaymentCondition\ngate in SP-04.13 safe to charge against.\npayment_required_body (solid-pod-rs/src/payments.rs:265) is the HTTP-402 body;\npay_info (:278) backs GET /pay/.info."
```

## SP-07.14 The `/pay/*` route family

```mermaid
flowchart LR
    REG["handlers::pay::register<br/>solid-pod-rs-server/src/handlers/pay.rs:1645"]
    BAL["GET /pay/.balance -> handle_balance<br/>solid-pod-rs-server/src/handlers/pay.rs:363"]
    DEP["POST /pay/.deposit -> handle_deposit<br/>solid-pod-rs-server/src/handlers/pay.rs:436"]
    ADDR["GET /pay/.address -> handle_address<br/>solid-pod-rs-server/src/handlers/pay.rs:682"]
    OFF["GET /pay/.offers -> handle_offers<br/>solid-pod-rs-server/src/handlers/pay.rs:774"]
    SELL["POST /pay/.sell -> handle_sell<br/>solid-pod-rs-server/src/handlers/pay.rs:809"]
    SWAP["POST /pay/.swap -> handle_swap<br/>solid-pod-rs-server/src/handlers/pay.rs:864"]
    POOLG["GET /pay/.pool -> handle_pool_get<br/>solid-pod-rs-server/src/handlers/pay.rs:927"]
    POOLP["POST /pay/.pool -> handle_pool_post<br/>solid-pod-rs-server/src/handlers/pay.rs:982"]
    BUY["POST /pay/.buy -> handle_buy<br/>solid-pod-rs-server/src/handlers/pay.rs:1295"]
    WD["POST /pay/.withdraw -> handle_withdraw<br/>solid-pod-rs-server/src/handlers/pay.rs:1384"]
    WDS["POST /pay/.withdraw-sats -> handle_withdraw_sats<br/>solid-pod-rs-server/src/handlers/pay.rs:1487"]

    REG --> BAL
    REG --> DEP
    REG --> ADDR
    REG --> OFF
    REG --> SELL
    REG --> SWAP
    REG --> POOLG
    REG --> POOLP
    REG --> BUY
    REG --> WD
    REG --> WDS

    N["Every write route resolves the caller's did:nostr first — require_did and<br/>require_did_with_body (solid-pod-rs-server/src/handlers/pay.rs:326), so a pay<br/>action is always attributable to a signing key.<br/>is_valid_did_nostr (:752) is the syntax gate."]
    REG -.-> N
    N2["DIVERGENCE: the whole /pay/* surface is registered with the SAME gating as<br/>/pay/.info — always on, with no payments feature flag<br/>(solid-pod-rs-server/src/lib.rs:4643). The README's status section says not to<br/>carry value through these routes until the audit findings are fixed."]
    REG -.-> N2
```

## SP-07.15 Payment-state atomicity

```mermaid
sequenceDiagram
    autonumber
    participant H as a /pay/* handler
    participant L as PAYMENT_STATE_LOCK<br/>solid-pod-rs-server/src/lib.rs:200
    participant PS as StoragePaymentStore<br/>solid-pod-rs-server/src/handlers/pay.rs:128
    participant S as Storage
    participant RC as recover_payment_intents<br/>solid-pod-rs-server/src/handlers/pay.rs:231

    H->>L: acquire the process-wide mutex
    Note over L: The storage API has NO cross-resource CAS primitive, so the guard must be<br/>held across the replay check, the ledger/order/pool/trail mutation and the<br/>persist.
    H->>PS: check_replay(key)<br/>solid-pod-rs-server/src/handlers/pay.rs:279
    H->>PS: read_state<br/>solid-pod-rs-server/src/handlers/pay.rs:168
    H->>H: mutate the ledger, order book or pool
    H->>PS: commit_state<br/>solid-pod-rs-server/src/handlers/pay.rs:187
    PS->>S: persist
    H->>PS: record_replay(key)<br/>solid-pod-rs-server/src/handlers/pay.rs:283
    H->>L: release

    Note over RC: A durable intent record gives crash recovery where an EXTERNAL broadcast is<br/>involved: raw_transaction_txid (solid-pod-rs-server/src/handlers/pay.rs:218)<br/>re-derives the txid so a re-run can tell "already broadcast" from "never sent".
    Note over L: DIVERGENCE: the lock is PROCESS-wide. Two replicas serving the same pod<br/>storage share no lock, and the README's audit line records "non-atomic payment<br/>state" as a reproduced critical finding.
```

## SP-07.16 Order book and AMM

```mermaid
classDiagram
    class OrderBook {
        +create_order  solid-pod-rs/src/trading.rs:180
        +list_offers  solid-pod-rs/src/trading.rs:206
        +cancel_order  solid-pod-rs/src/trading.rs:218
        +execute_swap  solid-pod-rs/src/trading.rs:240
    }
    class SellOrder {
        solid-pod-rs/src/trading.rs:145
    }
    class AmmPool {
        +new(a, b, fee_bps)  solid-pod-rs/src/trading.rs:342
        +add_liquidity  solid-pod-rs/src/trading.rs:359
        +remove_liquidity  solid-pod-rs/src/trading.rs:408
        +swap — constant product  solid-pod-rs/src/trading.rs:459
        +pool_info  solid-pod-rs/src/trading.rs:531
    }
    class Exchange {
        +get_or_create_pool  solid-pod-rs/src/trading.rs:568
        +get_pool  solid-pod-rs/src/trading.rs:581
    }
    class MultiCurrency {
        +get_currency_balance  solid-pod-rs/src/trading.rs:34
        +credit_currency  solid-pod-rs/src/trading.rs:43
        +debit_currency  solid-pod-rs/src/trading.rs:87
    }
    Exchange *-- AmmPool
    OrderBook *-- SellOrder
    OrderBook ..> MultiCurrency
    AmmPool ..> MultiCurrency
    note for AmmPool "pool_key (solid-pod-rs/src/trading.rs:598) normalises the currency pair so\n(A,B) and (B,A) resolve to one pool; isqrt_u128 (:607) is the integer square\nroot the LP-share maths needs — everything is integer arithmetic, no floats\nin a value path."
```

## SP-07.17 The unverified deposit stand-in

```mermaid
flowchart TD
    IN["POST /pay/.deposit -> handle_deposit<br/>solid-pod-rs-server/src/handlers/pay.rs:436"]
    SNIFF{"body_is_mrc20?<br/>solid-pod-rs-server/src/handlers/pay.rs:529"}
    MRC["handle_mrc20_deposit — the VERIFIED path<br/>solid-pod-rs-server/src/handlers/pay.rs:563"]
    FLAG{"deposit_txo_standin_enabled?<br/>solid-pod-rs-server/src/lib.rs:390"}
    R501["501 Not Implemented"]
    STAND["credit (vout + 1) * 1000 sats with NO chain check<br/>solid-pod-rs-server/src/main.rs:124"]

    IN --> SNIFF
    SNIFF -- yes --> MRC
    SNIFF -- no --> FLAG
    FLAG -- off, the DEFAULT --> R501
    FLAG -- explicitly on --> STAND

    N["INVARIANT: the stand-in is OFF by default and the binary logs a loud warning<br/>when it is switched on (solid-pod-rs-server/src/main.rs:307). It is a free-money<br/>oracle until backed by a live UTXO existence, value and ownership check."]
    STAND -.-> N
    N2["The genuinely verified MRC20 deposit path is unaffected by the flag — the two<br/>branches are separated by a body sniff, not by a shared code path."]
    MRC -.-> N2
    N3["pod_issuer_pubkey (solid-pod-rs-server/src/handlers/pay.rs:543) identifies the<br/>pod's own issuing key on the verified branch."]
    MRC -.-> N3
```
