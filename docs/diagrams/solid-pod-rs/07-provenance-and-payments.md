---
id: SP-07
title: Provenance — git-marks, block-trails, Bitcoin anchoring — and the web ledger
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2004, ADR-2007, ADR-2008]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs/src/provenance.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/mrc20.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/blocktrail.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/bitcoin_tx.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/payments.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/trading.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/anchor.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/handlers/prov.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/handlers/pay.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/mempool.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/trail_store.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/main.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/wac/conditions.rs
  - ../solid-pod-rs/crates/solid-pod-rs-git/src/api.rs
  - ../solid-pod-rs/crates/solid-pod-rs/docs/adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md
verified_commit: d64131b
---

## SP-07.1 The two provenance tiers

```mermaid
flowchart LR
    W["an LDP write that already succeeded"]
    GM["git-mark — cheap, ALWAYS attempted<br/>solid-pod-rs/src/provenance.rs:473"]
    AN["block-trail anchor — expensive, opt-in<br/>solid-pod-rs/src/provenance.rs:482"]
    SC["PROV-O sidecar at <resource>.prov.ttl<br/>solid-pod-rs/src/provenance.rs:1112"]
    MK["ProvenanceMark<br/>solid-pod-rs/src/provenance.rs:47"]

    W --> GM --> MK
    GM --> AN --> MK
    MK --> SC

    N["INVARIANT: the anchored state_hash IS the git commit SHA, so the Bitcoin UTXO<br/>commits to the git history — the two primitives form one chain rather than two<br/>parallel records.<br/>solid-pod-rs/src/provenance.rs:484"]
    AN -.-> N
    N2["GitMark (solid-pod-rs/src/provenance.rs:64) and BlockTrailAnchor<br/>(solid-pod-rs/src/provenance.rs:166) are the two tier records;<br/>GitMarkEnvelope (:94) and BlocktrailEnvelope (:209, since alpha.12 an alias of blocktrail::Blocktrail<br/>re-exported at :199) are their wire forms."]
    MK -.-> N2
```

## SP-07.2 git_mark_write — the skip ladder

```mermaid
flowchart TD
    IN["git_mark_write(state, resource_path, agent, message)<br/>solid-pod-rs-server/src/lib.rs:3444"]
    SIDE{"path ends .acl, .meta or .prov.ttl?<br/>solid-pod-rs-server/src/lib.rs:3455"}
    S1["Skipped: ExcludedPath<br/>solid-pod-rs/src/provenance.rs:635"]
    CONT{"path ends with a slash?<br/>solid-pod-rs-server/src/lib.rs:3462"}
    S2["Skipped: Container<br/>solid-pod-rs/src/provenance.rs:637"]
    ROOT{"data_root configured?<br/>solid-pod-rs-server/src/lib.rs:3467"}
    S3["Skipped: NotConfigured<br/>solid-pod-rs/src/provenance.rs:629"]
    SPLIT{"path splits into pod plus rest?<br/>solid-pod-rs-server/src/lib.rs:3473"}
    S4["Skipped: UnresolvablePath<br/>solid-pod-rs/src/provenance.rs:639"]
    GITD{"data_root/{pod}/.git is a directory?<br/>solid-pod-rs-server/src/lib.rs:3484"}
    S5["Skipped: NotGitBacked<br/>solid-pod-rs/src/provenance.rs:632"]
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
    N2["A skip is not a fault. ProvenanceSkip exists precisely so a correctly<br/>unmarked write is never reported as a provenance failure.<br/>solid-pod-rs/src/provenance.rs:624"]
    S5 -.-> N2
    N3["DIVERGENCE (ADR-2004): with the server's empty default feature set this whole<br/>function is the no-op shim (solid-pod-rs-server/src/lib.rs:3650), so a default<br/>build records ZERO marks. Every provenance claim carries a --features git caveat."]
    IN -.-> N3
```

## SP-07.3 Anchor policy resolution

```mermaid
flowchart TD
    ACL["the resource's effective ACL"]
    COND["acl:ProvenanceAnchor condition<br/>solid-pod-rs/src/wac/conditions.rs:61"]
    MODE["anchor_mode_of -> AnchorMode<br/>solid-pod-rs/src/wac/anchor.rs:119"]
    RES["resolve_anchor_policy<br/>solid-pod-rs-server/src/handlers/prov.rs:83"]
    NEVER["AnchorPolicy::Never — the default for an ordinary write<br/>solid-pod-rs/src/provenance.rs:323"]
    ALWAYS["AnchorPolicy::Always<br/>solid-pod-rs/src/provenance.rs:325"]
    HV["AnchorPolicy::HighValue<br/>solid-pod-rs/src/provenance.rs:328"]
    EP["AnchorPolicy::Epoch<br/>solid-pod-rs/src/provenance.rs:331"]
    INLINE["anchors_inline(high_value)<br/>solid-pod-rs/src/provenance.rs:343"]
    BLD["build_anchorer — only when the policy is not Never<br/>solid-pod-rs-server/src/handlers/prov.rs:124<br/>called from solid-pod-rs-server/src/lib.rs:3507"]
    BMR["an MRC20 trail already issued under the name (needs the pay-token):<br/>MempoolBlockAnchorer<br/>solid-pod-rs-server/src/handlers/prov.rs:143"]
    BGM["otherwise a started git-mark trail: GitmarkAnchorer, else None<br/>solid-pod-rs-server/src/handlers/prov.rs:153"]

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
    BLD --> BMR
    BLD --> BGM

    N["Never and Epoch never anchor INLINE — Epoch defers to the accumulator.<br/>Always is unconditional; HighValue is inline only when the resource is flagged.<br/>solid-pod-rs/src/provenance.rs:345"]
    INLINE -.-> N
    N2["The handler rewrites Epoch to Never before calling record, then batches the SHA<br/>itself, so record's inline path only ever sees Never, Always or HighValue.<br/>solid-pod-rs-server/src/lib.rs:3527"]
    EP -.-> N2
    N3["The trail is named by the request ticker, else the pay-token ticker, else the default<br/>name prov (solid-pod-rs-server/src/trail_store.rs:128). Since alpha.12 a pod with<br/>no pay-token can anchor, provided a git-mark trail of that name has been started."]
    BLD -.-> N3
    N4["DRIFT: the _prov/anchor handler still answers 'no pay-token / trail not minted'<br/>and comments that the pod 'must have a minted trail'<br/>(solid-pod-rs-server/src/handlers/prov.rs:383), but build_anchorer no longer<br/>requires a pay-token (:128). The message misdirects an operator whose pod has<br/>neither trail kind started."]
    BGM -.-> N4
    N5["OPEN: nothing outside the tests calls GitmarkAnchorer::genesis<br/>(solid-pod-rs-server/src/mempool.rs:943), and no route, CLI verb or boot step<br/>writes a git-mark trail file (solid-pod-rs-server/src/trail_store.rs:191). A new pod<br/>therefore has no way to reach the git-mark branch above from this repository."]
    BGM -.-> N5
```

## SP-07.4 record_receipt — typed, per-tier outcomes (ADR-2004)

```mermaid
stateDiagram-v2
    [*] --> Attempt: record_receipt(WriteRecord)<br/>solid-pod-rs/src/provenance.rs:516
    Attempt --> MarkFailed: marker.mark_write returned an error<br/>solid-pod-rs/src/provenance.rs:526
    Attempt --> Marked: a commit SHA exists

    MarkFailed --> ResourceStored: stage resource-stored plus mark_error<br/>solid-pod-rs/src/provenance.rs:589
    Marked --> NoAnchor: policy did not want one
    Marked --> AnchorTried: policy wanted one

    NoAnchor --> LocalMarkCommitted: stage local-mark-committed<br/>solid-pod-rs/src/provenance.rs:592
    AnchorTried --> AnchorFailed: anchor_error recorded, the MARK STANDS
    AnchorTried --> Submitted: stage anchor-submitted<br/>solid-pod-rs/src/provenance.rs:595
    Submitted --> Confirmed: stage anchor-confirmed<br/>solid-pod-rs/src/provenance.rs:598
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
      (solid-pod-rs/src/provenance.rs:514).
    end note
    note right of AnchorFailed
      A failed anchor never suppresses a successful git-mark
      (solid-pod-rs/src/provenance.rs:512), and the server logs it separately
      (solid-pod-rs-server/src/lib.rs:3568).
    end note
```

## SP-07.5 The receipt on the wire

```mermaid
sequenceDiagram
    autonumber
    participant H as LDP write handler
    participant G as git_mark_write<br/>solid-pod-rs-server/src/lib.rs:3444
    participant R as ProvenanceReceipt<br/>solid-pod-rs/src/provenance.rs:673
    participant S as Storage
    participant C as Client

    H->>G: mark the write
    G->>R: record_receipt then restore the full pod-relative path<br/>solid-pod-rs-server/src/lib.rs:3553
    opt policy is Epoch
        G->>G: epoch_push_and_maybe_anchor — one tx notarises many commits<br/>solid-pod-rs-server/src/handlers/prov.rs:188
    end
    G->>S: write <resource>.prov.ttl<br/>solid-pod-rs-server/src/lib.rs:3618
    alt sidecar write failed
        G->>R: mark_error set, stage stays local-mark-committed<br/>solid-pod-rs-server/src/lib.rs:3632
    end
    G-->>H: receipt
    H->>C: X-Provenance = receipt.summary()<br/>solid-pod-rs/src/provenance.rs:753
    H->>C: X-Provenance-Commit = receipt.commit_sha()<br/>solid-pod-rs/src/provenance.rs:743
    Note over G: The sidecar write also fires the FS-watch StorageEvent the Updates-via<br/>notification stream relays, so subscribers see the new mark. It ends in<br/>.prov.ttl, so the skip guard in SP-07.2 stops the recursion.
    Note over H: header_safe (solid-pod-rs/src/provenance.rs:774) sanitises the summary before<br/>it becomes a header value.
```

## SP-07.6 The `_prov` query surface

```mermaid
flowchart LR
    R1["GET /{pod}/_prov/{commit_sha} -> handle_resolve<br/>solid-pod-rs-server/src/handlers/prov.rs:234"]
    R2["POST /{pod}/_prov/anchor -> handle_anchor<br/>solid-pod-rs-server/src/handlers/prov.rs:327"]
    REG["handlers::prov::register<br/>solid-pod-rs-server/src/handlers/prov.rs:540"]
    RC["resolve_commit against the pod repo<br/>solid-pod-rs-git/src/api.rs:374"]
    PRICE["anchor_price_sats<br/>solid-pod-rs-server/src/handlers/prov.rs:471"]
    DEB["debit then refund on failure<br/>solid-pod-rs-server/src/handlers/prov.rs:489"]
    SIDE["the .prov.ttl sidecar GET is an ORDINARY LDP read"]

    REG --> R1 --> RC
    REG --> R2 --> PRICE --> DEB
    R1 -.-> SIDE

    N["DIVERGENCE (baseline open item REC-11): the provenance query surface is<br/>POINT-LOOKUP only — one commit SHA, or a per-resource sidecar. There is no<br/>pod-wide _prov enumeration, so 'one queryable trace over a whole pod' is not<br/>delivered by this repo."]
    R1 -.-> N
    N2["The explicit git-mark to Bitcoin-anchor upgrade is payment-gated and refunds on<br/>failure — refund (solid-pod-rs-server/src/handlers/prov.rs:519) is the<br/>compensating half of debit."]
    DEB -.-> N2
    N3["is_sidecar (solid-pod-rs-server/src/handlers/prov.rs:465) keeps the anchor<br/>endpoint off control-plane paths."]
    R2 -.-> N3
```

## SP-07.7 Epoch batching — one transaction, many commits

```mermaid
sequenceDiagram
    autonumber
    participant G as git_mark_write
    participant L as load_epoch_commits<br/>solid-pod-rs-server/src/handlers/prov.rs:159
    participant A as EpochAccumulator<br/>solid-pod-rs/src/provenance.rs:884
    participant M as merkle_root<br/>solid-pod-rs/src/provenance.rs:825
    participant AN as BlockAnchorer
    participant S as save_epoch_commits<br/>solid-pod-rs-server/src/handlers/prov.rs:167

    G->>L: read the pod's current epoch commit list
    G->>A: push the fresh commit SHA<br/>solid-pod-rs/src/provenance.rs:914
    alt the epoch is below JSS_PROV_EPOCH_SIZE
        A-->>S: persist and stop<br/>solid-pod-rs-server/src/handlers/prov.rs:50
    else the epoch fills
        A->>M: merkle_leaf per commit, then the root<br/>solid-pod-rs/src/provenance.rs:853
        M->>AN: anchor the ROOT once
        AN-->>A: ClosedEpoch<br/>solid-pod-rs/src/provenance.rs:895
        A->>S: reset the accumulator
    end
    Note over AN: A failed batch anchor is swallowed — it never fails the write nor the<br/>git-mark (solid-pod-rs-server/src/lib.rs:3626).
    Note over M: MerkleProof (solid-pod-rs/src/provenance.rs:861) lets one commit prove its<br/>membership in the anchored root without revealing the rest of the epoch.<br/>merkle_root is deterministic and order-sensitive<br/>(solid-pod-rs/src/provenance.rs:1745).
```

## SP-07.9 MRC20 state chain and Bitcoin anchoring

```mermaid
flowchart TD
    JCS["jcs — RFC 8785 canonical JSON<br/>solid-pod-rs/src/mrc20.rs:44"]
    SHA["sha256_hex over the canonical form<br/>solid-pod-rs/src/mrc20.rs:78"]
    ST["Mrc20State<br/>solid-pod-rs/src/mrc20.rs:116"]
    VAL["validate_mrc20_state<br/>solid-pod-rs/src/mrc20.rs:200"]
    LNK["verify_state_link — prev-state hash chaining<br/>solid-pod-rs/src/mrc20.rs:217"]
    TR["Mrc20Trail<br/>solid-pod-rs/src/mrc20.rs:163"]
    DEP["verify_mrc20_deposit<br/>solid-pod-rs/src/mrc20.rs:262"]
    KEY["bt_derive_chained_pubkey / privkey<br/>solid-pod-rs/src/mrc20.rs:561"]
    ADDR["bt_address — bech32m taproot<br/>solid-pod-rs/src/mrc20.rs:766"]
    VER["verify_mrc20_anchor — state chain, a head UTXO read via MempoolLookup,<br/>then a walk of every earlier mark<br/>solid-pod-rs/src/mrc20.rs:1004"]
    WALK["verify_anchor_chain — back from the head outpoint to genesis<br/>solid-pod-rs/src/blocktrail.rs:1068"]
    ML["MempoolLookup trait<br/>solid-pod-rs/src/mrc20.rs:372"]

    JCS --> SHA --> ST --> VAL --> LNK --> TR
    TR --> DEP
    KEY --> ADDR --> VER --> ML
    VER --> WALK

    N["INVARIANT: the state hash is taken over the RFC 8785 canonical serialisation,<br/>not over the raw bytes — two semantically identical JSON documents must hash<br/>identically or the chain breaks on a formatting change."]
    JCS -.-> N
    N2["MRC20_PROFILE mono.mrc20.v0.1 and TRANSFER_OP name the wire contract<br/>solid-pod-rs/src/mrc20.rs:34 and :36"]
    ST -.-> N2
    N3["extract_transfers_to and total_transferred_to are the read side<br/>solid-pod-rs/src/mrc20.rs:240"]
    DEP -.-> N3
    N4["CHANGED in alpha.12: a UTXO at the head address no longer suffices. Each earlier mark must<br/>be the key its prefix of states derives and spend the mark before it, else the call<br/>fails with 'does not verify link by link'. A head still in the mempool is still<br/>accepted (intact, not verified). Per-link detail: SP-07.19.<br/>solid-pod-rs/src/mrc20.rs:1004"]
    VER -.-> N4
```

## SP-07.10 The rust-bitcoin transaction builder (ADR-2008 D1, landed)

```mermaid
flowchart LR
    BLD["build_transaction<br/>solid-pod-rs/src/bitcoin_tx.rs:281"]
    P2TR["p2tr_script<br/>solid-pod-rs/src/bitcoin_tx.rs:124"]
    VS["verify_keypath_signature<br/>solid-pod-rs/src/bitcoin_tx.rs:388"]
    FEE["DEFAULT_FEE_SATS 300, DUST_LIMIT_SATS 546<br/>solid-pod-rs/src/bitcoin_tx.rs:80"]
    CO["checked_output — refuses to mint value<br/>solid-pod-rs/src/bitcoin_tx.rs:1081"]
    MINT["mint_token<br/>solid-pod-rs/src/bitcoin_tx.rs:539"]
    XFER["transfer_token_with_key<br/>solid-pod-rs/src/bitcoin_tx.rs:639"]
    ANC["anchor_state<br/>solid-pod-rs/src/bitcoin_tx.rs:745"]
    VCH["build_withdraw_voucher<br/>solid-pod-rs/src/bitcoin_tx.rs:1137"]
    TXO["parse_txo_voucher<br/>solid-pod-rs/src/bitcoin_tx.rs:433"]
    BC["MempoolBroadcast trait<br/>solid-pod-rs/src/bitcoin_tx.rs:502"]
    GG["gitmark_genesis — voucher to the first mark, state is the commit as text<br/>solid-pod-rs/src/bitcoin_tx.rs:869"]
    GA["gitmark_advance — spend the newest mark to the next<br/>solid-pod-rs/src/bitcoin_tx.rs:929"]
    DTI["decode_tx_info — raw hex to the TxInfo view, inputs included<br/>solid-pod-rs/src/bitcoin_tx.rs:1028"]

    P2TR --> BLD --> CO
    BLD --> MINT
    BLD --> XFER
    BLD --> ANC
    BLD --> VCH
    BLD --> BC
    BLD --> GG
    BLD --> GA
    DTI -.-> BC
    VS --> BLD
    FEE --> BLD
    TXO --> BC

    N["The former DIVERGENCE is closed (ADR-2008 D1, landed at 0befa5b): serialisation,<br/>CompactSize, txid order, the P2TR script, the BIP-341 TapSighash and TapTweak,<br/>BIP-340 signing and bech32m now come from rust-bitcoin and libsecp256k1<br/>(solid-pod-rs/src/bitcoin_tx.rs:18). tagged_hash, xonly_of, add_mod_n,<br/>neg_mod_n and the hand-rolled bech32m encoder are DELETED. The module still<br/>decides only WHAT to build; determinism is kept with aux_rand = 0^32 so the<br/>cross-implementation golden fixture stays byte-identical to JSS."]
    BLD -.-> N
    N3["alpha.12 adds the git-mark composers beside anchor_state. A git-mark state is the commit<br/>hash as text with no MRC20 wrapper, 40 or 64 lowercase hex (check_commit,<br/>solid-pod-rs/src/bitcoin_tx.rs:844). gitmark_advance refuses to extend a trail<br/>whose newest mark on-chain is not the key its states derive. BuiltTx now carries the<br/>txid (solid-pod-rs/src/bitcoin_tx.rs:174)."]
    GA -.-> N3
    N2["DEBT: the write side remains a native-only concern — the module is gated<br/>cfg(not(target_arch = wasm32)) behind the mrc20 feature<br/>(solid-pod-rs/src/bitcoin_tx.rs:53), so wasm consumers cannot build transactions."]
    BLD -.-> N2
```

## SP-07.11 Mempool endpoint selection (ADR-2007)

```mermaid
sequenceDiagram
    autonumber
    participant M as main<br/>solid-pod-rs-server/src/main.rs:210
    participant SEL as select_mempool_endpoint<br/>solid-pod-rs-server/src/mempool.rs:264
    participant INF as infer_network<br/>solid-pod-rs-server/src/mempool.rs:168
    participant LOG as log_mempool_selection_once<br/>solid-pod-rs-server/src/mempool.rs:329
    participant C as MempoolHttpClient<br/>solid-pod-rs-server/src/mempool.rs:340

    M->>SEL: the configured URL, or None
    SEL->>INF: derive the Bitcoin network from the URL
    INF-->>SEL: BitcoinNetwork<br/>solid-pod-rs-server/src/mempool.rs:113
    SEL-->>M: MempoolSelection with a MempoolConfigSource<br/>solid-pod-rs-server/src/mempool.rs:81
    M->>LOG: record base URL, inferred network and explicit-vs-defaulted, ONCE
    LOG-->>M: a warning when the network is unknown or the endpoint defaulted<br/>solid-pod-rs-server/src/mempool.rs:293
    C->>C: transaction_exists distinguishes 404 from an ambiguous failure<br/>solid-pod-rs-server/src/mempool.rs:410

    Note over SEL: DEFAULT_MEMPOOL_URL is the PUBLIC mempool.space testnet4 explorer<br/>(solid-pod-rs-server/src/mempool.rs:60), read from JSS_PAY_MEMPOOL_URL<br/>(solid-pod-rs-server/src/mempool.rs:56).
    Note over M: DIVERGENCE (ADR-2007): ONE URL, no fallback chain. Without the startup log an<br/>operator cannot tell from the logs which chain anchors are being written to.<br/>The LAN Bitcoin node cutover remains deployment config, not a crate default.
    Note over C: to_manifest_json (solid-pod-rs-server/src/mempool.rs:247) makes the selection<br/>machine-readable for a deployment manifest.
```

## SP-07.12 The block anchorer

```mermaid
sequenceDiagram
    autonumber
    participant P as ProvenanceLog
    participant A as MempoolBlockAnchorer::anchor (MRC20 trails already issued)<br/>solid-pod-rs-server/src/mempool.rs:676
    participant TS as trail_store<br/>solid-pod-rs-server/src/trail_store.rs:90
    participant M as mempool REST
    participant V as verify<br/>solid-pod-rs-server/src/mempool.rs:749

    P->>A: (ticker, commit_sha as the state hash, network)
    A->>TS: load_trail for the ticker<br/>solid-pod-rs-server/src/trail_store.rs:28
    TS-->>A: StoredTrail, private key included<br/>solid-pod-rs-server/src/trail_store.rs:38
    A->>M: build, sign and broadcast the anchoring tx
    M-->>A: txid
    A->>TS: save_trail with the new state<br/>solid-pod-rs-server/src/trail_store.rs:106
    A-->>P: BlockTrailAnchor
    P->>V: later, verify the anchor
    V-->>P: true when every mark back to genesis is intact, via verify_anchor_report<br/>solid-pod-rs-server/src/mempool.rs:785

    Note over TS: to_public (solid-pod-rs-server/src/trail_store.rs:59) strips the private key<br/>before a trail is served — merge_public (:77) folds a public update back in<br/>WITHOUT losing the secret. A test pins that the key never leaks<br/>(solid-pod-rs-server/src/trail_store.rs:263).
```
- Changed in alpha.12: the former head-only `verify` (one UTXO at the derived address) is gone: both anchorers now walk the trail from the anchor's outpoint through `verify_anchor_report` (`solid-pod-rs-server/src/mempool.rs:785`), and an anchor stays verifiable after a later anchor spends it. New trails use `GitmarkAnchorer`, SP-07.20.
- **Debt:** `load_trail` maps every storage read error to `Ok(None)` (`solid-pod-rs-server/src/trail_store.rs:94-100`), so a transient or permissions failure reads identically to "no trail exists yet" — a durability fault and a genuinely fresh ticker are indistinguishable to the anchorer. `load_gitmark_trail` copies the fault (`solid-pod-rs-server/src/trail_store.rs:186`), so a transient storage fault makes `build_anchorer` return `None`, which the anchor handler reports as 'not configured'.

## SP-07.13 The web ledger

```mermaid
classDiagram
    direction TB
    class WebLedger {
        +get_balance(did)  solid-pod-rs/src/payments.rs:412
        +credit_by_outpoint(txid, vout, ...)  solid-pod-rs/src/payments.rs:450
        +debit_by_payout(amount, txid)  solid-pod-rs/src/payments.rs:501
        +reverse_payout(txid)  solid-pod-rs/src/payments.rs:536
        +check_genesis()  solid-pod-rs/src/payments.rs:372
    }
    class LedgerEntry {
        solid-pod-rs/src/payments.rs:35
        +LedgerAmount  solid-pod-rs/src/payments.rs:47
        +chain_balance(chain)  solid-pod-rs/src/payments.rs:83
    }
    class LedgerGenesis {
        solid-pod-rs/src/payments.rs:121
        operator, name, currency, created, confirmations
    }
    class PayConfig {
        solid-pod-rs/src/payments.rs:638
        +ChainConfig  solid-pod-rs/src/payments.rs:688
    }
    class PaymentStore {
        <<trait>>
        solid-pod-rs/src/payments.rs:911
    }
    class Identity {
        +pubkey_to_did  solid-pod-rs/src/payments.rs:923
        +did_to_pubkey  solid-pod-rs/src/payments.rs:928
    }
    WebLedger *-- LedgerEntry
    WebLedger ..> LedgerGenesis
    WebLedger ..> PaymentStore
    PayConfig ..> WebLedger
    WebLedger ..> Identity
    note for WebLedger "TELLER-shaped — see the paragraphs below"
```
- **What it shows:** the ledger is TELLER-shaped (solidpayorg/teller parity): a fresh ledger is born with a genesis whose identity is `sha256(JCS(genesis))`, and `check_genesis` (`solid-pod-rs/src/payments.rs:372`) refuses a document whose hash is not the hash of its genesis. The only credit is `credit_by_outpoint` — the deposit's outpoint is the receipt, and a second credit for a recorded outpoint is a no-op (`ReceiptOutcome::AlreadyApplied`, `solid-pod-rs/src/payments.rs:450`). Debits are recorded as payout receipts; `reverse_payout` (`:536`) undoes a payout whose transaction never reached the chain. ADR-2008 D4/D5 (chain-derived view, credit/debit removal) are NOT built — this is still a stored number. `payment_required_body` (`solid-pod-rs/src/payments.rs:738`) is the HTTP-402 body; `pay_info` (`:751`) backs GET /pay/.info.

## SP-07.14 The `/pay/*` route family

```mermaid
flowchart LR
    REG["handlers::pay::register<br/>solid-pod-rs-server/src/handlers/pay.rs:1717"]
    BAL["GET /pay/.balance -> handle_balance<br/>solid-pod-rs-server/src/handlers/pay.rs:392"]
    DEP["POST /pay/.deposit -> handle_deposit<br/>solid-pod-rs-server/src/handlers/pay.rs:452"]
    ADDR["GET /pay/.address -> handle_address<br/>solid-pod-rs-server/src/handlers/pay.rs:733"]
    OFF["GET /pay/.offers -> handle_offers<br/>solid-pod-rs-server/src/handlers/pay.rs:825"]
    SELL["POST /pay/.sell -> handle_sell<br/>solid-pod-rs-server/src/handlers/pay.rs:860"]
    SWAP["POST /pay/.swap -> handle_swap<br/>solid-pod-rs-server/src/handlers/pay.rs:915"]
    POOLG["GET /pay/.pool -> handle_pool_get<br/>solid-pod-rs-server/src/handlers/pay.rs:978"]
    POOLP["POST /pay/.pool -> handle_pool_post<br/>solid-pod-rs-server/src/handlers/pay.rs:1033"]
    BUY["POST /pay/.buy -> handle_buy<br/>solid-pod-rs-server/src/handlers/pay.rs:1362"]
    WD["POST /pay/.withdraw -> handle_withdraw<br/>solid-pod-rs-server/src/handlers/pay.rs:1451"]
    WDS["POST /pay/.withdraw-sats -> handle_withdraw_sats<br/>solid-pod-rs-server/src/handlers/pay.rs:1554"]

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

    N["Every write route resolves the caller's did:nostr first — require_did and<br/>require_did_with_body (solid-pod-rs-server/src/handlers/pay.rs:349), so a pay<br/>action is always attributable to a signing key.<br/>is_valid_did_nostr (:803) is the syntax gate."]
    REG -.-> N
    N2["DIVERGENCE: the whole /pay/* surface is registered with the SAME gating as<br/>/pay/.info — always on, with no payments feature flag<br/>(solid-pod-rs-server/src/lib.rs:4625). The README's status section says not to<br/>carry value through these routes until the audit findings are fixed."]
    REG -.-> N2
```

## SP-07.15 Payment-state atomicity

```mermaid
sequenceDiagram
    autonumber
    participant H as a /pay/* handler
    participant L as PAYMENT_STATE_LOCK<br/>solid-pod-rs-server/src/lib.rs:200
    participant PS as StoragePaymentStore<br/>solid-pod-rs-server/src/handlers/pay.rs:129
    participant S as Storage
    participant RC as recover_payment_intents<br/>solid-pod-rs-server/src/handlers/pay.rs:246

    H->>L: acquire the process-wide mutex
    Note over L: The storage API has NO cross-resource CAS primitive, so the guard must be<br/>held across the replay check, the ledger/order/pool/trail mutation and the<br/>persist.
    H->>PS: check_replay(key)<br/>solid-pod-rs-server/src/handlers/pay.rs:302
    H->>PS: read_state<br/>solid-pod-rs-server/src/handlers/pay.rs:169
    H->>H: mutate the ledger, order book or pool
    H->>PS: commit_state<br/>solid-pod-rs-server/src/handlers/pay.rs:198
    PS->>S: persist
    H->>PS: record_replay(key)<br/>solid-pod-rs-server/src/handlers/pay.rs:306
    H->>L: release

    Note over RC: A durable intent record gives crash recovery where an EXTERNAL broadcast is<br/>involved, raw_transaction_txid (solid-pod-rs-server/src/handlers/pay.rs:229)<br/>re-derives the txid so a re-run can tell "already broadcast" from "never sent".<br/>A definite 404 reverses the recorded payout via reverse_payout, an intent with<br/>no payout receipt is RETAINED with a warning — the ledger cannot restore a<br/>debit it has no receipt for (solid-pod-rs-server/src/handlers/pay.rs:246).
    Note over L: DIVERGENCE: the lock is PROCESS-wide. Two replicas serving the same pod<br/>storage share no lock, and the README's audit line records "non-atomic payment<br/>state" as a reproduced critical finding.
```

## SP-07.16 Order book and AMM

```mermaid
classDiagram
    class OrderBook {
        +create_order  solid-pod-rs/src/trading.rs:183
        +list_offers  solid-pod-rs/src/trading.rs:209
        +cancel_order  solid-pod-rs/src/trading.rs:221
        +execute_swap  solid-pod-rs/src/trading.rs:243
    }
    class SellOrder {
        solid-pod-rs/src/trading.rs:148
    }
    class AmmPool {
        +new(a, b, fee_bps)  solid-pod-rs/src/trading.rs:345
        +add_liquidity  solid-pod-rs/src/trading.rs:362
        +remove_liquidity  solid-pod-rs/src/trading.rs:411
        +swap — constant product  solid-pod-rs/src/trading.rs:462
        +pool_info  solid-pod-rs/src/trading.rs:534
    }
    class Exchange {
        +get_or_create_pool  solid-pod-rs/src/trading.rs:571
        +get_pool  solid-pod-rs/src/trading.rs:584
    }
    class WebLedgerTrading {
        <<impl WebLedger>>
        +get_currency_balance  solid-pod-rs/src/trading.rs:37
        -credit_currency  solid-pod-rs/src/trading.rs:46
        -debit_currency  solid-pod-rs/src/trading.rs:90
    }
    Exchange *-- AmmPool
    OrderBook *-- SellOrder
    OrderBook ..> WebLedgerTrading
    AmmPool ..> WebLedgerTrading
    note for AmmPool "pool_key (solid-pod-rs/src/trading.rs:601) normalises the currency pair so\n(A,B) and (B,A) resolve to one pool; isqrt_u128 (:610) is the integer square\nroot the LP-share maths needs — everything is integer arithmetic, no floats\nin a value path. Per-currency balances are crate-private helpers on WebLedger:\noutside this crate a balance is raised only by credit_by_outpoint."
```

## SP-07.17 Deposits are MRC20-only — the stand-in is deleted (ADR-2008 D6)

```mermaid
flowchart TD
    IN["POST /pay/.deposit -> handle_deposit<br/>solid-pod-rs-server/src/handlers/pay.rs:452"]
    SNIFF{"body_is_mrc20?<br/>solid-pod-rs-server/src/handlers/pay.rs:480"}
    MRC["handle_mrc20_deposit — the only credit path<br/>solid-pod-rs-server/src/handlers/pay.rs:521"]
    R501["501 Not Implemented — unconditionally:<br/>no flag, no branch, no payment state written<br/>solid-pod-rs-server/src/handlers/pay.rs:467"]

    IN --> SNIFF
    SNIFF -- yes --> MRC
    SNIFF -- no --> R501

    N["INVARIANT: the Phase-0 stand-in that credited (vout + 1) * 1000 sats behind<br/>an operator flag was DELETED at e62d028, not left default-off (ADR-2008 D6):<br/>a pod must not carry a reachable free-money oracle. A non-MRC20 body gets 501<br/>and writes no payment state."]
    R501 -.-> N
    N2["ISSUER AND TICKER BINDING: the trail must be anchored on the pod's configured<br/>issuer key — pod_issuer_pubkey<br/>(solid-pod-rs-server/src/handlers/pay.rs:494), accepts_issuer check :546 —<br/>and carry the configured ticker. A self-issued trail with the right ticker is<br/>a different token: 403. Another ticker: 400."]
    MRC -.-> N2
    N3["CREDIT BY OUTPOINT RECEIPT: after verify_mrc20_anchor (which since alpha.12 walks every mark of the<br/>trail, not the head alone) and a replay guard on<br/>the anchor outpoint, credit_by_outpoint credits the caller's TICKER balance,<br/>never the satoshi balance<br/>(solid-pod-rs-server/src/handlers/pay.rs:652). One coin can never pay twice."]
    MRC -.-> N3
```

## SP-07.18 The ledger today, and the remaining chain-view work (ADR-2008, partial)

```mermaid
flowchart TB
    subgraph LIVE["LIVE at d64131b — a stored number this crate mutates"]
        GB["get_balance reads the stored map<br/>solid-pod-rs/src/payments.rs:412"]
        CR["credit_by_outpoint — the only credit, outpoint<br/>as receipt; verified MRC20 deposits only<br/>solid-pod-rs/src/payments.rs:450"]
        DB["debit_by_payout records a payout receipt;<br/>reverse_payout undoes an unconfirmed payout<br/>solid-pod-rs/src/payments.rs:501"]
        WL["write_ledger on PaymentStore is still the authority<br/>solid-pod-rs/src/payments.rs:913"]
        ST["Landed: rust-bitcoin port D1, golden tests D2,<br/>TXO stand-in deleted D6 — verified at e62d028<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:97"]
        ST -.-> CR
        CR --> WL
        DB --> WL
        GB --> WL
    end
    subgraph PROP["NOT BUILT — decision_status proposed, implementation_status partial,<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:5-6"]
        P1["D4: get_balance reads through sidestr-node and folds<br/>the UTXOs for that did:nostr, bounded cache<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:55"]
        P2["D5: credit and debit leave the public API — the only<br/>credit is a peg-in claim on the chain<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:60"]
        P3["write_ledger stops being an authority, writes<br/>become cache population<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:62"]
        P4["D7: non-atomic payment state fixed first<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:72"]
        P5["D8: both consumers adopt one post-port version<br/>in lockstep — an exit criterion, not a follow-up<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:76"]
        P1 --> P2 --> P3
        P4 --> P5
    end
    LIVE --> PROP

    N["INVARIANT (live): a staleness bound does not exist today because the balance is<br/>local — the proposal makes a stale figure an error rather than a slightly old<br/>number, which is a new failure mode callers do not handle yet.<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:57"]
    PROP -.-> N
    N2["EXTERNAL: ADR-2008 is this crate's projection of decisions taken elsewhere —<br/>agentbox ADR-2096 D3 and ADR-2099 D2/D3, and PRD-024 D0/D3/D4 — extending<br/>ADR-2007's single-explorer seam with the estate's own chain. MRC20 retires as a<br/>token rail but is retained as an anchoring primitive because block-trail anchors<br/>and the host's AnchorConfirmer depend on it.<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:15"]
    PROP -.-> N2
    N3["DIVERGENCE (D3 in part): the chained-key derivation ports and stays<br/>byte-identical, but the .buy / .withdraw MRC20 token routes are still live —<br/>their retirement belongs with D4/D5<br/>(adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:50,<br/>implementation note :144-145)."]
    LIVE -.-> N3
```
- **Tension (ADR-2008 vs code):** D3 declares MRC20 "retired as a token rail" (adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:50-52), yet `/pay/.buy` and `/pay/.withdraw` still execute token trades and withdrawals against the stored ledger (`solid-pod-rs-server/src/handlers/pay.rs:1362`, `:1451`) — the record itself concedes this is D3 "in part", pending D4/D5 (adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:98, :144-145).
- **Open:** the record makes fixing non-atomic payment state a precondition of the port (adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:72), but the lock it would have to replace is process-wide with no cross-resource CAS primitive behind it (`solid-pod-rs-server/src/lib.rs:200`) — nothing states what replaces it.
- **Invariant:** the PROPOSED half is drawn as not built; the record carries `decision_status: proposed` and `implementation_status: partial` (adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:5-6), and its Verification section states exactly what landed — "D1, D2 and D6 are built and verified at `e62d028`" while "D4, D5, D7 and D8 are not built" (adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:97-98).

## SP-07.19 Whole-trail verification — every mark walked back from the head (alpha.12)

```mermaid
sequenceDiagram
    autonumber
    participant V as verify<br/>solid-pod-rs-server/src/mempool.rs:749
    participant R as verify_anchor_report<br/>solid-pod-rs-server/src/mempool.rs:785
    participant W as verify_anchor_chain<br/>solid-pod-rs/src/blocktrail.rs:1068
    participant M as MempoolLookup
    participant T as TrailReport<br/>solid-pod-rs/src/blocktrail.rs:765

    V->>R: the anchor with its portable proof
    R-->>V: None when pubkey or state_strings are absent<br/>solid-pod-rs-server/src/mempool.rs:792
    R-->>V: None when the recorded address is not the derivation, a forged anchor<br/>solid-pod-rs-server/src/mempool.rs:803
    R->>W: pubkey, state_strings and the anchor's txid and vout<br/>solid-pod-rs-server/src/mempool.rs:809
    loop mark k-1 down to 0
        W->>M: tx for the current outpoint<br/>solid-pod-rs/src/blocktrail.rs:1106
        M-->>W: TxInfo with vin, vout and confirmation
        W->>W: the output script must be the key this prefix of states derives<br/>solid-pod-rs/src/blocktrail.rs:1113
        W->>W: the spent input becomes the previous mark<br/>solid-pod-rs/src/blocktrail.rs:1136
    end
    W-->>R: TrailReport, one MarkReport per state
    R-->>V: Some(report)
    V->>T: is_intact decides the boolean<br/>solid-pod-rs-server/src/mempool.rs:752
    Note over T: is_verified needs every mark confirmed (solid-pod-rs/src/blocktrail.rs:841), is_intact also<br/>accepts a head still in the mempool (solid-pod-rs/src/blocktrail.rs:849). verify answers<br/>with is_intact, so an unconfirmed head passes.
    Note over W: INVARIANT: the amount is not checked on this path, because an MRC20 trail keeps only its head<br/>outpoint (solid-pod-rs/src/blocktrail.rs:1068). The key and the spend order are the whole check.
    Note over W: A mark that does not spend its predecessor is BrokenLink and counts as a failure<br/>(solid-pod-rs/src/blocktrail.rs:647). The blocktrails/verify page shows the broken link but<br/>still labels the mark verified, so this crate is deliberately stricter than its reference.
```
- **What it shows:** `verify` on either anchorer is a walk, not a lookup. From the anchor's own outpoint back to genesis every mark must exist, be the key its prefix of states derives, and be spent by the next.
- **Why it is this way:** a UTXO at the head address proves only that the head key was paid, so the earlier version accepted a trail whose history was missing or paid other keys (0.5.0-alpha.12 CHANGELOG, `Changed`). Verification follows blocktrails/verify 043e7af, with fixtures generated from the page itself.

## SP-07.20 GitmarkAnchorer — the git-mark profile for new trails (alpha.12)

```mermaid
flowchart TB
    GEN["genesis(name, voucher, commit, network)<br/>solid-pod-rs-server/src/mempool.rs:943"]
    EXISTS{"trail already stored?<br/>solid-pod-rs-server/src/mempool.rs:959"}
    GG["gitmark_genesis — voucher to the first mark<br/>solid-pod-rs/src/bitcoin_tx.rs:869"]
    ANC["anchor(ticker, commit, network)<br/>solid-pod-rs-server/src/mempool.rs:994"]
    NOGEN["Err: trail has no genesis on this pod<br/>solid-pod-rs-server/src/mempool.rs:1006"]
    NETC{"stored.network equals the requested network?<br/>solid-pod-rs-server/src/mempool.rs:1012"}
    GA["gitmark_advance — newest mark must be the derived key, else refuse<br/>solid-pod-rs/src/bitcoin_tx.rs:929"]
    BAS["broadcast_and_save<br/>solid-pod-rs-server/src/mempool.rs:890"]
    BC["broadcast_tx returns the txid, written into the new txo entry<br/>solid-pod-rs-server/src/mempool.rs:906"]
    SAVE["save_gitmark_trail<br/>solid-pod-rs-server/src/trail_store.rs:191"]
    P1["/.well-known/gitmark/name.json with the base secret<br/>solid-pod-rs-server/src/trail_store.rs:133"]
    P2["/.well-known/gitmark/name/blocktrails.json, no secret<br/>solid-pod-rs-server/src/trail_store.rs:140"]
    ANCHOR["BlockTrailAnchor with state_strings and the trail's base key<br/>solid-pod-rs-server/src/mempool.rs:890"]

    GEN --> EXISTS
    EXISTS -- yes --> ERR1["Err: already exists"]
    EXISTS -- no --> GG --> BAS
    ANC -- no trail --> NOGEN
    ANC --> NETC
    NETC -- no --> ERR2["Err: network mismatch"]
    NETC -- yes --> GA --> BAS
    BAS --> BC --> SAVE
    SAVE --> P1
    SAVE --> P2
    BAS --> ANCHOR

    N["DEBT: anchor holds no lock across load, advance, broadcast and save. Two concurrent<br/>anchors on one trail would both spend the same head, and the second broadcast<br/>fails as a double spend after the first has saved.<br/>solid-pod-rs-server/src/mempool.rs:994"]
    ANC -.-> N
    N2["OPEN: nothing outside the tests calls genesis, so no pod reaches this path from<br/>this repository (see SP-07.3).<br/>solid-pod-rs-server/src/mempool.rs:943"]
    GEN -.-> N2
    N3["INVARIANT: an issued MRC20 trail is never moved to this profile. build_anchorer keeps<br/>MempoolBlockAnchorer whenever one exists under the name.<br/>solid-pod-rs-server/src/handlers/prov.rs:141"]
    BAS -.-> N3
    N4["The gitmark network name encodes addresses with the gm HRP, bc and tb unchanged<br/>solid-pod-rs/src/blocktrail.rs:60"]
    GG -.-> N4
```
- **What it shows:** the git-mark anchorer starts a trail from a funding voucher, advances it one commit per anchor, and publishes a secret-free `blocktrails.json` beside the secret-bearing trail file where a verifier can read it.
- **Why it is this way:** new trails follow blocktrails/git-mark b852d7d, with the commit as the state and no MRC20 wrapper, while issued MRC20 trails keep their on-chain keys and `urn:mono:op:anchor` states untouched (0.5.0-alpha.12 CHANGELOG, `Added`). Both anchorers share one read side, `verify_anchor_report`.
