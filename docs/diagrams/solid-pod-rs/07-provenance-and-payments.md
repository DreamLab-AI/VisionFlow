---
id: SP-07
title: Provenance — git-marks, block-trails, Bitcoin anchoring — and the web ledger
area: solid-pod-rs
governing: [../solid-pod-rs/README.md, ../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2004, ADR-2007, ADR-2008]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs/src/provenance.rs
  - ../solid-pod-rs/README.md
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
verified_commit: 9b6143ce55b242c322c92d8c6bcc0c1a5371b451
---

**Repair note — 2026-11-03:** selected counterexamples below have a [scoped successor verification](15-authority-and-payment-repairs.md). This historical topic retains its declared source revision; the successor states the repaired contracts and acceptance boundaries.

## SP-07.1 The two provenance tiers

```mermaid
flowchart LR
    W["eligible stored resource write<br/>git feature and repository required<br/>solid-pod-rs-server/src/lib.rs:3917"]
    GM["record_receipt attempts local mark<br/>solid-pod-rs/src/provenance.rs:516"]
    MK["receipt retains mark and per-tier errors<br/>solid-pod-rs/src/provenance.rs:558"]
    AN["inline policy may anchor the commit SHA<br/>solid-pod-rs/src/provenance.rs:544"]
    SC["write PROV-O sidecar after mark<br/>solid-pod-rs-server/src/lib.rs:4091"]
    W --> GM --> MK --> SC
    GM --> AN --> MK
    N["INVARIANT: inline anchoring receives the git commit SHA.<br/>Epoch batching instead anchors a Merkle root over commits.<br/>solid-pod-rs-server/src/handlers/prov.rs:313"]
    AN -.-> N
    D["DRIFT: DELETE bypasses marking and leaves the old sidecar.<br/>Not every LDP mutation has a receipt.<br/>solid-pod-rs-server/src/lib.rs:2277"]
    W -.-> D
```

- **What it shows:** resource storage precedes provenance. An eligible local mark can be accompanied by an optional inline anchor — epoch anchoring is a separate batch path.
- **Why it is this way:** ADR-2004 keeps Git opt-in and makes achieved tiers observable without changing a successful storage response. It does not make storage, marking and sidecar persistence atomic.

## SP-07.2 git_mark_write — the skip ladder

```mermaid
flowchart TD
    IN["git_mark_write(state, resource_path, agent, message)<br/>solid-pod-rs-server/src/lib.rs:3917"]
    SIDE{"path ends .acl, .meta or .prov.ttl?<br/>solid-pod-rs-server/src/lib.rs:3928"}
    S1["Skipped: ExcludedPath<br/>solid-pod-rs/src/provenance.rs:635"]
    CONT{"path ends with a slash?<br/>solid-pod-rs-server/src/lib.rs:3935"}
    S2["Skipped: Container<br/>solid-pod-rs/src/provenance.rs:637"]
    ROOT{"data_root configured?<br/>solid-pod-rs-server/src/lib.rs:3940"}
    S3["Skipped: NotConfigured<br/>solid-pod-rs/src/provenance.rs:629"]
    SPLIT{"path splits into pod plus rest?<br/>solid-pod-rs-server/src/lib.rs:3946"}
    S4["Skipped: UnresolvablePath<br/>solid-pod-rs/src/provenance.rs:639"]
    GITD{"data_root/{pod}/.git is a directory?<br/>solid-pod-rs-server/src/lib.rs:3957"}
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
    N3["Cargo default: this function is the no-op shim, returning NotConfigured.<br/>This is a build-profile fact — consumers can enable git explicitly.<br/>solid-pod-rs-server/src/lib.rs:4123"]
    IN -.-> N3
```

- **What it shows:** the skip ladder applies when a handler calls the hook — it does not establish mutation coverage. The default build skips even a configured filesystem repository.
- **Why it is this way:** ADR-2004 distinguishes ineligible writes from failed attempts. Agentbox enables Git in its package — its path patch and runtime configuration need separate verification.

## SP-07.3 Anchor policy resolution

```mermaid
flowchart TD
    RES["resolve_anchor_policy: first anchor condition across authorisations<br/>solid-pod-rs-server/src/handlers/prov.rs:90"]
    NEVER["absent condition or ACL read error: Never<br/>solid-pod-rs-server/src/handlers/prov.rs:96"]
    HV["inline mode: HighValue<br/>solid-pod-rs-server/src/handlers/prov.rs:110"]
    EP["epoch mode: Epoch<br/>solid-pod-rs-server/src/handlers/prov.rs:111"]
    BLD["build_anchorer when policy is not Never<br/>solid-pod-rs-server/src/lib.rs:3980"]
    MRC["issued MRC20 trail: MempoolBlockAnchorer<br/>solid-pod-rs-server/src/handlers/prov.rs:200"]
    GM["started git-mark trail: GitmarkAnchorer<br/>solid-pod-rs-server/src/handlers/prov.rs:212"]
    ERR["unreadable trail: UnavailableAnchorer<br/>solid-pod-rs-server/src/handlers/prov.rs:161"]
    NONE["no trail: no bundle<br/>solid-pod-rs-server/src/handlers/prov.rs:210"]
    FLAG["DRIFT: no bundle clears high_value —<br/>missing inline anchor is not reported<br/>solid-pod-rs-server/src/lib.rs:4003"]
    BATCH["Epoch becomes Never for record_receipt —<br/>batch call only when bundle exists<br/>solid-pod-rs-server/src/lib.rs:4000<br/>solid-pod-rs-server/src/lib.rs:4061"]
    RES --> NEVER
    RES --> HV --> BLD
    RES --> EP --> BLD
    BLD --> MRC
    BLD --> GM
    BLD --> ERR
    BLD --> NONE --> FLAG
    EP --> BATCH
```

- **What it shows:** the HTTP resolver returns Never, HighValue or Epoch — the library's Always variant is not selected here. The first matching authorisation condition determines the policy, independently of which rule granted the write.
- **Why it is this way:** inline and batch work share provider construction, but the server currently drops the requested-policy distinction when no provider exists. An unreadable inline trail does produce an anchor error.
- **Drift:** the explicit anchor endpoint still says no pay-token or minted trail at solid-pod-rs-server/src/handlers/prov.rs:501, although build_anchorer also accepts a started git-mark trail at solid-pod-rs-server/src/handlers/prov.rs:212.
- **Open:** the native-server source search found only test calls to GitmarkAnchorer::genesis at solid-pod-rs-server/src/mempool.rs:1400. No native route or boot call initiating that trail was found — external provisioning remains unassessed.

## SP-07.4 record_receipt — typed, per-tier outcomes (ADR-2004)

```mermaid
flowchart TD
    R["record_receipt attempts mark<br/>solid-pod-rs/src/provenance.rs:516"]
    FAIL["ResourceStored plus mark_error<br/>solid-pod-rs/src/provenance.rs:526"]
    WANT{"mark exists — inline anchor requested?<br/>solid-pod-rs/src/provenance.rs:542"}
    LOCAL["LocalMarkCommitted<br/>solid-pod-rs/src/provenance.rs:721"]
    TRY["try provider — absence is an error<br/>solid-pod-rs/src/provenance.rs:549"]
    ERR["LocalMarkCommitted plus anchor_error<br/>solid-pod-rs/src/provenance.rs:545"]
    CLASS{"returned anchor has blockheight?<br/>solid-pod-rs/src/provenance.rs:723"}
    SUB["AnchorSubmitted<br/>solid-pod-rs/src/provenance.rs:726"]
    CONF["AnchorConfirmed<br/>solid-pod-rs/src/provenance.rs:724"]
    R -- mark failed --> FAIL
    R -- mark succeeded --> WANT
    WANT -- no --> LOCAL
    WANT -- yes --> TRY
    TRY -- provider absent or failed --> ERR
    TRY -- anchor returned --> CLASS
    CLASS -- no --> SUB
    CLASS -- yes --> CONF
    N["stage() classifies existing receipt fields.<br/>No chain polling or confirmation transition.<br/>solid-pod-rs/src/provenance.rs:715"]
    CLASS -.-> N
```

- **What it shows:** the library returns the achieved tier plus separate errors after resource storage. Submitted and confirmed are alternative classifications of the returned anchor.
- **Why it is this way:** ADR-2004 separates local history from anchoring evidence without changing the already successful resource write. Stub tests cover classification — they do not establish live chain confirmation.

## SP-07.5 The receipt on the wire

```mermaid
sequenceDiagram
    autonumber
    participant H as Eligible write handler
    participant G as git_mark_write
    participant R as ProvenanceReceipt
    participant S as Storage
    participant C as Client
    Note over H,S: Resource bytes are stored before the hook runs.
    H->>G: mark the write<br/>solid-pod-rs-server/src/lib.rs:3917
    G->>R: compose receipt for eligible write<br/>solid-pod-rs-server/src/lib.rs:4023
    alt receipt has no mark
        G-->>H: return failed-mark receipt
    else local mark exists
        opt policy is Epoch and anchorer bundle exists
            G->>G: accumulate or anchor epoch<br/>solid-pod-rs-server/src/lib.rs:4063
            Note over G,R: DRIFT: success or failure only logged —<br/>receipt is not updated<br/>solid-pod-rs-server/src/lib.rs:4079
        end
        G->>S: write resource.prov.ttl<br/>solid-pod-rs-server/src/lib.rs:4091
        opt sidecar write fails
            G->>R: add mark_error — keep achieved mark and anchor<br/>solid-pod-rs-server/src/lib.rs:4105
        end
        G-->>H: receipt
    end
    H->>C: X-Provenance summary — commit header when available<br/>solid-pod-rs/src/provenance.rs:753<br/>solid-pod-rs/src/provenance.rs:743
    Note over R,C: header_safe bounds printable error text<br/>solid-pod-rs/src/provenance.rs:774
```

- **What it shows:** eligible PUT/POST/PATCH paths return the receipt while keeping the successful storage status. Earlier skip guards return before composition. The hook's epoch result is not attached to the response or its per-resource sidecar.
- **Why it is this way:** the receipt exposes local marking and sidecar failures without rolling back storage. A crash before the receipt reaches the client still needs durable reconciliation — a response header cannot close that window.

## SP-07.6 Provenance queries and explicit paid anchors

```mermaid
flowchart TD
    REG["register before LDP catch-all<br/>solid-pod-rs-server/src/handlers/prov.rs:747"]
    GET["GET commit: resolve Git metadata and first content path<br/>solid-pod-rs-server/src/handlers/prov.rs:351"]
    SC["read current resource sidecar directly from storage<br/>solid-pod-rs-server/src/handlers/prov.rs:391"]
    LEAK["DRIFT: no resource WAC gate or sidecar commit check<br/>anchored is a substring check for bt:txid<br/>solid-pod-rs-server/src/handlers/prov.rs:413"]
    POST["POST anchor: NIP-98 owner, resolve commit and trail<br/>solid-pod-rs-server/src/handlers/prov.rs:454"]
    DEBIT["charge before anchor, no logical request receipt<br/>solid-pod-rs-server/src/handlers/prov.rs:512"]
    ANCHOR["anchor commit hash<br/>solid-pod-rs-server/src/handlers/prov.rs:523"]
    PENDING["pending error: 202, keep charge<br/>solid-pod-rs-server/src/handlers/prov.rs:527"]
    REFUND["other error: refund or try retaining debt<br/>solid-pod-rs-server/src/handlers/prov.rs:543"]
    WRITE["accepted: replace current sidecar even for old commit<br/>write failure logged, still 200 and charged<br/>solid-pod-rs-server/src/handlers/prov.rs:584"]
    REG --> GET --> SC --> LEAK
    REG --> POST --> DEBIT --> ANCHOR
    ANCHOR -- ambiguous --> PENDING
    ANCHOR -- error --> REFUND
    ANCHOR -- accepted --> WRITE
```

- **What it shows:** the point query bypasses the ordinary LDP read path. A known old commit can return a newer sidecar, and an owner can overwrite that sidecar by anchoring the old commit. The endpoint does not deduplicate logical requests.
- **Why it is this way:** the explicit upgrade composes a local charge, chain submission and mutable sidecar. NIP-98 rejects reuse of an authentication event, but a retry with a new event is a fresh charge.
- **Debt:** a failed refund is only *attempted* in the pending file; a second storage failure is logged at solid-pod-rs-server/src/handlers/prov.rs:692. No repayment worker was found in the crate search. This is not a guaranteed durable refund.
- **Tension:** private resource and sidecar reads returned 401 in the audit fixture, while the unauthenticated commit query returned their metadata and sidecar text. No pod-wide query enumeration or live deployment exposure is claimed.

## SP-07.7 Epoch batching — one transaction, many commits

```mermaid
flowchart TD
    LOCK["process-local epoch lock and load pending commits<br/>solid-pod-rs-server/src/handlers/prov.rs:264"]
    DONE["remove commits named by a saved completion marker<br/>solid-pod-rs-server/src/handlers/prov.rs:272"]
    CAP{"new commit fits below 4 times epoch size?<br/>solid-pod-rs-server/src/handlers/prov.rs:289"}
    ADD["append new commit<br/>solid-pod-rs-server/src/handlers/prov.rs:290"]
    DROP["DEBT: omit new commit, still retry old batch<br/>solid-pod-rs-server/src/handlers/prov.rs:294"]
    SAVE["persist pending list before close<br/>solid-pod-rs-server/src/handlers/prov.rs:301"]
    FULL{"batch reaches epoch size?<br/>solid-pod-rs-server/src/handlers/prov.rs:302"}
    ROOT["close accumulator and anchor Merkle root<br/>solid-pod-rs-server/src/handlers/prov.rs:311<br/>solid-pod-rs-server/src/handlers/prov.rs:313"]
    RESET["on accepted anchor, reset pending list<br/>solid-pod-rs-server/src/handlers/prov.rs:319"]
    MARK["reset failed: save completed commit list<br/>solid-pod-rs-server/src/handlers/prov.rs:320"]
    FAIL["DEBT: reset and marker both failed<br/>later push can anchor again<br/>solid-pod-rs-server/src/handlers/prov.rs:328"]
    LOCK --> DONE --> CAP
    CAP -- yes --> ADD --> SAVE
    CAP -- no --> DROP --> SAVE
    SAVE --> FULL
    FULL -- no --> WAIT["return accumulated"]
    FULL -- yes --> ROOT
    ROOT -- failure --> KEEP["keep pending list and return error"]
    ROOT -- accepted --> RESET
    RESET -- failed --> MARK
    MARK -- failed --> FAIL
```

- **What it shows:** pending commits survive a failed close; a saved completion marker can repair a failed reset. The backlog limit omits new commits, including the write that triggers successful draining of a full backlog.
- **Why it is this way:** batch anchoring bounds transaction cost and retained backlog. The returned anchor is discarded here, and broadcast acceptance does not establish confirmation. Per-resource receipts still omit epoch outcomes (SP-07.5).
- **Open:** lock_path is process-local at solid-pod-rs-server/src/trail_store.rs:340. Multi-process safety, durable proof export and process-restart reconciliation are not established by the in-memory fault tests.

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
    VER["verify_mrc20_anchor — state chain, a head UTXO read via MempoolLookup,<br/>then a walk of every earlier mark<br/>solid-pod-rs/src/mrc20.rs:1034"]
    WALK["verify_anchor_chain — back from the head outpoint to genesis<br/>solid-pod-rs/src/blocktrail.rs:1068"]
    ML["MempoolLookup trait<br/>solid-pod-rs/src/mrc20.rs:372"]

    JCS --> SHA --> ST --> VAL --> LNK --> TR
    TR --> DEP
    KEY --> ADDR --> VER --> ML
    VER --> WALK

    N["INVARIANT: the state hash is taken over the RFC 8785 canonical serialisation,<br/>not over the raw bytes — two semantically identical JSON documents must hash<br/>identically or the chain breaks on a formatting change. The form hashed is the<br/>typed Mrc20State re-serialised, so a field the struct does not declare is dropped<br/>before hashing (solid-pod-rs/src/mrc20.rs:218)."]
    JCS -.-> N
    N2["MRC20_PROFILE mono.mrc20.v0.1 and TRANSFER_OP name the wire contract<br/>solid-pod-rs/src/mrc20.rs:34 and :36"]
    ST -.-> N2
    N3["extract_transfers_to and total_transferred_to are the read side<br/>solid-pod-rs/src/mrc20.rs:240"]
    DEP -.-> N3
    N4["CHANGED in alpha.12: a UTXO at the head address no longer suffices. Each earlier mark must<br/>be the key its prefix of states derives and spend the mark before it, else the call<br/>fails with 'does not verify link by link'. A head still in the mempool is still<br/>accepted (intact, not verified). Per-link detail: SP-07.19.<br/>solid-pod-rs/src/mrc20.rs:1034"]
    VER -.-> N4
    N5["The walk covers only the stateStrings the caller supplies. Only the LAST is compared<br/>with JCS(state) (solid-pod-rs/src/mrc20.rs:1061); nothing requires JCS(prevState) or a<br/>seq-0 genesis among them, or one string per state since genesis. With one string there<br/>is no earlier mark to walk (SP-07.19)."]
    VER -.-> N5
```

- **Tension:** "every earlier mark" is only as long as the proof the caller sends. A one-element `stateStrings` proof turns the walk into the head-only check that alpha.12 replaced, because mark 0's input is never examined (solid-pod-rs/src/blocktrail.rs:1114). Its head address is derived from the public issuer key alone (solid-pod-rs/src/mrc20.rs:766), so funding it needs no issuer secret. Only a mark after genesis needs that secret to spend its predecessor. The deposit consequence is recorded at SP-07.17. Closing it needs `verify_mrc20_anchor` to require at least two strings, with the second-to-last equal to `JCS(prevState)`, or to bind the proof to a trail the pod has recorded. This is source tracing only: no probe was run.

## SP-07.10 The rust-bitcoin transaction builder (ADR-2008 D1, landed)

```mermaid
flowchart TD
    INPUT["build_transaction: nonempty inputs, one secret key<br/>solid-pod-rs/src/bitcoin_tx.rs:281"]
    KEY["first input script decides key-path tweak<br/>xonly_of uses libsecp256k1 and still exists<br/>solid-pod-rs/src/bitcoin_tx.rs:130<br/>solid-pod-rs/src/bitcoin_tx.rs:293"]
    TX["rust-bitcoin Transaction, TxIn, TxOut and serialisation<br/>per-value money range checks<br/>solid-pod-rs/src/bitcoin_tx.rs:308<br/>solid-pod-rs/src/bitcoin_tx.rs:372"]
    HASH["SighashCache with all prevouts, SIGHASH_DEFAULT<br/>solid-pod-rs/src/bitcoin_tx.rs:340"]
    SIGN["libsecp256k1 Schnorr with zero auxiliary randomness<br/>solid-pod-rs/src/bitcoin_tx.rs:218"]
    RESULT["witnesses, computed txid, raw and unsigned hex<br/>solid-pod-rs/src/bitcoin_tx.rs:362"]
    CHECK["golden cases A/B/C and published BIP vectors<br/>solid-pod-rs/src/bitcoin_tx.rs:1244<br/>solid-pod-rs/src/bitcoin_tx.rs:1417"]
    INPUT --> KEY --> TX --> HASH --> SIGN --> RESULT
    CHECK -.-> RESULT
```

- **What it shows:** the primitive port is implemented. The xonly_of wrapper remains; the removed code is the handwritten primitive arithmetic and encoders. The golden fixtures use deterministic signing for cross-implementation equality.
- **Why it is this way:** ADR-2008 D1/D2 replaces crypto primitives with maintained libraries while retaining the transaction policy and compatibility surface. One key signs all inputs; callers must group compatible inputs, as the builder documentation states at solid-pod-rs/src/bitcoin_tx.rs:230.
- **Open:** native vector tests do not prove fee adequacy, chain acceptance, consumer activation or WASM transaction building. The write module remains native-only at solid-pod-rs/src/bitcoin_tx.rs:53.

## SP-07.11 Mempool endpoint selection (ADR-2007)

```mermaid
flowchart TD
    STATE["fresh AppState has mempool_url None<br/>solid-pod-rs-server/src/lib.rs:417"]
    BOOT["main selects from that state field<br/>solid-pod-rs-server/src/main.rs:327"]
    LOG["log selection once per process<br/>solid-pod-rs-server/src/mempool.rs:362"]
    ENV["request client resolves environment override<br/>solid-pod-rs-server/src/mempool.rs:434"]
    DIFF["DRIFT: startup can log default testnet4<br/>while requests use explicit environment URL<br/>solid-pod-rs-server/src/mempool.rs:314"]
    INF["infer_network uses URL path and host<br/>not a chain identity query<br/>solid-pod-rs-server/src/mempool.rs:201"]
    HTTP["one base URL, default redirect policy<br/>solid-pod-rs-server/src/mempool.rs:87"]
    PRES["presence: 404 false, other errors ambiguous,<br/>any successful status true without body validation<br/>solid-pod-rs-server/src/mempool.rs:456"]
    STATE --> BOOT --> LOG
    ENV --> LOG
    BOOT -.-> DIFF
    ENV -.-> DIFF
    ENV --> INF
    ENV --> HTTP --> PRES
```

- **What it shows:** startup and request construction resolve different inputs. Once-only logging can retain the startup default. The client keeps one configured base but follows a local cross-origin redirect in the audit probe.
- **Why it is this way:** ADR-2007 rejects ordered endpoint failover. That decision does not pin redirect destinations or turn URL-based network inference into chain verification.
- **Open:** to_manifest_json at solid-pod-rs-server/src/mempool.rs:280 exposes the selection value, but a deployed queryable manifest and chain identity remain unverified.

## SP-07.12 The block anchorer

```mermaid
sequenceDiagram
    autonumber
    participant P as Caller
    participant A as MempoolBlockAnchorer
    participant S as Trail storage
    participant M as Explorer REST
    P->>A: ticker, commit state, requested network
    A->>S: process-local trail lock<br/>solid-pod-rs-server/src/mempool.rs:755
    A->>A: settle earlier intent first<br/>solid-pod-rs-server/src/mempool.rs:757
    A->>S: load trail and compare requested network<br/>solid-pod-rs-server/src/mempool.rs:761<br/>solid-pod-rs-server/src/mempool.rs:768
    A->>S: save signed transaction and proposed trail before broadcast<br/>solid-pod-rs-server/src/mempool.rs:801
    A->>M: broadcast and require expected txid<br/>solid-pod-rs-server/src/mempool.rs:815
    alt broadcast fails or returns unexpected txid
        A-->>P: pending error, intent retained<br/>solid-pod-rs-server/src/mempool.rs:817
    else accepted txid
        A->>S: save trail, then try clearing intent<br/>solid-pod-rs-server/src/mempool.rs:827
        Note over A,S: Save failure is logged, intent retained.<br/>Return still succeeds after accepted broadcast.<br/>solid-pod-rs-server/src/mempool.rs:835
        A-->>P: anchor with no block height<br/>solid-pod-rs-server/src/mempool.rs:847
    end
    Note over A,M: Later verify walks commitments using explorer data.<br/>is_intact permits unconfirmed marks.<br/>solid-pod-rs-server/src/mempool.rs:856
```

- **What it shows:** issued MRC20 trails retain their profile. Broadcast ambiguity and a failed save after acceptance have different caller outcomes; acceptance is not chain confirmation.
- **Why it is this way:** the retained intent supports retry before another advance, without refunding an accepted transaction solely because local persistence failed. Existing fixture tests cover this path.
- **Open:** settle_trail_intent validates retained transaction identity and head input at solid-pod-rs-server/src/mempool.rs:1129, then trusts the longer proposed trail at solid-pod-rs-server/src/mempool.rs:1147. The analogous git-mark metadata failure is reproduced in SP-07.20; altered MRC20 state is not freshly probed.

## SP-07.13 The stored web ledger and its public operations

```mermaid
flowchart TD
    LEDGER["WebLedger stores balances and receipts<br/>get_balance is an exact-key local lookup<br/>solid-pod-rs/src/payments.rs:412"]
    ID["check_genesis checks genesis hash when present<br/>legacy ledger without genesis is accepted<br/>solid-pod-rs/src/payments.rs:372"]
    CREDIT["credit_by_outpoint: validate shape and amount,<br/>deduplicate outpoint, update chosen currency<br/>solid-pod-rs/src/payments.rs:450"]
    PAYOUT["debit_by_payout: debit once by txid<br/>reverse_payout restores its recorded amount<br/>solid-pod-rs/src/payments.rs:501<br/>solid-pod-rs/src/payments.rs:536"]
    CHARGE["charge and refund: local service-fee mutations<br/>Charge is an in-memory capability<br/>solid-pod-rs/src/payments.rs:548<br/>solid-pod-rs/src/payments.rs:557"]
    STORE["PaymentStore still reads and writes ledger authority<br/>solid-pod-rs/src/payments.rs:911"]
    VERIFY["MRC20 HTTP handler verifies before crediting token units<br/>solid-pod-rs-server/src/handlers/pay.rs:608<br/>solid-pod-rs-server/src/handlers/pay.rs:655"]
    ID --> LEDGER
    VERIFY --> CREDIT --> LEDGER
    PAYOUT --> LEDGER
    CHARGE --> LEDGER
    LEDGER --> STORE
```

- **What it shows:** outpoint and payout receipts are local replay controls. The public credit_by_outpoint method performs no network lookup; its caller supplies the verified facts. The handler does verify before it credits, but what verify_mrc20_anchor accepts is weaker than "verified" suggests: a one-string trail passes, so the credited amount is self-asserted (SP-07.17 Tension). Service charges and refunds are additional balance mutations.
- **Why it is this way:** Teller-shaped receipts improve the stored ledger without delivering ADR-2008's chain-derived authority. Bare credit/debit are now crate-private at solid-pod-rs/src/payments.rs:565 and solid-pod-rs/src/payments.rs:579, but public receipt operations and write_ledger still change stored balances.
- **Invariant:** the MRC20 deposit route refuses a ticker that aliases sats at solid-pod-rs-server/src/handlers/pay.rs:591. Its credit goes to token units, not the satoshi balance. Genesis identity does not authenticate the changing balances.

## SP-07.14 The `/pay/*` route family

```mermaid
flowchart LR
    REG["handlers::pay::register<br/>solid-pod-rs-server/src/handlers/pay.rs:1751"]
    BAL["GET /pay/.balance -> handle_balance<br/>solid-pod-rs-server/src/handlers/pay.rs:395"]
    DEP["POST /pay/.deposit -> handle_deposit<br/>solid-pod-rs-server/src/handlers/pay.rs:455"]
    ADDR["GET /pay/.address -> handle_address<br/>solid-pod-rs-server/src/handlers/pay.rs:736"]
    OFF["GET /pay/.offers -> handle_offers<br/>solid-pod-rs-server/src/handlers/pay.rs:828"]
    SELL["POST /pay/.sell -> handle_sell<br/>solid-pod-rs-server/src/handlers/pay.rs:863"]
    SWAP["POST /pay/.swap -> handle_swap<br/>solid-pod-rs-server/src/handlers/pay.rs:918"]
    POOLG["GET /pay/.pool -> handle_pool_get<br/>solid-pod-rs-server/src/handlers/pay.rs:981"]
    POOLP["POST /pay/.pool -> handle_pool_post<br/>solid-pod-rs-server/src/handlers/pay.rs:1036"]
    BUY["POST /pay/.buy -> handle_buy<br/>solid-pod-rs-server/src/handlers/pay.rs:1396"]
    WD["POST /pay/.withdraw -> handle_withdraw<br/>solid-pod-rs-server/src/handlers/pay.rs:1485"]
    WDS["POST /pay/.withdraw-sats -> handle_withdraw_sats<br/>solid-pod-rs-server/src/handlers/pay.rs:1588"]

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

    N["Every write route, and GET /pay/.balance, resolves the caller first: require_did and<br/>require_did_with_body (solid-pod-rs-server/src/handlers/pay.rs:352) answer 401 without<br/>a verified NIP-98 header. Outside dev-bearer builds a Bearer header is unauthenticated<br/>(solid-pod-rs-server/src/lib.rs:611), so the actor is a signing key. is_valid_did_nostr<br/>(solid-pod-rs-server/src/handlers/pay.rs:806) checks only the GET /pay/.address query<br/>(solid-pod-rs-server/src/handlers/pay.rs:776). The receipt mutators narrow an account to<br/>did:nostr with account_of (solid-pod-rs/src/payments.rs:458, :507, :601)."]
    REG -.-> N
    N2["DIVERGENCE: the whole /pay/* surface is registered with the SAME gating as<br/>/pay/.info — always on, with no payments feature flag<br/>(solid-pod-rs-server/src/lib.rs:5282)."]
    REG -.-> N2
```

- **Drift:** the README opens by saying payments are added "through explicit features" (../solid-pod-rs/README.md:3), and its payments paragraph describes deposits, withdrawals, the order book and the AMM with no caution (../solid-pod-rs/README.md:141). The `/pay/*` routes are mounted unconditionally (solid-pod-rs-server/src/lib.rs:5282). An earlier version of this panel cited a README warning against carrying value through these routes. That sentence is in neither 09ec90f nor 9b6143c.

## SP-07.15 Payment-state commits and the read-failure gap

```mermaid
flowchart TD
    LOCK["handlers acquire process-local PAYMENT_STATE_LOCK<br/>solid-pod-rs-server/src/lib.rs:208"]
    READ["read_state_unchecked reads canonical state.json<br/>solid-pod-rs-server/src/handlers/pay.rs:179"]
    PARSE["successful read: parse or return error<br/>then check ledger genesis<br/>solid-pod-rs-server/src/handlers/pay.rs:181<br/>solid-pod-rs-server/src/handlers/pay.rs:169"]
    FALL["DRIFT: any get error falls back to legacy files<br/>version zero and no pending intents<br/>solid-pod-rs-server/src/handlers/pay.rs:184"]
    MUTATE["mutate ledger, replay, orders or exchange"]
    SAVE["commit_state: increment version and put bundled state<br/>solid-pod-rs-server/src/handlers/pay.rs:198"]
    MIRROR["write legacy mirrors best-effort afterwards<br/>solid-pod-rs-server/src/handlers/pay.rs:210"]
    INTENT["token payout stores debit plus intent before broadcast<br/>solid-pod-rs-server/src/handlers/pay.rs:1320"]
    LOCK --> READ
    READ -- bytes --> PARSE --> MUTATE
    READ -- any storage error --> FALL --> MUTATE
    MUTATE --> SAVE --> MIRROR
    INTENT --> SAVE
```

- **What it shows:** bundled state prevents a local transition being split between authoritative ledger, replay and market files. But a read outage can select stale mirrors and the next successful write can replace the canonical state with them, losing intents and resetting the version.
- **Why it is this way:** the implementation has a useful single-document commit and process-local serialisation. It has no cross-process compare-and-swap, and its compatibility fallback is broader than its own missing-file-only contract.
- **Debt:** recovery at solid-pod-rs-server/src/handlers/pay.rs:242 trusts explorer presence, advances a saved trail when behind and reverses a payout on 404. A missing payout receipt retains the intent. These are bounded recovery rules, not proof of finality or safe absence across explorers.
- **Invariant:** token transfer waits at most two seconds for its trail lock while holding the payment lock, then returns 503 with Retry-After at solid-pod-rs-server/src/handlers/pay.rs:1253. The existing tests establish this local bound.

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
    note for AmmPool "pool_key (solid-pod-rs/src/trading.rs:601) normalises the currency pair so\n(A,B) and (B,A) resolve to one pool; isqrt_u128 (:610) is the integer square\nroot the LP-share maths needs — everything is integer arithmetic, no floats\nin a value path. Per-currency credit/debit are crate-private helpers on WebLedger.\nOutside this crate the only new value is credit_by_outpoint; the public swap,\nremove_liquidity and execute_swap move value through those helpers, and the\npublic refund (solid-pod-rs/src/payments.rs:557) and reverse_payout (:536) restore it."
```

## SP-07.17 Deposits are MRC20-only — the stand-in is deleted (ADR-2008 D6)

```mermaid
flowchart TD
    IN["POST /pay/.deposit -> handle_deposit<br/>solid-pod-rs-server/src/handlers/pay.rs:455"]
    SNIFF{"body_is_mrc20?<br/>solid-pod-rs-server/src/handlers/pay.rs:483"}
    MRC["handle_mrc20_deposit — the only credit path<br/>solid-pod-rs-server/src/handlers/pay.rs:524"]
    R501["501 Not Implemented for any authenticated non-MRC20 body<br/>(401 comes first without NIP-98, :460): no flag, no branch, no payment state written<br/>solid-pod-rs-server/src/handlers/pay.rs:470"]

    IN --> SNIFF
    SNIFF -- yes --> MRC
    SNIFF -- no --> R501

    N["INVARIANT: the Phase-0 stand-in that credited (vout + 1) * 1000 sats behind<br/>an operator flag was DELETED at e62d028, not left default-off (ADR-2008 D6):<br/>a pod must not carry a reachable free-money oracle. A non-MRC20 body gets 501<br/>and writes no payment state."]
    R501 -.-> N
    N2["ISSUER AND TICKER BINDING: the trail's anchor pubkey must be the configured<br/>pay_config.token issuer or one of its accepted_issuers — accepts_issuer<br/>(solid-pod-rs-server/src/handlers/pay.rs:549, solid-pod-rs/src/payments.rs:677) —<br/>and the state must carry the configured ticker. A self-issued trail with the right<br/>ticker is a different token: 403. Another ticker: 400. pod_issuer_pubkey<br/>(solid-pod-rs-server/src/handlers/pay.rs:497) serves /pay/.address, not this path."]
    MRC -.-> N2
    N3["CREDIT BY OUTPOINT RECEIPT: after verify_mrc20_anchor (which since alpha.12 walks every mark of the<br/>trail, not the head alone) and a replay guard on<br/>the anchor outpoint, credit_by_outpoint credits the caller's TICKER balance,<br/>never the satoshi balance<br/>(solid-pod-rs-server/src/handlers/pay.rs:655). One anchor outpoint is credited once,<br/>under the process-local PAYMENT_STATE_LOCK (solid-pod-rs-server/src/handlers/pay.rs:464)."]
    MRC -.-> N3
```

- **Tension:** the INVARIANT names what D6 removed: the stand-in. It does not show that the remaining credit path is closed. The credited amount is whatever `state.ops` transfer to the pod's address (solid-pod-rs/src/mrc20.rs:253), and no balance or supply rule limits it beyond the 21-million-coin ceiling per credit (solid-pod-rs/src/payments.rs:469, :622). The whole proof binding is SP-07.9's walk over caller-supplied `stateStrings`. That walk needs no issuer secret when the proof has one string (SP-07.9, SP-07.19). Nothing on this path rejects such a proof (solid-pod-rs-server/src/handlers/pay.rs:603-620). The anchor network comes from the body, defaulting to testnet4 (solid-pod-rs-server/src/handlers/pay.rs:566), so the configured mempool endpoint bounds the chain, not the code. The server's route tests have no one-string or head-only negative case. Source tracing at 9b6143c only: no request was sent. Closure is the SP-07.9 binding plus a route test that refuses a one-string proof.

## SP-07.18 ADR-2008 — completed port and unfinished ledger authority

```mermaid
flowchart TD
    PORT["implemented: rust-bitcoin primitives and golden parity D1/D2<br/>solid-pod-rs/src/bitcoin_tx.rs:281<br/>solid-pod-rs/src/bitcoin_tx.rs:1244"]
    TXO["D6 stand-in deletion implemented: non-MRC20 deposit 501<br/>solid-pod-rs-server/src/handlers/pay.rs:470"]
    MRC["D3/D6 partial: MRC20 credits token units,<br/>buy and withdraw routes remain<br/>solid-pod-rs-server/src/handlers/pay.rs:655<br/>solid-pod-rs-server/src/handlers/pay.rs:1761"]
    API["D5 partial: bare credit/debit now crate-private,<br/>public receipt mutations and write_ledger remain<br/>solid-pod-rs/src/payments.rs:565<br/>solid-pod-rs/src/payments.rs:913"]
    LOCAL["D4 absent: stored local balance, no height-stamped chain view<br/>solid-pod-rs/src/payments.rs:412"]
    ATOMIC["D7 partial: bundled state and local lock,<br/>read-failure fallback can overwrite authority<br/>solid-pod-rs-server/src/handlers/pay.rs:179"]
    PLAN["remaining decision: chain-resolved balance and spends,<br/>cache-only writes and coordinated consumers<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:55<br/>adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:76"]
    PORT --> TXO --> MRC --> API --> LOCAL --> ATOMIC --> PLAN
```

- **What it shows:** code has moved beyond the October implementation note, but not to a chain-derived ledger. D6's stand-in deletion is complete while its requested removal of MRC20 crediting is not. D5's old function-name grep now passes without satisfying its authority requirement.
- **Why it is this way:** this remains a proposed, partially implemented, staged decision at adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:5. The audit distinguishes landed library changes from chain activation and consumer versions.
- **Drift:** the implementation note still says bare credit/debit are public and D7 is not built at adr/ADR-2008-port-bitcoin-tx-to-rust-bitcoin-and-make-the-web-ledger-a-chain-view.md:147. The current code has narrowed those methods and added bundled commits, with the remaining failure in SP-07.15.
- **Open:** D8 remains unmet in the audited consumer lockfiles: VisionClaw resolves alpha.17 and the forum alpha.18. Detailed consumer source receipts and the whole-section assessment are in the dated payment audit; this topic does not certify either deployment.

## SP-07.19 Whole-trail verification — every mark walked back from the head (alpha.12)

```mermaid
sequenceDiagram
    autonumber
    participant V as verify<br/>solid-pod-rs-server/src/mempool.rs:853
    participant R as verify_anchor_report<br/>solid-pod-rs-server/src/mempool.rs:889
    participant W as verify_anchor_chain<br/>solid-pod-rs/src/blocktrail.rs:1068
    participant M as MempoolLookup
    participant T as TrailReport<br/>solid-pod-rs/src/blocktrail.rs:765

    V->>R: the anchor with its portable proof
    R-->>V: None when pubkey or state_strings are absent<br/>solid-pod-rs-server/src/mempool.rs:896
    R-->>V: None when the recorded address is not the derivation, a forged anchor<br/>solid-pod-rs-server/src/mempool.rs:907
    R->>W: pubkey, state_strings and the anchor's txid and vout<br/>solid-pod-rs-server/src/mempool.rs:913
    loop mark k-1 down to 0
        W->>M: tx for the current outpoint<br/>solid-pod-rs/src/blocktrail.rs:1106
        M-->>W: TxInfo with vin, vout and confirmation
        W->>W: the output script must be the key this prefix of states derives<br/>solid-pod-rs/src/blocktrail.rs:1113
        W->>W: the spent input becomes the previous mark<br/>solid-pod-rs/src/blocktrail.rs:1136
    end
    W-->>R: TrailReport, one MarkReport per state
    R-->>V: Some(report)
    V->>T: is_intact decides the boolean<br/>solid-pod-rs-server/src/mempool.rs:856
    Note over T: is_verified needs every mark confirmed (solid-pod-rs/src/blocktrail.rs:841), is_intact also<br/>accepts marks still in the mempool, not only the head (solid-pod-rs/src/blocktrail.rs:849). verify<br/>answers with is_intact, so an unconfirmed head passes.
    Note over W: INVARIANT: the amount is not checked on this path, because an MRC20 trail keeps only its head<br/>outpoint (solid-pod-rs/src/blocktrail.rs:1068). The key and the spend order are the whole check.
    Note over W: On this walk a mark with inputs is linked by construction (solid-pod-rs/src/blocktrail.rs:1137).<br/>An input that is not the predecessor shows as WrongKey or NotFound at the predecessor's index.<br/>BrokenLink (solid-pod-rs/src/blocktrail.rs:647) arises only for a mark with no inputs (:1116).<br/>Either way it fails. Mark 0's input is never examined (:1114), so a one-string proof is a head check.
```
- **What it shows:** `verify` on either anchorer is a walk, not a lookup. From the anchor's own outpoint back to genesis every mark must exist, be the key its prefix of states derives, and be spent by the next.
- **Why it is this way:** a UTXO at the head address proves only that the head key was paid, so the earlier version accepted a trail whose history was missing or paid other keys (0.5.0-alpha.12 CHANGELOG, `Changed`). Verification follows blocktrails/verify 043e7af, with fixtures generated from the page itself. On the full-trail walk, `verify_blocktrail` counts BrokenLink as a failure where the page still labels such a mark verified. Here the crate is deliberately stricter than its reference.
- **Tension:** the walk is as long as `state_strings`, and the caller supplies them. See SP-07.9 for the missing binding and SP-07.17 for its deposit consequence.

## SP-07.20 GitmarkAnchorer — the git-mark profile for new trails (alpha.12)

```mermaid
flowchart TD
    CALL["genesis or anchor takes process-local trail lock<br/>solid-pod-rs-server/src/mempool.rs:1410<br/>solid-pod-rs-server/src/mempool.rs:1462"]
    REC["load earlier intent and query presence<br/>solid-pod-rs-server/src/mempool.rs:1255<br/>solid-pod-rs-server/src/mempool.rs:1261"]
    CHECK["when rawHex exists, check txid and head input<br/>solid-pod-rs-server/src/mempool.rs:1276"]
    PRESENT["if unknown, resend retained raw transaction<br/>legacy without rawHex is dropped<br/>solid-pod-rs-server/src/mempool.rs:1288"]
    SAVE["if accepted and private count is behind, save proposed trail<br/>solid-pod-rs-server/src/mempool.rs:1298"]
    CLEAR["clear intent<br/>solid-pod-rs-server/src/mempool.rs:1303"]
    BUILD["new genesis or checked extension<br/>solid-pod-rs-server/src/mempool.rs:1421<br/>solid-pod-rs-server/src/mempool.rs:1478"]
    INT["save new intent before broadcast<br/>solid-pod-rs-server/src/mempool.rs:1343"]
    BC["broadcast and require expected txid<br/>solid-pod-rs-server/src/mempool.rs:1358"]
    FILES["save private trail then public proof<br/>solid-pod-rs-server/src/trail_store.rs:354"]
    RET["return accepted anchor, even after save failure<br/>solid-pod-rs-server/src/mempool.rs:1375"]
    CALL --> REC --> CHECK --> PRESENT --> SAVE --> CLEAR --> BUILD --> INT --> BC --> FILES --> RET
    BAD["DRIFT: altered proposed states can be saved and published.<br/>Transaction identity check does not bind that metadata.<br/>solid-pod-rs-server/src/mempool.rs:1011"]
    SAVE -.-> BAD
    PART["DEBT: private save succeeded, public save failed.<br/>Equal private count skips repair before clearing intent.<br/>solid-pod-rs-server/src/mempool.rs:1298"]
    CLEAR -.-> PART
```

- **What it shows:** recovery precedes a new advance. The raw transaction guard rejects a wrong txid, but the audit probe recovers altered state metadata and publishes a non-intact trail. A later extension refuses it.
- **Why it is this way:** durable intents protect against lost broadcast responses and failed private saves. Recovery uses private mark count to decide whether persistence is complete, which misses a failed public proof write.
- **Invariant:** a new extension compares the latest output with the key derived from the trail states at solid-pod-rs/src/bitcoin_tx.rs:963. That later check does not prevent recovery from first replacing stored metadata.
- **Open:** native server source contains no non-test genesis caller (SP-07.3). Full process restart, multi-process locking and external provisioning remain unverified. The fixture transport used here is not a consensus node.
