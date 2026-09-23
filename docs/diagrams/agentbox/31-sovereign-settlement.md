---
id: AB-31
title: Sovereign settlement over sidestr sidechains — the PRD-024 design and its eight decisions
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/docs/PROTOCOL-registry.md
adrs: [ADR-2096, ADR-2097, ADR-2098, ADR-2099, ADR-2100, ADR-2101, ADR-2102, ADR-2103]
sources:
  - ../project/agentbox/docs/proposals/sovereign-settlement.md
  - ../project/agentbox/docs/proposals/sovereign-settlement-domain.md
  - ../project/agentbox/docs/proposals/sovereign-settlement-research/README.md
  - ../project/agentbox/docs/adr/ADR-2096-sidestr-sidechains-are-the-sole-value-instrument.md
  - ../project/agentbox/docs/adr/ADR-2097-sidestr-rail-supersedes-lightning-first.md
  - ../project/agentbox/docs/adr/ADR-2098-chain-and-asset-urn-kinds-and-the-chain-nostr-plane.md
  - ../project/agentbox/docs/adr/ADR-2099-the-chain-is-the-ledger-of-record.md
  - ../project/agentbox/docs/adr/ADR-2100-every-settlement-passes-the-authority-gate.md
  - ../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md
  - ../project/agentbox/docs/adr/ADR-2102-assets-are-bridged-in-rgb-as-a-wrapped-asset.md
  - ../project/agentbox/docs/adr/ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/docs/PROTOCOL-registry.md
  - ../project/agentbox/docs/developer/economy-loop.md
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/management-api/routes/payments.js
  - ../project/agentbox/management-api/routes/broker-bridge.js
  - ../project/agentbox/management-api/routes/llm-marketplace.js
  - ../project/agentbox/management-api/lib/uris.js
  - ../project/agentbox/services/nostr-pod-bridge/src/contract.rs
verified_commit: 1639f86abded1441ce148d6c47924dfaf34f96af
---

## For developers

This topic is the PRD-024 design, read at `1639f86ab` when nothing of it was running: every one of the eight decisions was minted `proposed` with `implementation_status: none` (ADR-2096-sidestr-sidechains-are-the-sole-value-instrument.md:5-7), carried by the four governing documents in explicitly marked PROPOSED sections that join the compliance surface only on ratification (GOVERNANCE-capabilities.md:489-491). Read every node and Note below as a design that cites the record proposing it.
Part of it has since been built. AB-32 is the sealed root chain, AB-33 the four published crates, AB-34 what is actually running, AB-35 the reviewed level-2 shape — each stamped at `ec60a8f14`. Where this topic and those disagree, they are the current claim and this one is the design it came from.

**Drift (this topic vs ADR-2112):** "AB-33 the four published crates" is superseded — on 2026-09-23 ADR-2112 moved the crates (now five) to `DreamLab-AI/sidestr-rs`; agentbox hosts the chain instance only. See SR-01.

## For the business

The estate can already charge for work and already refuses work it has not been paid for, but the balance it debits is a JSON document rather than value anywhere (sovereign-settlement.md:51-55). The proposal is to run the estate's own settlement chains so that a balance becomes something a third party can verify, with every payment above a threshold stopping for a human signature first. It is a costed plan in five ordered phases, only the last of which touches real money, and it has not been approved; the first two phases have since partly begun (AB-32, AB-34).

## AB-31.1 PROPOSED - the target architecture as PRD-024 draws it

```mermaid
flowchart TB
    PARENT["PROPOSED parent chain, configured<br/>sovereign-settlement.md:130-132"]
    subgraph root["PROPOSED root chain sidestr:dreamlab - sovereign-settlement.md:133-137"]
        RC["chain document: parent, headerProfile, signers, threshold,<br/>currencyPin, cashOut, p21Receipt - sovereign-settlement.md:134"]
        PROD["producer: the upstream JS sidecar at P0 to P2,<br/>the Rust producer at P3 - sovereign-settlement.md:135"]
        NODE["sidestr-node validator and mirror, loopback port 9097,<br/>LAN only through the nip98-proxy chain upstream<br/>sovereign-settlement.md:136"]
    end
    subgraph kids["PROPOSED ephemeral child chains - sovereign-settlement.md:138-140"]
        C1["one per AoE session, bound at session create,<br/>closed when the session closes - sovereign-settlement.md:139"]
    end
    subgraph br["PROPOSED bridge, an isolated process - sovereign-settlement.md:141-143"]
        BR["RGB consignment in becomes a WRAPPED asset claim,<br/>burn becomes a consignment out - sovereign-settlement.md:142"]
    end
    subgraph api["PROPOSED management-api surfaces - sovereign-settlement.md:144-149"]
        W["/v1/wallet/* under NIP-98, spend key selected by DID<br/>sovereign-settlement.md:145"]
        G["lib/authority.js under payment_settlement<br/>sovereign-settlement.md:147"]
        U["lib/uris.js gains the chain and asset kinds<br/>sovereign-settlement.md:148"]
    end
    subgraph views["PROPOSED derived views, never authoritative - sovereign-settlement.md:150-154"]
        V1["solid-pod-rs WebLedger view - sovereign-settlement.md:151"]
        V2["forum D1 view - sovereign-settlement.md:152"]
        V3["VisionClaw file store deleted, proxied to the wallet<br/>sovereign-settlement.md:153"]
    end
    PARENT -->|"peg-in, peg-out, checkpoint"| RC
    RC --> PROD --> NODE
    RC -->|"nested parent"| C1
    BR --> PROD
    W --> G --> NODE
    NODE --> views
    U -.-> W
    subgraph note["Status"]
        direction TB
        N1["PROPOSED: no sidechain gate, no sidestr-node, sidestr-producer or<br/>sidestr-bridge program, no crates/sidestr workspace and no loopback<br/>port 9097 bind exists today - BASELINE-container.md:192"]
        N2["PROPOSED: this document is scope, not authority. The eight ADRs are the<br/>decisions and the governing documents are the compliance surface<br/>sovereign-settlement.md:21-23"]
        N1 ~~~ N2
    end
```

## AB-31.2 PROPOSED - what exists, what is missing

```mermaid
flowchart TB
    subgraph have["LIVE today - sovereign-settlement.md:53-62"]
        H1["sell-side HTTP 402 with a real Web Ledger debit<br/>sovereign-settlement.md:55"]
        H2["buy-side classifier, deterministic spend policy, native payer,<br/>receipts on every attempt - sovereign-settlement.md:56"]
        H3["a DECLARED zero-tolerance settlement class<br/>agentbox/agentbox.toml:980"]
        H4["the blocktrail single-use-anchor seam<br/>agentbox/services/nostr-pod-bridge/src/contract.rs:139"]
        H5["three independent did:nostr keyed ledgers, none synced<br/>sovereign-settlement.md:61"]
    end
    subgraph gap["The gap each one leaves"]
        G1["the balance being debited is a JSON document,<br/>not value anywhere - sovereign-settlement.md:55"]
        G2["x402 and l402 are payable false, the Lightning rail<br/>was never built - sovereign-settlement.md:56"]
        G3["DIVERGENCE: routes/payments.js never requires lib/authority.js,<br/>so the class gates NOTHING - sovereign-settlement.md:57"]
        G4["txo is constructed EMPTY and never populated<br/>contract.rs:139"]
        G5["the host deposit path is a stub and the three stores<br/>carry version skew - sovereign-settlement.md:61"]
    end
    H1 --> G1
    H2 --> G2
    H3 --> G3
    H4 --> G4
    H5 --> G5
    G3 --> FIX["PROPOSED ADR-2100 closes the authority gap<br/>GOVERNANCE-capabilities.md:485-487"]
    G4 --> FIX2["PROPOSED ADR-2099 opens the txo seam onto the chain<br/>ADR-2099-the-chain-is-the-ledger-of-record.md:3"]
```

**Drift (manifest vs routes):** `../project/agentbox/agentbox.toml:980` declares `payment_settlement = "zero-tolerance"` and `../project/agentbox/agentbox.toml:1028-1030` gives it `verifiability = inspectable` with `stakes = critical`, but the only two routes that require the gate are `../project/agentbox/management-api/routes/broker-bridge.js:48` and `../project/agentbox/management-api/routes/llm-marketplace.js:43`; the payment routes at `../project/agentbox/management-api/routes/payments.js:10-17` do not, so the class gates nothing today (recorded at `../project/agentbox/docs/GOVERNANCE-capabilities.md:485-487`).

**Debt:** `../project/agentbox/services/nostr-pod-bridge/src/contract.rs:139` ships the blocktrail `txo` field as an always-empty vector, a seam reserved years before anything could fill it.

## AB-31.3 PROPOSED - the eight decisions and what each one owns

```mermaid
flowchart TB
    PRD["PRD-024, scope not authority<br/>sovereign-settlement.md:2-3, drives the eight below<br/>sovereign-settlement.md:6"]
    PRD --> A96["PROPOSED ADR-2096 - sidestr chains are the SOLE value instrument,<br/>consensus and wallet code clean-room in Rust<br/>ADR-2096-sidestr-sidechains-are-the-sole-value-instrument.md:35-37"]
    PRD --> A97["PROPOSED ADR-2097 - the sidestr rail SUPERSEDES Lightning-first,<br/>pay402 gains a fixtured sidestr scheme<br/>ADR-2097-sidestr-rail-supersedes-lightning-first.md:33-35"]
    PRD --> A98["PROPOSED ADR-2098 - chain and asset URN kinds, the sidestr Nostr<br/>kinds registered, chain traffic on its own program<br/>ADR-2098-chain-and-asset-urn-kinds-and-the-chain-nostr-plane.md:34-39"]
    PRD --> A99["PROPOSED ADR-2099 - the chain is the LEDGER OF RECORD, a balance<br/>is a UTXO fold, every existing ledger becomes a view<br/>ADR-2099-the-chain-is-the-ledger-of-record.md:33-36"]
    PRD --> A100["PROPOSED ADR-2100 - every settlement passes the authority gate,<br/>the budget is durable, no gate fails open<br/>ADR-2100-every-settlement-passes-the-authority-gate.md:34-40"]
    PRD --> A101["PROPOSED ADR-2101 - one root chain, ephemeral child chains bound<br/>at session create, domain-separated keys<br/>ADR-2101-federation-topology-and-key-separation.md:36-39"]
    PRD --> A102["PROPOSED ADR-2102 - value enters ONLY by peg-in or bridge, RGB<br/>never enters as an in-chain VM<br/>ADR-2102-assets-are-bridged-in-rgb-as-a-wrapped-asset.md:35-39"]
    PRD --> A103["PROPOSED ADR-2103 - parent chain and header profile are manifest<br/>configuration behind the P21 gate<br/>ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:37-39"]
    subgraph land["Where they land"]
        direction TB
        L1["four governing documents plus the ADR ledger<br/>sovereign-settlement.md:11"]
        L2["domain model DDD-022, bounded context BC25<br/>sovereign-settlement-domain.md:5"]
        L1 ~~~ L2
    end
    A103 --> land
```

**Open:** every one of the eight records carries an empty `verified_commit` and `verified_paths` (`../project/agentbox/docs/adr/ADR-2096-sidestr-sidechains-are-the-sole-value-instrument.md:10-11`), which is correct for a decision with no code, and means nothing in this topic can be re-derived from a symbol until P0 lands.

## AB-31.4 PROPOSED - the ratification lifecycle of a governing-document section

```mermaid
stateDiagram-v2
    [*] --> Researched
    Researched: a Fable-coordinated mesh audits every payments, identity<br/>and provenance surface across six repositories
    note right of Researched
        Four Sonnet researchers, three Opus planners and the
        coordinator's fact base with the owner's binding
        decisions D0 to D6. Working inputs, not authority.
        sovereign-settlement-research/README.md:3-7
    end note
    Researched --> Reviewed
    Reviewed: two independent adversarial reviews, a Sonnet red team<br/>and GPT-6 Astra through the Codex consultant
    note right of Reviewed
        The red team found two mechanical blockers, a
        double-booked Nostr kind and a propagated wrong line
        citation, both fixed. sovereign-settlement.md:337-340
    end note
    Reviewed --> Proposed
    Proposed: the eight ADRs minted decision_status proposed,<br/>implementation_status none, activation_status inactive
    note right of Proposed
        Marked sections are added to the four governing documents
        and every addition says PROPOSED. The live compliance
        surfaces are unchanged. GOVERNANCE-capabilities.md:8
    end note
    Proposed --> Candidate
    Candidate: the proposed invariants are CANDIDATES, and no gate,<br/>test or review may cite them as binding
    note right of Candidate
        GOVERNANCE-capabilities.md:489-491
    end note
    Candidate --> Ratified : the owner ratifies PRD-024
    Candidate --> Withdrawn : the owner does not
    Ratified: the invariants join the Invariants compliance surface VERBATIM
    Ratified --> [*]
    Withdrawn --> [*]
```

**Invariant:** a proposed section states its own non-authority in the document that carries it, so a reader who lands on it by search cannot mistake it for a live rule (`../project/agentbox/docs/BASELINE-container.md:192`, `../project/agentbox/docs/GOVERNANCE-capabilities.md:489-491`).

## AB-31.5 PROPOSED - a settlement through the authority gate

```mermaid
sequenceDiagram
    autonumber
    participant AG as an agent asking to spend
    participant WAL as PROPOSED /v1/wallet/*<br/>agentbox/docs/proposals/sovereign-settlement.md:145
    participant POL as spend-policy, the only authoriser<br/>agentbox/docs/adr/ADR-2100-every-settlement-passes-the-authority-gate.md:41
    participant GATE as PROPOSED lib/authority.js under payment_settlement<br/>agentbox/docs/GOVERNANCE-capabilities.md:493
    participant HUM as the approving human
    participant NODE as PROPOSED sidestr-node<br/>agentbox/docs/BASELINE-container.md:320

    AG->>WAL: request a spend
    WAL->>POL: max_sats_per_call, daily_budget_sats, origin allowlist, threshold
    Note over POL: PROPOSED ADR-2100 D2: a model may REQUEST a spend and can never<br/>AUTHORISE one above policy, and the daily budget becomes durable on the<br/>existing memory adapter slot rather than a sixth one<br/>(ADR-2100-every-settlement-passes-the-authority-gate.md:41-45)
    POL-->>WAL: within policy, or refused
    WAL->>GATE: emit a kind-31402 carrying the task-property triple
    alt above the configured approval threshold
        GATE->>HUM: block until a SIGNED kind-31403 comes back
        HUM-->>GATE: signed decision
    end
    alt denied
        GATE->>GATE: journal the deny as a hash-chained authority.deny entry
        GATE-->>AG: refused, and the outcome is mirrored to the approving human
        Note over GATE: PROPOSED ADR-2100 D4: every outcome mints a receipt, INCLUDING<br/>denied and failed (GOVERNANCE-capabilities.md:514)
    else approved
        GATE->>NODE: the spend reaches the chain
        NODE-->>AG: a receipt citing the chain transaction
    end
    Note over GATE: PROPOSED ADR-2100 D1: no new governance mechanism is added, the<br/>EXISTING one is called - /v1/wallet/* and /v1/chain/* from their first<br/>commit, retrofitted onto /v1/pay/*<br/>(GOVERNANCE-capabilities.md:493-500)
    Note over NODE: PROPOSED ADR-2100 D3: settlement FAILS CLOSED - the cost gate is<br/>forced closed on any chain-settling path, and a stopped node is refused<br/>rather than waved through (GOVERNANCE-capabilities.md:510, GOVERNANCE-capabilities.md:551)
```

**Invariant (proposed):** spend authorisation counts authorising principals and never accounts, reusing the colloquy rule, so an operator's fifty agents are one voice (`../project/agentbox/docs/GOVERNANCE-capabilities.md:517-519`).

## AB-31.6 PROPOSED - the chain as the ledger of record

```mermaid
flowchart TB
    FOLD["PROPOSED: balance(did) is a FOLD over the UTXOs whose script is the<br/>derived spend key, across the root chain and that principal's live child<br/>chains. There is NO wallet object.<br/>ADR-2099-the-chain-is-the-ledger-of-record.md:33-36"]
    FOLD --> V1["solid-pod-rs WebLedger reads THROUGH the node, with a bounded cache<br/>and a staleness bound that is an ERROR, not a slightly old number<br/>ADR-2099-the-chain-is-the-ledger-of-record.md:37-39"]
    FOLD --> V2["the forum D1 adapter becomes a view - the only one of the three that<br/>debits atomically today, and consensus ordering replaces that atomicity,<br/>a change in MECHANISM not a loss of the property<br/>ADR-2099-the-chain-is-the-ledger-of-record.md:39-43"]
    FOLD --> V3["the host file payment store and its pay handler are DELETED and<br/>replaced by a proxy to the wallet<br/>ADR-2099-the-chain-is-the-ledger-of-record.md:43-44"]
    subgraph rule["The rule that makes a view safe"]
        direction TB
        R1["PROPOSED INVARIANT: no code path may spend, credit, debit or gate on a<br/>view without resolving to the chain<br/>ADR-2099-the-chain-is-the-ledger-of-record.md:45"]
        R2["PROPOSED: a fold is not a thing you can spend from - every one of the<br/>three ledgers becomes a read-through view, or it stops being consulted<br/>sovereign-settlement-domain.md:35-39"]
        R3["PROPOSED vocabulary: until the confirmer is green against the node the<br/>words are ANCHOR, PEG and CLAIM, never SEAL<br/>GOVERNANCE-capabilities.md:541-543"]
        R1 ~~~ R2 ~~~ R3
    end
    V1 --> rule
    V2 --> rule
    V3 --> rule
```

## AB-31.7 PROPOSED - key separation and the account binding

```mermaid
flowchart TB
    KID["k_id, the sovereign identity key<br/>INGRESS-identity.md:407 - signs identity events and the binding.<br/>NEVER spends, NEVER seals a block"]
    KID -->|"derive_subkey with a sidestr/spend domain"| KSP["k_spend(chain)<br/>INGRESS-identity.md:408 - spends UTXOs on EXACTLY that chain"]
    KID -->|"derive_subkey with a sidestr/sign domain"| KSG["k_sign(chain)<br/>INGRESS-identity.md:409 - federated instance operators only,<br/>seals blocks on EXACTLY that chain"]
    KSP --> BIND["PROPOSED kind 38110 sidestr-account-binding<br/>addressable, d is chain id plus did hex, content is the derived<br/>spend pubkey, signed by k_id<br/>PROTOCOL-registry.md:143"]
    BIND --> REG["PROPOSED: allocated from the free 38106 to 38201 range inside the<br/>agentbox-owned block, so NOTHING outside this repo moves for it<br/>PROTOCOL-registry.md:143"]
    subgraph external["Sibling records, asserted by this repo about others"]
        direction TB
        E1["EXTERNAL: solid-pod-rs ADR-2008 - the rust-bitcoin port and the<br/>ledger becoming a view - sovereign-settlement-domain.md:8"]
        E2["EXTERNAL: VisionFlow host ADR-2111 - the file payment store retired<br/>sovereign-settlement-domain.md:8"]
        E3["EXTERNAL: nostr-rust-forum ADR-2012 - the D1 ledger becomes a view,<br/>derive_subkey as the domain-separation primitive<br/>sovereign-settlement-domain.md:8"]
        E4["EXTERNAL: VisionFlow canon ADR-2012 - canon alignment<br/>sovereign-settlement-domain.md:8"]
        E1 ~~~ E2 ~~~ E3 ~~~ E4
    end
    KSG --> external
```

**Invariant (proposed):** the identity key never spends and never seals, which is what keeps a compromised wallet from being a compromised identity (`../project/agentbox/docs/INGRESS-identity.md:407`).

## AB-31.8 PROPOSED - the Nostr kind plane and two URN kinds

```mermaid
flowchart TB
    subgraph ext["EXTERNAL kinds, owned by the upstream sidestr spec"]
        K1["23500 transaction, throwaway key per event<br/>PROTOCOL-registry.md:136"]
        K2["23501 faucet, testnet only, compiled out for mainnet variants<br/>PROTOCOL-registry.md:137"]
        K3["23510 to 23514 level-2 signing round, signer instances only<br/>PROTOCOL-registry.md:138"]
        K4["33333 chain tip<br/>PROTOCOL-registry.md:139"]
        K5["33500 rule document and 33501 genesis - NO upstream wire example,<br/>our codec is conformant to SPEC prose ONLY<br/>PROTOCOL-registry.md:140"]
        K6["33502 DUAL-SCHEMA peg record or desk pledge - the decoder returns<br/>PegRecord, Pledge or Ambiguous and NEVER guesses<br/>PROTOCOL-registry.md:142"]
    end
    subgraph ours["agentbox-owned"]
        K7["38110 sidestr-account-binding - see AB-31.7<br/>PROTOCOL-registry.md:143"]
    end
    subgraph urns["PROPOSED URN kinds, minted only through uris.js"]
        U1["chain - no owner scope, not content-addressed, the id IS the chain<br/>name upstream uses - PROTOCOL-registry.md:164, PROTOCOL-registry.md:167"]
        U2["asset - owner-scoped to the issuer, content-addressed over the origin<br/>contract id, on the knowledge-kind precedent<br/>PROTOCOL-registry.md:165"]
    end
    ext --> WARN["PROPOSED: the EXTERNAL classification is load-bearing, not a<br/>formality - the spec says its kinds and document shapes are<br/>provisional and we do not control their evolution<br/>PROTOCOL-registry.md:145-149"]
    ours --> WARN
    urns --> MINT["every durable identifier is minted through the existing table<br/>agentbox/management-api/lib/uris.js:87"]
    WARN --> ACC["PROPOSED: acceptance is OPEN - this allocation is not fixture-backed<br/>PROTOCOL-registry.md:151"]
```

**Open:** the kind allocation is recorded but not fixture-backed, so nothing yet proves a decoder round-trips an upstream event (`../project/agentbox/docs/PROTOCOL-registry.md:151-152`).

## AB-31.9 PROPOSED - the five phases, and the one that touches real value

```mermaid
stateDiagram-v2
    [*] --> P0
    P0: P0 Foundation - the record pack, two clean-room crates,<br/>the solid-pod-rs rust-bitcoin port, the manifest block
    note right of P0
        Exit evidence: validation against a live upstream chain,
        golden fixtures, cargo doc clean, the upstream proposal
        filed. sovereign-settlement.md:329
    end note
    P0 --> P1
    P1: P1 Root chain - mint the root chain, supervise the node and the<br/>JS producer, mirror on loopback, add the URN kinds and the wallet
    note right of P1
        Exit evidence: genesis validated from cold by both the node
        and an independent explorer, a peg-in claimed and spendable,
        a peg-out paid on the parent, and the RuVector recall gate
        still in band. sovereign-settlement.md:330
    end note
    P1 --> P2
    P2: P2 Chain is truth - the pay402 scheme, the authority wiring,<br/>the durable budget, the ledgers become views, the txo seam opens
    note right of P2
        Exit evidence: agent A pays agent B a thousand test sats
        through a 402 with a receipt citing the chain transaction, a
        spend above threshold blocks on a signed decision and
        journals a deny, and three-ledger equality holds on a
        hundred random principals. sovereign-settlement.md:331
    end note
    P2 --> P3
    P2 --> P4
    P3: P3 Child chains and the Rust producer - session-bound child<br/>chains, nested-parent validation, principal collapse
    note right of P3
        Fifty agents under one principal count as one voice.
        sovereign-settlement.md:332
    end note
    P4: P4 Bridge and the mainnet gate - the only phase that touches<br/>real value, and it cannot start until the gate exists AS CODE
    note right of P4
        Strict order P0 then P1 then P2, and only then P3 and P4 in
        either order. P0 to P3 run on the configured testnet parent.
        sovereign-settlement.md:324-325, sovereign-settlement.md:333
    end note
    P3 --> [*]
    P4 --> [*]
```

**Invariant (proposed):** the mainnet phase cannot begin until the regulatory gate exists as code rather than as an assertion, which is the correction of a prior record that called the same containment "architecturally enforced" while it was not built (`../project/agentbox/docs/proposals/sovereign-settlement.md:324-325`, `../project/agentbox/docs/proposals/sovereign-settlement.md:60`).

**Open:** PRD-024 lists its own outstanding questions and ranked risks (`../project/agentbox/docs/proposals/sovereign-settlement.md:420`, `../project/agentbox/docs/proposals/sovereign-settlement.md:456`); none is answered by anything in this repository today.

**Drift (this topic vs the repository since 1639f86ab):** the account binding moved from kind `38110` to `38420` (AB-34.4), `crates/sidestr/` now exists with four published crates (AB-33.1), and the root chain named here as proposed has been sealed (AB-32.1) — AB-31.1's and AB-31.2's status notes are true only at this topic's declared revision.
