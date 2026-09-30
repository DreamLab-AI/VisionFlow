---
id: AB-31
title: Sovereign settlement over sidestr sidechains — the PRD-024 design and its eight decisions
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/docs/PROTOCOL-registry.md
adrs: [ADR-2096, ADR-2097, ADR-2098, ADR-2099, ADR-2100, ADR-2101, ADR-2102, ADR-2103, ADR-2105]
sources:
  - ../project/agentbox/docs/proposals/sovereign-settlement.md
  - ../project/agentbox/docs/proposals/sovereign-settlement-domain.md
  - ../project/agentbox/docs/proposals/sovereign-settlement-research/README.md
  - ../project/agentbox/docs/adr/ADR-2096-sidestr-sidechains-are-the-sole-value-instrument.md
  - ../project/agentbox/docs/adr/ADR-2098-chain-and-asset-urn-kinds-and-the-chain-nostr-plane.md
  - ../project/agentbox/docs/adr/ADR-2099-the-chain-is-the-ledger-of-record.md
  - ../project/agentbox/docs/adr/ADR-2100-every-settlement-passes-the-authority-gate.md
  - ../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md
  - ../project/agentbox/docs/adr/ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/docs/PROTOCOL-registry.md
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/management-api/routes/payments.js
  - ../project/agentbox/management-api/routes/broker-bridge.js
  - ../project/agentbox/management-api/routes/llm-marketplace.js
  - ../project/agentbox/management-api/lib/uris.js
  - ../project/agentbox/services/nostr-pod-bridge/src/contract.rs
  - ../project/agentbox/docs/adr/ADR-2105-agentbox-kind-bands-and-the-colloquy-move.md
verified_commit: 6a4ad132f2dc5ddaedd05c679fdd10066bf30a0f
---

## For developers

This topic is the PRD-024 design, read at `1639f86ab` when nothing of it was running: every one of the eight decisions was minted `proposed` with `implementation_status: none` (all eight still carry `decision_status: proposed`; ADR-2096 and ADR-2103 have since moved to `implementation_status: partial` as P0/P1 landed — ADR-2096-sidestr-sidechains-are-the-sole-value-instrument.md:5-6, ADR-2103-parent-chain-and-header-profile-are-configuration-behind-the-p21-gate.md:5-6), carried by the four governing documents in explicitly marked PROPOSED sections that join the compliance surface only on ratification (GOVERNANCE-capabilities.md:503-505). Read every node and Note below as a design that cites the record proposing it.
Part of it has since been built. AB-32 is the sealed root chain, AB-33 the four published crates, AB-34 what is actually running, AB-35 the reviewed level-2 shape — each stamped at `ec60a8f14`. Where this topic and those disagree, they are the current claim and this one is the design it came from.

**Drift (this topic vs ADR-2112):** "AB-33 the four published crates" is superseded — on 2026-09-23 ADR-2112 moved the crates (now five) to `DreamLab-AI/sidestr-rs`; agentbox hosts the chain instance only. See SR-01.

## For the business

The estate can already charge for work and already refuses work it has not been paid for, but the balance it debits is a JSON document rather than value anywhere (sovereign-settlement.md:51-55). The proposal is to run the estate's own settlement chains so that a balance becomes something a third party can verify, with every payment above a threshold stopping for a human signature first. It is a costed plan in five ordered phases, only the last of which touches real money, and it has not been approved; the first two phases have since partly begun (AB-32, AB-34).

## AB-31.1 PROPOSED - the target architecture as PRD-024 draws it

```mermaid
flowchart TB
    subgraph parent["PROPOSED configured parent - sovereign-settlement.md:132-134"]
        BTC["Bitcoin Core node, testnet4-blake2b default,<br/>mainnet variants gated - sovereign-settlement.md:133"]
    end
    subgraph root["PROPOSED root chain sidestr:dreamlab - sovereign-settlement.md:135-139"]
        RC["chain document: parent, headerProfile, signers, threshold,<br/>currencyPin, cashOut, p21Receipt - sovereign-settlement.md:136"]
        PROD["producer: the upstream JS sidecar at P0 to P2,<br/>the Rust producer at P3 - sovereign-settlement.md:137"]
        NODE["sidestr-node validator and mirror, loopback port 9097,<br/>LAN only through the nip98-proxy chain upstream<br/>sovereign-settlement.md:138"]
    end
    subgraph kids["PROPOSED ephemeral child chains - sovereign-settlement.md:140-142"]
        C1["one per AoE session, bound at session create,<br/>closed when the session closes - sovereign-settlement.md:141"]
    end
    subgraph br["PROPOSED bridge, an isolated process - sovereign-settlement.md:143-145"]
        BR["RGB consignment in becomes a WRAPPED asset claim,<br/>burn becomes a consignment out - sovereign-settlement.md:144"]
    end
    subgraph api["PROPOSED management-api surfaces - sovereign-settlement.md:146-151"]
        W["/v1/wallet/* under NIP-98, spend key selected by DID<br/>sovereign-settlement.md:147"]
        G["lib/authority.js under payment_settlement<br/>sovereign-settlement.md:149"]
        U["lib/uris.js gains the chain and asset kinds<br/>sovereign-settlement.md:150"]
    end
    subgraph views["PROPOSED derived views, never authoritative - sovereign-settlement.md:152-156"]
        V1["solid-pod-rs WebLedger view - sovereign-settlement.md:153"]
        V2["forum D1 view - sovereign-settlement.md:154"]
        V3["VisionClaw file store deleted, proxied to the wallet<br/>sovereign-settlement.md:155"]
    end
    parent -->|"peg-in, peg-out, checkpoint"| RC
    RC --> PROD --> NODE
    RC -->|"nested parent"| C1
    BR --> PROD
    W --> G --> NODE
    NODE --> views
    U -.-> W
    subgraph note["Status"]
        direction TB
        N1["PROPOSED: no sidechain gate, no sidestr-node, sidestr-producer or<br/>sidestr-bridge program and no loopback port 9097 bind exists today.<br/>The Rust crates are published from sidestr-rs (ADR-2112), not linked<br/>into the image - BASELINE-container.md:196"]
        N2["PROPOSED: this document is scope, not authority. The eight ADRs are the<br/>decisions and the governing documents are the compliance surface<br/>sovereign-settlement.md:21-23"]
        N3["the eight decisions land as DDD-022, bounded context BC25<br/>sovereign-settlement-domain.md:5"]
        N1 ~~~ N2 ~~~ N3
    end
```

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
        citation, both fixed. sovereign-settlement.md:345-346
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
        GOVERNANCE-capabilities.md:503-505
    end note
    Candidate --> Ratified : the owner ratifies PRD-024
    Candidate --> Withdrawn : the owner does not
    Ratified: the invariants join the Invariants compliance surface VERBATIM
    Ratified --> [*]
    Withdrawn --> [*]
```

**Invariant:** a proposed section states its own non-authority in the document that carries it, so a reader who lands on it by search cannot mistake it for a live rule (`../project/agentbox/docs/BASELINE-container.md:196`, `../project/agentbox/docs/GOVERNANCE-capabilities.md:503-505`).

## AB-31.5 PROPOSED - a settlement through the authority gate

```mermaid
sequenceDiagram
    autonumber
    participant AG as an agent asking to spend
    participant WAL as PROPOSED /v1/wallet/*<br/>agentbox/docs/proposals/sovereign-settlement.md:147
    participant POL as spend-policy, the only authoriser<br/>agentbox/docs/adr/ADR-2100-every-settlement-passes-the-authority-gate.md:41
    participant GATE as PROPOSED lib/authority.js under payment_settlement<br/>agentbox/docs/GOVERNANCE-capabilities.md:507-508
    participant HUM as the approving human
    participant NODE as PROPOSED sidestr-node<br/>agentbox/docs/BASELINE-container.md:335

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
        Note over GATE: PROPOSED ADR-2100 D4: every outcome mints a receipt, INCLUDING<br/>denied and failed (GOVERNANCE-capabilities.md:528)
    else approved
        GATE->>NODE: the spend reaches the chain
        NODE-->>AG: a receipt citing the chain transaction
    end
    Note over GATE: PROPOSED ADR-2100 D1: no new governance mechanism is added, the<br/>EXISTING one is called - /v1/wallet/* and /v1/chain/* from their first<br/>commit, retrofitted onto /v1/pay/*<br/>(GOVERNANCE-capabilities.md:507-515)
    Note over NODE: PROPOSED ADR-2100 D3: settlement FAILS CLOSED - the cost gate is<br/>forced closed on any chain-settling path, and a stopped node is refused<br/>rather than waved through (GOVERNANCE-capabilities.md:524, GOVERNANCE-capabilities.md:565)
```

**Invariant (proposed):** spend authorisation counts authorising principals and never accounts, reusing the colloquy rule, so an operator's fifty agents are one voice (`../project/agentbox/docs/GOVERNANCE-capabilities.md:531-533`).

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
    rule --> DEBT["Debt: the blocktrail txo field is constructed EMPTY and never<br/>populated - a seam reserved years before anything could fill it<br/>agentbox/services/nostr-pod-bridge/src/contract.rs:139"]
```

## AB-31.7 PROPOSED - key separation and the account binding

```mermaid
flowchart TB
    KID["k_id, the sovereign identity key<br/>INGRESS-identity.md:431 - signs identity events and the binding.<br/>NEVER spends, NEVER seals a block"]
    KID -->|"derive_subkey with a sidestr/spend domain"| KSP["k_spend(chain)<br/>INGRESS-identity.md:432 - spends UTXOs on EXACTLY that chain"]
    KID -->|"derive_subkey with a sidestr/sign domain"| KSG["k_sign(chain)<br/>INGRESS-identity.md:433 - federated instance operators only,<br/>seals blocks on EXACTLY that chain"]
    KSP --> BIND["PROPOSED kind 38420 sidestr-account-binding (ADR-2105 moved it off<br/>38110, which sat inside the agent-response reservation)<br/>addressable, d is chain id plus did hex, content is the derived<br/>spend pubkey, signed by k_id<br/>PROTOCOL-registry.md:170"]
    BIND --> REG["PROPOSED: allocated from the agentbox band 38400 to 38499, the<br/>first hundred no record reserves, so NOTHING outside this repo<br/>moves for it - PROTOCOL-registry.md:92, PROTOCOL-registry.md:170"]
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

**Invariant (proposed):** the identity key never spends and never seals, which is what keeps a compromised wallet from being a compromised identity (`../project/agentbox/docs/INGRESS-identity.md:431`).

## AB-31.8 PROPOSED - the Nostr kind plane and two URN kinds

```mermaid
flowchart TB
    subgraph ext["EXTERNAL kinds, owned by the upstream sidestr spec"]
        K1["23500 transaction, throwaway key per event<br/>PROTOCOL-registry.md:163"]
        K2["23501 faucet, testnet only, compiled out for mainnet variants<br/>PROTOCOL-registry.md:164"]
        K3["23510 to 23514 level-2 signing round, signer instances only<br/>PROTOCOL-registry.md:165"]
        K4["33333 chain tip<br/>PROTOCOL-registry.md:166"]
        K5["33500 rule document and 33501 genesis - NO upstream wire example,<br/>our codec is conformant to SPEC prose ONLY<br/>PROTOCOL-registry.md:167-168"]
        K6["33502 DUAL-SCHEMA peg record or desk pledge - the decoder returns<br/>PegRecord, Pledge or Ambiguous and NEVER guesses<br/>PROTOCOL-registry.md:169"]
    end
    subgraph ours["agentbox-owned, band 38400-38499 (ADR-2105) - see AB-31.7"]
        K7["38420 sidestr-account-binding, moved off 38110<br/>PROTOCOL-registry.md:170"]
        subgraph domevents["settlement domain events, DDD-022"]
            direction TB
            K8["38421 PegOutDefaulted"]
            K9["38422 ChildChainOpened"]
            K10["38423 ChildChainClosing"]
            K11["38424 ChainTombstoned"]
            K12["38425 SettlementRecorded<br/>PROTOCOL-registry.md:171"]
            K8 ~~~ K9 ~~~ K10 ~~~ K11 ~~~ K12
        end
    end
    subgraph urns["PROPOSED URN kinds, minted only through uris.js"]
        U1["chain - no owner scope, not content-addressed, the id IS the chain<br/>name upstream uses - PROTOCOL-registry.md:192, PROTOCOL-registry.md:198"]
        U2["asset - owner-scoped to the issuer, content-addressed over the origin<br/>contract id, on the knowledge-kind precedent<br/>PROTOCOL-registry.md:193"]
    end
    ext --> WARN["PROPOSED: the EXTERNAL classification is load-bearing, not a<br/>formality - the spec says its kinds and document shapes are<br/>provisional and we do not control their evolution<br/>PROTOCOL-registry.md:173-177"]
    ours --> WARN
    urns --> MINT["every durable identifier is minted through the existing table<br/>agentbox/management-api/lib/uris.js:87"]
    WARN --> ACC["PROPOSED: acceptance is OPEN - this allocation is not fixture-backed<br/>PROTOCOL-registry.md:179"]
```

**Open:** the kind allocation is recorded but not fixture-backed, so nothing yet proves a decoder round-trips an upstream event (`../project/agentbox/docs/PROTOCOL-registry.md:179-182`).

**Drift:** the account binding and the five domain events moved from a claimed-free `38106`-`38201` range (which overlapped the ADR-009 agent-response reservation) to the clean `38400`-`38499` band, and gained five concrete kind numbers where the design had left the domain events as a range (`../project/agentbox/docs/adr/ADR-2105-agentbox-kind-bands-and-the-colloquy-move.md:1-30`).

## AB-31.9 PROPOSED - the five phases, and the one that touches real value

```mermaid
stateDiagram-v2
    [*] --> P0
    P0: P0 Foundation - the record pack, two clean-room crates,<br/>the solid-pod-rs rust-bitcoin port, the manifest block
    note right of P0
        Exit evidence: validation against a live upstream chain,
        golden fixtures, cargo doc clean, the upstream proposal
        filed. sovereign-settlement.md:334
    end note
    P0 --> P1
    P1: P1 Root chain - mint the root chain, supervise the node and the<br/>JS producer, mirror on loopback, add the URN kinds and the wallet
    note right of P1
        Exit evidence: genesis validated from cold by both the node
        and an independent explorer, a peg-in claimed and spendable,
        a peg-out paid on the parent, and the RuVector recall gate
        still in band. sovereign-settlement.md:335
    end note
    P1 --> P2
    P2: P2 Chain is truth - the pay402 scheme, the authority wiring,<br/>the durable budget, the ledgers become views, the txo seam opens
    note right of P2
        Exit evidence: agent A pays agent B a thousand test sats
        through a 402 with a receipt citing the chain transaction, a
        spend above threshold blocks on a signed decision and
        journals a deny, and three-ledger equality holds on a
        hundred random principals. sovereign-settlement.md:337
    end note
    P2 --> P3
    P2 --> P4
    P3: P3 Child chains and the Rust producer - session-bound child<br/>chains, nested-parent validation, principal collapse
    note right of P3
        Fifty agents under one principal count as one voice.
        sovereign-settlement.md:338
    end note
    P4: P4 Bridge and the mainnet gate - the only phase that touches<br/>real value, and it cannot start until the gate exists AS CODE
    note right of P4
        Strict order P0 then P1 then P2, and only then P3 and P4 in
        either order. P0 to P3 run on the configured testnet parent.
        sovereign-settlement.md:329-330, sovereign-settlement.md:339
    end note
    P3 --> [*]
    P4 --> [*]
```

**Invariant (proposed):** the mainnet phase cannot begin until the regulatory gate exists as code rather than as an assertion, which is the correction of a prior record that called the same containment "architecturally enforced" while it was not built (`../project/agentbox/docs/proposals/sovereign-settlement.md:329-330`, `../project/agentbox/docs/proposals/sovereign-settlement.md:60`).

**Open:** PRD-024 lists its own outstanding questions and ranked risks (`../project/agentbox/docs/proposals/sovereign-settlement.md:462`, `../project/agentbox/docs/proposals/sovereign-settlement.md:426`); none is answered by anything in this repository today.

**Drift (this topic vs the repository since 1639f86ab):** the account binding moved from kind `38110` to `38420` (AB-34.4), `crates/sidestr/` now exists with four published crates (AB-33.7), and the root chain named here as proposed has been sealed (AB-32.1) — AB-31.1's and AB-31.6's status notes are true only at this topic's declared revision.
