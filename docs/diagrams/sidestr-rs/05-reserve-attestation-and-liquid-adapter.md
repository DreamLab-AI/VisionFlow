---
id: SR-05
title: Origin-neutral reserve attestations and the private Liquid adapter
area: sidestr-rs
governing:
  - ../sidestr-rs/docs/adr/ADR-0003-keep-reserve-attestations-origin-neutral-and-private.md
  - ../project/agentbox/docs/adr/ADR-2117-private-owner-usd-unit-bridged-through-rgb-into-sidestr.md
adrs: [ADR-2117]
sources:
  - ../sidestr-rs/sidestr-reserve/src/lib.rs
  - ../sidestr-rs/sidestr-bridge-liquid/src/lib.rs
  - ../sidestr-rs/docs/adr/ADR-0003-keep-reserve-attestations-origin-neutral-and-private.md
  - ../project/agentbox/docs/adr/ADR-2117-private-owner-usd-unit-bridged-through-rgb-into-sidestr.md
  - ../project/docs/TODO-unified.md
verified_commit: {sidestr-rs: 3aadeb7a26ff60c18614113bb986c1ba0d33d115, visionclaw: d5ecd38a2de3012e42509c27b950f6cd17e3fee4, agentbox: 5d5d083e2e77ea448d27e6e2dae95822eb960922}
---

## For developers

`sidestr-reserve` defines a deterministic statement about a reserve origin, its replay-keyed credits and the final origin tip. `sidestr-bridge-liquid` is one adapter that reads a Liquid wallet through LWK and produces that statement. Neither crate implements or activates the consuming `bridge` rule.

## For the business

Source can describe and sign what a private reserve wallet held at a specific origin-chain tip. It does not fund a reserve, issue a redeemable asset or put a bridge into service. Both crates remain unpublished and the format is a private project contract.

## SR-05.1 Adapter boundary

```mermaid
flowchart LR
    O["reserve origin"] --> AD["origin-specific adapter<br/>sidestr-bridge-liquid/src/lib.rs:21"]
    AD --> OR["Origin<br/>network, asset, decimals"]
    AD --> CR["final credits<br/>replay-keyed ids"]
    AD --> TIP["final origin tip"]
    OR --> AT["ReserveAttestation"]
    CR --> AT
    TIP --> AT
    AT -. "future consumer" .-> BR["bridge rule"]
```

**What it shows.** The neutral crate identifies an origin by network, asset and decimals and identifies each credit by an origin-appropriate replay key (`../sidestr-rs/sidestr-reserve/src/lib.rs:132`, `../sidestr-rs/sidestr-reserve/src/lib.rs:223`). A Liquid-specific adapter supplies the wallet, sync and registry interpretation (`../sidestr-rs/sidestr-bridge-liquid/src/lib.rs:14`).

**Why it is this way.** Another origin can provide a sibling adapter while a future bridge rule consumes one stable statement shape. The local ADR keeps origin dependencies isolated (`../sidestr-rs/docs/adr/ADR-0003-keep-reserve-attestations-origin-neutral-and-private.md:26`).

## SR-05.2 Deterministic statement and signature

```mermaid
flowchart LR
    IN["origin, tip, credits,<br/>source and time<br/>sidestr-reserve/src/lib.rs:277"] --> SORT["sort credit ids and<br/>refuse duplicates"]
    SORT --> JSON["canonical compact JSON"]
    JSON --> SHA["SHA-256 digest"]
    SHA --> SIG["BIP-340 signer hook"]
    SIG --> CHECK["verify before return"]
```

**What it shows.** Attestation construction sorts credits, refuses duplicate ids and totals amounts with checked arithmetic (`../sidestr-rs/sidestr-reserve/src/lib.rs:277`). Canonical JSON fixes key and credit order before SHA-256 and BIP-340 signing (`../sidestr-rs/sidestr-reserve/src/lib.rs:323`, `../sidestr-rs/sidestr-reserve/src/lib.rs:368`).

**Why it is this way.** Independent adapters and validators must reproduce exactly the bytes that were signed, and a credit must not support two mints.

## SR-05.3 Private, unpublished and inactive

```mermaid
stateDiagram-v2
    [*] --> Implemented
    Implemented: neutral attestation and Liquid adapter source
    Implemented --> Unpublished
    Unpublished: publish equals false
    Unpublished --> Unfunded
    Unfunded: no reserve value and no live bridge rule
    Unfunded --> Gated: explicit estate decision and evidence required
    note right of Unpublished
        sidestr-bridge-liquid/src/lib.rs:73
    end note
    Gated --> [*]
```

**What it shows.** The adapter's crate documentation says it is private, unfunded, unpublished and not live (`../sidestr-rs/sidestr-bridge-liquid/src/lib.rs:5`, `../sidestr-rs/sidestr-bridge-liquid/src/lib.rs:73`). The local ADR states that neither crate activates a bridge rule, funds a reserve or deploys a chain (`../sidestr-rs/docs/adr/ADR-0003-keep-reserve-attestations-origin-neutral-and-private.md:32`).

**Why it is this way.** A signed data format is only one input to a custody system. Funding, issuance, supply validation and activation require separate estate evidence under ADR-2117.

**Open:** the master board keeps reserve custody and service-account isolation open; publishing source does not close those runtime gates (`../project/docs/TODO-unified.md:97`).
