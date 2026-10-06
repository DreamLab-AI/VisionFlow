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
verified_commit: {sidestr-rs: a7aadd68d536167507e00e1ce6237fbecb5b46fa, visionclaw: af3dff3f25300cf12bceda5650688ec223270eca, agentbox: 9fd49a935611a4f1b3591030d52fe13610179397}
---

## For developers

`sidestr-reserve` defines a deterministic statement about a reserve origin, its replay-keyed credits and the final origin tip. Its canonical bytes are RFC 8785 (JCS) JSON, held to `serde_jcs` and to the teller's `jcs` by a cross-check suite, and the digest a key signs is a BIP-340-style tagged SHA-256. `sidestr-bridge-liquid` is one adapter that reads a Liquid wallet through LWK and produces that statement. Neither crate implements or activates the consuming `bridge` rule.

## For the business

Source can describe and sign what a private reserve wallet held at a specific origin-chain tip. It does not fund a reserve, issue a redeemable asset or put a bridge into service. Both crates remain unpublished and the format is a private project contract.

## SR-05.1 Adapter boundary

```mermaid
flowchart LR
    O["reserve origin"] --> AD["origin-specific adapter<br/>sidestr-bridge-liquid/src/lib.rs:25"]
    AD --> OR["Origin<br/>network, asset, decimals"]
    AD --> CR["final credits<br/>replay-keyed ids"]
    AD --> TIP["final origin tip"]
    OR --> AT["ReserveAttestation"]
    CR --> AT
    TIP --> AT
    AT -. "future consumer" .-> BR["bridge rule"]
```

**What it shows.** The neutral crate identifies an origin by network, asset and decimals and identifies each credit by an origin-appropriate replay key (`../sidestr-rs/sidestr-reserve/src/lib.rs:351`, `../sidestr-rs/sidestr-reserve/src/lib.rs:436`). A Liquid-specific adapter supplies the wallet, sync and registry interpretation (`../sidestr-rs/sidestr-bridge-liquid/src/lib.rs:17`).

**Why it is this way.** Another origin can provide a sibling adapter while a future bridge rule consumes one stable statement shape. The local ADR keeps origin dependencies isolated (`../sidestr-rs/docs/adr/ADR-0003-keep-reserve-attestations-origin-neutral-and-private.md:26`).

## SR-05.2 Deterministic statement and signature

```mermaid
flowchart LR
    IN["origin, tip, credits,<br/>source and time<br/>sidestr-reserve/src/lib.rs:499"] --> SORT["sort credit ids and<br/>refuse duplicates"]
    SORT --> JSON["RFC 8785 JCS canonical JSON<br/>sidestr-reserve/src/lib.rs:577"]
    JSON --> SHA["tagged SHA-256 digest<br/>sidestr-reserve/src/lib.rs:584"]
    SHA --> SIG["BIP-340 signer hook<br/>sidestr-reserve/src/lib.rs:611"]
    SIG --> CHECK["verify before return"]
```

**What it shows.** Attestation construction sorts credit ids through a set, refuses duplicates and totals amounts with checked arithmetic (`../sidestr-rs/sidestr-reserve/src/lib.rs:499`). Canonical form is RFC 8785 (JCS) JSON — keys ascending by byte order, no whitespace — fixed before a BIP-340-style tagged SHA-256 digest and BIP-340 signing (`../sidestr-rs/sidestr-reserve/src/lib.rs:577`, `../sidestr-rs/sidestr-reserve/src/lib.rs:584`, `../sidestr-rs/sidestr-reserve/src/lib.rs:611`). The bytes are held to `serde_jcs` and to the JCS of the solidpayorg teller `7c00cea` by `tests/jcs.rs` (`../sidestr-rs/sidestr-reserve/src/lib.rs:60`).

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
