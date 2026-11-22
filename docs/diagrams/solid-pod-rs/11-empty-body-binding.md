---
id: SP-11
title: Empty-body request commitments
area: solid-pod-rs
governing: [../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2006]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs/src/auth/nip98.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/tests/nip98_empty_body.rs
verified_commit: a28a74832dd0a6e47a531e34e4adc46a3946dfd3
---

## For developers

Every supplied body requires its payload hash, including empty bytes; explicit None skips binding. The minting helper agrees with the verifier. SP-05 draws the wider NIP-98 path.

## For the business

A signed private-document update cannot be reused to remove its content. Intentional empty writes require the empty-body hash. Released in 0.5.0-alpha.19; consumer adoption and deployment are not re-checked here.

## SP-11.1 Commit to the bytes before consuming the request

```mermaid
flowchart TD
    MINT["mint payload tag for supplied bytes<br/>including empty bytes<br/>auth/nip98.rs:308"]
    PUT["private resource PUT supplies raw body<br/>lib.rs:2193"]
    BODY["Some bytes: require matching payload hash<br/>auth/nip98.rs:195"]
    REJECT["missing tag or mismatched bytes: return error<br/>auth/nip98.rs:198<br/>auth/nip98.rs:201"]
    SIGN["verify signature and canonical event id<br/>auth/nip98.rs:216"]
    CLAIM["process-local replay admission<br/>lib.rs:957"]
    WAC["enforce write access<br/>lib.rs:2195"]
    WRITE["validate resource and persist bytes<br/>lib.rs:2222<br/>lib.rs:2229"]
    TEST["private PUT regression: bytes and token state<br/>tests/nip98_empty_body.rs:15"]
    TEST -.-> PUT
    MINT --> PUT --> BODY
    BODY -->|missing or mismatched| REJECT
    BODY -->|matching| SIGN
    SIGN -->|valid| CLAIM
    CLAIM -->|admitted| WAC
    WAC -->|authorised| WRITE
```

Binding rejection precedes replay admission. SP-ID-01 is repaired; the replay TTL is now floored at the freshness window (SP-ID-02, SP-05), while restart/replica scope and GitLenient scoping remain open. See the [repair evidence](../../estate-review/2026-11-01-solid-empty-body-repair.md).
