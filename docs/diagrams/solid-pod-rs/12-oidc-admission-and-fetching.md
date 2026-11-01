---
id: SP-12
title: OIDC consumer admission and guarded document fetching
area: solid-pod-rs
governing: [../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md]
adrs: [ADR-2003]
sources:
  - ../solid-pod-rs/crates/solid-pod-rs/src/oidc/mod.rs
  - ../solid-pod-rs/crates/solid-pod-rs/src/oidc/jwks.rs
  - ../solid-pod-rs/crates/solid-pod-rs-server/src/lib.rs
  - ../solid-pod-rs/crates/solid-pod-rs/tests/oidc_jwks_ssrf.rs
verified_commit: fcc624a3e952e05923d93d7ea038294bcddb25ae
---

## For developers

The library and resource server are separate admission paths. Historical SP-05 records the older verifier/fetcher revision; these panels record the merged redirect repair and the remaining missing resource dispatch.

## For the business

The producer now refuses redirected discovery and key documents. OIDC components do not yet provide an external-client private-resource journey on the main pod server. Publication, consumer upgrades and deployment remain open.

## SP-12.1 Refuse unapproved redirect destinations

```mermaid
flowchart TD
    ISS["fetch issuer discovery<br/>oidc/jwks.rs:222"]
    CHECK["resolve and check issuer destination<br/>oidc/jwks.rs:228"]
    CLIENT["build client with approved address<br/>disable automatic redirects<br/>oidc/jwks.rs:213<br/>oidc/jwks.rs:214"]
    DOC["require successful direct response<br/>validate returned issuer<br/>oidc/jwks.rs:266<br/>oidc/jwks.rs:281"]
    KEYS["check returned jwks_uri separately<br/>oidc/jwks.rs:310"]
    DIRECT["same guarded client for JWKS fetch<br/>oidc/jwks.rs:326"]
    ISS --> CHECK --> CLIENT --> DOC --> KEYS --> DIRECT
```

Direct-fetch success and both denied-host redirect regressions are retained in the [assessment](../../estate-review/2026-11-01-solid-oidc-boundaries.md). Direct address pinning does not establish an inherited proxy's routing policy.

## SP-12.2 Library verification does not wire resource authentication

```mermaid
flowchart TD
    PROOF["library verifies DPoP proof<br/>optional ath and replay checks<br/>oidc/mod.rs:406"]
    TOKEN["library checks token signature,<br/>issuer, audience, time and key binding<br/>oidc/mod.rs:836"]
    PROFILE["caller must check WebID issuer authority<br/>oidc/mod.rs:984"]
    GET["actual private GET<br/>extract identity then enforce read access<br/>lib.rs:1478<br/>lib.rs:1485"]
    AUTH["resource extractor handles Nostr<br/>production Bearer refused, no DPoP dispatch<br/>lib.rs:603<br/>lib.rs:635"]
    DENY["unauthenticated request denied<br/>challenge nevertheless advertises DPoP<br/>lib.rs:1140<br/>lib.rs:1160"]
    PROOF --> TOKEN --> PROFILE
    GET --> AUTH -->|no identity| DENY
```

The disposable route probe verifies its ES256 token/proof in the library, then observes HTTP 401; the Nostr positive control reads the same private bytes. This is an integration finding, not an authorisation bypass claim.
