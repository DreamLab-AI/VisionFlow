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
verified_commit: a28a74832dd0a6e47a531e34e4adc46a3946dfd3
---

## For developers

The library and resource server are separate admission paths. SP-05 draws the whole identity surface; these panels record the redirect repair and the still-missing resource dispatch of OIDC tokens.

## For the business

The producer refuses redirected discovery and key documents (released in 0.5.0-alpha.19). OIDC components still provide no external-client private-resource journey on the main pod server, and its 401 no longer advertises one. Consumer upgrades and deployment are not re-checked here.

## SP-12.1 Refuse unapproved redirect destinations

```mermaid
flowchart TD
    ISS["fetch issuer discovery<br/>oidc/jwks.rs:224"]
    CHECK["resolve and check issuer destination<br/>oidc/jwks.rs:230"]
    CLIENT["build client with approved address,<br/>never through a proxy, no automatic redirects<br/>oidc/jwks.rs:201<br/>oidc/jwks.rs:214<br/>oidc/jwks.rs:215<br/>oidc/jwks.rs:216"]
    DOC["require successful direct response<br/>validate returned issuer<br/>oidc/jwks.rs:269<br/>oidc/jwks.rs:283"]
    KEYS["check returned jwks_uri separately<br/>oidc/jwks.rs:312"]
    DIRECT["same guarded client for JWKS fetch<br/>oidc/jwks.rs:328"]
    ISS --> CHECK --> CLIENT --> DOC --> KEYS --> DIRECT
```

Direct-fetch success and both denied-host redirect regressions are retained in the [assessment](../../estate-review/2026-11-01-solid-oidc-boundaries.md). The pinned client sets `no_proxy`, so an inherited proxy cannot re-route the approved address.

## SP-12.2 Library verification does not wire resource authentication

```mermaid
flowchart TD
    PROOF["library verifies DPoP proof<br/>optional ath and replay checks<br/>oidc/mod.rs:406"]
    TOKEN["library checks token signature,<br/>issuer, audience, time and key binding<br/>oidc/mod.rs:929"]
    PROFILE["caller must check WebID issuer authority<br/>oidc/mod.rs:1094"]
    GET["actual private GET<br/>extract identity then enforce read access<br/>lib.rs:1935<br/>lib.rs:1942"]
    AUTH["resource extractor handles Nostr<br/>production Bearer refused, no DPoP dispatch<br/>lib.rs:919<br/>lib.rs:927<br/>lib.rs:933"]
    DENY["unauthenticated request denied<br/>challenge names only Nostr<br/>(plus Bearer under dev-bearer), no DPoP<br/>lib.rs:1572<br/>lib.rs:1592<br/>lib.rs:1553<br/>lib.rs:1558"]
    PROOF --> TOKEN --> PROFILE
    GET --> AUTH -->|no identity| DENY
```

The disposable route probe verifies its ES256 token/proof in the library, then observes HTTP 401; the Nostr positive control reads the same private bytes. This is an integration finding, not an authorisation bypass claim.
