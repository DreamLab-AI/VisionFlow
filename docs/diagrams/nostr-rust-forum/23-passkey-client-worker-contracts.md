---
id: NF-23
title: Passkey client-worker contracts, PRF inputs and validation limits
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
  - ../nostr-rust-forum/docs/adr/ADR-2003-derive-subkey-raw-hmac-js-parity.md
adrs: [ADR-2003]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/schema.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/webauthn.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/http.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/passkey.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs
verified_commit: {nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df}
---

## For developers

NF-23.1/23.2 have a [scoped source-repair successor, NF-29](29-external-review-repairs.md#nf-291-registration-keeps-the-issued-salt-and-proves-the-derived-identity). NF-23.1 and NF-23.2 are marked resolved at the re-verified stamp 6064622. Other discovery, recovery and authenticator boundaries below are not restamped or closed.

[The passkey review](../../estate-review/2026-11-01-forum-passkey-contracts.md) adds 10 audit cases to the actual worker's 65 passing native WebAuthn tests, plus four isolated SQLite scenarios. No live Worker, browser ceremony or authenticator ran. Request/header findings are source composition; the salt probe holds PRF output constant to isolate downstream derivation drift.

## For the business

The acceptance journey is creation, logout, return to the same public identity and one authorised forum action. A functioning cryptographic primitive does not prove that the browser and server agree on the inputs needed to recover that identity.

## NF-23.1 Both registration clients omit the required authorisation header

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs:181
    A["RESOLVED at nostr-rust-forum 6064622: forum registration verify<br/>uses fetch_with_nip98_post"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/http.rs:156
    B["Request is signed with the derived key<br/>and carries a NIP-98 Authorization"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/passkey.rs:356
    C["BBS registration verify<br/>also uses fetch_with_nip98_post"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/lib.rs:259
    D["Router dispatches register_verify"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:893
    E["Missing Authorization returns 401<br/>before attestation validation"]
    A --> B --> D --> E
    C --> D
```

The earlier caller/handler mismatch is resolved at 6064622: both clients sign register/verify, and the worker's matching-author proof is retained (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:905-906`). This is not a live request trace.

## NF-23.2 Issued and stored salts do not describe the same identity

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:783
    A["Options generates random salt A"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/webauthn.rs:65
    B["Client decodes A for PRF input"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs:153
    C["Client derives identity using PRF output<br/>and encoded A as HKDF salt"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1077
    D["RESOLVED at nostr-rust-forum 6064622: verify stores A, the issued salt<br/>read back with the challenge"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1186
    E["Login options returns the stored salt A"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs:274
    F["Login derives with A<br/>same-identity guarantee holds"]
    A --> B --> C
    C -. "if verification is reached" .-> D --> E --> F
```

At the earlier stamp native controls proved different roots for A and B even with identical PRF output; at 6064622 the stored salt is the issued one, which the native test `registration_and_login_use_the_same_server_issued_prf_input` asserts (`../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:723`). No authenticator PRF was emulated and no salt was changed in production.

## NF-23.3 Empty credential lists do not close every discovery path

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1161
    A["Login options always returns<br/>empty allowCredentials"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1179
    B["Unknown pubkey gets deterministic fallback salt"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:78
    C["Missing server secret defaults to empty<br/>fallback becomes publicly reproducible"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/lib.rs:275
    D["Separate public POST /auth/lookup"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1503
    E["Known credential returns owner<br/>unknown credential returns 404"]
    A --> B
    C --> B
    D --> E
```

The common IP rate limiter still applies. The lookup finding concerns supplied credential IDs, not guessing arbitrary high-entropy IDs. The missing-secret condition was tested with synthetic values; deployed configuration is unknown.

## NF-23.4 Capability and transport heuristics are not identity checks

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs:334
    A["Creation guard inspects function text<br/>rejects missing native-code marker"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/webauthn.rs:123
    B["Creation must yield PRF results.first<br/>enabled-only result is rejected"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/webauthn.rs:205
    C["Login checks cross-platform attachment"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/webauthn.rs:220
    D["Tries response.getTransports<br/>call failure permits continuation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs:293
    E["Verification body uses derived pubkey<br/>no prior requested-identity equality check"]
    A --> B
    C --> D --> E
```

Creation-time PRF evaluation is optional in the specification. `getTransports` belongs to attestation responses, not standard assertion responses. These source/spec discrepancies require tested compatibility and identity checks; no hardware support claim follows from this diagram.

## NF-23.5 Point validation and attestation parsing have narrower guarantees

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:606
    A["Parse fmt and authData<br/>skip other attestation fields"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:991
    B["Handler accepts none or packed string<br/>no packed-statement signature verification"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:304
    C["COSE parser extracts x and y<br/>other metadata skipped, last duplicate wins"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:297
    D["P-256 library validates point"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:518
    E["Unknown byte/text value skip trusts length<br/>does not check remaining payload"]
    A --> B --> C --> D
    A -. "unknown field" .-> E
```

Native probes accept contradictory COSE metadata and a truncated final unknown field; an invalid point and truncated known authData are rejected. This is not a demonstrated bypass of NIP-98 ownership or login signature verification. Packed attestation needs an explicit verification policy or rejection.

## NF-23.6 Challenge reads and consumption are separate operations

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/schema.rs:33
    A["Bootstrap credential table<br/>no owner or credential uniqueness constraint"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1351
    B["Read challenge age and owner bucket"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1407
    C["Check counter unless new count is zero<br/>then verify assertion signature"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1464
    D["Batch unconditional counter update<br/>and challenge deletion"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1469
    E["Batch completion accepted<br/>deleted-row count not checked"]
    A -. "schema context" .-> B --> C --> D --> E
```

SQLite permits duplicate owner rows under this bootstrap and two batches after a chosen two-reader schedule. Serial reuse correctly finds no challenge. D1 scheduling, deployed constraints and NIP-98 replay behaviour were not executed; both modelled requests assume independently valid tokens.
