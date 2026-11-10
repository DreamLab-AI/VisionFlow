---
id: NF-21
title: Subkey parity, consumer identity and rotation boundaries
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2003-derive-subkey-raw-hmac-js-parity.md
adrs: [ADR-2003, ADR-094, ADR-100]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/tests/identity_subkey_vectors.rs
  - ../nostr-rust-forum/scripts/identity-vector-parity.mjs
  - ../nostr-rust-forum/.github/workflows/ci.yml
  - ../nostr-rust-forum/docs/archive/adr/ADR-100-key-lifecycle.md
  - ../project/agentbox/config/hooks/nostr-live-mirror.cjs
  - ../project/agentbox/config/nostr-gateway/nostr-send.cjs
  - ../project/agentbox/config/nostr-gateway/gateway.cjs
verified_commit: {nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93}
---

## For developers

[Section review and adversarial evidence](../../estate-review/2026-11-01-forum-subkey-derivation.md) cover ADR-2003 and archived ADR-094/100. Thirty native tests, twelve extracted JavaScript probes and eight runner scenarios distinguish primitive parity from consumer selection and release enforcement. No production entrypoint or live key is used.

## For the business

A replacement key helps only when every component adopts it and the old key loses the relevant rights. Show the expected public identity and the rotation state; a passing derivation fixture alone cannot establish either.

## NF-21.1 Primitive equality has exact input and validation boundaries

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs:49
    A["Rust validates the root scalar"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs:254
    B["HMAC-SHA256 over exact UTF-8 tag"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs:263
    C["Rust validates child scalar<br/>no modular reduction"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/tests/identity_subkey_vectors.rs:151
    D["Rust checks shared vectors"]
    %% Source: ../nostr-rust-forum/scripts/identity-vector-parity.mjs:106
    E["Node separately mirrors primitive<br/>and checks the same fixture"]
    %% Source: ../project/agentbox/config/hooks/nostr-live-mirror.cjs:69
    F["Consumer envFirst trims tags<br/>before derivation"]
    A --> B --> C --> D
    E -. shared vectors .-> D
    F --> B
```

## NF-21.2 The parity runner does not execute production consumers

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/scripts/identity-vector-parity.mjs:198
    A["Try reading three agentbox source files"]
    %% Source: ../nostr-rust-forum/scripts/identity-vector-parity.mjs:200
    B["Read failure: successful skip"]
    %% Source: ../nostr-rust-forum/scripts/identity-vector-parity.mjs:204
    C["Readable: record hash<br/>no expected hash comparison"]
    %% Source: ../nostr-rust-forum/scripts/identity-vector-parity.mjs:205
    D["Search source substring<br/>comments can satisfy it"]
    %% Source: ../nostr-rust-forum/scripts/identity-vector-parity.mjs:224
    E["Any HKDF text fails<br/>including unrelated comments"]
    %% Source: ../nostr-rust-forum/.github/workflows/ci.yml:283
    F["CI checks out agentbox without explicit ref"]
    %% Source: ../nostr-rust-forum/.github/workflows/ci.yml:306
    G["Aggregate ci-pass omits drift-gates<br/>external rules are separate;<br/>see dated report"]
    A --> B
    A --> C --> D --> E
    F --> A
    F -. enforcement boundary .-> G
```

## NF-21.3 Sender and gateway disagree on identity modes

```mermaid
flowchart TB
    %% Source: ../project/agentbox/config/nostr-gateway/nostr-send.cjs:36
    A["AGENTBOX_GATEWAY_IDENTITY"]
    %% Source: ../project/agentbox/config/nostr-gateway/nostr-send.cjs:38
    B["Sender: child selects derived key<br/>default tag agentbox-mirror-v1"]
    %% Source: ../project/agentbox/config/nostr-gateway/nostr-send.cjs:42
    C["Sender: gateway and default select root"]
    %% Source: ../project/agentbox/config/nostr-gateway/gateway.cjs:167
    D["Gateway: gateway selects derived key<br/>default tag agentbox-gateway-v1"]
    %% Source: ../project/agentbox/config/nostr-gateway/gateway.cjs:169
    E["Gateway: child and default select root"]
    A --> B
    A --> C
    A --> D
    A --> E
```

## NF-21.4 Tag rotation is not consumer adoption or revocation

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/tests/identity_subkey_vectors.rs:266
    A["Same root and new tag<br/>produce a different child"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs:66
    B["Old secret remains able to sign"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs:254
    C["Old root and tag still reproduce old child"]
    %% Source: ../project/agentbox/config/hooks/nostr-live-mirror.cjs:213
    D["Warm mirror cache returns first result<br/>before rereading tag or root"]
    %% Source: ../nostr-rust-forum/docs/archive/adr/ADR-100-key-lifecycle.md:78
    E["Lifecycle intent: adopt replacement<br/>and withdraw old authority separately"]
    A -. does not disable .-> B
    C --> B
    D -. may retain .-> B
    A --> E
```
