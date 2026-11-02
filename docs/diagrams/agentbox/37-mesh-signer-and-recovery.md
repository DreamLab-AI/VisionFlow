---
id: AB-37
title: Proposed mesh signer and current recovery boundaries
area: agentbox
governing:
  - ../project/agentbox/docs/PROTOCOL-registry.md
adrs: [ADR-2162, ADR-2061, ADR-2101, ADR-2122, ADR-2027]
sources:
  - ../project/agentbox/docs/PROTOCOL-registry.md
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/schema/agentbox.toml.schema.json
  - ../project/agentbox/config/custody/identity-port-acl.json
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/port.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/server.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/receipts.rs
  - ../project/agentbox/services/secret-backup/src/main.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/config.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/envelope.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/transport.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/gift_wrap.rs
verified_commit: {agentbox: c57d9c0508f4044915f9aecea4b9aa5629cc6a35, nostr-rust-forum: c39f41f104b114d6a3fa29013daaf6fa58b45080}
---

## For developers

The [mesh audit](../../estate-review/2026-11-01-mesh-signer-and-custody.md) runs 65 isolated tests against current modules and library APIs. Proposed nodes describe requirements; no mesh service or production signer was activated.

## For the business

Sharing spaces across nodes still needs compatible configuration, limited signing authority and reliable recovery. A signed event or successful local socket call alone does not prove that a peer applied a change or removed an old link.

## AB-37.1 Current configuration and codec do not implement the proposed peer contract

```mermaid
flowchart TD
    MAN["Agentbox moderation is boolean<br/>agentbox.toml:282"] --> FAIL["Library honor_remote_moderation is a list of DIDs<br/>config.rs:67-69"]
    SCHEMA["Schema modes: standalone or client<br/>agentbox.toml.schema.json:2585,2586"]
    PEERS["PROPOSED pinned peer records and projection<br/>PROTOCOL-registry.md:253,254"] -.-> SCHEMA
    PEERS -.-> LIB["Current roster uses peer_relays<br/>transport.rs:642,643"]
    LIB --> CACHE["SeenIds is a bounded memory cache<br/>transport.rs:584,595,605"]
    TYPES["Current envelope has six variants<br/>envelope.rs:81,83,85,87,89,91,93"]
```

**What it shows.** Direct configuration reuse fails; current envelope parsing rejects all three proposed control types. **Why it is this way.** The proposal requires schema, codec and service integration beyond the existing library's relay roster and loop-prevention cache.

## AB-37.2 Current identity-port policy stops at account, operation, key and kind

```mermaid
flowchart TD
    SOCKET["Kernel peer credentials<br/>server.rs:106"] --> UID["Authorise uid and named operation<br/>port.rs:361,362,378"]
    UID --> GRANT["Kind must be in grant<br/>port.rs:534-536"]
    GRANT --> SIGN["Caller tags, content and timestamp passed through<br/>port.rs:559,562,563,564"]
    SIGN --> RECEIPT["Append receipt before returning result<br/>port.rs:320,326,328<br/>receipts.rs:132,136"]
    CLOSED["Checked-in grants omit kind 30078<br/>identity-port-acl.json:39,40"]
    REQUIRED["PROPOSED typed catalogue and control signer<br/>independent key, VisionClaw-only client<br/>PROTOCOL-registry.md:255"]
    SIGN -.-> REQUIRED
```

**What it shows.** A synthetic 30078 grant signs an unrelated namespace with no revision; the current checked-in grant refuses that kind. **Why it is this way.** A generic kind allowlist does not enforce catalogue semantics, and UID authorisation requires an exclusive account to identify one workload.

## AB-37.3 Current backup selection is narrower than role-key custody

```mermaid
flowchart TD
    ROLE["PROPOSED independent node key under role secrets<br/>PROTOCOL-registry.md:255"] -.-> ROOT["Current backup scans configured root<br/>main.rs:154,156"]
    ROOT --> FILTER["Fixed filename allowlist<br/>main.rs:58,66,178"]
    FILTER --> AGE["Selected files become encrypted tar stream<br/>main.rs:248,253,256,260,262"]
    FILTER --> OMIT["Other basenames are skipped<br/>main.rs:179"]
    AGE --> NAMES["Optional names-only sidecar<br/>main.rs:264,266,272"]
    AGE --> RESTORE["Decrypt, refuse unsafe paths, extract and harden each entry<br/>main.rs:309,362,369,372"]
```

**What it shows.** Synthetic federation-key filenames are omitted even inside the scan root; selected configuration files still round-trip. **Why it is this way.** The backup tool deliberately uses a small allowlist. The proposed key needs explicit inclusion, recovery-state inventory and an off-host restore rehearsal.

## AB-37.4 Proposed control authority must become a durable delivery protocol

```mermaid
flowchart TD
    PIN["PROPOSED operator-pinned node key<br/>PROTOCOL-registry.md:253,254"] -.-> SEAL["Current decoder authenticates seal author<br/>transport.rs:474,476,503,504"]
    SEAL --> TYPE["PROPOSED three typed controls<br/>and referenced catalogue validation<br/>PROTOCOL-registry.md:255"]
    TYPE -.-> JOURNAL["REQUIRED durable inbox and watermark<br/>gap recovery, acknowledgement, idempotent apply"]
    JOURNAL -.-> VC["PROPOSED VisionClaw application"]
    WRAP["Outer gift wrap uses a throwaway key<br/>gift_wrap.rs:327"] --> SEAL
    CACHE["Current deduplication is in-memory SeenIds<br/>transport.rs:637,647,664"] -.-> JOURNAL
```

**What it shows.** Seal identity and a pinned peer are only the beginning of control admission; the durable queue remains a requirement. **Why it is this way.** Advancing a watermark before a missing lower-sequence retraction arrives can lose that command. Restart, ordering and restore tests must establish the complete delivery contract.
