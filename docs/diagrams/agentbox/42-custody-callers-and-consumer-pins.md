---
id: AB-42
title: Typed custody callers and immutable consumer sources
area: agentbox
governing:
  - ../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md
  - ../project/agentbox/docs/adr/ADR-2145-the-solid-pod-forge-and-mirror-are-compiled-in-and-inert-until-named.md
adrs: [ADR-2122, ADR-2145]
sources:
  - ../project/agentbox/management-api/lib/identity-port.js
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/server.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/port.rs
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/mod.rs
  - ../project/agentbox/lib/solid-pod-rs.nix
  - ../project/agentbox/lib/nostr-pod-bridge.nix
  - ../project/agentbox/lib/colloquy.nix
verified_commit: 720e0d89656a4360998b3a716d0f3ee8cac66f93
---

## For developers

This successor records W3b's source wiring and the paired dependency acceptance. AB-36 retains its historical verification; source readiness does not change the role-isolation activation decision.

## For the business

The selected callers can request narrow identity operations without receiving held keys. Owner provisioning, the flag-on rehearsal and deployment are still required before this becomes an active custody boundary.

## AB-42.1 Bounded calls reach the existing receipt boundary

```mermaid
sequenceDiagram
    participant Caller as Selected consumer
    participant Client as Bounded port client
    participant Port as Identity port
    participant Receipt as Local receipt store
    %% Source: ../project/agentbox/management-api/lib/identity-port.js:20
    Caller->>Client: Purpose-bound request over stdin
    %% Source: ../project/agentbox/services/nostr-pod-bridge/src/identity_port/server.rs:106
    Client->>Port: Unix socket with kernel peer credentials
    %% Source: ../project/agentbox/services/nostr-pod-bridge/src/identity_port/port.rs:419
    Port->>Port: Check UID, operation grant and typed parameters
    %% Source: ../project/agentbox/services/nostr-pod-bridge/src/identity_port/port.rs:377
    Port->>Receipt: Persist admission receipt before returning result
    alt Receipt write succeeds
        Port-->>Caller: Admitted operation result
    else Refused or unavailable
        %% Source: ../project/agentbox/management-api/lib/identity-port.js:16
        Port-->>Caller: Error without key-file fallback
    end
```

**What it shows:** Typed wrap/open, forum sealing and governance operations use the existing held-key port and maintained signing primitives. This panel does not claim separate identities for callers sharing one UID.

**Why it is this way:** ADR-2122 separates purpose and key selection from runtime provisioning. A receipt precedes response release; it does not prove delivery to a remote relay.

## AB-42.2 Consumer source revisions move together

```mermaid
flowchart LR
    %% Source: ../project/agentbox/lib/solid-pod-rs.nix:64
    S["Solid 27e18995; alpha.21"] --> Server["Native pod server"]
    %% Source: ../project/agentbox/lib/nostr-pod-bridge.nix:64
    S --> Bridge["Identity and relay bridge"]
    %% Source: ../project/agentbox/lib/nostr-pod-bridge.nix:50
    F["Forum a3812b8e"] --> Bridge
    %% Source: ../project/agentbox/lib/colloquy.nix:46
    F --> C["Colloquy internal transport"]
    Bridge --> Gate["Exact archive layouts and committed Cargo locks"]
    C --> Gate
    Server --> Gate
```

**What it shows:** The bridge and server share one Solid revision; both forum consumers share one forum revision. Native acceptance caught and repaired the internal Colloquy signing-key dependency mismatch.

**Why it is this way:** Immutable source hashes and lockfiles close the gap between sibling-workspace compilation and the packaged consumer. Nix parsing and native tests do not establish a completed image build.

## AB-42.3 Source readiness stops before owner activation

```mermaid
flowchart TD
    %% Source: ../project/agentbox/services/nostr-pod-bridge/src/identity_port/mod.rs:94
    Flag{"Role isolation enabled?"} -->|No| Exit["Identity service exits without reading held keys"]
    Flag -->|Yes| Load["Load declared ACL, role keys and receipt store"]
    %% Source: ../project/agentbox/services/nostr-pod-bridge/src/identity_port/mod.rs:73
    Load --> Serve["Serve permitted typed operations"]
    Owner["Owner provisions identities and forum admission"] -.-> Rehearsal["Rebuild and run flag-on rehearsal"]
    Rehearsal -.-> Flag
```

**What it shows:** The implementation retains flag-off compatibility. Source tests used temporary fixture keys; no live identity, host socket permission or browser vault changed.

**Why it is this way:** Custody activation needs the existing owner-run acceptance procedure. The source merge is not evidence that Docker access or ambient credentials have been restricted in the running estate.
