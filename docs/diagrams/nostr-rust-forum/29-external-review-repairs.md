---
id: NF-29
title: External-review repair boundaries and deployment limits
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/audit/external-fixes-2026-11-03.md
adrs: [ADR-2003, ADR-2006, ADR-2009, ADR-2014]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs
  - ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/passkey.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/acl.rs
  - ../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/replay.rs
verified_commit: {nostr-rust-forum: d36fac107c686321719d02b94afd71b23350eb6d}
---

## For developers

This scoped successor records source repairs for selected findings in NF-18, NF-20, NF-23, NF-24 and NF-25. Historical panels retain their original evidence revisions. Neither this panel nor native/WASM gates establish a deployed Cloudflare Worker, real WebAuthn authenticator ceremony or browser logout journey. Other findings in those topics remain open unless explicitly named below.

## For the business

Registration now keeps the original identity input and proves the registering identity. Demotion uses one dynamic authority, revoked devices cannot silently re-enrol, and concurrent notifications retain work owed. Deployment still needs database bindings and schema reconciliation; old mismatched passkeys need authenticated recovery or re-enrolment.

## NF-29.1 Registration keeps the issued salt and proves the derived identity

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:783
    A["Options generates salt A"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:814
    B["Persist A against registration challenge"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/passkey.rs:177
    C["Forum client signs verify POST with derived identity"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/passkey.rs:357
    D["BBS client signs verify POST too"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:719
    E["Verify requires matching live account challenge and stored A"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/webauthn.rs:1081
    F["Conditional credential insert stores A; batch consumes challenge and salt"]
    A --> B --> E --> F
    C --> E
    D --> E
```

NF-23.1 and NF-23.2 are repaired in source. SQL fixtures check challenge expiry/account binding and registration/login derivation equality. Old credential salt values are not rewritten: the original PRF input cannot be reconstructed safely. Native checks and successful client WASM compilation do not prove real hardware behaviour.

## NF-29.2 Device lifecycle and dynamic administration settle at their owners

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:331
    A["Owner and device proof required; upsert updates only an active same-owner label"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/devices.rs:333
    B["Integer wire fields; invalid database numbers refuse serialisation"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/admin.rs:53
    C["Dynamic admin authority reads relay whitelist; static bootstrap authority remains explicit"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/lib.rs:2016
    D["Pod administration also uses RELAY_DB"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:263
    E["Invite cohorts union in one SQL update"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-auth-worker/src/invites.rs:734
    F["Revoke resolves unique invitation id or code"]
    A --> B
    C --> D
    E --- F
```

NF-20.2/20.3 and the split-admin authority path in NF-25 are repaired in source. A revoked device requires a fresh derived device identity. The pod deployment must bind `RELAY_DB` to the instance relay database; the kit template contains an explicit placeholder. Payment cleanup also separates authority queries from payment data and has a regression fixture for stale auth-member grants after relay demotion. Device cache revocation latency remains a separate boundary.

## NF-29.3 Notification claims and bounded sweeps retain concurrent work

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:200
    A["Drain compares counters, token hash and bump sequence; retry races, then 503"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/push.rs:700
    B["Delivery records the sequence captured before send; later bumps remain owed"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/push.rs:314
    C["Page stores teardown credentials alongside worker storage"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:562
    D["Capped or aborted sweep persists stable cursor"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/trust_sweep.rs:569
    E["Completion clears cursor for next full cycle"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/agent_disclosure.rs:88
    F["Configured ingress disclosure has registered false; durable registry row wins"]
    A --- B --- C
    D --> E
    F
```

The push drain/bump/logout source findings and NF-24 bounded-prefix starvation are repaired. Disclosure is additive: consumers deciding whether provisioning is complete must require `registered === true`; a configured ingress row alone is insufficient. No registry identity was provisioned by this repair.

## Validation and remaining boundaries

At the first repair revision, 7852fe8, 1,120 native tests passed across auth, BBS, pod and relay; all three workers passed warning-denied native Clippy and rustdoc; all workers and both browser clients compiled to WASM. The two browser clients had four pre-existing WASM Clippy warnings, so no blanket client Clippy pass is claimed. The optional bounded Deepsec gate timed out without a verdict and is not counted as passing evidence.

## NF-29.4 Forum workers consume the shared Solid denial and nonce contracts

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/acl.rs:40
    A["Shared classifier plus kit 64 KiB policy cap"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-pod-worker/src/acl.rs:314
    B["R2 and KV reads: only absent policy ascends; invalid or unavailable settles deny"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/replay.rs:88
    C["D1 adapter implements shared ReplayStore; duplicate nonce returns Replayed"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/replay.rs:47
    D["Unknown write metadata or backend error refuses authentication"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-rate-limit/src/replay.rs:106
    E["Existing kit verifier delegates to the shared contract"]
    A --> B
    E --> C --> D
```

The a3812b8 delta adds shared Solid alpha.19 adoption. The first three panels' sources are byte-identical to their 7852fe8 evidence revision. Cargo resolves alpha.19 from the registry with a checksum; no workspace patch remains. Production SQL fixtures prove that a duplicate nonce cannot refresh its expiry and unknown write metadata is not first-seen success. Classifier fixtures prove invalid-child terminal denial and absent-child inheritance.

Final adoption gates pass 1,123 native library tests, existing integration/doc tests, four workers' native all-target Clippy and WASM library Clippy with warnings denied, plus both client WASM builds. The repaired nonce module and pod were retested after recovery from a local disk-space failure. NF-19 signer retention, sealed-history migration, NIP-171 owner ceremonies, remaining cache/transport findings and live rollout are not closed here.
