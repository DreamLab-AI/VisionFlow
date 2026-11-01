---
id: NF-22
title: Recovery export, phone import and device attribution boundaries
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/archive/adr/ADR-095-recovery-device-onboarding-sheet.md
  - ../nostr-rust-forum/docs/archive/adr/ADR-098-connect-magic-link-onboarding.md
  - ../nostr-rust-forum/docs/archive/adr/ADR-099-revocable-device-keys.md
adrs: [ADR-095, ADR-098, ADR-099, ADR-2004]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/signup.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/connect.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/session.rs
  - ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/utils/devices.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs
verified_commit: c39f41f104b114d6a3fa29013daaf6fa58b45080
---

## For developers

[The section review](../../estate-review/2026-11-01-forum-recovery-and-connect.md) covers archived ADR-095/098/099 with nineteen native cases and five isolated Chrome cases. Extracted callbacks use mocked browser/reactive state; Chrome runs synthetic formatter output and History API controls. Neither suite executes the full signup, relay or phone journey.

## For the business

A saved recovery key, a locally imported account and a revocable phone are three different outcomes. Show which one the user has obtained. A download click does not prove a usable backup, and a successful local import does not prove access to the forum.

## NF-22.1 Connect calls strip before import but ignores strip failure

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/connect.rs:52
    A["Read fragment<br/>optional k= prefix; trim whitespace"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/connect.rs:75
    B["Try history.replaceState<br/>unavailable API and errors ignored"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs:438
    C["Validate hex or nsec<br/>32 bytes and valid scalar"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs:456
    D["Install local key and signer<br/>best-effort browser persistence"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/auth/mod.rs:485
    E["Local Authenticated state<br/>profile hydration starts separately"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/connect.rs:116
    F["Success warning and delayed navigation<br/>no relay acceptance awaited here"]
    A --> B
    B -->|success or ignored failure| C
    C -->|valid| D --> E --> F
```

Native probes reproduce failed clearing followed by import. A deliberately cross-origin Chrome replacement demonstrates `SecurityError` retaining the fragment, not a production routing defect. Clearing runs at mount time, not before every possible page observer.

## NF-22.2 Recovery HTML contains raw interpolated text beside the key

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/signup.rs:241
    A["Display name checked for trimmed byte length<br/>2 to 50; HTML content not rejected"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/signup.rs:645
    B["Original display-name signal<br/>passed to RecoverySheet"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:138
    C["format! builds HTML<br/>raw display name, identity and recovery text"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:229
    D["JS Blob and object URL<br/>downloadable HTML document"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:294
    E["Rust-generated QR SVG<br/>separate from text escaping"]
    A --> B --> C --> D
    E --> C
```

Chrome executes the synthetic display-name marker in this formatter output. An escaped test-only control does not. This is active-content injection in an exported secret-bearing document; remote delivery and real-secret exfiltration were not demonstrated. Pure-Rust QR generation does not make the exported DOM or Blob inaccessible to scripts.

## NF-22.3 The backup gate latches an attempted output and acknowledgement

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:321
    A["Download callback<br/>helper returns no success value"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:338
    B["Print callback<br/>print error discarded"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:335
    C["printed becomes true"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:303
    D["printed AND confirmed<br/>emits positive on_ready only"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/signup.rs:647
    E["Parent sheet_ready becomes true<br/>no negative callback resets it"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/pages/signup.rs:674
    F["Finish enabled by sheet_ready<br/>OR explicit advanced override"]
    A --> C
    B --> C --> D --> E --> F
```

A failed print and a missing window still set `printed`. Unchecking confirmation does not clear the parent latch. These are extracted-callback probes, not a complete reactive-runtime test. The advanced override is intentional; this gate is a user acknowledgement, not proof of file creation or account authorisation.

## NF-22.4 Master recovery and device enrolment have different exports

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:288
    A["Master nsec creates primary phone URL<br/>live origin plus connect fragment"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:309
    B["Download closure captures master fields<br/>no device or sweep inputs"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:355
    C["Device feature gate<br/>separate Generate action"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:381
    D["Await register_device_with_master"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:383
    E["Only success sets screen device URL and QR"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-forum-client/src/components/recovery_sheet.rs:590
    F["Screen tear-off card<br/>included in DOM print layout"]
    A --> B
    C --> D -->|success| E --> F
    A -. "remains alongside device card" .-> F
```

NF-20 documents the missing registration proof and incompatible list parser. This diagram shows the success branch in source, not a working deployed enrolment. Downloaded HTML retains the master link and lacks device/sweep sections. A master-key import cannot be contained by revoking an unrelated device row.

## NF-22.5 Owner policy does not rewrite the signed author or tags

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1054
    A["Receive signed device-authored event"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1121
    B["Resolve effective principal for policy<br/>selected kinds use owner lookup"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1552
    C["Pass original event to save_event"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/storage.rs:73
    D["Serialise original tags<br/>store original pubkey and signature"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:1554
    E["Broadcast original event after successful save"]
    A --> B --> C --> D
    C -->|save succeeds| E
```

This ordinary stored-event path has no appended owner-attribution tag. Ephemeral events broadcast directly without storage. Separate UI projections were not exhaustively reviewed here. Preserve signed event bytes and specify any owner presentation as an independent authenticated assertion; see NF-20 for policy/cache limits.
