---
id: NF-19
title: BBS signer — adoption, account lifetime and key custody
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2008-bbs-signer-holds-key-directly.md
adrs: [ADR-2008]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/nip07.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/config.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/app.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/relay.rs
  - ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/pwa.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs
verified_commit: {nostr-rust-forum: c39f41f104b114d6a3fa29013daaf6fa58b45080}
---

## For developers

[ADR-2008 section review](../../estate-review/2026-11-01-forum-bbs-signer.md) distinguishes tested local signer lifetime from browser-only storage, extension and socket findings. No live session was changed.

## For the business

Users need sign-out and backup to refer to the same account shown on screen. Shared branding and same-origin storage do not make the forum and BBS one session. Existing extension support reduces raw-key handling, but account binding and cancellation still need work.

## NF-19.1 Adoption chooses storage before checking the extension

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/app.rs:26
    A["Boot calls adopt_forum_session"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:393
    B["Read localStorage key first<br/>then sessionStorage key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:332
    C["Any valid local key wins<br/>no session metadata required"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:338
    D["Otherwise stored isNip07 and public key<br/>plus provider presence select extension"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:269
    E["Install stored public key<br/>no getPublicKey round-trip"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/app.rs:134
    F["PWA unwrap runs only if<br/>no signer was adopted"]
    A --> B --> C
    B --> D --> E
    C --> F
    E --> F
```

A non-empty but invalid localStorage key prevents reading a valid sessionStorage key; parsing then falls through to extension selection. The extension-selection helper accepts any non-empty stored public-key string, including malformed values.

## NF-19.2 Sign-out leaves recoverable credentials and the socket

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs:1976
    A["User selects sign out"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:229
    B["Clear BBS signer, public key and error"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/relay.rs:464
    C["Clear relay signer reference only<br/>no socket close or subscription reset"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:393
    D["Forum secret storage remains available<br/>for a later boot to adopt"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/pwa.rs:247
    E["Separate Forget this device operation<br/>attempts baked-record and profile deletion"]
    A --> B --> C
    B --> D
    A -. separate control .-> E
```

Ordinary sign-out does not remove the baked key either. Forget this device ignores IndexedDB deletion failures and does not clear forum raw-key storage. Neither operation establishes global account revocation.

## NF-19.3 A retained signer can act after the visible account changes

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:106
    A["get_signer clones an Rc handle"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs:942
    B["Composer captures handle and author<br/>when send starts"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:229
    C["Logout clears stored handles<br/>external clones remain alive"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs:1385
    D["Await signature with captured handle"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs:1400
    E["Publish returned event<br/>no current-account generation check"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/relay.rs:543
    F["AUTH also awaits a captured signer<br/>then uses current socket sender"]
    A --> B --> D --> E
    B -. while pending .-> C
    A --> F
```

Two native counterexamples produce strictly verified signatures with retained local signers after logout and account replacement. Publication and extension-prompt races remain source findings, not executed browser journeys.

## NF-19.4 Extension signing does not validate the returned intent

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/nip07.rs:97
    A["Signer stores supplied public key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/nip07.rs:109
    B["sign_event delegates to JS bridge<br/>stored public key is not compared"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/nip07.rs:167
    C["Resolve current window.nostr provider"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/nip07.rs:191
    D["Await signEvent Promise<br/>rejection becomes error"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/nip07.rs:202
    E["Deserialize returned event<br/>no signature or input-equivalence check"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs:1400
    F["Composer forwards result to relay"]
    A --> B --> C --> D --> E --> F
```

An extension account change may produce a different author from the visible account. Relay verification can reject invalid signatures; it cannot reconstruct the user’s intended author or content. No malicious extension or deployed account switch was exercised.

## NF-19.5 Backup reads storage rather than the active signer

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs:1972
    A["Signed-in panel offers backup<br/>when local_secret_hex returns a value"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:291
    B["Only require some active public key"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:295
    C["Read forum secret from storage<br/>no active-public-key comparison"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/screens.rs:1915
    D["Render supplied secret as nsec and hex<br/>copy controls retain String clones"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:147
    E["Generated key returned once to UI<br/>does not update forum storage"]
    A --> B --> C --> D
    E -. does not replace backup source .-> C
```

With forum key A still stored and local signer B installed, the backup path can show A while the account label names B. An in-memory pasted key has no later backup route when forum storage is empty. These are source-traced conditions; browser reproduction remains open.

## NF-19.6 Zeroisation covers selected objects and paths

```mermaid
flowchart TB
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:253
    A["Storage string passed by reference<br/>original String has ordinary drop"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:358
    B["Parser creates another hex String<br/>manually overwrites this copy"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:371
    C["Decoded Vec remains an ordinary buffer"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:382
    D["Invalid scalar exits through error<br/>before explicit arr.zeroize"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-bbs-client/src/signer.rs:384
    E["Successful parse zeroizes local array"]
    %% Source: ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs:40
    F["Owned SecretKey zeroizes on final drop"]
    A --> B --> C
    C --> D
    C --> E --> F
```

Final-key zeroisation does not prove erasure of browser storage, UI strings, decoded vectors, rejected-input copies or JavaScript allocations. This audit makes no heap-recovery or compiler-level erasure claim.
