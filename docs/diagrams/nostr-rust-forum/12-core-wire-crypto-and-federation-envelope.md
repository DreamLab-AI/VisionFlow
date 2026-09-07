---
id: NF-12
title: nostr-bbs-core wire and crypto primitives, and the nostr-bbs-mesh federation envelope
area: nostr-rust-forum
governing:
  - ../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md
  - ../nostr-rust-forum/docs/BASELINE-architecture.md
adrs: [ADR-2002, ADR-2003, ADR-2005]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/keys.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/event.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/signer.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/nip19.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/nip04.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/nip44.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/gift_wrap.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/nip98.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/did.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/deletion.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/groups.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/calendar.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/moderation_events.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/thread.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/types.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/wasm_bridge.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/feature_gate.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/boot_profile.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/d1_helpers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/cors.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/tests/identity_subkey_vectors.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/lib.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/envelope.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/jcs.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/delegation.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/config.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/transport.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/src/mock.rs
  - ../nostr-rust-forum/crates/nostr-bbs-mesh/tests/federation.rs
  - ../nostr-rust-forum/Cargo.toml
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/Cargo.toml
verified_commit: d48a7a546
---

## NF-12.1 Module map — what the workers and clients all link

```mermaid
flowchart TB
    subgraph wire["Wire primitives"]
        EV["event nostr-bbs-core/src/lib.rs:27"]
        SG["signer nostr-bbs-core/src/lib.rs:36"]
        TY["types nostr-bbs-core/src/lib.rs:38"]
        N19["nip19 bech32 nostr-bbs-core/src/lib.rs:33"]
    end
    subgraph crypto["Encryption and auth"]
        N04["nip04 nostr-bbs-core/src/lib.rs:32"]
        N44["nip44 nostr-bbs-core/src/lib.rs:34"]
        GW["gift_wrap NIP-59 nostr-bbs-core/src/lib.rs:28"]
        N98["nip98 HTTP auth nostr-bbs-core/src/lib.rs:35"]
        KY["keys nostr-bbs-core/src/lib.rs:30"]
    end
    subgraph domain["Domain kinds"]
        CAL["calendar NIP-52 nostr-bbs-core/src/lib.rs:25"]
        DEL["deletion NIP-09 nostr-bbs-core/src/lib.rs:26"]
        GRP["groups NIP-29 nostr-bbs-core/src/lib.rs:29"]
        MOD["moderation_events nostr-bbs-core/src/lib.rs:31"]
        THR["thread NIP-28 nostr-bbs-core/src/lib.rs:37"]
        GOV["governance nostr-bbs-core/src/lib.rs:46 - see NF-06"]
        KAN["kanban nostr-bbs-core/src/lib.rs:47"]
    end
    subgraph glue["Runtime glue"]
        DID["did nostr-bbs-core/src/lib.rs:44"]
        COR["cors nostr-bbs-core/src/lib.rs:42"]
        FG["feature_gate nostr-bbs-core/src/lib.rs:45"]
        BP["boot_profile nostr-bbs-core/src/lib.rs:41"]
        D1H["d1_helpers nostr-bbs-core/src/lib.rs:43"]
        ASH["admin_shared nostr-bbs-core/src/lib.rs:40 - see NF-08.7"]
        WB["wasm_bridge - wasm32 only nostr-bbs-core/src/lib.rs:48"]
    end

    N1["INVARIANT ADR-2002: this layer is what the upstream nostr absorption would REPLACE. Nothing here has<br/>been deleted - the canary gates that, see NF-01.5"]
    N2["wasm_bridge is compiled ONLY for wasm32 nostr-bbs-core/src/lib.rs:48 and exports NIP-44, subkey<br/>derivation and Schnorr signing to JS - nostr-bbs-core/src/wasm_bridge.rs:20 wasm_bridge.rs:88<br/>wasm_bridge.rs:228. ANOMALY O10 re-verified: no JS consumer exists in this repo."]
    N3["did wraps solid_pod_rs::did_nostr_types rather than re-encoding nostr-bbs-core/src/did.rs:13 -<br/>EXTERNAL: the encoder of record is the solid-pod-rs area (SP-*), see NF-02.9 and ES-04"]
```

## NF-12.2 Event identity and the signing path

```mermaid
sequenceDiagram
    autonumber
    participant A as Author
    participant U as UnsignedEvent<br/>nostr-bbs-core/src/event.rs:62
    participant ID as compute_event_id<br/>nostr-bbs-core/src/event.rs:78
    participant S as sign_event<br/>nostr-bbs-core/src/event.rs:122
    participant R as Relay
    participant V as verify_event_strict<br/>nostr-bbs-core/src/event.rs:205

    A->>U: kind, tags, content, created_at
    U->>ID: canonical serialisation
    ID->>ID: SHA-256 nostr-bbs-core/src/event.rs:88
    ID-->>S: 32-byte id
    S->>S: BIP-340 Schnorr sign nostr-bbs-core/src/event.rs:141
    S-->>R: NostrEvent nostr-bbs-core/src/event.rs:50
    R->>V: on admission
    V->>V: rebuild the id and compare, then Schnorr verify nostr-bbs-core/src/event.rs:238 nostr-bbs-core/src/event.rs:241

    Note over V: The single verification the WHOLE estate leans on. The relay calls it before any side effect (NF-03.4 step 4)
    Note over V: NIP-42 AUTH calls it (NF-03.2), and the forum client re-runs it on every inbound event rather than trusting the relay (NF-05.5)
    Note over S: sign_event_deterministic nostr-bbs-core/src/event.rs:160 exists for test vectors
    Note over S: sign_event_upstream nostr-bbs-core/src/event.rs:315 is the absorption seam onto the upstream nostr crate
    Note over U: The Signer trait nostr-bbs-core/src/signer.rs:75 lets a NIP-07 extension, a passkey key or a pasted nsec share one call site - see NF-05.4
```

## NF-12.3 The two key derivations, side by side

```mermaid
flowchart TB
    subgraph prf["derive_from_prf - passkey root"]
        P1["HKDF-SHA256, salt = per-identity derivation salt<br/>nostr-bbs-core/src/keys.rs:197"]
        P2["info = HKDF_INFO || counter, 32-byte OKM<br/>nostr-bbs-core/src/keys.rs:205 constant keys.rs:15"]
        P3["counter loop 0..=255 searching for a valid scalar<br/>nostr-bbs-core/src/keys.rs:201"]
    end
    subgraph sub["derive_subkey - purpose-scoped child"]
        S1["ONE raw HMAC-SHA-256, keyed by the root's 32 secret bytes<br/>nostr-bbs-core/src/keys.rs:253"]
        S2["message is the UTF-8 tag - no Extract/Expand<br/>nostr-bbs-core/src/keys.rs:255"]
        S3["exactly crypto.createHmac sha256 root update tag digest<br/>nostr-bbs-core/src/keys.rs:230"]
    end
    KAT["known-answer JS-parity vector<br/>nostr-bbs-core/src/keys.rs:478, digest keys.rs:483<br/>shared fixture nostr-bbs-core/tests/identity_subkey_vectors.rs:30"]

    sub --> KAT

    N1["INVARIANT: these are DIFFERENT constructions and must never be substituted - the doc says so at the<br/>definition site nostr-bbs-core/src/keys.rs:224"]
    N2["INVARIANT: a derived subkey is RECOVERABLE FROM THE ROOT by anyone holding the root<br/>nostr-bbs-core/src/keys.rs:240. It provides DOMAIN SEPARATION, not compromise isolation<br/>nostr-bbs-core/src/keys.rs:242 - rotation of the root is the only boundary."]
    N3["INVARIANT ADR-2003: the HMAC construction is a byte-for-byte cross-stack contract with agentbox's JS<br/>mirror derivation. Both halves read ONE versioned fixture, so they cannot drift - see NF-09.3.<br/>EXTERNAL: see AB-11 and ES-04"]
```

## NF-12.4 NIP-59 gift wrap — three layers, and what each hides

```mermaid
sequenceDiagram
    autonumber
    participant SD as Sender
    participant RU as rumor kind 14<br/>nostr-bbs-core/src/gift_wrap.rs:172
    participant SE as seal kind 13<br/>nostr-bbs-core/src/gift_wrap.rs:196
    participant WR as wrap kind 1059<br/>nostr-bbs-core/src/gift_wrap.rs:243
    participant RC as Recipient

    SD->>RU: create_rumor - UNSIGNED, carries the recipient p tag gift_wrap.rs:177
    RU->>SE: seal_rumor - signed by the sender's REAL key, NIP-44 encrypted
    SE->>WR: wrap_seal - a fresh THROWAWAY keypair gift_wrap.rs:246
    WR->>WR: NIP-44 encrypt the seal JSON, throwaway sk to recipient pk gift_wrap.rs:259
    WR->>WR: outer p tag names the recipient gift_wrap.rs:265
    WR-->>RC: kind 1059
    RC->>RC: unwrap_gift gift_wrap.rs:324
    RC->>RC: verify the SEAL's Schnorr signature before trusting seal.pubkey gift_wrap.rs:357
    RC->>RC: rumor.pubkey MUST equal the verified seal.pubkey gift_wrap.rs:379

    Note over WR: The ephemeral author is why relay admission cannot key on the author - it keys on the outer p tag instead, see NF-03.5, and delivery keys on the AUTHENTICATED session, see NF-11.5
    Note over RC: INVARIANT author binding: the rumor's claimed author must match the signature-verified seal author gift_wrap.rs:379 - without this check a seal could carry any rumor
    Note over SE: Kind constants: seal 13 gift_wrap.rs:30, wrap 1059 gift_wrap.rs:33. NIP-44 wire format is version || nonce 32 || ciphertext || mac 32 nostr-bbs-core/src/nip44.rs:11
```

## NF-12.5 NIP-98 — the ten checks every REST call passes

```mermaid
flowchart TB
    T["verify_token_full<br/>nostr-bbs-core/src/nip98.rs:414"]
    C1["Authorization must start with the Nostr prefix<br/>nostr-bbs-core/src/nip98.rs:424 constant nip98.rs:77"]
    C2["token and decoded JSON both capped at 64 KiB<br/>nostr-bbs-core/src/nip98.rs:429 nip98.rs:435 constant nip98.rs:74"]
    C3["kind must be 27235<br/>nostr-bbs-core/src/nip98.rs:441 constant nip98.rs:62"]
    C4["created_at within the tolerance window<br/>nostr-bbs-core/src/nip98.rs:451, default 60 s nip98.rs:65<br/>pubkey must be 64 hex nostr-bbs-core/src/nip98.rs:446"]
    C5["Schnorr signature and id integrity<br/>nostr-bbs-core/src/nip98.rs:459"]
    C6["u tag present and EXACTLY equal to the expected URL<br/>nostr-bbs-core/src/nip98.rs:465 nip98.rs:466"]
    C7["method tag present, compared case-insensitively<br/>nostr-bbs-core/src/nip98.rs:475 nip98.rs:477"]
    C8["payload tag REQUIRED when a body is present<br/>nostr-bbs-core/src/nip98.rs:487"]
    C9["payload hash must equal SHA-256 of the body<br/>nostr-bbs-core/src/nip98.rs:488"]

    T --> C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7 --> C8 --> C9

    N1["INVARIANT: the URL and method are bound into the SIGNATURE, so a token minted for one endpoint cannot<br/>be replayed against another. The payload hash extends that binding to the body."]
    N2["Replay is a SEPARATE concern layered on top: REPLAY_CACHE_TTL_SECS is twice the tolerance, a safe<br/>upper bound for cache entries nostr-bbs-core/src/nip98.rs:71. The single-use INSERT OR IGNORE lives in<br/>nostr-bbs-rate-limit - see NF-02.5 and NF-07.8"]
    N3["verify_nip98 nostr-bbs-core/src/nip98.rs:311 is the canonical entry point; every worker reaches this<br/>same implementation. EXTERNAL: agentbox terminates NIP-98 at its own proxy with an independent<br/>single-use cache - see AB-10"]
```

## NF-12.6 Domain kinds this crate owns

```mermaid
classDiagram
    class Moderation {
        BAN 30910 : moderation_events.rs:27
        MUTE 30911 : moderation_events.rs:31
        WARNING 30912 : moderation_events.rs:36
        REPORT 30913 : moderation_events.rs:39
        MODERATION_ACTION 30914 : moderation_events.rs:42
        REPORT_NIP56 1984 : moderation_events.rs:53
        ADMIN_ONLY_MOD_KINDS : moderation_events.rs:67
        enforcement : moderation_events.rs:261
    }
    class Calendar {
        DATE_EVENT 31922 : calendar.rs:34
        EVENT 31923 : calendar.rs:36
        RSVP 31925 : calendar.rs:38
        create_calendar_event : calendar.rs:207
        create_rsvp : calendar.rs:390
    }
    class Groups {
        GROUP_METADATA 39000 : groups.rs:30
        GROUP_ADMINS 39001 : groups.rs:31
        GROUP_MEMBERS 39002 : groups.rs:32
        build_group_metadata : groups.rs:254
    }
    class Deletion {
        KIND_DELETION 5 : deletion.rs:13
        create_deletion_event : deletion.rs:48
        target ids must be 64-hex : deletion.rs:68
    }
    class Thread {
        MAX_PARENT_HOPS 64 : thread.rs:40
        reply_parent NIP-10 : thread.rs:82
        group_threads : thread.rs:171
    }

    note for Moderation "REPORT is DELIBERATELY absent from<br/>ADMIN_ONLY_MOD_KINDS - members may report,<br/>only admins may act moderation_events.rs:67.<br/>The relay ban gate is the enforcement twin,<br/>see NF-03.6"
    note for Groups "The relay admits 39000-39002 from admin<br/>CLIENTS while these builders exist for relay-key<br/>signing - that mismatch is anomaly O2,<br/>see NF-03.4 and NF-10.3"
    note for Thread "MAX_PARENT_HOPS caps the reply walk so a<br/>malicious cycle cannot hang the client thread.rs:40"
```

## NF-12.7 The IS-Envelope — a published cross-repo wire contract

```mermaid
classDiagram
    class Envelope {
        from : canonical did:nostr hex, REQUIRED even under delegation — envelope.rs:164
        to : canonical did:nostr hex, REQUIRED — envelope.rs:162
        kind : EnvelopeKind — envelope.rs:81
        body : Value
        new : envelope.rs:197
        to_jcs_string : envelope.rs:348
    }
    class EnvelopeKind {
        Chat maps to Create
        ToolInvoke maps to Offer
        ToolResult maps to Add
        KnowledgeLink maps to Announce
        Moderation maps to Block
        MeshPing maps to View
    }
    class Delegation {
        DelegationToken : delegation.rs:49
        Conditions : delegation.rs:187
    }
    Envelope --> EnvelopeKind
    Envelope --> Delegation

    note for EnvelopeKind "Each kind declares the ActivityStreams 2.0 outer activity it maps to at the LDN boundary — the mapping table is at nostr-bbs-mesh/src/envelope.rs:97-103, kinds enumerated from envelope.rs:83"
    note for Envelope "Structural validation checks required fields, DID canonicality and the chain nostr-bbs-mesh/src/envelope.rs:255; a DID must be canonical did:nostr with 64 lowercase hex nostr-bbs-mesh/src/envelope.rs:463, normalised by nostr-bbs-mesh/src/envelope.rs:458"
    note for Delegation "EXTERNAL: this is a PUBLISHED contract other repos must match byte-for-byte, even though no transport ships here - see NF-12.8 and AB-13"
```

## NF-12.8 Why JCS — and why federation is still not wired

```mermaid
flowchart TB
    JCS["jcs::canonicalize - RFC 8785 subset<br/>nostr-bbs-mesh/src/jcs.rs:55"]
    WHY["Every envelope is JCS-serialised BEFORE becoming the content of a kind-14 rumor<br/>nostr-bbs-mesh/src/jcs.rs:4-6"]
    ID["so the outer kind-1059 event id - SHA-256 over canonical event JSON - is STABLE<br/>across independent encoders nostr-bbs-mesh/src/jcs.rs:6-7"]
    DEDUP["Two encoders producing semantically identical envelopes MUST emit byte-identical content,<br/>or the dedup primitive breaks nostr-bbs-mesh/src/jcs.rs:7-8"]
    HONEST["Declared a FAITHFUL SUBSET of RFC 8785, not the whole specification<br/>nostr-bbs-mesh/src/jcs.rs:11-12"]
    TRANS["MeshSocket nostr-bbs-mesh/src/transport.rs:237 | MeshTransport nostr-bbs-mesh/src/transport.rs:249<br/>RelayTransport generic over any socket nostr-bbs-mesh/src/transport.rs:323"]
    ONLY["The ONLY implementation in the tree is the test-only MockSocket<br/>nostr-bbs-mesh/src/mock.rs:249"]
    TEST["End-to-end federation tests DO exercise the full crypto stack<br/>nostr-bbs-mesh/tests/federation.rs:1"]
    CFG["MeshConfig nostr-bbs-mesh/src/config.rs:53 | MeshMode nostr-bbs-mesh/src/config.rs:98<br/>mesh_anchor_tags nostr-bbs-mesh/src/lib.rs:69"]

    JCS --> WHY --> ID --> DEDUP
    JCS --> HONEST
    TRANS --> ONLY --> TEST
    CFG --> TRANS

    N1["This is the crate's real value today: the envelope, its DID canonicality rules and its JCS encoding are<br/>a CONTRACT another repo can implement against, independent of whether this repo ships a transport."]
    N2["The transport half is genuinely absent - see NF-03.13 and NF-09.8. The relay declares the dependency<br/>nostr-bbs-relay-worker/Cargo.toml:26 but its own wiring is allow(dead_code)."]
    N3["EXTERNAL: agentbox speaks the same gift-wrap envelope shape on its own relay - see AB-13 and ES-03"]
```

## NF-12.9 Small shared surfaces with outsized reach

```mermaid
flowchart LR
    CORS["POD_CORS_HEADERS nostr-bbs-core/src/cors.rs:46<br/>STANDARD_CORS_HEADERS nostr-bbs-core/src/cors.rs:31"]
    FG["DEVICE_KEYS_ENABLED_VAR nostr-bbs-core/src/feature_gate.rs:42<br/>device_keys_enabled nostr-bbs-core/src/feature_gate.rs:64"]
    BP["BOOTPROFILE_MODE nostr-bbs-core/src/boot_profile.rs:41<br/>is_pwa_boot nostr-bbs-core/src/boot_profile.rs:97<br/>parse_boot_profile nostr-bbs-core/src/boot_profile.rs:107"]
    D1H["js_str D1 binding helper nostr-bbs-core/src/d1_helpers.rs:13"]
    N19["encode_npub nostr-bbs-core/src/nip19.rs:77 | decode_nsec nostr-bbs-core/src/nip19.rs:129<br/>encode_nprofile nostr-bbs-core/src/nip19.rs:180 | decode_naddr nostr-bbs-core/src/nip19.rs:249"]
    TYP["EventId nostr-bbs-core/src/types.rs:31 | PublicKey nostr-bbs-core/src/types.rs:108<br/>Timestamp nostr-bbs-core/src/types.rs:219 | Tag nostr-bbs-core/src/types.rs:253"]

    N1["feature_gate is why the two workers cannot disagree on what DEVICE_KEYS_ENABLED means - each reads its<br/>OWN binding but shares the PARSE rule, so ADR-2004's lockstep requirement holds by construction rather<br/>than by review - see NF-02.7 and NF-08.8"]
    N2["cors is why the pod worker's DPoP, Updates-Via, WAC and payment header envelope cannot drift per<br/>worker - see NF-04.1"]
    N3["boot_profile is consumed by the BBS client's zone-bound one-shot PWA - see NF-05.11"]
    N4["nip19 is the only place bech32 npub, nsec, nprofile and naddr are encoded; the recovery sheet's QR<br/>codes carry its output - see NF-05.8"]
```
