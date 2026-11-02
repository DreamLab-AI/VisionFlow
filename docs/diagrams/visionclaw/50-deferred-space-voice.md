---
id: VC-50
title: Deferred space voice and the scope of its authority proof
area: visionclaw
governing:
  - ../project/docs/IDENTITY-authority-chain.md
  - ../project/docs/XR-client.md
adrs: [ADR-2157, ADR-2160]
sources:
  - ../project/docs/IDENTITY-authority-chain.md
  - ../project/docs/XR-client.md
  - ../project/xr-client/rust/src/webrtc_audio.rs
  - ../project/crates/visionclaw-domain/src/config/services.rs
  - ../project/src/services/audio_router.rs
verified_commit: {visionclaw: 86cdd7635582}
---

## For developers

ADR-2160 explicitly defers voice. The [audit](../../estate-review/2026-11-01-room-admission-and-voice.md) tests existing state code and identifies a missing dependency issuer in the proposed deadline argument; it does not run or activate a call.

## For the business

Voice configuration and position tracking are present. A membership-controlled spatial call still needs media integration and a revocation contract covering all reconnect credentials.

## VC-50.1 Current code stores voice state without the proposed call authority

```mermaid
flowchart TD
    CONFIG["LiveKit URL, API key and secret fields<br/>services.rs:159,162,165,168"]
    ROUTER["Audio router stores optional participant id<br/>audio_router.rs:39,160,420"]
    TRACKS["Voice core: track positions and mute flags<br/>webrtc_audio.rs:43,62,88,99"] --> SNAP["Return track snapshot<br/>webrtc_audio.rs:113,115"]
    LISTENER["Store listener transform<br/>webrtc_audio.rs:103,105"] --> READ["Return listener snapshot<br/>webrtc_audio.rs:120,123"]
    GAP["Documented media transport gap<br/>XR-client.md:875,877"]
```

**What it shows.** These are existing fields and state operations, not a connected media pipeline. **Why it is this way.** The native media transport remains unwired; passing state tests do not establish HRTF, microphone capture or a server token issuer.

## VC-50.2 The proposal makes room admission a prerequisite for application minting

```mermaid
flowchart TD
    AUTH["PROPOSED: verified identity and RoomAdmission<br/>IDENTITY-authority-chain.md:240,242"] --> BOUND["PROPOSED: bound registered rooms only<br/>IDENTITY-authority-chain.md:242"]
    BOUND --> MINT["PROPOSED: room-scoped grant, TTL at most 60 seconds<br/>IDENTITY-authority-chain.md:242"]
    MINT --> KEY["PROPOSED: per-SFU signing key custody<br/>IDENTITY-authority-chain.md:242"]
    REFRESH["PROPOSED: refuse removed members a refresh<br/>IDENTITY-authority-chain.md:242"] --> AUTH
```

**What it shows.** The intended application credential path is specified but absent from current source. **Why it is this way.** The authority document describes a proposal; the audit identifies an additional dependency refresh path that its system-wide argument must cover.

## VC-50.3 Proposed leave and removal paths have different deadlines

```mermaid
flowchart TD
    VOL["PROPOSED: voluntary leave<br/>IDENTITY-authority-chain.md:242"] --> GRACE["PROPOSED: 25-second grace<br/>IDENTITY-authority-chain.md:242"]
    KICK["PROPOSED: membership removal or staleness<br/>IDENTITY-authority-chain.md:240,242"] --> REMOVE["PROPOSED: RemoveParticipant<br/>IDENTITY-authority-chain.md:242"]
    GRACE --> REMOVE
    LOOP["PROPOSED: participant reconciler every 10 seconds<br/>IDENTITY-authority-chain.md:242"] --> REMOVE
    PART["PROPOSED: API unreachable beyond 90 seconds<br/>IDENTITY-authority-chain.md:242"] --> ALARM["PROPOSED: stop minting and alarm<br/>connected window remains unenforced<br/>IDENTITY-authority-chain.md:242"]
```

**What it shows.** The proposal distinguishes voluntary reconnect grace, removal and an API-partition exception. **Why it is this way.** Token expiry does not disconnect an active participant; the separate 150-second rejoin claim is undercut by the issuer gap documented in the audit and must not be read as a current guarantee.
