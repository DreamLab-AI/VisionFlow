---
id: VC-43
title: Codescape presence admission, replacement and queue loss
area: visionclaw
governing:
  - ../project/codescape/docs/BASELINE-codescape.md
  - ../project/docs/adr/ADR-2143-codescape-section-separate-workspace-and-port.md
adrs: [ADR-2143, ADR-2159]
sources:
  - ../project/codescape/crates/codescape-server/src/auth.rs
  - ../project/codescape/crates/codescape-server/src/proxy.rs
  - ../project/codescape/crates/codescape-server/src/origin.rs
  - ../project/codescape/crates/codescape-server/src/routes/ws.rs
  - ../project/codescape/crates/codescape-server/src/routes/presence.rs
  - ../project/codescape/crates/codescape-server/src/xr_presence.rs
  - ../project/xr-client/rust/src/presence.rs
  - ../project/xr-client/scripts/graph_scene.gd
verified_commit: {codescape: e3b76f63e801c6a8460110b93e88a0b420e9b0b4, visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
---

## For developers

The Codescape hub relays poses and roster events. The [production-library probes](../../estate-review/2026-10-31-codescape-presence-and-xr.md) expose stale membership and queue-loss boundaries; they do not measure a live socket race.

## For the business

An avatar label is not authenticated identity here. Reconnects and slow readers can also leave a consumer with an inconsistent roster. The default XR scene uses VisionClaw presence, not this endpoint (`graph_scene.gd:2941,2945`).

## VC-43.1 Admission has an origin filter but no identity verifier

```mermaid
flowchart TD
    CFG{"require_auth?<br/>auth.rs:137"} -->|true| STOP["Refuse startup<br/>auth.rs:138"]
    CFG -->|false| OPEN["Open mode returns no pubkey<br/>auth.rs:124,127,142"]
    OPEN --> WS["Common open: verifier, origin, repo and connection slot<br/>routes/ws.rs:192"]
    PEER["Forwarded host only when proxy trust applies<br/>disabled by default, optional peer restriction<br/>proxy.rs:267,283"] --> WS
    WS --> CH["Challenge then auth envelope within 10 seconds<br/>routes/presence.rs:34,120,130"]
    CH --> LABEL["Signature ignored, DID resolved as label<br/>routes/presence.rs:98,136"]
    LABEL --> ROOM["Join hub and send initial roster<br/>routes/presence.rs:150,155"]
    ROOM --> POSE["Decode finite pose and rate-limit before relay<br/>routes/presence.rs:175,178,181"]
```

**What it shows.** Origin filtering and proxy trust do not establish a signed principal. Missing Origin is allowed for native clients (`origin.rs:127`); the audit confirms required-auth startup refusal and the separate ingest-token guard. **Why it is this way.** ADR-2143 A4 deliberately permits open dev-LAN access pending P6. This is a documented deployment boundary, not a newly discovered signature bypass.

## VC-43.2 Replacement changes the roster before the old handle loses relay ability

```mermaid
sequenceDiagram
    participant OLD as Old Membership<br/>xr_presence.rs:649
    participant HUB as PresenceHub.join<br/>xr_presence.rs:552
    participant NEW as Replacement Membership<br/>xr_presence.rs:615
    participant OBS as Existing observer<br/>xr_presence.rs:523
    HUB->>HUB: Remove entry with the same avatar (:565)
    HUB-->>OLD: try_send Close, failure ignored (:568)
    HUB-->>OBS: AvatarLeft then AvatarJoined (:569,602)
    HUB->>NEW: Allocate a different local_id and store new entry (:581,604)
    OLD->>HUB: relay checks held room and avatar only (:668)
    HUB-->>OBS: Pose carries removed local_id (:673,677)
    HUB-->>NEW: Same stale pose reaches replacement (:677)
    Note over OLD,HUB: Drop removes only the old local_id<br/>Replacement stays (:638,683)
```

**What it shows.** The probe relays from the old handle after replacement and observes the removed local id at both recipients. **Why it is this way.** `relay` does not check whether its local id remains in the room. The socket loop eventually consumes Close or channel termination (`routes/presence.rs:198`); the audit does not claim permanent survival of a replaced socket.

**Tension:** The replacement contract promises closure (`xr_presence.rs:550`), but roster removal does not revoke relay authority (`xr_presence.rs:668`).

## VC-43.3 One bounded queue drops both replaceable poses and roster transitions

```mermaid
flowchart TD
    FILL["64 pose messages fill observer queue<br/>xr_presence.rs:71,603"] --> SEND["send_all uses try_send for every message kind<br/>xr_presence.rs:521,524"]
    JOIN["New join announces AvatarJoined<br/>xr_presence.rs:594,602"] --> SEND
    LEAVE["Leave announces AvatarLeft<br/>xr_presence.rs:640"] --> SEND
    SEND --> FULL["Full queue loses announcement<br/>only successful sends counted<br/>xr_presence.rs:524"]
    FULL --> DRAIN["Observer drains queued poses<br/>route forwards queued bytes<br/>routes/presence.rs:198,200"]
    DRAIN --> LATER["New member pose carries local id<br/>xr_presence.rs:673"]
    LATER --> MAP["Client maps id from Joined or AvatarJoined<br/>xr-client/rust/src/presence.rs:591,599"]
    MAP --> MISS["Missing mapping falls back to decimal local id<br/>xr-client/rust/src/presence.rs:633,636"]
```

**What it shows.** The probe fills a queue using poses admitted by the real rate limiter over synthetic time, loses a join announcement, then receives that member's pose. Leave and replacement-Close loss are also reproduced. **Why it is this way.** The queue has no separate reliable roster channel or resynchronisation path. The XR client's numeric-id fallback is source evidence; no headset rendering outcome was tested.

**Drift:** The baseline says presence carries no avatars (`BASELINE-codescape.md:109`), although the current hub forwards them (`xr_presence.rs:668`).

**Tension (risk):** Both defects are fixed only upstream: standalone Codescape `d2c3c3b` adds a cancel fence and a separate control queue for roster messages, but VisionClaw `ec785afa9` still pins `e3b76f6`, where `relay` checks only room and avatar (`xr_presence.rs:668`) and `send_all` uses one `try_send` queue (`xr_presence.rs:524`).
