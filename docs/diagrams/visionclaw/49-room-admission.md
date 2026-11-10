---
id: VC-49
title: Current presence joins and proposed room admission
area: visionclaw
governing:
  - ../project/docs/IDENTITY-authority-chain.md
  - ../project/docs/IDENTIFIER-taxonomy.md
adrs: [ADR-2019, ADR-2157]
sources:
  - ../project/docs/IDENTITY-authority-chain.md
  - ../project/docs/IDENTIFIER-taxonomy.md
  - ../project/crates/visionclaw-xr-presence/src/types.rs
  - ../project/crates/visionclaw-xr-presence/src/room.rs
  - ../project/crates/visionclaw-xr-presence/src/ports/mod.rs
  - ../project/src/handlers/presence_handler.rs
  - ../project/src/actors/presence_actor.rs
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
---

## For developers

The [audit](../../estate-review/2026-11-01-room-admission-and-voice.md) runs the real actor in isolation and checks the proposed identifier vector. Proposed nodes are requirements; no membership implementation or live admission test is claimed.

## For the business

A signed identity can currently join a legacy presence room. Restricting a space to its members requires a separate authority check, durable replay protection and removal handling.

## VC-49.1 Current identity checks lead directly to the legacy room actor

```mermaid
flowchart TD
    ID["Challenge, signature, verified DID and replay checks<br/>presence_handler.rs:210,240<br/>presence_handler.rs:249,266"] --> PARSE["Parse legacy room URN<br/>presence_handler.rs:273"]
    PARSE --> REG["Get or create actor; replace disconnected address<br/>presence_handler.rs:281,290"]
    REG --> JOIN["Send verified DID in JoinRoom<br/>presence_handler.rs:305,306"]
    JOIN --> DUP["Aggregate rejects duplicate DID<br/>room.rs:42,45"]
    DUP --> MEMBERS["Insert member and subscriber<br/>room.rs:51,52<br/>presence_actor.rs:681"]
```

**What it shows.** The current join path has identity and duplicate checks but no membership gate or cap. **Why it is this way.** It implements the earlier legacy presence contract; the isolated actor accepted 33 distinct members and refused a duplicate.

## VC-49.2 The new room identifier has a valid vector but no current parser support

```mermaid
flowchart TD
    LEGACY["Current parser accepts sha256-12 suffix only<br/>types.rs:43,48,52"] --> KEEP["Retains supplied hexadecimal case<br/>types.rs:52,55"]
    KEEP --> HASH["Wire hash derives from retained string<br/>types.rs:64,65"]
    PROPOSED["PROPOSED: domain plus raw home key plus group bytes<br/>IDENTIFIER-taxonomy.md:211"] --> MINT["PROPOSED: one room_from_group constructor<br/>IDENTIFIER-taxonomy.md:209"]
    MINT --> BIND["PROPOSED: registered nip29 URN binds key and group<br/>IDENTIFIER-taxonomy.md:212"]
```

**What it shows.** The audit reproduces the proposed SHA-256 vector while the production parser rejects its result. **Why it is this way.** The new namespace is specified independently of the retained legacy form; a test-only hash calculation does not implement the constructor or registry.

## VC-49.3 Proposed admission includes freshness and a second check after registration

```mermaid
flowchart TD
    ID["PROPOSED: retain existing identity check<br/>IDENTITY-authority-chain.md:242"] --> ADMIT["PROPOSED: registered room and pinned home membership<br/>IDENTITY-authority-chain.md:242"]
    ADMIT --> LEASE["PROPOSED: freshness lease and revision ordering<br/>IDENTITY-authority-chain.md:242"]
    LEASE --> JOIN["PROPOSED: register then recheck admission<br/>IDENTITY-authority-chain.md:242"]
    JOIN --> RECHECK["PROPOSED: recheck live sessions every 10 seconds<br/>IDENTITY-authority-chain.md:242"]
    RECHECK --> CLOSE["PROPOSED: stale membership closes sessions<br/>IDENTITY-authority-chain.md:242"]
    ID --> LEGACY["PROPOSED: unbound presence only, capped<br/>bound_only migration at P2<br/>IDENTITY-authority-chain.md:246"]
```

**What it shows.** The proposed check applies both at entry and during a session. **Why it is this way.** A successful initial identity check cannot prove continuing membership; the 90-second presence deadline still needs implementation and adversarial timing evidence.

**Tension:** The proposal lets unbound rooms keep "the member cap" (`IDENTITY-authority-chain.md:246`), but `Room::join` checks only for a duplicate DID before inserting (`room.rs:42-52`), so no cap exists to keep.
