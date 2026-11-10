---
id: ES-13
title: Governance delivery — expiry, projection and restart checkpoints
area: estate
governing:
  - ../nostr-rust-forum/docs/adr/ADR-2013-ontology-governance-panel-promote-demote-expiry.md
adrs: [ADR-2013, ADR-2109]
sources:
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/receipts.rs
  - ../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/cron.rs
  - ../nostr-rust-forum/crates/nostr-bbs-core/src/ontology_governance.rs
  - ../project/agentbox/mcp/nostr-bridge/relay-consumer.js
  - ../project/agentbox/management-api/adapters/orchestrator/local-process-manager.js
verified_commit: {nostr-rust-forum: 6064622dbb1a7cc1ba2d072109f9e3e8a96157df, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93}
---

## For developers

A stored decision, a projected case transition and an applied corpus change have separate receipts. This topic follows the crossings where they can disagree, then follows the consumer's restart checkpoint. [Signed local fixtures](../../estate-review/2026-10-31-governance-ingress.md) exercise Agentbox ingress and the installed CLI; deployed relay, D1 and browser behaviour remain outside those fixtures.

## For the business

A human decision needs both a reliable effect and a reliable record. A deadline on the review case does not protect the corpus if its writer ignores that deadline. A delivery checkpoint must also distinguish completed work from a decision still waiting to finish, so restarting a service does not silently suppress it.

## ES-13.1 Storage broadcast, projection and mutation are separate boundaries

```mermaid
flowchart TD
    EV["Signed decision: event validation and author gates<br/>nip_handlers.rs:1063,1077,1344"] --> STORE["Save, acknowledge and broadcast<br/>nip_handlers.rs:1552"]
    STORE --> PROJECT["Later response projection<br/>nip_handlers.rs:1634"]
    PROJECT --> SQL["Commit only for matching request and allowed case state<br/>receipts.rs:185,188"]
    EXP["Expiry receipt then close open / under_review / reopened<br/>cron.rs:764,794,834"] --> SQL
    STORE --> AB["Consumer verifies signature and human roster<br/>relay-consumer.js:446,449,454"]
    AB --> FETCH["Fetch request by exact ID; verify signature<br/>relay-consumer.js:513,532"]
    FETCH --> APPLY["Explicit promote checks request tag and IRI,<br/>then applies without reading projection receipt or expiry<br/>local-process-manager.js:419,444,466"]
    APPLY -.-> GAP["TENSION: corpus mutation does not wait<br/>for the forum's successful state transition<br/>relay-consumer.js:497"]
    PARSE["Deadline parser byte-slices s[..19] and s[19..]: a multibyte character<br/>at byte 19 panics instead of returning None; the offset sign is applied correctly<br/>ontology_governance.rs:282,289,313,330"] --> EXP
```

**What it shows.** Projection failure can coexist with a stored, broadcast decision. The consumer's independent path does not turn a closed forum row into a mutation veto. The parser adds a separate malformed-input risk.

**Why it is this way.** ADR-2013 implements expiry as a case transition. Agentbox ADR-2109 applies the human event directly. The audit verifies the consumer gap with signed expired-proposal fixtures, while the whole deployed crossing remains untested. **Tension:** the Agentbox apply path has no expiry or projection input. The orchestrator checks the request's `d` tag and IRI, then calls `applyOntologyDecision` (`../project/agentbox/management-api/adapters/orchestrator/local-process-manager.js:444,449,466`), and no non-test agentbox source reads `stale_after`. A decision the forum has already closed as expired is still applied. **Debt:** `parse_rfc3339_utc` checks the byte length, then indexes the string with `s[..19]` and `&s[19..]` (`../nostr-rust-forum/crates/nostr-bbs-core/src/ontology_governance.rs:284,289,313`). A `stale_after` value whose 20th byte falls inside a multibyte UTF-8 character panics on the relay's proposal admission path (`../nostr-rust-forum/crates/nostr-bbs-relay-worker/src/relay_do/nip_handlers.rs:417`), where the doc comment promises `None` for anything the parser will not vouch for (`../nostr-rust-forum/crates/nostr-bbs-core/src/ontology_governance.rs:246-250`). `s.get(..19)` would close it. This is source reading only: no panic was reproduced.

## ES-13.2 A checkpoint can include a later unfinished decision

```mermaid
sequenceDiagram
    participant IN as Forum event callback
    participant SEEN as Shared seen map
    participant Q as Serial handler chain
    participant DISK as Cursor checkpoint
    participant NEW as Restored consumer
    IN->>SEEN: Add event A before dispatch<br/>relay-consumer.js:464
    IN->>Q: Queue A handler<br/>relay-consumer.js:496
    IN->>SEEN: Add event B while A runs<br/>relay-consumer.js:464
    IN->>Q: Queue B handler<br/>relay-consumer.js:496
    Q->>DISK: A settles: serialise whole seen map, including B<br/>relay-consumer.js:470,395
    Q->>Q: B starts, fixture blocks it before completion
    DISK->>NEW: Restore A and B as seen<br/>relay-consumer.js:369,376
    IN->>NEW: Redeliver B
    NEW->>NEW: Reject duplicate, no handler runs<br/>relay-consumer.js:459
    Note over DISK,NEW: TENSION: persisted seen does not imply completed handler<br/>relay-consumer.js:464,395
```

**What it shows.** The checkpoint records shared receipt state, not just completed work. A deterministic snapshot/restore fixture reproduced suppression of B before its original handler completed.

**Why it is this way.** The consumer serialises handlers to avoid concurrent Git index operations, but marks delivery in a shared map before that serial boundary. Its completion callback persists all deliveries. Governance records remain on disk, but omit the original signature; recovery by another reconciler was not demonstrated. **Drift:** the comment above `settle` says a crash mid-apply leaves the decision re-deliverable (`../project/agentbox/mcp/nostr-bridge/relay-consumer.js:466-469`). That holds only for the event whose handler is running when the process stops. A later event already added to `seen` (`../project/agentbox/mcp/nostr-bridge/relay-consumer.js:464`) is persisted by the earlier event's `settle` (`../project/agentbox/mcp/nostr-bridge/relay-consumer.js:473,395`) and is lost on restart.
