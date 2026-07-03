# Nostr Event-Kind Registry (cross-mesh, canonical)

| Field | Value |
|-------|-------|
| Status | Canonical index (Accepted, 2026-07-03) |
| Owner | **VisionFlow `docs/protocol/`** — this file is the single cross-mesh index of record |
| Scope | Every Nostr `kind` emitted, accepted, or federated by a DreamLab mesh substrate (VisionClaw, agentbox, nostr-rust-forum, solid-pod-rs) |
| Source-of-record | The originating ADR/PRD owns each range's *semantics*; this registry owns the *allocation table* and flags collisions/gaps. It does not redefine any range. |

## 1. Why this exists

Before this file the event-kind allocation was unowned and drifting: agentbox
federates a marketplace range (38300-38304) that the forum relay's mesh
allow-list does not carry, and no single document listed which substrate owns
which range. F4 assigns ownership of the *allocation table* to
`VisionFlow/docs/protocol/` (this file). Range semantics stay with the
originating ADR/PRD cited in the table.

## 2. Allocation table

Legend — **Fed?**: is the kind in the forum relay's cross-mesh federation
allow-list (`nostr-rust-forum/.../wrangler.toml` `MESH_FEDERATED_KINDS`)?

### 2.1 Base / non-addressable kinds

| Kind | Name | Substrate / owner | Spec | Fed? |
|------|------|-------------------|------|------|
| 0 | Metadata (profile) | all | NIP-01 | no |
| 1 | Short text note | all | NIP-01 | **yes** |
| 3 | Contacts | all | NIP-01 | no |
| 1059 | Gift wrap (sealed DM, session mirror) | all | NIP-59 | **yes** |
| 9000-9020 | Group moderation actions | nostr-rust-forum | NIP-29 | no |
| 27235 | HTTP Auth token | agentbox, all auth paths | NIP-98 | no (auth, not federated) |

### 2.2 Addressable / parameterized-replaceable (30000-39999, NIP-33)

| Kind | Name | Substrate / owner | Spec | Fed? |
|------|------|-------------------|------|------|
| 30001 | Generic list | all | NIP-51 | **yes** |
| 30033 | did:nostr mesh federated-broadcast anchor (DIDNostrMesh) | nostr-rust-forum `nostr-bbs-mesh` | ADR-074 §D2 | no (mesh transport is scaffold; see FREEZE/T5) |
| 30050 | App-defined addressable data (mesh allow-listed; no dedicated in-tree handler) | forum mesh config | — | **yes** |
| 30078 | Application-specific data | agentbox, all | NIP-78 | **yes** |
| 30840 | Mobile-bridge SessionEnd digest | agentbox `nostr-pod-bridge` | sovereign_mesh mobile_bridge | no |
| 30841 | Mobile-bridge companion (reserved) | agentbox | sovereign_mesh mobile_bridge | no |
| 30910 | Moderation: ban | nostr-rust-forum | PRD-009 moderation | **yes** |
| 30911 | Moderation: mute | nostr-rust-forum | PRD-009 moderation | no |
| 30912-30916 | Moderation: mute-variant / unban (30915) / unmute (30916) | nostr-rust-forum | PRD-009 moderation | no |
| 31400 | ACSP PanelDefinition | VisionClaw `acsp`, forum, agentbox | ADR-110 | **yes** |
| 31401 | ACSP PanelState | VisionClaw, forum, agentbox | ADR-110 | **yes** |
| 31402 | ACSP ActionRequest | VisionClaw, forum, agentbox | ADR-110 | **yes** |
| 31403 | ACSP ActionResponse (human decision) | VisionClaw, forum, agentbox | ADR-110 | **yes** |
| 31404 | ACSP PanelUpdate | VisionClaw, forum, agentbox | ADR-110 | **yes** |
| 31405 | ACSP PanelRetired | VisionClaw, forum, agentbox | ADR-110 | **yes** |
| 31922 | Date-based calendar event | nostr-rust-forum | NIP-52 | no |
| 31923 | Time-based calendar event | nostr-rust-forum | NIP-52 | no |
| 31925 | Calendar RSVP | nostr-rust-forum | NIP-52 | no |
| 38000-38099 | Agent-intent (inbound agent-action request) | agentbox | PRD-004 | **yes** (38000) |
| 38100-38199 | Agent-response (outbound response) | agentbox | PRD-004 | **yes** (38100) |
| 38200 | Agent job cost estimate | agentbox `nostr-bridge` | agentbox payments | no |
| 38201 | Agent job receipt / settlement | agentbox `nostr-bridge` | agentbox payments | no |
| 38300 | LLM marketplace: capability advertisement | agentbox `llm-marketplace` | ADR-021 | **no (gap — see §3)** |
| 38301 | LLM marketplace: capability request | agentbox | ADR-021 | **no (gap)** |
| 38302 | LLM marketplace: grant | agentbox | ADR-021 | **no (gap)** |
| 38303 | LLM marketplace: deny | agentbox | ADR-021 | **no (gap)** |
| 38304 | LLM marketplace: usage receipt | agentbox | ADR-021 | **no (gap)** |
| 38305 | LLM marketplace: reserved | agentbox | ADR-021 | no |
| 39000 | Group metadata (relay-signed) | nostr-rust-forum | NIP-29 | no |
| 39001 | Group admins list (relay-signed) | nostr-rust-forum | NIP-29 | no |
| 39002 | Group members list (relay-signed) | nostr-rust-forum | NIP-29 | no |

### 2.3 Reserved band conventions (agentbox custom, PRD-004 / ADR-021)

| Band | Purpose |
|------|---------|
| 38000-38099 | Agent-intent (inbound) |
| 38100-38199 | Agent-response (outbound) |
| 38200-38299 | Agent job/payment (estimate/settlement) |
| 38300-38399 | LLM resource marketplace (ADR-021) |

## 3. Known federation gap (unresolved — owner decision required)

agentbox's per-node relay `allowed_kinds` (`agentbox.toml`) accepts and emits
**38200, 38201, 38300-38305**, but the forum relay's cross-mesh allow-list
(`MESH_FEDERATED_KINDS` in `nostr-bbs-relay-worker/wrangler.toml`) currently
ends at **38100**:

```
MESH_FEDERATED_KINDS = "1,1059,30001,30050,30078,30910,31400,31401,31402,31403,31404,31405,38000,38100"
```

Consequence: an agentbox node can *locally* accept a kind-38300 LLM-capability
advertisement, but the forum mesh will not federate it to peer relays — the
marketplace is single-node-only across the mesh.

Raising the ceiling (adding `38200,38201,38300,38301,38302,38303,38304,38305`
to `MESH_FEDERATED_KINDS`) is a one-line config change, but it widens the set
of event kinds that cross relay boundaries and therefore the federation attack
surface (DoS / spam / cross-node resource-negotiation trust). That is a
**deployment policy decision for the mesh owner**, not a mechanical closeout
edit, and is deliberately left open here. When the owner decides, update both
`MESH_FEDERATED_KINDS` and the **Fed?** column above in the same change.

## 4. References

- agentbox ADR-009 (embedded Nostr relay, `allowed_kinds`), ADR-021 (LLM
  resource marketplace kinds), PRD-004 (external-agent messaging bands)
- VisionClaw ADR-110 (Agent Control Surface Protocol, kinds 31400-31405),
  ADR-074 §D2 (did:nostr / DIDNostrMesh kind-30033)
- nostr-rust-forum: NIP-29 groups, NIP-52 calendar, PRD-009 moderation
- NIP-01, NIP-33, NIP-51, NIP-59, NIP-78, NIP-98
