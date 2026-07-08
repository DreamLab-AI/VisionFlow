# P1 — COM-13: Agent-disclosure MUST (canon norm)

| Field | Value |
|-------|-------|
| Register item | COM-13 disclosure MUST (canon norm); register gap F2 (*agents indistinguishable from humans*) |
| Canon owner docs | `docs/PRD-gap-close-canon.md` §"Disclosure Norm (COM-13)"; `docs/ADR-005-gap-close-canon-decisions.md` §Decision 4; `docs/DDD-gap-close-canon-context.md` §5 (`DisclosureNorm`) |
| Canary | none — a normative position document, not a loop item (acceptance check, not a canary) |
| Branch | `gap-close/2026-07` |
| Base commit at work start | `f72d173cd57d48fcf66c1a8c42628e6e6d11511c` |
| Maturity (honest) | Canon clause `standalone` (authored this wave). The forum's reference implementation is `integrated`. Full norm reaches `integrated` when both the forum and VisionClaw cite this canon clause by name. |

## What was written

`docs/protocol/identity-spine.md` was descriptive only (canonical DID form +
verification-surface tables, zero normative language). A normative section was
added: **"Agent Disclosure Norm (COM-13) — normative"**, stated in RFC 2119 terms.
The document status line now records that it carries one normative clause.

The norm has four MUST/MUST NOT clauses, going beyond the PRD's single
interaction-time sentence to cover the author-render surface the task requires:

1. **MUST — interaction-time disclosure** (the PRD clause): an agent MUST publish
   its `did:nostr` and be resolvable to the authorising principal at interaction
   time, before its contribution is acted upon.
2. **MUST — author-render disclosure**: an agent MUST be identifiable *as an agent*
   at **every** author-render surface (message body, quoted reply, thread view,
   pinned item, topic list, notification, event card, headset nameplate, desktop
   node), naming the authorising principal.
3. **MUST — registry-sourced, not self-asserted**: the agent flag + principal MUST
   resolve from an authoritative registry keyed by `did:nostr`, not from a display
   name, avatar or message text.
4. **MUST NOT — silent degradation**: a surface that cannot resolve the registry
   MUST fail closed (withhold or mark unverified), never render agent-authored
   content as human-authored.

## Reference implementation cited (integrated)

The clause cites the forum's shipped `GET /api/agents/disclosure` as the reference
implementation, at `integrated`: a public, active-only projection of the agent
registry keyed by `did:nostr`, each entry naming the authorising principal, with the
`AgentBadge` mounted at every author-render surface. VisionClaw's COM-14 `did:nostr`
keying is named as the second consumer (desktop node nameplate + headset).

## Receipt (P0 wave record — the shipped reference implementation this cites)

From the sprint's P0 wave record (`project-state:gap-close-p0-wave-record-2026-07-08`):
```
forum: 7157a92 + fb7826859 COM-13/F2 — public GET /api/agents/disclosure
  active-only registry projection + AgentBadge wired at ALL author-render sites
  incl quoted_message/thread_view/pinned_messages/topic_list/event_card;
  census-verified 12 files 15 mounts.
VisionClaw: 4a595cc8f COM-14 consumer — did_nostr on both Agent structs, client
  agentIdentity.ts trust-key + short-DID nameplate.
```
These are the two substrates the norm names; the forum implements it, VisionClaw
carries the same identity to its surfaces.

## Acceptance mapping (PRD §"Disclosure Norm (COM-13)")

| Criterion | State |
|-----------|-------|
| The clause is present in `identity-spine.md` as a MUST | Done — four RFC 2119 clauses added. |
| Agents identifiable as agents at every author-render surface, registry-sourced, naming the authorising principal | Stated as an explicit MUST (clauses 2 + 3). |
| The forum's `GET /api/agents/disclosure` cited as the reference implementation (integrated) | Cited in §"Reference implementation (integrated)". |
| Norm reaches `integrated` when both substrates cite it | Recorded as the integration condition; canon clause held at `standalone` until then. |

## Falsification exposure

This item is falsified if `identity-spine.md` carries no agent-disclosure MUST
(it now does), or if the norm is claimed `integrated` before both substrates cite
the canon clause by name (it is held at `standalone` — the forum reference
implementation is `integrated`, the full norm is not yet claimed so).
