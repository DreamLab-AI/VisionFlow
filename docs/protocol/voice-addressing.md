# Voice Addressing and Turn-Taking

**Status:** Canon position (`standalone`); `theory→canon`. Carries normative clauses (RFC 2119 MUST / MUST NOT) on multi-agent voice addressing, turn-taking and repair
**Date:** 2026-07-08
**Discharges:** register gap **V2** — *no multi-agent addressing or turn-taking model* (`theory→canon`, Major)
**Governed by:** [PRD Gap-Close Canon](../PRD-gap-close-canon.md) §"Position Documents (F10, V2)"; [ADR-005](../ADR-005-gap-close-canon-decisions.md) §Decision 4
**Builds on:** [DID:Nostr Identity Spine](identity-spine.md) (identity primitive); COM-14 `did:nostr` keying of agent nodes (the addressing key)
**Book cross-reference:** "The Gap Register" (Chapter 14b, voice register)

## 1. What this is

`identity-spine.md` fixes *who* an agent is (`did:nostr`) but says nothing about
*addressing* it by voice: which agent a spoken command is directed at, which agent
holds the floor when several could answer, and what happens when the addressed
agent cannot understand. The register names this gap: *no multi-agent addressing or
turn-taking model* (Chapter 14b, voice register, Major). This document is the
addressing counterpart to the identity spine. VisionClaw's COM-15 implementation
cites it as its canon contract.

The book scoped the gap precisely. The one shipped voice loop —
`VoiceInterfaceActor` — is *"honestly the best-behaved thing in the register"*
(14b §Voice), with two properties this document promotes to norms: *"identity is
asserted from the operator's configured key and never from the transcript"* and an
unrecognised utterance *"falls back to a read-only query rather than a silent
mutation"*. What it lacked, and what the register flags, is a *multi-party* model:
*"Nothing tells the human which agent is listening: awareness cues are
one-directional, two competing swarm-keyword paths fire on the same audio with no
floor management, and multi-party addressing has no convention here at all"* (14b,
citing SpeakStaySilent2026), and *"there is no grounding or repair move ---
unrecognised speech degrades to silence or a read-only default, never a clarifying
question"* (14b, citing GroundingRifts2025).

## 2. The shipped primitive — PTT-to-selected-actor (COM-15)

The addressing primitive is **push-to-talk bound to the selected actor**, shipped
as COM-15 (landed P1). It is the base case on which the multi-party model is built.

- **Selection is the address.** Under PTT, the utterance is addressed to exactly
  the actor currently selected on the surface, resolved to that actor's
  `did:nostr` (COM-14). The transcript never names or re-targets the actor; the
  selection does. A transcript cannot assert an address it does not hold, exactly
  as it cannot assert an identity.
- **Signed and scoped.** The addressed utterance dispatches a scoped, signed ACSP
  request (kind 31402) to the selected actor's `did:nostr`, authored under the
  operator's configured key.

COM-15 is single-address: one operator, one selected actor, one floor. Everything
below defines the semantics that make that primitive well-formed and extends it to
the case where more than one actor could respond.

## 3. Normative model

### 3.1 Addressed-utterance semantics

**MUST — a PTT utterance is addressed to the selected actor only.** An utterance
captured under PTT MUST be addressed to exactly one actor: the one selected on the
surface at capture time, identified by `did:nostr`. It MUST NOT be broadcast to
multiple agents, and MUST NOT be re-addressed by any name, keyword or wake-word
appearing in the transcript.

**MUST — no selection, no mutation.** When no actor is selected, an utterance MUST
degrade to a read-only query against the operator's own settings assistant (the
shipped fallback), and MUST NOT mutate the state of, or dispatch a governed action
to, any agent. Selection is a precondition of any addressed, state-changing
utterance.

**MUST NOT — transcript-asserted identity or authority.** The operator's identity
and the utterance's authority MUST derive from the operator's configured key, never
from the transcript text. Wake-words and agent names in the transcript are content,
not credentials.

### 3.2 Turn-taking and floor control

**MUST — exactly one actor holds the floor.** For any addressed utterance, exactly
one actor holds the floor: the selected actor. The surface MUST NOT run two or more
recognition or routing paths that can each act on the same audio (the register's
*"two competing swarm-keyword paths fire on the same audio with no floor
management"* is the failure this forbids). A single floor token, held by the
selection, arbitrates.

**MUST — deterministic arbitration when several could respond.** When more than one
agent *could* respond to an utterance (overlapping capability, ambient keyword
match), the addressed actor is still the selected one, deterministically. Capability
overlap MUST NOT cause a race; it is resolved by selection, not by which path fires
first. To address a different actor, the operator re-selects and re-holds PTT.

**MUST — bidirectional listening cue.** Before an addressed utterance is acted upon,
the surface MUST show the human which actor is addressed and listening — the
`did:nostr`-resolved identity of the selected actor (COM-13 / identity-spine),
rendered on the surface the operator is watching. Awareness MUST be bidirectional:
the human knows which agent hears them, not only the agent knowing it was
addressed. This closes the register's *"nothing tells the human which agent is
listening"*.

**MUST — floor release is explicit.** The floor is held for the duration of PTT and
released on PTT release or on the addressed actor's turn completing. A held floor
MUST NOT be pre-empted by another agent's ambient activity; barge-in, where offered,
MUST be an explicit operator action, not an agent-initiated seizure.

### 3.3 Clarification and repair turn (V3)

**MUST — repair instead of silent degradation.** When the addressed actor cannot
ground the utterance — low transcription confidence, an ambiguous referent, an
out-of-grammar verb, or a selection that does not resolve to a `did:nostr` — it
MUST issue a **clarification/repair turn** (a spoken or surfaced question that names
what it could not ground) rather than degrade to silence or a silent read-only
default. This is the grounding move the register found absent, where voice agents
*"issue repair requests far less often than humans do"* (GroundingRifts2025).

**MUST — repair preserves the address and the floor.** A repair turn is addressed
back to the same operator and keeps the floor on the same actor; the operator's
answering utterance is grounded against the open repair, not treated as a fresh
unaddressed command. The repair turn MUST NOT silently switch the addressed actor.

**Scope note.** VisionClaw lands the repair turn (**V3**) this wave. The canon here
states the model — the repair turn is mandatory, addressed, and floor-preserving;
VisionClaw's V3 implements it and cites this section as its contract.

## 4. Relationship to the identity spine

Voice addressing is unbuildable without `did:nostr`-keyed agent nodes: the register
records that agent nodes keyed by ephemeral `task_id` *"cannot be addressed by ACSP
nor targeted by voice"* (14b §Desktop). COM-14 (the V1 addressing root) makes the
node addressable; `identity-spine.md` defines the key; COM-13 renders the listening
cue in §3.2. This document sits directly above that stack: it is the addressing and
turn-taking model, not a second identity model. Where identity-spine answers *who*,
this answers *addressed how, and whose turn*.

## 5. Consumer and integration condition

This position is authored `standalone`. It reaches `integrated` when **VisionClaw's
COM-15** cites it by name as the canon contract its PTT-to-selected-actor loop and
its V3 repair turn implement. The shipped COM-15 primitive (§2) and the V3 repair
turn (§3.3) are the two consumers this document names; citation by COM-15 promotes
it from `standalone` to `integrated`.

## 6. Falsification exposure

This position is falsified if a PTT utterance is broadcast to more than one agent or
re-addressed from the transcript; if an unselected utterance mutates agent state; if
two recognition paths can act on the same audio without a single floor token; if the
human is not shown which actor is addressed before it acts; if the addressed actor
degrades an ungrounded utterance to silence instead of a repair turn; or if this
document is claimed `integrated` before VisionClaw's COM-15 cites it by name.
