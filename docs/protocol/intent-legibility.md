# Pre-Action Intent Legibility — Declared Intent Before Acting

**Status:** Canon position (`standalone`); `theory→canon`. Carries normative clauses (RFC 2119 MUST / MUST NOT) on pre-action intent declaration in agent embodiment surfaces
**Date:** 2026-07-08
**Discharges:** register gap **D7** — *No pre-action intent legibility; embodiment shows only the past* (`theory→canon`, Major)
**Governed by:** [PRD Gap-Close Canon](../PRD-gap-close-canon.md) §"Owned Items" (D7 row); reconciled as a protocol position page per [ADR-005](../ADR-005-gap-close-canon-decisions.md) §Decision 4 (position documents as protocol pages, standalone-first)
**Builds on:** [DID:Nostr Identity Spine](identity-spine.md) (identity primitive); the `did:nostr`-keyed agent-events envelope (COM-14, the attribution root)
**Book cross-reference:** "The Gap Register" (Chapter 14b, desktop register); "Four Surfaces" (Chapter 12a)

## 1. What this is

The embodiment surface renders what an agent *did*. It has never rendered what an
agent is *about to do*. The register names the omission directly: *No pre-action
intent legibility; embodiment shows only the past* (Chapter 14b, desktop register,
Major), scored `theory→canon` because the remedy is a canon design position, not a
code fix the substrate can invent for itself. This document is that position. The
desktop observation surface renders against it; the agent-events envelope carries
the field it renders.

`identity-spine.md` fixes *who* an agent is (`did:nostr`) and COM-13 makes each
agent-authored act *individually* legible. Neither says anything about the *tense*
of what a surface shows: both describe the act that already happened. D7 governs the
forward view — the declared intent an agent states *before* it acts, so a human
watching the surface has an intervention window rather than only a record of
consequences. This is the projective layer (Endsley's Level 3) over the descriptive
past, not a second identity model.

## 2. Framing (from the book)

Chapter 14b diagnoses the desktop as the surface whose *"entire reason to exist"* is
to make distributed agent activity legible, and finds it renders only the past. The
register's own words fix the gap and the harm precisely:

- **The gap.** *"Embodiment renders only what already happened, so beams show that
  an agent touched a node but never what an agent is about to do, and with no
  plan-preview or intent overlay the human can react to consequences without
  pre-empting them, forgoing the intervention window that intent-legibility practice
  targets"* (14b §Desktop, citing Nielsen2025Intent).
- **The standing state.** Chapter 12a scores the same surface *"observation is
  shipped; steering is partial"* and places it exactly where *"projective situation
  awareness — Endsley's Level 3 — has no settled visual grammar."* D7 is the canon
  supplying that grammar's minimum: a declared intent, shown before the act.

The design answer is not a new autonomy: it is a *pre-action* legibility affordance.
An agent states, on the same signed envelope that will carry the act, what it is
about to do; the surface renders that statement ahead of the effect; the human keeps
the window to pre-empt rather than only to review. The whole value is in the tense —
declared *before*, or it is not intent legibility at all, only a restatement of the
past the register already faults.

## 3. The reference implementation — VisionClaw's declared-intent field (D7)

VisionClaw shipped the affordance this position governs this sprint; it is the
reference implementation, and the consumer whose citation promotes this page to
`integrated`.

- **Additive envelope field.** `AgentActionEnvelope` gains `intent: Option<String>`,
  additive and versioned (`#[serde(default, alias = "declared_intent")]`), so a
  producer that declares no intent still deserialises to `None`, both the register
  spelling `intent` and the fuller `declared_intent` parse, and the identity-blind
  binary projection is unaffected — intent is legibility metadata carried on the
  same envelope as the act, keyed to the acting agent's `did:nostr`
  (`src/agent_events/schema.rs`).
- **Pre-action display.** The `AgentDetailPanel` steering surface renders a distinct
  *"Declared Intent — About to: &lt;declared action&gt;"* block **above** the
  past-activity "Current Task", shown only when a declared intent is present and
  never fabricated (`AgentDetailPanel.tsx`, mapped through `BotsDataContext` from the
  node's `declared_intent` / `intent` metadata).

This document states the model the field and the panel implement; the sections below
are what a conformant declared-intent affordance must honour, not a description of
one product's UI.

## 4. Normative model

### 4.1 What an intent declaration MUST contain

**MUST — a declaration is a forward statement of the pending act.** A declared
intent MUST state, in human-readable terms, the action the agent is *about to*
take — the operation and the target it will act on — at a granularity sufficient for
a human to decide whether to intervene. *"Working"* or a bare status is not an intent
declaration; *"about to rewrite the budget node with Q3 figures"* is.

**MUST — the intent rides the agent-events envelope, attributed to the actor.** The
declaration MUST be carried as an additive field on the same signed agent-events
envelope that carries (or will carry) the action, keyed to the acting agent's
`did:nostr` (COM-14). The intent MUST be attributable to the exact identity that
will perform the act; it MUST NOT be carried on an out-of-band channel a surface
cannot bind to that identity. An intent no one can attribute is not legible.

**MUST — declaration precedes effect.** The intent MUST be emitted and renderable
*before* the action's effect is committed and rendered, so the human retains the
intervention window the register names. A declaration that arrives only with, or
after, the effect is not a pre-action declaration and does not discharge this
position.

### 4.2 When intent may be absent

**MAY — a non-mutating act carries no intent.** An agent performing a read-only or
otherwise non-mutating action (a query, an observation) need not declare an intent;
there is no consequence to pre-empt. The field is optional and legitimately absent
for such acts, which is why the envelope field is `Option` and additive: absence is
a valid, backward-compatible state, never an error and never a block on the act.

**MAY — a producer not yet upgraded carries no intent (transitional).** Until a
producer is upgraded to declare intent, the field is absent on its envelopes. This
transitional absence is legitimate, but it is still absence, not silence: §4.3
governs how a surface must render it. A consequential, mutating act SHOULD carry a
declared intent; where it does not, the surface MUST make the *absence* visible
rather than imply the act is benign.

### 4.3 The honesty rule — never fabricate an intent after the fact

**MUST NOT — no post-hoc intent.** An intent is a pre-action declaration or it is
nothing. A producer or surface MUST NOT synthesise, infer, or backfill an intent
after the action has occurred and present it as though it had been declared
beforehand. The declared-intent field carries only what the agent stated *before*
acting.

**MUST — declared, not inferred.** The intent shown to a human MUST be the agent's
own declared statement, carried on the envelope — never a surface's guess derived
from the action's observed effect. Inferring *"it must have intended X because it
did X"* collapses the pre-action legibility this position exists to provide back
into a restatement of the past, which is exactly the failure the register faults
(*"embodiment shows only the past"*).

**MUST — absence renders as absence.** When no intent was declared, a conformant
surface MUST render *nothing* in the declared-intent slot — no "About to" line, no
placeholder, no inferred stand-in. Absence of a declared intent is a legible,
honest state; filling it to complete a panel manufactures a claim the agent never
made.

## 5. What a conformant embodiment surface must show a human

| Signal | Requirement | Source norm |
|---|---|---|
| Declared intent (pre-action) | Rendered *before* the action's effect, when present | D7 §4.1 |
| Target + operation of the pending act | Named at a granularity a human can act on | D7 §4.1 |
| Attribution to the acting agent's `did:nostr` | Intent bound to the same identity that will act | COM-14 / identity-spine |
| Absence of a declared intent | Rendered as absence, never a fabricated placeholder | D7 §4.2, §4.3 |
| Past activity (what already happened) | Shown distinctly from, and below, the declared intent | D7 §4.3; register 14b |

A surface that renders only past activity for a consequential act, or that fills the
declared-intent slot with an inferred or post-hoc statement, is non-conformant
against this position even when every rendered item is COM-13-compliant on identity.

## 6. Relationship to embodiment and the identity spine

D7 sits directly above the identity stack and does not restate it. COM-14 makes the
agent node addressable and its envelope attributable; `identity-spine.md` defines the
`did:nostr` key; COM-13 renders the acting agent's identity. This document adds the
*temporal* layer to that same envelope: where identity-spine answers *who* and COM-13
answers *is this from an agent*, D7 answers *what is that agent about to do*. An
intent that cannot be attributed to a `did:nostr`-keyed actor is unbuildable as a
legibility signal, so a substrate that has not keyed its agent-events envelope by
`did:nostr` cannot implement D7; implement the identity keying first, and D7 rides
the envelope above it.

## 7. Consumer and integration condition

This position is authored `standalone` — a canon position now carried on the
agent-events envelope but promoted only on citation. It reaches `integrated` when
**VisionClaw's D7 implementation** cites it by name as the canon contract its
declared-intent envelope field (`AgentActionEnvelope.intent`) and its
`AgentDetailPanel` pre-action display honour. VisionClaw is the named substrate for
D7 (register: Desktop, VisionClaw); the reference implementation in §3 is the
consumer this document names, and citation by that implementation promotes this page
from `standalone` to `integrated`. Live population of the field is the producer's
(agentbox's) to supply on real envelopes; the canon position and the substrate
affordance do not wait on it, but a live end-to-end intent is what the substrate's
own canary observes.

## 8. Falsification exposure

This position is falsified if a surface presents a post-hoc or inferred intent as a
pre-action declaration; if a declared intent is rendered with or after the action's
effect rather than before it; if the absence of a declared intent is filled with a
fabricated or inferred placeholder; if the intent is not attributable to the acting
agent's `did:nostr`; if declared intent is not shown distinctly from past activity;
or if this document is claimed `integrated` before VisionClaw's D7 implementation
cites it by name.
</content>
</invoke>
