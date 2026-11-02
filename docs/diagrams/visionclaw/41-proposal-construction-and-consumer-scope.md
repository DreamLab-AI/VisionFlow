---
id: VC-41
title: Proposal construction — levels, consequence identity and grouped consumers
area: visionclaw
governing:
  - ../project/docs/VAULT-corpus-format.md
  - ../project/docs/adr/ADR-2116-ontology-proposals-are-forum-action-requests-not-prs.md
  - ../project/docs/adr/ADR-2147-inferences-carry-a-re-checkable-certificate-and-proposals-carry-a-consequence-delta.md
adrs: [ADR-2116, ADR-2147]
sources:
  - ../project/crates/vault/src/main.rs
  - ../project/crates/vault/src/propose.rs
  - ../project/crates/vault/src/nostr.rs
  - ../project/crates/vault/src/build/turtle.rs
  - ../project/crates/vault-core/src/proposal.rs
  - ../project/agentbox/management-api/lib/ontology-apply.js
  - ../ontomark/crates/rdfs-certificate/src/lib.rs
  - ../ontomark/crates/ontomark-compile/src/write/apply.rs
verified_commit: {visionclaw: 86cdd7635582, agentbox: c57d9c0508f4, ontomark: 4b9d15df4805}
---

## For developers

This topic maps proposal preparation and its apply consumer at current source pins. [The audit](../../estate-review/2026-10-31-proposal-construction.md) uses real signed CLI dry-runs and passes their grouped payloads into the real adapter. Signing, case identity, assessment and successful application are separate properties. Certificate source matches the consumed published crate; independent checker activation remains outside this pass.

## For the business

A reviewer can receive a valid signed proposal that the application cannot apply as a group. Its consequence count can include version metadata while excluding the changed logical expression. Review needs to show supported scope and the evidence attached to the signed request, rather than treating a stable case number as a fixed assessment.

## VC-41.1 Preparation and declared level

```mermaid
flowchart TD
    I["Full page or directory manifest<br/>vault/src/propose.rs:532,706"] --> D["Drop no-ops and collect affected pages<br/>vault/src/propose.rs:539,550"]
    D --> A["Assess base and patched corpus<br/>vault/src/propose.rs:557,558"]
    A --> B["New validation and conflict findings block<br/>reasoner refusal always blocks<br/>vault/src/propose.rs:174,188,199"]
    D --> L["Start with caller level<br/>content is CLI default<br/>vault/src/propose.rs:576; vault/src/main.rs:214"]
    L --> C["Escalate created pages with unknown/provisional keys<br/>amendments do not enter this loop<br/>vault/src/propose.rs:369,577,586"]
    B --> P["Assemble proposal and attach advisory consequences<br/>vault/src/propose.rs:599,611,615"]
    C --> P
    P -->|no blockers| T["Task stakes follow selected level<br/>nostr.rs:84; vault/src/main.rs:500"]
    T --> S["Sign complete request<br/>vault/src/main.rs:514,516"]
```

**What it shows.** Both corpus states are assessed, but validation blockers are deltas on touched pages. Consequences do not automatically raise the level. The audit's taxonomy and definition amendments remain content-level.

**Why it is this way.** ADR-2147 makes the consequence delta advisory. The producer infers schema level for a narrow creation case and otherwise trusts the caller's classification.

## VC-41.2 Same case, different assessment context

```mermaid
flowchart TD
    A["Level, IRI, page, hypothesis, diff<br/>plus group set and create kind<br/>vault-core/src/proposal.rs:365"] --> C["Stable case digest<br/>vault-core/src/proposal.rs:388"]
    BASE["Base and patched emitted graphs<br/>vault/src/propose.rs:118,555"] --> V["Graphs include content-derived versionIRI<br/>build/turtle.rs:495,513"]
    V --> R["Compare non-blank asserted and derived triples<br/>rdfs-certificate/src/lib.rs:603,630"]
    R --> META["Restriction filler probe: only versionIRI differs<br/>one added and one lost triple<br/>rdfs-certificate/src/lib.rs:642"]
    R --> E["Consequences and generation outside case digest<br/>vault-core/src/proposal.rs:260,365"]
    C --> FULL["Complete proposal serialised into signed event<br/>nostr.rs:151,189,190"]
    E --> FULL
```

**What it shows.** Changing a parent elsewhere alters consequences without changing the patch's case digest. The signed event still binds its complete body. Blank-node exclusion can coexist with a non-conservative verdict caused solely by version metadata.

**Why it is this way.** Stable patch identity and RDFS closure comparison were defined separately. The ontology header participates in the latter. Neither alone proves EL equivalence or identifies the review context by case digest.

## VC-41.3 A signed group cannot be applied by this consumer

```mermaid
flowchart TD
    M["Several changed pages in manifest<br/>vault/src/propose.rs:551"] --> G["Group subject is proposal IRI<br/>page repeats IRI and pages lists notes<br/>vault-core/src/proposal.rs:225,234"]
    G --> K["Any new page makes group kind create<br/>vault/src/propose.rs:590"]
    G --> S["Postable group can be signed<br/>vault/src/main.rs:500,516"]
    K --> S
    S --> A["Consumer dispatches create or amend<br/>ontology-apply.js:384,391"]
    A --> F["Require note-page slug to answer to subject IRI<br/>ontology-apply.js:570,699"]
    F --> X["DRIFT: real grouped payload fails identity check<br/>before invoking writer<br/>ontology-apply.js:571,700"]
    F -.-> N["Further guards refuse grouped application<br/>ontology-apply.js:574,702"]
```

**What it shows.** The producer's grouped subject is a decision; the consumer expects a note. Both grouped fixtures fail before CLI, Git or HTTP calls. Later group guards independently prohibit that scope.

**Why it is this way.** Grouping provides one judgement over a multi-page change, but the consumer retains one-page write/commit guards. The mismatch prevents application; it does not silently split a decision.

**Open:** A supported all-or-nothing group apply/recovery contract, or explicit pre-signing refusal, is not established. No deployed group was submitted in this audit.

## VC-41.4 Amendment changes the supposedly immutable resource

```mermaid
flowchart TD
    B["Existing Alpha resource is original IRI<br/>proposed page changes it<br/>vault/src/propose.rs:463"] --> P["Assess before and after without continuity comparison<br/>vault/src/propose.rs:557,558,568"]
    P --> S["Proposal subject remains original resource<br/>vault/src/propose.rs:663,665"]
    S --> OLD["Writer checks current resource against proposal IRI<br/>write/apply.rs:229,230"]
    OLD --> PATCH["Apply approved patch then parse new page<br/>write/apply.rs:242,258"]
    PATCH --> V["Check introduced errors and persist<br/>write/apply.rs:282,395"]
    V --> NEW["DRIFT: fixture page now has replacement resource<br/>no after-resource equality check<br/>write/apply.rs:280,282"]
```

**What it shows.** A real signed producer output changes the resource through the real CLI writer. The old-resource equality check passes because the page still has its original identity before application.

**Why it is this way.** Validation compares findings in separately parsed states. The governing corpus format's immutable-resource invariant needs a before/after identity check, which the inspected amendment path lacks. This probe does not execute live approval or attestation.
