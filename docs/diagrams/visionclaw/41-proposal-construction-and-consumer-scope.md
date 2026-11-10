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
  - ../project/crates/vault-core/src/vocabulary.rs
  - ../project/crates/vault-core/src/promotion.rs
  - ../project/crates/vault-core/src/consistency.rs
  - ../project/docs/VAULT-corpus-format.md
  - ../project/agentbox/management-api/lib/ontology-apply.js
  - ../ontomark/crates/rdfs-certificate/src/lib.rs
  - ../ontomark/crates/ontomark-compile/src/write/apply.rs
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93, ontomark: 6159bb6089753105ef839fd4aac46c1598a64bd2}
---

## For developers

This topic maps proposal preparation and its apply consumer at current source pins. [The audit](../../estate-review/2026-10-31-proposal-construction.md) uses real signed CLI dry-runs and passes their grouped payloads into the real adapter. Signing, case identity, assessment and successful application are separate properties. Certificate source matches the consumed published crate; independent checker activation remains outside this pass.

## For the business

A reviewer can receive a valid signed proposal that the application cannot apply as a group. Its consequence count can include version metadata while excluding the changed logical expression. Review needs to show supported scope and the evidence attached to the signed request, rather than treating a stable case number as a fixed assessment.

## VC-41.1 Preparation and declared level

```mermaid
flowchart TD
    I["Full page or directory manifest<br/>vault/src/propose.rs:532,715"] --> D["Drop no-ops and collect affected pages<br/>vault/src/propose.rs:539,550"]
    D --> A["Assess base and patched corpus<br/>vault/src/propose.rs:557,558"]
    A --> B["New validation and conflict findings block<br/>reasoner refusal always blocks<br/>vault/src/propose.rs:174,188,199"]
    D --> L["Start with caller level<br/>content is CLI default<br/>vault/src/propose.rs:584; vault/src/main.rs:214"]
    L --> C["Escalate created pages with unknown/provisional keys<br/>amendments do not enter this loop<br/>vault/src/propose.rs:369,585,594"]
    B --> P["Assemble proposal and attach advisory consequences<br/>vault/src/propose.rs:607,619,624"]
    C --> P
    P -->|no blockers| T["Task stakes follow selected level<br/>nostr.rs:84; vault/src/main.rs:509"]
    T --> S["Sign complete request<br/>vault/src/main.rs:539,541"]
```

**What it shows.** Both corpus states are assessed, but validation blockers are deltas on touched pages. Consequences do not automatically raise the level. The audit's taxonomy and definition amendments remain content-level.

**Why it is this way.** ADR-2147 makes the consequence delta advisory. The producer infers schema level for a narrow creation case and otherwise trusts the caller's classification.

## VC-41.2 Same case, different assessment context

```mermaid
flowchart TD
    A["Level, IRI, page, hypothesis, diff<br/>plus group set and create kind<br/>vault-core/src/proposal.rs:365"] --> C["Stable case digest<br/>vault-core/src/proposal.rs:388"]
    BASE["Base and patched emitted graphs<br/>vault/src/propose.rs:118,555"] --> V["Graphs include content-derived versionIRI<br/>build/turtle.rs:495,513"]
    V --> R["Compare non-blank asserted and derived triples<br/>rdfs-certificate/src/lib.rs:603,635"]
    R --> META["Restriction filler probe: only versionIRI differs<br/>one added and one lost triple<br/>rdfs-certificate/src/lib.rs:647"]
    R --> E["Consequences and generation outside case digest<br/>vault-core/src/proposal.rs:260,365"]
    C --> FULL["Complete proposal serialised into signed event<br/>nostr.rs:145,183,184"]
    E --> FULL
```

**What it shows.** Changing a parent elsewhere alters consequences without changing the patch's case digest. The signed event still binds its complete body. Blank-node exclusion can coexist with a non-conservative verdict caused solely by version metadata.

**Why it is this way.** Stable patch identity and RDFS closure comparison were defined separately. The ontology header participates in the latter. Neither alone proves EL equivalence or identifies the review context by case digest.

## VC-41.3 A grouped proposal is assessed but refused before signing

<!-- 2026-11-06 restamp ec785afa9: redrawn. 6b811f7a1 makes action_request refuse a grouped proposal, so the producer no longer signs what the consumer cannot apply -->
```mermaid
flowchart TD
    M["Several changed pages in manifest<br/>vault/src/propose.rs:551"] --> G["Group subject is proposal IRI<br/>page repeats IRI and pages lists notes<br/>vault-core/src/proposal.rs:225,234"]
    G --> K["Any new page makes group kind create<br/>vault/src/propose.rs:598"]
    G --> NOTE["CLI prints the group to stderr, unsigned<br/>vault/src/main.rs:502-507"]
    NOTE --> AR["action_request called before signing<br/>vault/src/main.rs:539"]
    K --> AR
    AR --> REF["Grouped proposal refused with an error, nothing signed<br/>nostr.rs:105-107, test nostr.rs:384-390"]
    AR -.->|single page only| S["Blocker check then sign<br/>nostr.rs:108; vault/src/main.rs:541"]
    S --> A["Consumer dispatches create or amend<br/>ontology-apply.js:384,391"]
    A --> F["Require note-page slug to answer to subject IRI<br/>ontology-apply.js:570,699"]
    F -.-> N["Group guards still refuse grouped application<br/>ontology-apply.js:574,702"]
```

**What it shows.** A grouped proposal is still built and assessed as one decision, but `action_request` now refuses it, so no grouped event reaches the consumer that cannot apply it.

**Why it is this way.** The producer-consumer mismatch recorded at the previous stamp is closed on the producer side (`6b811f7a1`, recorded in `docs/VAULT-corpus-format.md:100`). Grouping is kept for local assessment until the consumer applies a group atomically.

**Open:** A supported all-or-nothing group apply contract is still not established. The refusal (`crates/vault/src/nostr.rs:105-107`) postpones the question and does not answer it.

## VC-41.4 Amendment cannot remint the resource IRI (producer only)

<!-- 2026-11-06 restamp ec785afa9: redrawn. RESOURCE_IMMUTABLE blocker added in propose.rs by 6b811f7a1. The ontomark writer was not re-read (ontomark stamp unchanged) -->
<!-- 2026-11-06 restamp 809a859aa: redrawn. 9fa159971 - one blocker code, RESOURCE_IMMUTABLE in vault_core::promotion, with RESOURCE_MUTATED an accepted alias read through canonical_blocker_code -->
```mermaid
flowchart TD
    B["Existing page resource vs proposed page<br/>vault/src/propose.rs:463"] --> P["Assess before and after<br/>vault/src/propose.rs:557,558,568"]
    P --> RI["Per touched page compare old and patched resource<br/>vault/src/propose.rs:572-574"]
    RI -->|differs| BL["promotion.RESOURCE_IMMUTABLE blocker<br/>vault/src/propose.rs:575-576, test :826-842"]
    BL --> NP["Not postable, refused before signing<br/>nostr.rs:108-113"]
    RI -->|same| S["Proposal subject remains original resource<br/>vault/src/propose.rs:672,674"]
    S --> OLD["Writer checks current resource against proposal IRI<br/>write/apply.rs:229,230"]
    OLD --> PATCH["Apply approved patch then parse new page<br/>write/apply.rs:242,258"]
    PATCH --> V["Check introduced errors and persist<br/>write/apply.rs:291,404"]
    V --> NEW["TENSION: writer still has no after-resource equality check<br/>write/apply.rs:280,291"]
    K["the one code RESOURCE_IMMUTABLE<br/>vault-core/src/promotion.rs:134"] --> BL
    AL["BLOCKER_CODE_ALIASES maps RESOURCE_MUTATED to it,<br/>canonical_blocker_code reads it<br/>vault-core/src/promotion.rs:140, :151-156"] --> K
    VOC["corpus vocabulary still says RESOURCE_MUTATED,<br/>validation.blockers deserialises through the alias<br/>vault-core/src/vocabulary.rs:331, :335-343, test :1235"] --> AL
```

**What it shows.** The producer now blocks an amendment that changes a page's resource IRI. The consumer-side writer still checks only the old resource, so the guard lives in one place.

**Why it is this way.** The external audit repair (`6b811f7a1`) closed the path the previous stamp proved. Its record (`docs/VAULT-corpus-format.md:388`) credits `vault propose`, not the writer, and names `BLOCKER_CODE_ALIASES` as the one place an alias lives.

**Drift:** only the resource code has an alias. The corpus vocabulary, as the crate's test fixture reproduces it, also names `WHELK_INCONSISTENCY` in `validation.blockers` (`crates/vault-core/src/vocabulary.rs:927`, read back verbatim by the test at `:1229`), while the consistency probe emits `WHELK_INCONSISTENT` (`crates/vault-core/src/consistency.rs:43`) and `BLOCKER_CODE_ALIASES` has one entry (`crates/vault-core/src/promotion.rs:140`). Nothing gates on `validation.blockers` at this commit, so the mismatch is latent.
