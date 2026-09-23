---
id: AB-35
title: Level 2 — q against k, authorise then finalise, and the signer service outside the restore domain
area: agentbox
governing:
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/PROTOCOL-registry.md
adrs: [ADR-2098, ADR-2101, ADR-2105]
sources:
  - ../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md
  - ../project/agentbox/docs/proposals/sovereign-settlement-research/REVIEW-kofn-consensus-gpt6-astra.md
  - ../project/agentbox/crates/sidestr/sidestr-nostr/src/round.rs
  - ../project/agentbox/crates/sidestr/sidestr-nostr/src/kinds.rs
  - ../project/agentbox/docs/proposals/sovereign-settlement.md
  - ../project/agentbox/docs/INGRESS-identity.md
  - ../project/agentbox/docs/PROTOCOL-registry.md
verified_commit: ec60a8f14f4544520b4b1f6e8f5de2def4cfedcf
---

## For developers

Nothing here is built. A consultant review on 2026-09-22 (GPT-6 Astra through the Codex consultant tier, see AB-15.12 for that path) took ADR-2101's k-of-n federation apart and returned a different shape, which the record then adopted (`ADR-2101-federation-topology-and-key-separation.md:133-136`). The change that matters most: a threshold signature is not a consensus protocol, so `q` and `k` are two numbers, and the block a federation signs is not the block it finalises.

**Drift (this topic vs sidestr-rs):** "nothing here is built" held at `ec60a8f14`; since then `sidestr-round` (0.1.x, now 0.2.0) has shipped upstream's availability-tolerant round (not the reviewed BFT shape) and, with the other crates, moved to `DreamLab-AI/sidestr-rs` under ADR-2112. Not re-stamped here; see SR-01.6.

## For the business

The plan to have several machines jointly authorise payments was reviewed by an independent model and found to be under-specified in ways that could lose money rather than merely stall. The corrected design costs more to build, and the review separates what can wait until the chain carries real value from what cannot. Only the wire formats exist in code today.

## AB-35.1 Two thresholds, not one

```mermaid
flowchart TB
    Q["q, the CONSENSUS certificate threshold"]
    K["k, the BLOCK and PEG authorisation threshold"]
    Q --> S["safety needs 2q minus n greater than f, so the intersection<br/>of any two quorums holds an honest participant<br/>REVIEW-kofn-consensus-gpt6-astra.md:42"]
    Q --> L["liveness needs q at most n minus f"]
    S --> N["together these require n at least 3f plus 1<br/>REVIEW-kofn-consensus-gpt6-astra.md:46"]
    L --> N
    K --> WRONG["2k minus n greater than f ALONE is not a smaller<br/>Byzantine model: it buys neither safety nor liveness<br/>ADR-2101-federation-topology-and-key-separation.md:140"]
    N --> FIVE["for five instances: q equals 4, k equals 4, f equals 1<br/>ADR-2101-federation-topology-and-key-separation.md:141"]
    WRONG --> FIVE
    FIVE --> TRAP["do NOT mechanically use 2f plus 1 when n is above 3f plus 1:<br/>three votes out of five are insufficient for f equals 1<br/>REVIEW-kofn-consensus-gpt6-astra.md:56"]
    FIVE --> CUST["a 3-of-5 CUSTODY threshold may sit behind 4-of-5 consensus<br/>only if every honest signer requires a decision certificate<br/>before signing - ADR-2101-federation-topology-and-key-separation.md:141-142"]
    FIVE --> THREE["three instances is a CRASH-ONLY research profile, f equals 0,<br/>and must be declared as one<br/>ADR-2101-federation-topology-and-key-separation.md:143"]
```

**Invariant:** the safety inequality and the liveness inequality must both hold, which is what forces `n` at least `3f + 1`; assuming only the safety one produces a model with neither property (`../project/agentbox/docs/proposals/sovereign-settlement-research/REVIEW-kofn-consensus-gpt6-astra.md:46`).

**Open:** PRD-024 question 13 defers the fault model itself — how many Byzantine signers, and how independent the operators, credentials, release channels and parent nodes behind the `n` keys actually are (`../project/agentbox/docs/proposals/sovereign-settlement.md:494-496`).

## AB-35.2 Two decisions per block — authorise the template, then finalise the block

```mermaid
sequenceDiagram
    autonumber
    participant PP as the proposer for a height
    participant CS as the consensus protocol, ABOVE the signature<br/>docs/adr/ADR-2101-federation-topology-and-key-separation.md:149
    participant SG as the other signers
    participant CH as the chain

    PP->>CS: propose a canonical UNSIGNED template
    Note over CS: AUTHORISE_TEMPLATE binds the template, the height, the exact<br/>predecessor hash, chain and genesis, epoch and rules<br/>REVIEW-kofn-consensus-gpt6-astra.md:203
    CS->>CS: decide with q
    CS-->>SG: the template decision certificate
    Note over SG: an honest signer releases its BIP-325 signature ONLY for an<br/>authorised template (ADR-2101-federation-topology-and-key-separation.md:159-160)
    SG-->>PP: partial signatures, and ANY k valid ones seal the block
    Note over PP: a signing subset is NEVER chosen before signatures exist<br/>docs/adr/ADR-2101-federation-topology-and-key-separation.md:161
    PP->>PP: seal, which recomputes the merkle root and grinds a nonce<br/>AFTER the witness is inserted
    Note over PP: so the template's identity is NOT the sealed block's hash, and<br/>different valid witness subsets give DIFFERENT hashes<br/>docs/adr/ADR-2101-federation-topology-and-key-separation.md:156-158
    PP->>CS: the sealed block
    CS->>CS: decide FINALISE_BLOCK over the exact sealed hash<br/>and the template decision (REVIEW-kofn-consensus-gpt6-astra.md:206)
    CS-->>CH: external release and the next height use the FINALISED hash<br/>docs/adr/ADR-2101-federation-topology-and-key-separation.md:160-161
```

**Invariant:** because sealing changes the block's hash, one consensus decision cannot cover both the thing signed and the thing published — a design with a single decision per block is unsound however large its quorum (`../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md:156-158`).

**Tension (ADR-2101 amendment vs the review):** the earlier amendment's rule that a signer never signs two proposals at one height is over-stated — honest replicas may vote for different candidates in different views, and what is actually forbidden is a second value in one `(epoch, view, phase, instance)` slot, breaking a lock, or signing a conflicting decided block (`../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md:144-147`); `sidestr-nostr`'s codecs still carry the over-stated rule in their own prose (`../project/agentbox/crates/sidestr/sidestr-nostr/src/round.rs:20-22`).

## AB-35.3 The wire problem — five ephemeral kinds carrying a safety protocol

```mermaid
flowchart TB
    subgraph have["What exists in code: codecs only"]
        C1["23510 proposal, 23511 partial signature, 23514 sealed block<br/>crates/sidestr/sidestr-nostr/src/round.rs:4-8"]
        C2["23512 peg-out PSBT keyed by the burn's outpoint,<br/>23513 the co-signature<br/>crates/sidestr/sidestr-nostr/src/round.rs:9-13"]
        C3["the ROUND LOGIC is deliberately NOT here: who may propose<br/>when, one signature per height, the re-sign-after-timeout<br/>rule, the seal - crates/sidestr/sidestr-nostr/src/round.rs:15-18"]
    end
    subgraph broken["Why those kinds cannot carry it"]
        B1["23510 to 23514 are EPHEMERAL under NIP-01: relays are not<br/>expected to retain them<br/>REVIEW-kofn-consensus-gpt6-astra.md:217"]
        B2["chain is not a single-letter tag, so it is not a portable<br/>indexed filter - REVIEW-kofn-consensus-gpt6-astra.md:217"]
        B1 ~~~ B2
    end
    subgraph want["The profile the review asks for"]
        W1["REGULAR stored kinds in the estate's own band, with<br/>single-letter routing tags<br/>ADR-2101-federation-topology-and-key-separation.md:162-164"]
        W2["a canonical signed payload: protocol version, scope, epoch<br/>and configuration hash, instance and height, view and phase,<br/>candidate and predecessor digests, justification digest,<br/>logical signer id and incarnation<br/>ADR-2101-federation-topology-and-key-separation.md:164-166"]
        W3["an OUTER transport signature and an INNER consensus<br/>signature - ADR-2101-federation-topology-and-key-separation.md:166"]
        W4["certificates persisted locally and re-published, fetched by<br/>DIGEST, with no since or created_at rule anywhere in the<br/>safety path - ADR-2101-federation-topology-and-key-separation.md:167-168"]
    end
    have --> broken
    broken --> want
    want --> BAND["the estate band for this is unallocated: 38420 to 38425 are<br/>spent on the binding and the five domain events<br/>see AB-34.4"]
```

**Debt:** the five round kinds are ported and tested as wire formats while the protocol they were designed for has been rejected, so `sidestr-nostr` ships codecs for a shape the governing record no longer intends to run (`../project/agentbox/crates/sidestr/sidestr-nostr/src/round.rs:15-22`, `../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md:148`).

**Open:** the stored kinds the review requires have no allocation in the registry, whose sidestr section still records only the external `2xxxx`/`3xxxx` set and `38420`-`38425` (`../project/agentbox/docs/PROTOCOL-registry.md:163-171`).

## AB-35.4 What a finality proof is, and what a quorum certificate is not

```mermaid
flowchart TB
    F1["genesis and profile"] --> F2["configuration chain"]
    F2 --> F3["template decision certificate"]
    F3 --> F4["sealed header, bound to the template"]
    F4 --> F5["FINALISE_BLOCK certificate"]
    F5 --> F6["inclusion proof"]
    F6 --> DONE["the finality proof, in that order<br/>ADR-2101-federation-topology-and-key-separation.md:169-171"]
    subgraph notproof["What does NOT establish it"]
        NP1["a quorum certificate is NOT a proof of UTXO execution:<br/>a light client still trusts the federation's VALIDITY<br/>attestations - ADR-2101-federation-topology-and-key-separation.md:171-172"]
        NP2["next-coinbase embedding is ARCHIVAL REINFORCEMENT,<br/>not finality - ADR-2101-federation-topology-and-key-separation.md:172"]
        NP1 ~~~ NP2
    end
    DONE --> notproof
    notproof --> PEG["peg-out is an ORDERED PAYMENT INTENT with a durable payout<br/>state machine. One signature per burn is replaced by an<br/>exclusivity rule that survives fee replacement<br/>ADR-2101-federation-topology-and-key-separation.md:173-174"]
```

**Invariant:** validity and finality are different claims, and conflating them is what makes a quorum certificate look like proof that the chain's rules were applied (`../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md:171-172`).

## AB-35.5 The signer service, outside the restore domain

```mermaid
flowchart TB
    PROB["A journal counter is NOT anti-rollback: an epoch or<br/>incarnation stored in the restored snapshot rolls back WITH<br/>the snapshot, and a fencing event does not revoke a private<br/>key's ability to make valid Bitcoin signatures<br/>REVIEW-kofn-consensus-gpt6-astra.md:316"]
    PROB --> SVC["put the safety journal and the signing authority in a<br/>SEPARATE signer service whose state is not restored with the<br/>application host - REVIEW-kofn-consensus-gpt6-astra.md:318"]
    SVC --> HOST["the application host becomes a REPLACEABLE REQUESTER<br/>REVIEW-kofn-consensus-gpt6-astra.md:318"]
    SVC --> ENF["the service itself enforces slots, locks, decision checks,<br/>payout policy and incarnation fencing<br/>ADR-2101-federation-topology-and-key-separation.md:176-177"]
    ENF --> INSUFF["a generic sign-digest endpoint is INSUFFICIENT<br/>ADR-2101-federation-topology-and-key-separation.md:177-178"]
    SVC --> FENCE["parent custody fencing ultimately means key isolation or<br/>MOVING THE FUNDS - under an isolated view, recovery HALTS<br/>ADR-2101-federation-topology-and-key-separation.md:178-179"]
    subgraph wait["What may wait on testnet - ADR-2101-federation-topology-and-key-separation.md:187-188"]
        W1["HSMs, independent operators, DKG, aggregation,<br/>parent checkpointing, a production DR service"]
    end
    subgraph nowait["What may NOT - ADR-2101-federation-topology-and-key-separation.md:188-190"]
        X1["durable votes"]
        X2["removal of timeout re-signing"]
        X3["the conflicting-certificate test"]
        X4["the validity-versus-finality distinction"]
        X1 ~~~ X2 ~~~ X3 ~~~ X4
    end
    FENCE --> wait
    FENCE --> nowait
```

**Invariant:** block-and-peg authorisation is ONE custody role, independent of identity and of bridge custody, because the same-descriptor convention makes the block challenge and the peg output necessarily share keys (`../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md:180-182`); AB-31.7 draws the identity, spend and sign separation this sits beside.

**Open:** ADR-2101 remains `proposed` with its ratification evidence unrun — three signer keys on three hosts sealing a block with the proposer down, and a known-answer test that `k_spend` and `k_sign` differ from `k_id` and from each other (`../project/agentbox/docs/adr/ADR-2101-federation-topology-and-key-separation.md:205-209`).
