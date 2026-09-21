---
id: ADR-2012
title: The financial substrate is our own sidestr sidechains: record the decision in canon, mirror the kind registry, and retire Lightning-first and the SHA-256d-only anchoring note
date: 2026-09-21
decision_status: proposed
implementation_status: none
activation_status: inactive
supersedes: []
superseded_by: []
verified_commit: ec7f353da39e5883ff63ef24ab2fab5e2f69d341
owner: jjohare
review_trigger: the first sidestr chain sealed by a federated instance; upstream sidestr changing a kind number or the chain-document shape; any proposal to reinstate Lightning, NWC or an EVM rail as a planned rail; USDT-on-RGB confirmed live on mainnet
repo: visionflow
domain: BASELINE-visionflow.md
lineage: "Canon entry for PRD-024 (agentbox docs/proposals/sovereign-settlement.md). Carries no implementation; it records the ecosystem-level decision and the cross-repo routing that agentbox ADR-2096 to ADR-2103 and the sibling records implement. Extends the event-kind registry's allocation-table ownership (F4) to the sidestr kinds."
---

# ADR-2012 — The financial substrate is our own sidestr sidechains: record the decision in canon, mirror the kind registry, and retire Lightning-first and the SHA-256d-only anchoring note

## Context

This repo owns cross-repo canon: the honest status ledger (`README.md:240-256`), the
compatibility matrix (`docs/architecture/compatibility-matrix.md:10`), the cross-mesh event-kind
allocation table (`docs/protocol/event-kind-registry.md:1-8`) and the three-axis status contract
(`docs/architecture/adr-status-contract.md`). Canon currently asserts that the one `did:nostr`
keypair is "login + WAC principal + provenance author + DID subject + payment account"
(`README.md:134`, `README.md:248`) while naming no instrument behind the payment account, and
the published site states Lightning over L402 and NWC as "the rail today"
(`website/static/index.html:597`, echoed at `:1015-1016` as "the native NWC rail is the next
phase"), sourced from agentbox PRD-015 v1.2 (`docs/PRD-website.md:23-25`). No such rail was ever
built. The same sentence pins anchoring to a SHA-256d Bitcoin parent, "on testnet4 by default
and mainnet by explicit operator choice" (`website/static/index.html:597`). The owner decided on
2026-09-21 that DreamLab's own sidestr sidechains are the key and only value instrument
(PRD-024 D0 to D6).

## Decision

1. **Canon records one financial substrate: our own sidestr sidechains.** Value in this
   ecosystem is a UTXO on a chain DreamLab signs. The `did:nostr` payment-account claim is now
   backed by a named instrument and must be stated that way wherever canon repeats it
   (`README.md:134`, `README.md:248`, `docs/protocol/identity-spine.md`, the pitch and website
   copy). This record ratifies nothing on its own: every axis stays `proposed / none /
   inactive` until the implementing records are accepted and their evidence filed.
2. **Lightning-first is retired from canon.** The "Lightning over L402 and NWC as the rail
   today" and "the native NWC rail is the next phase" claims are withdrawn as unbuilt, not
   merely rescheduled; `website/static/index.html:597,1015-1016` and `docs/PRD-website.md:23-25`
   are corrected in the same change. Lightning may return only as an optional bridge on-ramp
   into a chain. `x402` and `l402` keep classifying and stay unpayable. An EVM rail stays
   rejected (agentbox PRD-015 C11 stands unamended).
3. **The SHA-256d-only anchoring note is retired.** The parent network and the sidechain header
   family are validated configuration, not a fixed property of the estate: the set is
   `{btc:testnet4-blake2b, btc:mainnet-blake2b, btc:testnet4, btc:mainnet}` with header profiles
   `{knots-blake2b-v2, sha256d}`, defaulting to upstream's BLAKE2b family, bound into the chain
   document before its genesis hash. Canon states the choice and its liveness character rather
   than a single hash family. Mainnet variants sit behind an implemented owner-and-legal gate.
4. **The kind registry mirrors the sidestr kinds.** `docs/protocol/event-kind-registry.md`
   gains rows for 23500, 23501, 23510 to 23514, 33333, 33500, 33501 and 33502 marked
   **externally owned** and provisional (upstream spec v0.0.1, field names and kinds not final),
   and for **38420 `sidestr-account-binding`** inside agentbox's 38400 to 38499 band as
   estate-owned. 33502 carries two upstream schemas and the registry says so rather than
   choosing one. The registry owns the allocation table; semantics stay with the originating
   record, per its own source-of-record rule (`docs/protocol/event-kind-registry.md:8`).
5. **Custody is stated honestly in canon.** DreamLab is its own federation: the root chain is a
   level-2 k-of-n signer set of instances we operate and is custodial; child chains are level 1,
   custodied by the root signers; an asset bridge is a custodian of the origin asset. The
   level 1 to 2 to 3 ladder is the hardening path, and canon does not describe any rung as
   trustless before it exists.
6. **No regulatory claim moves.** No cell of the VisionClaw ADR-124 §7 matrix is exempted by a
   sidechain, a bridge, RGB or client-side validation, and a fiat-referenced stablecoin adds the
   FCA stablecoin regime. Canon may not be cited as relief.
7. **Cross-repo routing.** Canon records which sibling record carries each part, so a reader
   lands on the implementing decision rather than on this summary:

   | Part | Carrier |
   |---|---|
   | Sole value instrument; clean-room Rust crates; rust-bitcoin accepted; `evm`, `pool` and `desk` rules excluded | agentbox ADR-2096 |
   | Lightning-first superseded; `pay402` gains a fixtured `sidestr` scheme | agentbox ADR-2097 |
   | `chain` and `asset` URN kinds; Nostr kind registration; chain traffic on its own program; agentbox ADR-2012 relay allowlist narrowed to identity ingress | agentbox ADR-2098 |
   | The chain is the ledger of record; balances are UTXO folds; the blocktrail `txo[]` seam opens | agentbox ADR-2099 |
   | Every settlement passes the `payment_settlement` authority gate; durable spend budget; no settlement gate fails open | agentbox ADR-2100 |
   | Root chain topology; ephemeral child chains bound at session create; domain-separated spend and signer keys amending ADR-033 | agentbox ADR-2101 |
   | Assets bridged in; RGB only as a wrapped asset behind an isolated process; ADR-124 re-sequenced for the bridged case alone | agentbox ADR-2102 |
   | Parent network and header profile as manifest configuration; the P21 gate made real on-seal | agentbox ADR-2103 |
   | rust-bitcoin port; `WebLedger` as a derived view; `credit` / `debit` removed; TXO stand-in deleted | solid-pod-rs ADR-2008 |
   | `FsPaymentStore` and `/pay/*` deleted; `AnchorConfirmer` implemented; `extraction/solid-pod-rs` deleted; contracts naming disambiguated | VisionClaw ADR-2111 |
   | D1 ledger demoted to a derived view; `solid-pod-rs` lockstep pin; `derive_subkey` frozen as a Published Language | nostr-rust-forum ADR-2012 |
   | The programme, phases and open questions | agentbox PRD-024 |

## Consequences

Canon gains its first named financial substrate, which closes the standing finding that the
payment-account claim had no living anchor on any side. Three public claims are withdrawn as
unbuilt, and the drift counter and the site copy both have to move in the same change, which
is the honest cost of having asserted a rail before building one. The ecosystem takes a
dependency on a single-author upstream specification that describes itself as provisional; the
mitigation recorded here is that consensus code is ours, clean-room and fixture-pinned. Every
repo's `solid-pod-rs` pin becomes coupled through the lockstep rule. Nothing in this record
promotes any implementation axis: under the status contract, a source-level decision cannot
promote activation, and missing deployment evidence stays unknown rather than inferred.

## Verification

Proposed; nothing built, and this record carries no implementation of its own. Ratification
evidence will be:

- `grep -rn "NWC\|Lightning" website/static/index.html docs/PRD-website.md` returning only
  bridge-on-ramp or historical phrasing, with the drift counter green at that commit.
- `docs/protocol/event-kind-registry.md` carrying the sidestr rows and 38420, with no collision
  flagged against agentbox's 38000 to 38201 block or the forum's 31400 to 31405.
- `README.md:248` naming sidestr as the instrument behind the payment-account claim, with a
  maturity word matching the implementing records' filed axes rather than exceeding them.
- The compatibility matrix gaining a settlement row whose evidence column cites the sibling
  records above by id and their verified commits.
- Each carrier record in the routing table resolvable by id in its repo's generated ADR index.
