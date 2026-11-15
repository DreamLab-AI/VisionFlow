---
id: ADR-2024
title: Separate provenance roles and bind certification to reviewed content
date: 2026-11-15
decision_status: accepted
implementation_status: partial
activation_status: inactive
supersedes: []
superseded_by: []
amends: [ADR-2023]
verified_commit:
verified_paths: []
owner: jjohare
review_trigger: A consumer changes provenance interpretation, certification digest encoding or legacy stamp handling
repo: visionflow
evidence_class: source+test
acceptance_boundary: Owner-authorised contract and partial producer implementation; consumer integration and release not complete
---

# ADR-2024 — Separate provenance roles and bind certification to reviewed content

## Context

The owner authorised this contract and its implementation on 15 November 2026.
Loom ADR-141 defaults missing unit provenance to `process:vault`, with no verification.
Ontomark ADR-OM-012 admits dated human certification but records that subsequent edits
can leave changed values certified. Bundle compilation, content attribution, logical
derivation, human review and governance admission make different claims.

## Decision

1. Content attribution and source evidence remain explicit page metadata. Missing
   attribution remains unknown. Compilation never invents a human or agent author.
2. Bundle build provenance names the compiler and its version separately from source
   generation, input digests and artefact hashes. A build receipt proves processing,
   not authorship, truth, review or governance approval.
3. Logical derivation records retain their checker, rule/profile scope and limitations.
   Governance admission retains its signer and receipt separately from human review.
4. Explicit page `generated` and `verified` records survive parsing, editing and export.
   Legacy `process:vault` remains historical processing attribution. Legacy bundles
   remain readable; missing build metadata is unknown, never guessed from the reader.
5. New human review stamps bind to a versioned SHA-256 digest of their reviewed scope.
   Key-scoped review hashes only the named values; whole-page review includes all
   frontmatter except the review list itself, and the body. Missing values, scalar
   types, sequence order and nested structures are distinguished. Map order is not
   semantic; keys are ordered before hashing. No hand-written cryptography is used.
6. A changed reviewed value makes that stamp stale. An unrelated field edit does not
   invalidate key-scoped review. Historical stamps are retained, not silently rewritten.
   Undigested legacy stamps remain visible as unbound historical review and continue
   protecting their fields against automated ingest; they cannot claim fresh binding.
   A stale stamp likewise keeps automated ingest from overwriting reviewed fields.
7. Digest attachment is an explicit review operation, never an automatic re-certification
   during build or an ordinary edit. Human actor and date remain required; a digest is
   an integrity binding, not a signature or proof that the actor actually reviewed it.
8. Ontomark owns the portable digest and status rules. Compiler, ingest and consumer
   adapters use shared fixtures. Public metadata never discloses private content;
   working-vault input stays within the private machine-bundle boundary.

## Consequences

Consumers can distinguish known processing from unknown authorship and distinguish
historical review from review of current values. Existing evidence and legacy readers
remain usable. Strict freshness is additive and must be displayed honestly; no bulk
certification, identity reminting or corpus reinterpretation is authorised by this ADR.
Versioned JSON-LD contexts retain immutable existing meanings; new fields require a
versioned context. Source completion, publication and deployment have separate receipts.

## Verification

Required controls: legacy/no-producer bundle; explicit producer round-trip; absent
content attribution; preservation of explicit attestations; whole-page and per-key
bindings; unchanged values; changed covered value/body; unrelated key edit; map-order
invariance; sequence/type distinctions; malformed digest; agent/process refusal;
legacy unbound and stale ingest protection; mixed current/stale stamps; public/private
projection canaries; native/WASM parity and supported consumer builds.

Implementation has not yet been verified. This section must record actual commands,
revisions, results and release/consumer boundaries before completion is claimed.

## Implementation checkpoint

The producer branch implements shared content-binding digests, current/stale/unbound
counts in compiler and ingest, explicit bundle `built_by` metadata, and a read-only
`review-binding` CLI operation. Parser unit tests, compiler no-default-feature suite,
focused ingest and cross-adapter tests, scoped Clippy and CLI compilation passed.

Open delivery work: publish freshness through a versioned API/context; propagate the
producer and freshness contracts through Loom and VisionClaw; add CLI end-to-end,
malformed-scope, private projection and WASM conformance checks; reconcile all required
release goldens; version and release the changed crates; run consumer compatibility
and required independent review. No source merge, publication or deployment is claimed.
