---
id: ADR-2014
title: Own shared ontology capabilities in one Rust workspace
date: 2026-10-11
decision_status: accepted
implementation_status: partial
activation_status: staged
supersedes: []
superseded_by: []
depends_on: [ADR-2013]
verified_commit: 813a759
verified_paths: []
owner: jjohare
review_trigger: workspace creation, package ownership, licence or consumer contract change
repo: visionflow
domain: BASELINE-visionflow.md
---

# ADR-2014 — Own shared ontology capabilities in one Rust workspace

## Context

The owner requests a general-purpose, local Obsidian application and greater reuse across the estate. Markdown with machine-readable headers remains authoritative. Existing boundaries may move, obsolete implementations may be removed, and beneficial schema changes may be bulk-migrated. This proposal follows a source investigation of VisionClaw, visionGraph, Agentbox, Loom and vowl-wasm; it does not certify the running estate.

## Decision

Use [DreamLab-AI/ontomark](https://github.com/DreamLab-AI/ontomark) as the independently buildable source owner for shared semantics, Markdown processing, validation, supported reasoning, compilation, graph contracts and portable host bindings. The parser, model, reasoner, compiler and NGG1 packages are named `ontomark-*`; the portable saturation engine is `whelk-el`. Obsidian is a first-class application of this workspace; its needs do not cap the capabilities available to other consumers.

Use Rust libraries directly in native consumers and compile portable capabilities to WASM for Obsidian and browser consumers. TypeScript supplies host APIs and presentation. Native consumers need not execute WASM to share the same implementation. Dependency direction is host adapter → capability → semantic model, with no reverse dependency on Obsidian, VisionClaw, Loom, a database or an estate endpoint.

The proposed capability boundaries are:

| Boundary | Responsibility |
|---|---|
| Markdown document model | Parse headers and source spans; preserve unknown metadata and prose; lossless guarded patches |
| Ontology model | Typed entities, axioms, evidence, stable identities and explicit resolution outcomes |
| Validation and reasoning | Schema/policy validation separately from declared logical capabilities; shared Whelk adapter |
| Compilation | Local/public projection, canonical artefacts, generation identity and diagnostics |
| Graph protocol and rendering | `ontomark-ngg1` owns the binary codec; Ontomark `packages/graph` owns the framework-neutral interactive view; VOWL retains compatibility and physics |
| Optional capabilities | Queries, Markdown-assisted proposals, import/export formats and provenance |
| Host adapters | CLI filesystem/Git, Obsidian vault, web artefacts, service APIs and governed promotion |

These are dependency boundaries, not a requirement to publish a crate for every row. Split packages when target support, dependency weight, licensing or independent consumers justify it. Keep native networking, signing, storage and threading out of the portable default dependency closure. Generate binding schemas/types and test serialisation across targets.

Keep mature upstream primitives as dependencies: Horned-OWL for OWL representation, Whelk for the supported reasoning fragment, and RDF/SPARQL components as needed. Reuse permissive patterns with attribution and tests. Do not fork a standards implementation merely to put its source in this repository. VOWL's existing identities remain compatibility consumers of `ontomark-ngg1`; its physics API stays VOWL-owned. Ontomark `packages/graph` owns the shared Canvas renderer, hierarchy adapter and focus traversal consumed by the publisher and standalone Obsidian plugin. Routing, tier loading, worker simulation, vault events and edit UI stay with the host. These boundaries prevent duplicate codecs or renderer copies without treating physics constraints as logical entailments.

Profiles contain vocabulary, layout, namespace projections, validation policy and publication policy. An ordinary vault works without estate configuration; the estate profile preserves or deliberately migrates its richer semantics. Optional Rust capabilities can exceed the Obsidian surface. Markdown and headers are the durable source; graph/query caches and published bundles are derived and rebuildable.

Migrate consumers through released crates, bindings and CLI artefacts with a compatibility manifest and shared conformance fixtures. Temporary compatibility adapters have named removal criteria. Completion includes deletion of superseded parsers, writers and semantic transformations after their consumers pass replacement tests. Preserve distinct logical entailments and presentation/physics constraints even when they share a model.

## Consequences and tensions

One source owner reduces implementation drift but introduces release coordination. Preserve package identities where useful; permit breaking internal APIs behind explicit migrations. The selected core uses MIT OR Apache-2.0 under the rights holder's explicit grant in [Ontomark NOTICE](https://github.com/DreamLab-AI/ontomark/blob/340b8a0/NOTICE). Governance posting, Nostr signing and excluded AGPL code remain outside the permissive compiler. VOWL remains MIT and Loom MIT/Apache. Whelk's Rust port MIT notice and Scala BSD-3-Clause notice are preserved; corrected `whelk-el` and `ontomark-reason` 0.1.1 feed compiler 0.1.2. Relocation alone never changes third-party licence terms.

WASM-first core design is appropriate; mandatory WASM execution for every native consumer is not. Large model runtimes and platform integration are adapters. No service or model is required for parsing, editing, validation, supported reasoning or graph rendering.

## Verification and delivery

The [investigation and delivery gates](../../../visionGraph/planning/ontology-workspace/README.md) record source findings, the existing 21-test renderer baseline and open target checks. The [semantic-model proposal](ADR-2015-typed-ontology-model-and-migration.md) and [Obsidian capability proposal](ADR-2016-local-markdown-ontology-application.md) define the remaining contracts. The initial investigation remains historical. This ownership decision is accepted with partial implementation; the full estate migration and the semantic-model migration obligations remain open. Producer evidence establishes specific released capabilities, not an estate-wide activation claim. Cross-family implementation audit remains a separate gate.

Acceptance requires clean-checkout native/WASM builds, supported-inference parity, installed offline plugin tests, consumer conformance, migration/rollback receipts, and removal of the inventoried duplicate implementations. Source completion, release publication and runtime activation are tracked separately.

## Implementation continuation — 2026-10-11

[Core release documentation](https://github.com/DreamLab-AI/ontomark/blob/4abbce9/README.md)
records released parser/model/NGG1 0.1.0, corrected Whelk/reasoner 0.1.1 and
compiler 0.1.2. The strict C3 marker and scaffold/prose readers are shared wire
contracts. The opt-in typed projection uses an injected model/profile and explicit
authoring/public scope, without filesystem writes. Missing or duplicate identities
are refused; private targets and values are withheld before RDF and reasoning.
Unsupported logical input and source omissions remain explicit incompleteness.

[Typed migration evidence](https://github.com/DreamLab-AI/ontomark/blob/671b6be/docs/api/WP-C-TYPED.md)
distinguishes that API from the retained estate CLI compatibility route. The shared
native adapter passes the committed visionGraph `b5e36abd1` comparison with zero
differences across 32 artefact hashes and 12 command outputs. That check deliberately
uses `--no-redact-private-links`; default public privacy has separate canaries. The
typed rule declaring SKOS-only model classes as `owl:Class` does not establish a
completed native migration of the 690 recorded legacy declarations.

[Shared graph ownership](https://github.com/DreamLab-AI/ontomark/blob/7f5b2bb/docs/adr/ADR-OM-002-shared-graph-and-obsidian-boundary.md)
and [standalone desktop acceptance](https://github.com/DreamLab-AI/ontomark-obsidian/blob/ebf8fd6/docs/desktop-acceptance.md)
supply consumer-specific evidence. Remaining work includes the normative estate
profile migration, other consumer adapters and removal of remaining duplicate
transformations. Assisted ingest ([ADR-OM-003](https://github.com/DreamLab-AI/ontomark/blob/main/docs/adr/ADR-OM-003-native-assisted-ingest.md)) is an optional shared capability of the workspace: a pure, host-driven plan/accept core any consumer may adopt; adopting it is each consumer's own gate. Neither these receipts nor publication alone close those gates.
