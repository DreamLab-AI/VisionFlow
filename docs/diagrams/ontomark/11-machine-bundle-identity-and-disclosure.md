---
id: OM-11
title: Machine bundle identity and disclosure boundaries
area: ontomark
governing:
  - ../ontomark/docs/adr/ADR-OM-005-estate-alignment-one-producer-explicit-projections.md
adrs: [ADR-OM-005]
sources:
  - ../ontomark/crates/ontomark-compile/src/ir/projection.rs
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/build/generation.rs
  - ../ontomark/crates/ontomark-compile/src/build/redact.rs
  - ../ontomark/crates/ontomark-compile/src/build/page_api.rs
verified_commit: {ontomark: 6159bb6089753105ef839fd4aac46c1598a64bd2}
---

## For developers

Machine scope adds public working notes as plain records while preserving the knowledge projection. The [ADR and probe review](../../estate-review/2026-11-01-ontomark-machine-bundle-contract.md) checks its collision gate, redaction boundary and generation fields. Existing byte hashes identify outputs, but the source label and content digest do not describe every effective build input.

## For the business

A working note can be visible to the internal graph without becoming an ontology class. That separation needs an explicit disclosure policy and a receipt identifying the files the reader received. A successful build does not establish that every private name has disappeared from public prose.

## OM-11.1 A short folder hash can collide; the publication gate refuses it

```mermaid
flowchart TD
    PATH["Working page path<br/>projection.rs:669"] --> SEG["Slugify each path segment<br/>projection.rs:670,679"]
    SEG --> EMPTY{"ASCII slug empty?<br/>projection.rs:681"}
    EMPTY -->|no| KEEP["Keep normalised slug<br/>projection.rs:682"]
    EMPTY -->|yes| HASH["x- plus first four SHA-256 bytes<br/>projection.rs:685-687"]
    KEEP --> CHECK["Check final working slugs among surviving public records<br/>projection.rs:856-861"]
    HASH --> CHECK
    CHECK --> BAD["Duplicate slug returns PublicationBlocked::Collision<br/>projection.rs:862-866"]
    CHECK --> OK["Accepted record gets working prefix and page IRI<br/>projection.rs:877-882"]
    HASH -.-> LIMIT["TENSION: 32-bit suffix is not unique<br/>two distinct non-ASCII paths can share it<br/>projection.rs:686"]
```

**What it shows.** Two synthetic non-ASCII folder names mapped to `working/x-24f80a23/foo`. The real build refused the duplicate before creating output. The gate, rather than collision-free naming, protects against overwrite.

**Why it is this way.** ADR-OM-005 requires path-based working identities. Normalisation and finite hash suffixes need a separate collision check; the implementation has one.

## OM-11.2 Knowledge and working records use different private-name marker sets

```mermaid
flowchart TD
    LINKS["Cross-vault private-link resolver<br/>mod.rs:721-726"] --> K["Project knowledge first through its own record set<br/>projection.rs:751,752"]
    K --> KP["Knowledge projection supplies ontology outputs and page records<br/>mod.rs:772,779,875"]
    LINKS --> W["Clean working links and build records<br/>projection.rs:779-781"]
    W --> UNION["Union record set builds a private prose-marker redactor<br/>projection.rs:838-851"]
    UNION --> WP["Apply union redactor only to surviving working records<br/>projection.rs:870-876"]
    KP --> API["Merge knowledge and plain working records for page API<br/>projection.rs:712-718; mod.rs:910-914"]
    WP --> API
    API --> TITLE["Page title comes from the projected record<br/>page_api.rs:89"]
    K -.-> GAP["TENSION: bare private working name can survive<br/>in a public knowledge title<br/>knowledge redactor is not the union redactor<br/>projection.rs:751,851,891"]
```

**What it shows.** The same private working-note title was masked in a public working title and retained in a public knowledge title. Both public and machine builds retained the latter; the private body's canary stayed absent.

**Why it is this way.** D4 preserves the knowledge projection across scopes. Cross-vault link cleaning and plain-text marker masking are different operations; the union marker set is applied after the knowledge projection has already been produced.

## OM-11.3 Source labels, content digests and output hashes have distinct jobs

```mermaid
flowchart TD
    SOURCE["Caller source override or discovered Git state<br/>mod.rs:1041-1058"] --> ID["Generation prefix plus commit and dirty suffix<br/>mod.rs:1059,1060; generation.rs:163"]
    PAGES["Hashes of loaded pages and journals<br/>mod.rs:1079-1094"] --> CONTENT["content_digest<br/>mod.rs:1101"]
    VOCAB["Hash of reserialised parsed vocabulary<br/>mod.rs:1095-1098"] --> CONTENT
    PROFILE["Profile influences emitted context, page IRIs and graph<br/>mod.rs:779,934; projection.rs:878"] --> OUTPUT["Staged outputs carry per-file SHA-256 and byte counts<br/>generation.rs:39-45"]
    PROFILE -.-> MISS["TENSION: profile is absent from content_digest inputs<br/>same digest can accompany changed output<br/>mod.rs:1079-1101"]
    ID --> MARK["Root and data markers carry identity, scope and artefacts<br/>mod.rs:1107-1134"]
    CONTENT --> MARK
    OUTPUT --> MARK
    SOURCE -.-> TRUST["Caller-provided commit and dirty flag are accepted directly<br/>mod.rs:1041,1042"]
```

**What it shows.** A changed page namespace preserved generation ID and content digest while changing page IRIs and artefact hashes. A vocabulary comment changed input bytes without changing the digest, because the vocabulary is reserialised. An arbitrary caller-supplied commit label was retained.

**Why it is this way.** The build exposes reproducibility controls to its host. ADR-OM-005's generation loop needs the host to verify source assertions and retain the effective configuration and output receipt; these fields are not interchangeable evidence.
