---
id: VG-03
title: Publisher assembly order and the limits of its gates
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-001, ADR-VG-002, ADR-VG-004, ADR-VG-008]
sources:
  - ../visionGraph/.github/workflows/publish.yml
  - ../visionGraph/publishing-tools/WasmVOWL/modern/package.json
  - ../visionGraph/publishing-tools/WasmVOWL/modern/scripts/check-space-domains.mjs
  - ../visionGraph/publishing-tools/WasmVOWL/modern/scripts/check-supply-chain-domain.mjs
  - ../visionGraph/quartz/scripts/check-secrets.sh
  - ../visionGraph/quartz/scripts/check-urls.sh
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
verified_commit: {visiongraph: 556a0156b5c326fb61141ed2ab1760dd8d9c4dd6, ontomark: 6159bb6089753105ef839fd4aac46c1598a64bd2}
---

**Repair note — 2026-11-03:** selected counterexamples below have a [scoped successor verification](08-promoted-certificate-consumption.md). This historical topic retains its declared source revision; the successor states the repaired contracts and acceptance boundaries.

## For developers

Follow the configured publisher from its released compiler to the final site and downstream dispatch; the [publication and identity audit](../../estate-review/2026-11-01-visiongraph-publication-and-identity.md) separates source checks, staging receipts and consumer acceptance.

## For the business

The build combines public knowledge, notes and an interactive explorer, but a successful stage check does not prove final page completeness, privacy or that another application has loaded the new content.

## VG-03.1 One configured job assembles three outputs

```mermaid
flowchart TB
    START["Matching main push or manual dispatch<br/>Serial deploy-ontology concurrency group"]
    CLI["Consumer tests, checksum-pinned ontomark v0.7.0<br/>source validation and collision lint"]
    BUILD["All-vault public build and content staging<br/>certificate re-check and body-residue gate"]
    DATA["Overlay static contexts<br/>machine volume and domain checks"]
    NOTES["Quartz builds notes from staged content<br/>copy machine files, then gzip selected outputs"]
    SPA["Size check, then explorer build<br/>static routes and notices"]
    FINAL["Browser smoke, credential scan, layout check<br/>deploy to knowledgeGraph gh-pages"]
    EVENT["Send corpus-sync with source repository and SHA<br/>request delivery is not downstream completion"]
    START --> CLI --> BUILD --> DATA --> NOTES --> SPA --> FINAL --> EVENT
    %% Source: .github/workflows/publish.yml:20
    %% Source: .github/workflows/publish.yml:37
    %% Source: .github/workflows/publish.yml:48
    %% Source: .github/workflows/publish.yml:88
    %% Source: .github/workflows/publish.yml:144
    %% Source: .github/workflows/publish.yml:175
    %% Source: .github/workflows/publish.yml:217
    %% Source: .github/workflows/publish.yml:248
    %% Source: .github/workflows/publish.yml:261
    %% Source: .github/workflows/publish.yml:331
    %% Source: .github/workflows/publish.yml:375
    %% Source: .github/workflows/publish.yml:536
```

**What it shows:** the actual ordering, including the source collision gate, post-build body check and post-deploy event; the event carries the source commit, not the assembled site digest (`publish.yml:144`, `publish.yml:248`, `publish.yml:536`).
**Why it is this way:** one job owns the public site while consumers refresh through separate paths; the pinned compiler has no source-build fallback (`publish.yml:42`, `publish.yml:98`).

## VG-03.3 Explorer assembly preserves the machine tree

```mermaid
flowchart TB
    INPUT["Read built ontology, graph tiers and search index<br/>stage local preview copies"]
    DEPS["npm ci<br/>ontomark-graph 0.2.0 and vowl-wasm 0.2.2 tarballs"]
    CHECK["TypeScript, runtime-notices Rust checks<br/>unit tests, then Vite production build"]
    COPY["Copy shell and assets into www<br/>do not copy preview api or data trees"]
    ROUTES["Five static route shells with hash comparisons<br/>legacy explorer redirect"]
    NOTICE["Require named notices, copy and hash-compare<br/>browser smoke against assembled www"]
    INPUT --> DEPS --> CHECK --> COPY --> ROUTES --> NOTICE
    %% Source: .github/workflows/publish.yml:421
    %% Source: publishing-tools/WasmVOWL/modern/package.json:24
    %% Source: .github/workflows/publish.yml:430
    %% Source: .github/workflows/publish.yml:441
    %% Source: .github/workflows/publish.yml:455
    %% Source: .github/workflows/publish.yml:464
    %% Source: .github/workflows/publish.yml:477
    %% Source: .github/workflows/publish.yml:492
```

**What it shows:** the configured integration and output-copy boundary; dependency versions are source pins, not a newly measured browser-performance result (`package.json:24`, `publish.yml:441`).
**Why it is this way:** the compiler's machine artefacts remain authoritative while the host app owns navigation, rendering and attribution (`publish.yml:441`, `publish.yml:477`).

## VG-03.5 Source, certificate and domain gates prove different things

```mermaid
flowchart TB
    VALIDATE["validate --vault all<br/>nonzero exit refuses the job, warnings remain nonfatal"]
    BUILD["build --vault all --with-markdown-mirror<br/>--publish-out and --certificate-out"]
    CERT["oo-cert v2.0.1 checksum-verified<br/>asserted and derivation files, JSON ok required"]
    STAGE["Require pages and working staging directories<br/>machine paths and three minimum-volume checks"]
    SPACE["Space and Earth IDs, counts and catalogue checks<br/>tier path existence is not binary validation"]
    SUPPLY["Supply ID and present-category checks<br/>missing catalogue skips identity checks"]
    VALIDATE --> BUILD --> CERT --> STAGE --> SPACE --> SUPPLY
    %% Source: .github/workflows/publish.yml:127
    %% Source: .github/workflows/publish.yml:175
    %% Source: .github/workflows/publish.yml:202
    %% Source: .github/workflows/publish.yml:219
    %% Source: .github/workflows/publish.yml:229
    %% Source: .github/workflows/publish.yml:271
    %% Source: .github/workflows/publish.yml:304
    %% Source: publishing-tools/WasmVOWL/modern/scripts/check-space-domains.mjs:14
    %% Source: publishing-tools/WasmVOWL/modern/scripts/check-supply-chain-domain.mjs:31
    %% Source: publishing-tools/WasmVOWL/modern/scripts/check-supply-chain-domain.mjs:39
    %% Source: publishing-tools/WasmVOWL/modern/scripts/check-supply-chain-domain.mjs:50
```

**What it shows:** checks for source errors, derivation validity, output volume and selected domain metadata; none independently certifies scientific truth or all emitted bytes (`publish.yml:130`, `publish.yml:229`, `publish.yml:304`).
**Why it is this way:** the workflow guards distinct failure modes; the domain scripts' actual checks are narrower than their names and the earlier diagram's fifteen-category guarantee (`check-supply-chain-domain.mjs:39`).

## VG-03.6 Receipts precede transformations and final acceptance

```mermaid
flowchart TB
    OVERLAY["Copy static contexts into site-data<br/>after the compiler created its generation markers"]
    RECEIPT["Inventory files and hash-check copy into www"]
    ZIP["Compress page JSON, Markdown and ontology JSON<br/>raw filenames replaced by gzip paths"]
    SIZE["Measure site size before SPA, route and notice additions"]
    ASSEMBLE["Complete explorer assembly and browser smoke"]
    SCAN["Credential self-test then grep-based final scan<br/>DEBT: compressed-only content is not scanned as text"]
    LAYOUT["Layout checks raw-or-gzip path presence<br/>notes completeness is a count comparison"]
    DEPLOY["Deploy assembled tree<br/>these checks do not regenerate a final-site manifest"]
    OVERLAY --> RECEIPT --> ZIP --> SIZE --> ASSEMBLE --> SCAN --> LAYOUT --> DEPLOY
    %% Source: crates/ontomark-compile/src/build/mod.rs:1133
    %% Source: .github/workflows/publish.yml:261
    %% Source: .github/workflows/publish.yml:317
    %% Source: .github/workflows/publish.yml:343
    %% Source: .github/workflows/publish.yml:358
    %% Source: .github/workflows/publish.yml:375
    %% Source: .github/workflows/publish.yml:421
    %% Source: .github/workflows/publish.yml:504
    %% Source: quartz/scripts/check-secrets.sh:150
    %% Source: quartz/scripts/check-urls.sh:90
    %% Source: quartz/scripts/check-urls.sh:115
    %% Source: .github/workflows/publish.yml:526
```

**What it shows:** each receipt covers the bytes at its own stage; the retained KG-P probes reproduce compression, count-substitution and final-size gaps at this same source revision.
**Why it is this way:** copying, compression and app assembly are separate steps, so a successful earlier hash check does not describe every file finally deployed (`publish.yml:343`, `publish.yml:358`, `publish.yml:441`).

**Debt:** push filters omit working pages and root assets despite consuming them; matching knowledge or workflow changes and manual runs can still publish those inputs (`publish.yml:24`, `publish.yml:158`, `publish.yml:195`).
