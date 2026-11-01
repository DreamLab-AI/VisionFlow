---
id: VG-01
title: Vault structure, publication boundaries and domain gates
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-001, ADR-VG-002, ADR-VG-003, ADR-VG-004, ADR-VG-008]
sources:
  - ../visionGraph/README.md
  - ../visionGraph/CLAUDE.md
  - ../visionGraph/docs/PUBLICATION-contract.md
  - ../visionGraph/docs/adr/README.md
  - ../visionGraph/docs/adr/ADR-VG-003-obsidian-only-authoring.md
  - ../visionGraph/docs/adr/ADR-VG-004-space-and-earth-domains.md
  - ../visionGraph/docs/adr/ADR-VG-008-supply-chains-and-logistics-domain.md
  - ../visionGraph/publishing-tools/WasmVOWL/modern/scripts/check-supply-chain-domain.mjs
  - ../visionGraph/publishing-tools/WasmVOWL/modern/scripts/check-space-domains.mjs
  - ../visionGraph/vault.toml
  - ../visionGraph/ontology/vocabulary.yaml
  - ../visionGraph/.github/workflows/publish.yml
  - ../visionGraph/scripts/check-bases.sh
  - ../visionGraph/quartz/quartz/plugins/filters/explicit.ts
  - ../visionGraph/quartz/quartz.config.ts
verified_commit: 556a0156b5c326fb61141ed2ab1760dd8d9c4dd6
---

## For developers

Trace the two authoring roles, the publication override and the limits of domain and Bases checks; the [dated source review](../../estate-review/2026-11-01-visiongraph-vault-and-domain-contracts.md) assesses all six panels with thirteen scoped probes.

## For the business

The vault supports private work and public knowledge, but neither a successful view check nor a domain tile proves a complete, usable publication; curator intent, emitted identities and downstream acceptance need separate evidence.

## VG-01.1 Current repository layout and asset ownership

```mermaid
flowchart TB
    K["knowledge/ governed role<br/>Pages at the vault root<br/>vault.toml:27-39"]
    W["working/ curator role<br/>Notes, episodes, drafts and journals<br/>vault.toml:41-50"]
    AS["One repo-level assets/ store<br/>working/assets is a symlink<br/>knowledge/assets is a synced real subset<br/>vault.toml:56-66"]
    VOC["ontology/vocabulary.yaml<br/>Vocabulary version and namespace<br/>vault.toml:17-21"]
    %% Source: publish.yml:175-181,329-335,421-490
    PUB["Publisher uses ontomark, Quartz and explorer<br/>Publication workflow"]
    K --> AS
    W --> AS
    VOC --> K
    K --> PUB
    W --> PUB
    DRIFT["DRIFT: README calls knowledge/index.md committed<br/>Manifest says the index is a build artefact<br/>visionGraph/README.md:18, vault.toml:30-31"]
    K --- DRIFT
```

**What it shows:** The flattened vaults share an asset store through different mechanisms; neither the old `pages/` layout nor a knowledge-assets symlink describes the current tree.
**Why it is this way:** The asset section records the October layout change, while ADR-VG-003 retires the legacy authoring path; the report verifies tracked tree types and the missing committed index.

## VG-01.3 Publication staging and the Quartz inclusion predicate

```mermaid
flowchart TB
    SRC["Authored knowledge and working vaults<br/>Workflow explicitly selects --vault all<br/>publish.yml:175-181"]
    STAGE["Compiler machine outputs and Quartz staging<br/>Public working content is included by this route<br/>publish.yml:155-159"]
    QUARTZ["Quartz configured with key public<br/>quartz.config.ts:126"]
    FILTER["ExplicitPublish accepts boolean true<br/>OR the exact string true<br/>explicit.ts:23-25"]
    SITE["www/ deployed to knowledgeGraph gh-pages<br/>Then a VisionClaw refresh request is sent<br/>publish.yml:526-542"]
    SRC --> STAGE --> QUARTZ --> FILTER --> SITE
    LIMIT["TENSION: README describes typed true only<br/>The downstream predicate is more permissive<br/>visionGraph/README.md:143-145<br/>Upstream compiler filtering is a separate boundary"]
    FILTER --- LIMIT
```

**What it shows:** The publisher chooses both vaults and Quartz applies its own predicate to the staging tree; an exact string true passes that downstream predicate.
**Why it is this way:** Staging and rendering are distinct steps, and a broader downstream predicate does not by itself demonstrate that a malformed source flag can pass the upstream compiler.

## VG-01.4 Historical findings and current acceptance obligations

```mermaid
flowchart TB
    HIST["September publication evidence is historical<br/>Retired Python behaviour is not a current runbook<br/>PUBLICATION-contract.md:3-8"]
    %% Source: ADR-VG-003-obsidian-only-authoring.md:25-33
    RETIRE["ADR-VG-003 retires Logseq authoring and fallbacks<br/>Keep rejection tests and provenance history<br/>ADR-VG-003 decision"]
    GATE["Current workflow validates through ontomark<br/>Then checks body repair and other release gates<br/>publish.yml:127-145,248-253"]
    ACCEPT["Inclusion and generation decisions remain proposed<br/>ADR-VG-001 and ADR-VG-002: partial, inactive<br/>docs/adr/README.md:13-14"]
    HIST --> RETIRE --> GATE --> ACCEPT
    LIMIT["OPEN: retiring old code does not certify<br/>visibility, identity or consumer activation<br/>PUBLICATION-contract.md:20-26"]
    ACCEPT --- LIMIT
```

**What it shows:** The historical failure set and the current implementation have different evidence boundaries; old tests are neither current failures nor successor acceptance.
**Why it is this way:** The publication contract marks its historical scope, and the operative decision index still leaves inclusion and consumer-generation acceptance open.

## VG-01.5 Manifest declarations and workflow overrides

```mermaid
flowchart TB
    K["knowledge role declares unknown_keys=fail<br/>Required metadata and public gate<br/>vault.toml:27-39"]
    W["working role declares unknown_keys=allow<br/>publishable=false and reasoned=false<br/>vault.toml:41-50"]
    DEFAULT["Build defaults to knowledge, public_only=true<br/>Domain-root disjointness disabled<br/>vault.toml:74-80"]
    PUBLISH["Publish section says knowledge-only content<br/>and working excluded<br/>vault.toml:143-152"]
    ACTUAL["TENSION: active workflow passes --vault all<br/>and stages public working content too<br/>publish.yml:155-181"]
    K --> DEFAULT
    W --> DEFAULT
    DEFAULT --> PUBLISH --> ACTUAL
    PIN["DRIFT: README says v0.2.0, rules say v0.5.0<br/>Workflow pins ontomark v0.7.0 and SHA-256<br/>visionGraph/README.md:66, CLAUDE.md:23, publish.yml:48-51"]
    ACTUAL --- PIN
```

**What it shows:** Manifest values express defaults and policy declarations; the active workflow selects a broader publication route and a newer compiler than the accompanying prose names.
**Why it is this way:** The migration retained old manifest comments and documentation while the publisher evolved; this map records those tensions without declaring every manifest field a proven enforcement point.

## VG-01.6 Bases checks are a bounded linter

```mermaid
flowchart TB
    FILES["Four knowledge and six working .base files<br/>Checker walks both bases/ directories<br/>check-bases.sh:8"]
    YAML["Parse YAML, restrict top-level keys<br/>Require views sequence<br/>check-bases.sh:9-14"]
    VIEW["Check listed type and name-key presence<br/>check-bases.sh:15-18"]
    FORM["Compare declared formulas with references in<br/>view order, sort and property keys<br/>check-bases.sh:19-23"]
    PASS["Ten committed files pass this checker<br/>Not an Obsidian UI execution test"]
    FILES --> YAML --> VIEW --> FORM --> PASS
    GAP1["DEBT: blank name hides an invalid view type<br/>The failure test inspects extracted name text<br/>check-bases.sh:15-16"]
    GAP2["DEBT: filter/formula-expression references<br/>are outside the extracted reference set<br/>check-bases.sh:20-21"]
    VIEW --- GAP1
    FORM --- GAP2
```

**What it shows:** The checker gives useful structural feedback but accepts two reproduced cases that its broad schema-conformance description suggests it covers.
**Why it is this way:** Shell tests use an extracted display name to detect invalid types, and formula discovery inspects selected fields instead of parsing all expressions.

## VG-01.7 Domain expansion gates and their unverified geometry

```mermaid
flowchart TB
    ROOTS["Nine vocabulary roots<br/>Space and Earth appended, then supply chains<br/>vocabulary.yaml:350-353"]
    %% Source: check-space-domains.mjs:14-35,40-75
    SPACE["Space gate: IDs 6 and 7, minimum counts,<br/>root titles and catalogue-backed page checks<br/>Space domain gate"]
    %% Source: check-supply-chain-domain.mjs:31-46,62-77
    SUP["Supply gate: ID 8, root and one-member minimum<br/>Check present category IDs and new/rehome pages<br/>Supply-chain domain gate"]
    %% Source: check-space-domains.mjs:25, check-supply-chain-domain.mjs:36
    TIERS["DEBT: tier existence is checked, bytes are not decoded<br/>Empty and invalid tiers pass scoped probes<br/>Space gate: tier presence<br/>Supply gate: tier presence"]
    %% Source: check-supply-chain-domain.mjs:42-54
    SKIP["DEBT: absent supply catalogue exits successfully<br/>No requirement that all 15 category IDs exist<br/>Supply gate: optional catalogue"]
    ROOTS --> SPACE --> TIERS
    ROOTS --> SUP --> SKIP
    %% Source: ADR-VG-004-space-and-earth-domains.md:43-44
    NOTE["OPEN: stable IDs and page metadata do not prove<br/>competency answers or meaningful bridges<br/>ADR-VG-004 acceptance"]
    TIERS --> NOTE
    SKIP --> NOTE
```

**What it shows:** Domain metadata and catalogue identities are checked, but binary tier integrity and complete category identity are outside these particular gates.
**Why it is this way:** The expansion records require useful cross-domain behaviour as well as stable identifiers; the gates only establish part of that acceptance contract, and later workflow checks must be assessed separately.
