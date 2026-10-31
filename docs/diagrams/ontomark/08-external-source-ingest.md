---
id: OM-08
title: External-source ingest — acquisition, acceptance and bundle identity
area: ontomark
governing:
  - ../ontomark/docs/adr/ADR-OM-010-external-source-ingest-lands-as-proposals.md
adrs: [ADR-OM-010]
sources:
  - ../ontomark/crates/ontomark-cli/src/ingest_cmd.rs
  - ../ontomark/crates/ontomark-cli/src/ingest_profile.rs
  - ../ontomark/crates/ontomark-source/src/url.rs
  - ../ontomark/crates/ontomark-source/src/ledger.rs
  - ../ontomark/crates/ontomark-ingest/src/accept.rs
  - ../ontomark/crates/ontomark-jev/src/config.rs
verified_commit: {ontomark: 0caae5d4b695cb720942b9b9212017c00a80ad41}
---

## For developers

The current `ingest` verb reads external evidence, asks an explicitly selected model and prepares full proposed pages for a separate governance submission. This topic supplements the historical OM-05 command inventory. The [audit and executable probes](../../estate-review/2026-10-31-external-source-ingest.md) distinguish input identity, extraction completeness, proposal bytes and publication. The ADR remains inactive: a passing local dry-run does not prove a signed forum-to-corpus journey.

## For the business

A researcher can turn a document or web page into suggested relationships on existing notes, with quotes available for review. The source text and some note context reach the selected model. A located quote may still contradict the suggestion, and a repeat run can replace an existing review bundle. Reviewers need the actual proposal and its evidence, rather than treating “complete” or a request identifier as a quality guarantee.

## OM-08.1 Acquisition precedes model budgets

```mermaid
flowchart TD
    A["Require target or discovery option<br/>ingest_cmd.rs:502"] --> E["Unless plan-only, check model endpoint<br/>declared local, local-looking host or opt-in<br/>ingest_cmd.rs:508; config.rs:141,231"]
    E --> V["Knowledge only<br/>working and all refused<br/>ingest_cmd.rs:528"]
    V --> S["List and fetch all selected sources<br/>ingest_cmd.rs:193,577"]
    S --> U["URL source: HEAD then GET<br/>five redirects, bounded body<br/>url.rs:57,105,124,154"]
    S --> P["Plan-only returns after reading sources<br/>no model call or bundle write<br/>ingest_cmd.rs:627"]
    S --> C["Target label, keys, existing values<br/>and source chunk become request<br/>ingest_cmd.rs:340,378"]
    C --> H["Response cache checked first<br/>ingest_cmd.rs:450"]
    H --> B["On miss, optional call and character budgets<br/>then HTTP with redirects disabled<br/>ingest_cmd.rs:462,465,415"]
```

**What it shows.** Source acquisition and model transport have separate policies. The endpoint check is bypassed for plan-only, but source fetching still occurs. Discovery's cap is per source; explicit targets are separate.

**Why it is this way.** ADR-OM-010 keeps source acquisition in the host and plan/accept in the pure library. Its budgets constrain model requests, not the total cost of acquiring and rendering sources.

## OM-08.2 Acceptance and completion measure different things

```mermaid
flowchart TD
    R["Model reply<br/>ingest_cmd.rs:721"] --> J{"Assertion JSON parses?<br/>ingest_cmd.rs:396,728"}
    J -->|yes| A["Recheck target digest, source version and chunks<br/>accept.rs:329,354,358,366"]
    J -->|no| D["Record dropped chunk only<br/>ingest_cmd.rs:735"]
    A --> G["Allowed key, grounded name, kind and located quote<br/>accept.rs:415,422,444,466"]
    G --> M["Quote must mention the value<br/>accept.rs:533"]
    M --> O["Proposed patch and evidence<br/>ingest_cmd.rs:745,748,802"]
    D --> F["No skipped pair or stop added<br/>complete can remain true<br/>ingest_cmd.rs:821"]
    F --> L["Source recorded as finished<br/>ingest_cmd.rs:842"]
    M -.-> N["TENSION: mention check does not prove relation meaning<br/>a negated relation can pass<br/>accept.rs:533"]
    O -.-> Q["Review list selects literal conflict reasons<br/>host profile offers relations<br/>ingest_cmd.rs:791; ingest_profile.rs:54,93"]
```

**What it shows.** Digest and span checks protect structural integrity. The reproduced negative sentence still grounds an accepted positive relation. A malformed response is exposed as a dropped chunk but does not make the host's summary incomplete.

**Why it is this way.** The pure engine grounds model-supplied values; the host defines run accounting. ADR-OM-010's conflict-review promise is narrower in the current relation-only implementation than a general contradiction check.

## OM-08.3 Request IDs do not identify immutable bundles

```mermaid
flowchart TD
    I["Sources, target digests, model and options<br/>ingest_cmd.rs:606"] --> ID["Request ID determines default directory<br/>ingest_cmd.rs:769"]
    EP["Endpoint included in parser and response cache<br/>ingest_cmd.rs:570,446"] -.-> GAP["DRIFT: endpoint omitted from request ID<br/>ingest_cmd.rs:606"]
    ID --> DEL["Remove existing manifest directory<br/>ingest_cmd.rs:771,772"]
    DEL --> W["Write changed pages, then evidence<br/>ingest_cmd.rs:782,785,835"]
    W --> CMD["Print vault propose command<br/>ingest_cmd.rs:867"]
    W -.-> B["TENSION: incomplete uncached rerun<br/>can replace complete bundle under same ID<br/>ingest_cmd.rs:723,769,772"]
    OBS["Classify observed plus removed IDs<br/>ledger.rs:224,255"] --> ZIP["Zip with observed items only<br/>ingest_cmd.rs:223"]
    ZIP --> LOST["Removed entries dropped from host result<br/>only finished observed sources recorded<br/>ingest_cmd.rs:841,843"]
```

**What it shows.** Endpoint-specific cache and parser identity do not extend to the bundle ID. The audit reproduces endpoint and budget/cache changes replacing a previously complete manifest. Removal classification is implemented in the library but lost by the host.

**Why it is this way.** The request ID deliberately excludes budgets and output paths; the host reconstructs bundles in place. ADR-OM-010 does not yet define an immutable output identity, atomic bundle publication or the listing scope needed to interpret removed sources.

**Open:** Complete source snapshots, prior target bytes and model requests are not retained in the bundle (`ingest_cmd.rs:802-838`). Replay requires matching inputs from outside it. No live proposal activation or recovery journey is certified by these diagrams.
