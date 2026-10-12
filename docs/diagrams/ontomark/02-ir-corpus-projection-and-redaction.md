---
id: OM-02
title: IR corpus, public projection and private-link redaction
area: ontomark
governing:
  - docs/engineering/sovereign-corpus-contracts.md
adrs: [ADR-OM-001, ADR-OM-005]
sources:
  - ../ontomark/crates/ontomark-compile/src/ir/model.rs
  - ../ontomark/crates/ontomark-compile/src/ir/projection.rs
  - ../ontomark/crates/ontomark-compile/src/ir/closure.rs
  - ../ontomark/crates/ontomark-compile/src/build/redact.rs
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/build/publish.rs
  - ../ontomark/crates/ontomark-compile/src/build/indexes.rs
  - ../ontomark/crates/ontomark-compile/src/build/okf.rs
  - ../ontomark/crates/ontomark-compile/src/build/page_api.rs
  - ../ontomark/crates/ontomark-compile/src/input/page.rs
  - ../ontomark/crates/ontomark-compile/src/input/frontmatter.rs
  - ../ontomark/crates/ontomark-compile/src/input/slug.rs
  - ../ontomark/crates/ontomark-compile/src/migrate/compile.rs
  - ../ontomark/crates/ontomark-compile/src/shacl.rs
  - ../ontomark/crates/ontomark-compile/src/check/validate.rs
  - ../ontomark/crates/ontomark-compile/src/profile.rs
  - ../ontomark/crates/ontomark-compile/examples/shacl_export.rs
  - ../ontomark/crates/ontomark-compile/tests/jsonld_matches_turtle.rs
  - ../ontomark/crates/ontomark-compile/tests/publish_gate.rs
verified_commit: ee9e04a25715e70d3de211c17a012e878d14f339
---

## OM-02.1 Corpus::build — Vault to ClassRecord, every field's source and every resolver

```mermaid
flowchart TD
    VAULT["Vault.pages — loaded by Vault::load or Vault::load_with_walk<br/>input/page.rs:413, input/page.rs:436"]
    NAMES["NameIndex::build — lowercased id, title, slug, aliases; first page wins<br/>ir/model.rs:329, ir/model.rs:226-234, ir/model.rs:224"]
    RES["resources[i] = Page::resource namespace<br/>ir/model.rs:330-334, input/page.rs:217"]
    RESOLVE["resolve target,label closure<br/>1 names.get target - resources[i]<br/>2 tail_iris by label, 3 tail_iris by target<br/>4 namespace + slugify target<br/>ir/model.rs:336-347, ir/model.rs:338-342"]
    LINKSOF["links_of page,key — wikilinks through resolve<br/>ir/model.rs:349-355"]
    REL["relations — 12 RELATION_KEYS plus disjointWith, empty lists absent<br/>ir/model.rs:363-373, ir/model.rs:36"]
    ISA["sub_class_of from is-a, instance_of from instance-of<br/>ir/model.rs:412-413"]
    CURATED["links — Page::outbound_links, slug-VERBATIM, bypasses NameIndex<br/>namespace + target_slug w.target<br/>ir/model.rs:420-427, input/page.rs:252"]
    DEFS["defines_as — definition_of through the SAME resolve<br/>ir/model.rs:430, ir/model.rs:297-323"]
    SCALARS["page_id, title, slug, public, domain, quality, legacy_term_id, body<br/>ir/model.rs:380-393, ir/model.rs:406-407, ir/model.rs:428"]
    SLUGSRC["slug = Page::slug declared else slugify title<br/>public = Page::is_public<br/>input/page.rs:207, input/page.rs:231"]
    IRIS["iri = resources[i]; page_iri = page_resource else resources[i]<br/>ir/model.rs:388-391"]
    TEXTS["label else title; definition else leading_paragraph; maturity else draft<br/>ir/model.rs:382-385, ir/model.rs:398-405, input/page.rs:283"]
    GEN["generated_by = Stamp::from_yaml generated, by only<br/>ir/model.rs:408-411"]
    HASONT["has_ontology = has resource OR has type<br/>ir/model.rs:429"]
    DISAMB["disambiguate_public_slugs — declared slug keeps it, else resource tail, else first<br/>ir/model.rs:434, ir/model.rs:261-292"]
    BYSLUG["by_class_slug — has_ontology AND iri non-empty, first wins<br/>ir/model.rs:436-441"]
    OUT["Corpus records + by_class_slug<br/>ir/model.rs:443-446"]
    VAULT --> NAMES --> RESOLVE
    VAULT --> RES --> RESOLVE
    RESOLVE --> LINKSOF --> REL
    LINKSOF --> ISA
    RESOLVE --> DEFS
    VAULT --> CURATED
    VAULT --> SCALARS --> SLUGSRC
    RES --> IRIS
    VAULT --> TEXTS
    VAULT --> GEN
    VAULT --> HASONT
    REL --> DISAMB
    ISA --> DISAMB
    CURATED --> DISAMB
    DEFS --> DISAMB
    SLUGSRC --> DISAMB
    IRIS --> DISAMB
    TEXTS --> DISAMB
    GEN --> DISAMB
    HASONT --> DISAMB
    DISAMB --> BYSLUG --> OUT
    n1["DIVERGENCE: THREE wikilink-target resolvers. Corpus::build resolve ir/model.rs:336, PrivateLinks::resolve build/redact.rs:210, and shacl Resolver::page shacl.rs:1201 (no slug key, case-sensitive id match at shacl.rs:1203). The first two share NameIndex, the third does not."]
    n2["DIVERGENCE: PrivateLinks strips a #fragment and a .md suffix before lookup build/redact.rs:219, build/redact.rs:227; Corpus::build passes the target straight to names.get ir/model.rs:338, so a fragment link redacted as private is resolved here to namespace + slugify of the whole string."]
    n3["DIVERGENCE: curated links are slug-verbatim ir/model.rs:424, every other relation goes through resolve ir/model.rs:338. Two target-to-IRI derivations in one function; deliberate for backlink keys ir/model.rs:415-419."]
```
- `Corpus::build` is the only constructor of `ClassRecord` outside tests; every artefact is a pure function of it (`ir/model.rs:4-7`).
- `slug` is the page publish slug, `class_slug()` is the IRI tail; they differ on 96 pages (`ir/model.rs:185-190`), and `disambiguate_public_slugs` rewrites `slug` from the IRI tail on a clash (`ir/model.rs:285-287`).
- Private records are never renamed; a public/private clash is left to the projection (`ir/model.rs:257-260`).

## OM-02.2 The record selectors — which accessor is right for what

```mermaid
flowchart LR
    REC["Corpus.records — every record, no filter<br/>ir/model.rs:203"]
    PUB["Corpus::public — public flag only<br/>ir/model.rs:450-452"]
    PCL["Corpus::public_classes — public AND has_ontology AND iri non-empty AND Class<br/>ir/model.rs:456-460"]
    BYS["Corpus::by_slug — class slug, first definition wins, includes private and Individual<br/>ir/model.rs:464-466, ir/model.rs:436-441"]
    C1["closure::compute iterates records with has_ontology and iri, Individuals included<br/>ir/closure.rs:99-102"]
    C2["page_count = listed.records.len, listed is the merged corpus under machine scope<br/>build/mod.rs:659-660"]
    C3["class_count = public_classes.count<br/>build/mod.rs:654"]
    C4["OKF bundle classes and scaffold-index candidates<br/>build/okf.rs:74, build/indexes.rs:102"]
    C5["page API, mirror loop, search index<br/>build/page_api.rs:77, build/mod.rs:757, build/mod.rs:767, build/indexes.rs:255, build/indexes.rs:310"]
    C6["validate: parent label check on the RAW corpus<br/>check/validate.rs:549"]
    REC --> C1
    REC --> C2
    PCL --> C3
    PCL --> C4
    PUB --> C5
    BYS --> C6
    n1["INVARIANT: every consumer except validate receives the PROJECTED corpus, where records are already public only ir/projection.rs:502-506, so the public flag test in public and public_classes is a second, redundant public gate ir/model.rs:451, ir/model.rs:458."]
    n2["DIVERGENCE: closure and by_class_slug admit Individuals ir/closure.rs:99-102, ir/model.rs:438, while public_classes excludes them ir/model.rs:458; class_count and the closure therefore count different sets."]
    n3["INVARIANT: validate runs on the raw corpus build/mod.rs:576, before projection build/mod.rs:602, so by_slug there can resolve a PRIVATE record check/validate.rs:549."]
```
- Use `public_classes` for anything that is an OWL class count or class listing; `public` for per-page surfaces; `records` only before projection or for the closure.
- No selector is dead: `public` (5 non-test callers), `public_classes` (3), `by_slug` (1, `check/validate.rs:549`).

## OM-02.3 The two projection paths — project_with_redactor vs project_vault_with_links

```mermaid
flowchart TD
    SEL{"build::run selects by redact_private_links<br/>profile default true<br/>build/mod.rs:556-558, build/mod.rs:602-617, profile.rs:417"}
    subgraph LEGACY["Compat path — runs on the RAW corpus"]
      PWR["project_with_redactor profile, raw, excluded<br/>ir/projection.rs:495, build/mod.rs:616"]
    end
    subgraph LINKS["Link-aware path — rebuilds the corpus from a cleaned Vault clone"]
      L1["project profile, Corpus::build vault, extra — result DISCARDED, collision check only<br/>ir/projection.rs:607"]
      L2["vault.clone; held_back pages forced public:false<br/>ir/projection.rs:617-625, build/publish.rs:134"]
      L3["skip private pages<br/>ir/projection.rs:626"]
      L4["links pinned to non-private outbound_links, even when empty<br/>ir/projection.rs:631-636, ir/projection.rs:650-652"]
      L5["frontmatter.retain — defines-as dropped whole, other values through frontmatter_value<br/>ir/projection.rs:637-649, build/redact.rs:276"]
      L6["body = PrivateLinks::body, Tally thrown away<br/>ir/projection.rs:653, build/redact.rs:307"]
      L7["Corpus::build cleaned vault, THEN project_with_redactor<br/>ir/projection.rs:609"]
      L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7
    end
    CORE["project_with_redactor core: Split::checked partition and collision, Split::redactor markers, clean_record, corpus_of by_class_slug<br/>ir/projection.rs:495, ir/projection.rs:537, ir/projection.rs:561, ir/projection.rs:512"]
    WRAP["project — thin wrapper dropping the redactor<br/>ir/projection.rs:443-449"]
    MACH["machine scope: project_machine runs the link-aware path on knowledge, then adds working plain records, see OM-06.2<br/>ir/projection.rs:751, build/mod.rs:603-605"]
    SEL -->|false| PWR --> CORE
    SEL -->|true| L1
    SEL -->|scope machine| MACH --> L1
    L7 --> CORE
    WRAP --> CORE
    PROD["non-test callers of project: migrate compile, shacl_export example, jsonld test<br/>migrate/compile.rs:101, examples/shacl_export.rs:76, tests/jsonld_matches_turtle.rs:80"]
    PROD --> WRAP
    n1["DIVERGENCE: the compat path never forces misc/ pages private (held_back is only applied in the link path ir/projection.rs:623); a public misc/ page stays public there, while stage_redacted skips it build/publish.rs:216."]
    n2["DIVERGENCE: the compat path exists only to keep the legacy projection bytes build/mod.rs:615, an opt-out of the default redact_private_links=true profile.rs:417. A projection path kept for byte-parity."]
    n3["DIVERGENCE: Corpus::build runs up to THREE times per build: raw build/mod.rs:520, collision probe ir/projection.rs:607, cleaned ir/projection.rs:609."]
```
- Both paths end in the same `project_with_redactor`; the link path only differs in WHAT corpus it is handed (links and frontmatter pre-redacted at the `Vault` level).
- `project_vault_with_links` is called from `build/mod.rs:613` (and from `project_machine`, `ir/projection.rs:762`); `project_with_redactor` is called directly once, `build/mod.rs:616`.
- The `ProseRedactor` is the same `Redactor` that cleaned the records (`ir/projection.rs:507`), reused for the markdown mirror (`build/mod.rs:786`).

## OM-02.4 Redaction order — private-link rewrite, marker redaction, closure

```mermaid
sequenceDiagram
    autonumber
    participant RUN as build::run_with<br/>build/mod.rs:494
    participant PL as PrivateLinks<br/>build/redact.rs:154
    participant PUB as stage_redacted<br/>build/publish.rs:194
    participant VAL as validate<br/>check/validate.rs:549
    participant PRJ as projection<br/>ir/projection.rs:600
    participant RDX as Redactor<br/>ir/projection.rs:296
    participant CLO as closure::compute<br/>ir/closure.rs:91

    RUN->>RUN: walk_pages once, then inspect_walk census (build/mod.rs:512, build/mod.rs:513)
    RUN->>RUN: Vault::load_with_walk then Corpus::build raw (build/mod.rs:514, build/mod.rs:520)
    RUN->>PL: PrivateLinks::new over knowledge, working, sibling vault (build/mod.rs:568-573)
    RUN->>PUB: stage_redacted rewrites publish/ markdown (build/mod.rs:574)
    RUN->>VAL: validate on the RAW corpus, errors abort (build/mod.rs:576-582)
    RUN->>PRJ: excluded_identities_of then project_vault_with_links (build/mod.rs:599, build/mod.rs:613)<br/>machine scope forks to project_machine, see OM-06.2 (build/mod.rs:605)
    PRJ->>PL: is_private, frontmatter_value, body — Step 1 typed edges and body links (ir/projection.rs:634-653)
    PRJ->>RDX: Step 2 Split::redactor builds the marker alternation<br/>over private ids, names, http projections, Redactor::new returns Result (ir/projection.rs:561-586)
    RDX->>RDX: clean_record drops private refs, redacts every string (ir/projection.rs:502-506, ir/projection.rs:377)<br/>clean_definition withholds defines-as whole (ir/projection.rs:413-416)
    PRJ-->>RUN: Corpus and ProseRedactor (ir/projection.rs:507)
    RUN->>CLO: closure::compute over the PROJECTED corpus (build/mod.rs:619)
    Note over CLO: not filtered again, matches build.py — ir/closure.rs:87-89
    Note over RUN: INVARIANT: both redactions precede the closure, so a private intermediate class never enters inferred ancestry — ir/projection.rs:13-15
    Note over PL: DIVERGENCE: the mirror re-runs links.body on the RAW page.body and re-applies the prose redactor, a second body-cleaning pass whose only reason is the Tally — build/mod.rs:786, ir/projection.rs:653
    Note over RDX: INVARIANT: the redactor fails CLOSED. A marker alternation past REDACTOR_SIZE_LIMIT<br/>is PublicationBlocked::Redactor and the build refuses, a None pattern means only<br/>an empty marker set — ir/projection.rs:288, ir/projection.rs:316-318, ir/projection.rs:586, ir/projection.rs:311-312
```
- Step 1 (`PrivateLinks`) is structural: it rewrites wikilinks and frontmatter before `Corpus::build`; Step 2 (`Redactor`) is lexical: it masks identity strings after `Corpus::build`.
- `PrivateLinks` resolves its own vault first, then the other (`build/redact.rs:192-194`); a journal is always non-public (`build/redact.rs:170`).
- A definition naming a private class is withheld whole (`ir/projection.rs:413-416`), at both layers (`ir/projection.rs:638-642`).

## OM-02.5 One walk of pages/ — walk_pages feeding the census, the vault load and the held-page identities

```mermaid
flowchart TD
    DIR["options.vault_root/pages<br/>build/mod.rs:511"]
    W1["walk_pages — ONE listing: follow_links false, walk ERRORS propagate, hidden dirs become SkipReason::HiddenDirectory, a symlink or device .md becomes SkipReason::NotARegularFile<br/>input/page.rs:532, input/page.rs:537-540, input/page.rs:548-557, input/page.rs:560, build/mod.rs:512"]
    CEN["inspect_walk — census over that walk: skips counted as excluded_directory and excluded_not_regular_file, an unreadable file or an unterminated fence is invalid_input, only YAML true or false counts, any other public value is invalid_publication_flag and refuses the build<br/>ir/projection.rs:95, ir/projection.rs:103-109, ir/projection.rs:117-127, ir/projection.rs:138-142, ir/projection.rs:145-154, build/mod.rs:513"]
    LOAD["Vault::load_with_walk — parses the SAME file list, held pages included<br/>input/page.rs:436, build/mod.rs:514"]
    PUB["Page::is_public — only a real YAML boolean true grants publication, the string arm is gone<br/>input/page.rs:231-236, input/frontmatter.rs:247-251"]
    EXC["excluded_identities_of — held pages read from the loaded vault, no second walk: id, slug, resource, title<br/>ir/projection.rs:227, ir/projection.rs:241-248, build/mod.rs:599"]
    MIG["migrate compile: a SECOND copy of the same four-value set, from the in-memory vault<br/>migrate/compile.rs:90-98"]
    USE["extra_private into the collision check and the marker set<br/>ir/projection.rs:546, ir/projection.rs:572"]
    DIR --> W1
    W1 --> CEN
    W1 --> LOAD --> PUB
    LOAD --> EXC --> USE
    MIG --> USE
    n1["INVARIANT: ONE public rule. The census requires a real boolean ir/projection.rs:138-141, Page::is_public matches Bool(true) only input/page.rs:232-235, validate reports INVALID_PUBLIC_FLAG check/validate.rs:359, so the migrate and example paths read the same gate migrate/compile.rs:101."]
    n2["INVARIANT: one walk per vault and one error policy, an unreadable subtree fails the census and the load alike input/page.rs:537-540, build/mod.rs:512-514. Machine scope walks working/pages the same way build/mod.rs:542-544."]
    n4["DIVERGENCE: dead branch — the census sums keys starting conflicting_ ir/projection.rs:147 but nothing ever bumps one, and Census.rejected is the constant 0 ir/projection.rs:171."]
    n5["DIVERGENCE: held-back set mismatch — excluded_identities_of and the census use is_unpublished (_misc only, top level) input/page.rs:583, ir/projection.rs:113, ir/projection.rs:234; staging and the link path use held_back (_misc and misc, any depth) build/publish.rs:130, build/publish.rs:134."]
```
- The census exists so the parser cannot silently drop a file and so a non-boolean flag is a refusal (`ir/projection.rs:6-9`); its counts are written to `api/census.json` (`build/mod.rs:711`).
- `excluded_identities_of` is what stops a link from publishing a held page: its values join `private_ids` (`ir/projection.rs:546`); the path-walking `excluded_identities` (`ir/projection.rs:201`) has no caller left in the workspace.
- A dot-directory is deliberately NOT a held page (`ir/projection.rs:189-196`), so it is absent from `excluded_identities_of` but counted as `excluded_directory` by the census (`ir/projection.rs:106`).

## OM-02.6 Register — parallel implementations and dead code, ranked

```mermaid
flowchart TD
    R1["1 three wikilink resolvers: Corpus resolve, PrivateLinks resolve, shacl Resolver<br/>ir/model.rs:336, build/redact.rs:210, shacl.rs:1201"]
    R2["2 two projection paths plus the machine fork, compat one kept for byte-parity, and held_back applied in the link path only<br/>build/mod.rs:602-617, ir/projection.rs:623"]
    R3["3 three held-back/identity derivations: is_unpublished, held_back, migrate copy<br/>input/page.rs:583, build/publish.rs:134, migrate/compile.rs:38-98"]
    R4["4 three Corpus::build per build over ONE walk<br/>build/mod.rs:512, build/mod.rs:520, ir/projection.rs:607, ir/projection.rs:609"]
    R5["5 slug and IRI derivations: Page::slug, Page::resource, inline format, target_slug<br/>input/page.rs:207, input/page.rs:217, ir/model.rs:342, ir/model.rs:424"]
    R6["6 by_class_slug built in two places, indexes re-derives a link's slug from IRI or label<br/>ir/model.rs:436-441, ir/projection.rs:512-518, build/indexes.rs:55-60"]
    D1["DEAD: parse_page has no caller in the workspace<br/>input/page.rs:598"]
    D2["DEAD: excluded_identities path form has no caller, excluded_identities_of replaced it<br/>ir/projection.rs:201"]
    D3["DEAD: Census.rejected constant 0 and the conflicting_ counter<br/>ir/projection.rs:171, ir/projection.rs:147"]
    D4["TEST-ONLY: project has no production caller in vault build, only migrate, an example and a test<br/>migrate/compile.rs:101, examples/shacl_export.rs:76, tests/jsonld_matches_turtle.rs:80"]
    X1["RESOLVED in 8d654d1: fail-open redactor, the coercing public gate, three separate directory walks, and the dead iri_for, page_files, has_secret and unsatisfiable_message are gone from the tree"]
    n1["EXTERNAL: parse_page documents VisionClaw CorpusSource::LocalDirectory as its caller input/page.rs:593 — no caller exists in the ontomark workspace, so this repo cannot verify it."]
    R1 --> R2 --> R3 --> R4 --> R5 --> R6
    R6 --> D1 --> D2 --> D3 --> D4
    D4 --> X1
```
- Ranking is by blast radius on the publication boundary, not by code size; R1 to R3 can mis-redact or mis-hold, R4 to R6 cost only duplication. The former top two (fail-open redactor, coercing public gate) were closed by the OM-00 reconcile.
- Grep scope was the whole ontomark workspace (`crates/*`); the wasm, ingest, parse and model crates call none of these functions.
