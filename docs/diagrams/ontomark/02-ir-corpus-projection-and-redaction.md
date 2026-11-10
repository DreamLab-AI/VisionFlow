---
id: OM-02
title: IR corpus, public projection and private-link redaction
area: ontomark
governing:
  - docs/engineering/sovereign-corpus-contracts.md
adrs: [ADR-OM-001, ADR-OM-005, ADR-OM-007, ADR-OM-008]
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
verified_commit: 6159bb6089753105ef839fd4aac46c1598a64bd2
---

## OM-02.1 Corpus::build — Vault to ClassRecord, every field's source and every resolver

```mermaid
flowchart TD
    VAULT["Vault.pages — loaded by Vault::load or Vault::load_with_walk<br/>input/page.rs:464, input/page.rs:506"]
    NAMES["NameIndex::build — lowercased id, title, slug, aliases; first page wins<br/>ir/model.rs:331, ir/model.rs:228-236, ir/model.rs:226"]
    RES["resources[i] = Page::resource namespace<br/>ir/model.rs:332-336, input/page.rs:255"]
    RESOLVE["resolve target,label closure<br/>1 names.get target - resources[i]<br/>2 tail_iris by label, 3 tail_iris by target<br/>4 namespace + slugify target<br/>ir/model.rs:338-349, ir/model.rs:340-344"]
    LINKSOF["links_of page,key — wikilinks through resolve<br/>ir/model.rs:351-357"]
    REL["relations — 12 RELATION_KEYS plus disjointWith, empty lists absent<br/>ir/model.rs:365-375, ir/model.rs:36"]
    ISA["sub_class_of from is-a, instance_of from instance-of<br/>ir/model.rs:415-416"]
    CURATED["links — Page::outbound_links, slug-VERBATIM, bypasses NameIndex<br/>namespace + target_slug w.target<br/>ir/model.rs:423-430, input/page.rs:290"]
    DEFS["defines_as — definition_of through the SAME resolve<br/>ir/model.rs:433, ir/model.rs:299-325"]
    SCALARS["page_id, title, slug, public, domain, quality, legacy_term_id, body<br/>ir/model.rs:382-395, ir/model.rs:408-409, ir/model.rs:431"]
    SLUGSRC["slug = Page::slug declared else slugify title<br/>public = Page::is_public<br/>input/page.rs:245, input/page.rs:269"]
    IRIS["iri = resources[i]; page_iri = page_resource else resources[i]<br/>ir/model.rs:390-393"]
    TEXTS["label else title; definition else leading_paragraph; maturity else draft<br/>ir/model.rs:384-387, ir/model.rs:400-407, input/page.rs:321"]
    GEN["generated_by = Stamp::from_yaml generated, by only<br/>ir/model.rs:410-413"]
    HASONT["has_ontology = has resource OR has type<br/>ir/model.rs:432"]
    DISAMB["disambiguate_public_slugs — declared slug keeps it, else resource tail, else first<br/>ir/model.rs:437, ir/model.rs:263-294"]
    BYSLUG["by_class_slug — has_ontology AND iri non-empty, first wins<br/>ir/model.rs:439-444"]
    OUT["Corpus records + by_class_slug<br/>ir/model.rs:446-449"]
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
    n1["DIVERGENCE: THREE wikilink-target resolvers. Corpus::build resolve ir/model.rs:338, PrivateLinks::resolve build/redact.rs:210, and shacl Resolver::page shacl.rs:1201 (no slug key, case-sensitive id match at shacl.rs:1203). The first two share NameIndex, the third does not."]
    n2["DIVERGENCE: PrivateLinks strips a #fragment and a .md suffix before lookup build/redact.rs:219, build/redact.rs:227; Corpus::build passes the target straight to names.get ir/model.rs:340, so a fragment link redacted as private is resolved here to namespace + slugify of the whole string."]
    n3["DIVERGENCE: curated links are slug-verbatim ir/model.rs:427, every other relation goes through resolve ir/model.rs:340. Two target-to-IRI derivations in one function; deliberate for backlink keys ir/model.rs:418-422."]
```
- `Corpus::build` is the only constructor of `ClassRecord` outside tests; every artefact is a pure function of it (`ir/model.rs:4-7`).
- `slug` is the page publish slug, `class_slug()` is the IRI tail; they differ on 96 pages (`ir/model.rs:187-192`), and `disambiguate_public_slugs` rewrites `slug` from the IRI tail on a clash (`ir/model.rs:287-289`).
- Private records are never renamed; a public/private clash is left to the projection (`ir/model.rs:259-262`).

## OM-02.2 The record selectors — which accessor is right for what

```mermaid
flowchart LR
    REC["Corpus.records — every record, no filter<br/>ir/model.rs:205"]
    PUB["Corpus::public — public flag only<br/>ir/model.rs:453-455"]
    PCL["Corpus::public_classes — public AND has_ontology AND iri non-empty AND Class<br/>ir/model.rs:459-463"]
    BYS["Corpus::by_slug — class slug, first definition wins, includes private and Individual<br/>ir/model.rs:467-469, ir/model.rs:439-444"]
    C1["closure::compute iterates records with has_ontology and iri, Individuals included<br/>ir/closure.rs:99-102"]
    C2["page_count = listed.records.len, listed is the merged corpus under machine scope<br/>build/mod.rs:875-876"]
    C3["class_count = public_classes.count<br/>build/mod.rs:870"]
    C4["OKF bundle classes and scaffold-index candidates<br/>build/okf.rs:188, build/indexes.rs:102"]
    C5["page API, mirror loop, search index<br/>build/page_api.rs:77, build/mod.rs:974, build/mod.rs:984, build/indexes.rs:256, build/indexes.rs:311"]
    C6["validate: parent label check on the RAW corpus<br/>check/validate.rs:569"]
    REC --> C1
    REC --> C2
    PCL --> C3
    PCL --> C4
    PUB --> C5
    BYS --> C6
    n1["INVARIANT: every consumer except validate receives the PROJECTED corpus, where records are already public only ir/projection.rs:500-504, so the public flag test in public and public_classes is a second, redundant public gate ir/model.rs:454, ir/model.rs:461."]
    n2["DIVERGENCE: closure and by_class_slug admit Individuals ir/closure.rs:99-102, ir/model.rs:441, while public_classes excludes them ir/model.rs:461; class_count and the closure therefore count different sets."]
    n3["INVARIANT: validate runs on the raw corpus build/mod.rs:729, before projection build/mod.rs:755, so by_slug there can resolve a PRIVATE record check/validate.rs:569."]
```
- Use `public_classes` for anything that is an OWL class count or class listing; `public` for per-page surfaces; `records` only before projection or for the closure.
- No selector is dead: `public` (5 non-test callers), `public_classes` (3), `by_slug` (1, `check/validate.rs:569`).

## OM-02.3 The two projection paths — project_with_redactor vs project_vault_with_links

```mermaid
flowchart TD
    SEL{"build::run selects by redact_private_links<br/>profile default true<br/>build/mod.rs:709-711, build/mod.rs:755-770, profile.rs:433"}
    subgraph LEGACY["Compat path — runs on the RAW corpus"]
      PWR["project_with_redactor profile, raw, excluded<br/>ir/projection.rs:493, build/mod.rs:769"]
    end
    subgraph LINKS["Link-aware path — rebuilds the corpus from a cleaned Vault clone"]
      L1["project profile, Corpus::build vault, extra — result DISCARDED, collision check only<br/>ir/projection.rs:605"]
      L2["vault.clone; held_back pages forced public:false<br/>ir/projection.rs:615-622, build/publish.rs:138"]
      L3["skip private pages<br/>ir/projection.rs:620"]
      L4["links pinned to non-private outbound_links, even when empty<br/>ir/projection.rs:625-630, ir/projection.rs:644-646"]
      L5["frontmatter.retain — defines-as dropped whole, other values through frontmatter_value<br/>ir/projection.rs:631-643, build/redact.rs:276"]
      L6["body = PrivateLinks::body, Tally thrown away<br/>ir/projection.rs:647, build/redact.rs:307"]
      L7["Corpus::build cleaned vault, THEN project_with_redactor<br/>ir/projection.rs:607"]
      L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7
    end
    CORE["project_with_redactor core: Split::checked partition and collision, Split::redactor markers, clean_record, corpus_of by_class_slug<br/>ir/projection.rs:493, ir/projection.rs:535, ir/projection.rs:559, ir/projection.rs:510"]
    WRAP["project — thin wrapper dropping the redactor<br/>ir/projection.rs:441-447"]
    MACH["machine scope: project_machine runs the link-aware path on knowledge, then adds working plain records, see OM-06.2<br/>ir/projection.rs:741, build/mod.rs:756-758"]
    SEL -->|false| PWR --> CORE
    SEL -->|true| L1
    SEL -->|scope machine| MACH --> L1
    L7 --> CORE
    WRAP --> CORE
    PROD["non-test callers of project: migrate compile, shacl_export example, jsonld test<br/>migrate/compile.rs:116, examples/shacl_export.rs:76, tests/jsonld_matches_turtle.rs:80"]
    PROD --> WRAP
    n1["DIVERGENCE: the compat path never forces misc/ pages private (held_back is only applied in the link path ir/projection.rs:617); a public misc/ page stays public there, while stage_redacted skips it build/publish.rs:220."]
    n2["DIVERGENCE: the compat path exists only to keep the legacy projection bytes build/mod.rs:768, an opt-out of the default redact_private_links=true profile.rs:433. A projection path kept for byte-parity."]
    n3["DIVERGENCE: Corpus::build runs THREE times on the public link path: raw build/mod.rs:671, collision probe ir/projection.rs:605, cleaned ir/projection.rs:607; machine scope adds the working validate build/mod.rs:742 and four more in project_machine ir/projection.rs:764-781."]
```
- Both paths end in the same `project_with_redactor`; the link path only differs in WHAT corpus it is handed (links and frontmatter pre-redacted at the `Vault` level).
- `project_vault_with_links` is called from `build/mod.rs:766` (and from `project_machine`, `ir/projection.rs:752`); `project_with_redactor` is called directly once, `build/mod.rs:769`.
- The `ProseRedactor` is the same `Redactor` that cleaned the records (`ir/projection.rs:505`), reused for the markdown mirror (`build/mod.rs:1003`).

## OM-02.4 Redaction order — private-link rewrite, marker redaction, closure

```mermaid
sequenceDiagram
    autonumber
    participant RUN as build::run_with<br/>build/mod.rs:631
    participant PL as PrivateLinks<br/>build/redact.rs:154
    participant PUB as stage_redacted<br/>build/publish.rs:200
    participant VAL as validate<br/>check/validate.rs:569
    participant PRJ as projection<br/>ir/projection.rs:598
    participant RDX as Redactor<br/>ir/projection.rs:293
    participant CLO as closure::compute<br/>ir/closure.rs:91

    RUN->>RUN: walk_pages once, then inspect_walk census (build/mod.rs:661, build/mod.rs:662)
    RUN->>RUN: Vault::load_with_walk then Corpus::build raw (build/mod.rs:663, build/mod.rs:671)
    RUN->>PL: PrivateLinks::new over knowledge, working, sibling vault (build/mod.rs:721-726)
    RUN->>PUB: stage_redacted rewrites publish/ markdown (build/mod.rs:727)
    RUN->>VAL: validate on the RAW corpus, errors abort (build/mod.rs:729-735)
    RUN->>PRJ: excluded_identities_of then project_vault_with_links (build/mod.rs:752, build/mod.rs:766)<br/>machine scope forks to project_machine, see OM-06.2 (build/mod.rs:758)
    PRJ->>PL: is_private, frontmatter_value, body — Step 1 typed edges and body links (ir/projection.rs:628-647)
    PRJ->>RDX: Step 2 Split::redactor builds the marker alternation<br/>over private ids, names, http projections, Redactor::new returns Result (ir/projection.rs:559-584)
    RDX->>RDX: clean_record drops private refs, redacts every string (ir/projection.rs:500-504, ir/projection.rs:374)<br/>clean_definition withholds defines-as whole (ir/projection.rs:411-414)
    PRJ-->>RUN: Corpus and ProseRedactor (ir/projection.rs:505)
    RUN->>CLO: closure::compute over the PROJECTED corpus (build/mod.rs:772)
    Note over CLO: not filtered again, matches build.py — ir/closure.rs:87-89
    Note over RUN: INVARIANT: both redactions precede the closure, so a private intermediate class never enters inferred ancestry — ir/projection.rs:13-15
    Note over PL: DIVERGENCE: the mirror re-runs links.body on the RAW page.body and re-applies the prose redactor, a second body-cleaning pass whose only reason is the Tally — build/mod.rs:1003, ir/projection.rs:647
    Note over RDX: INVARIANT: the redactor fails CLOSED. A marker alternation past REDACTOR_SIZE_LIMIT<br/>is PublicationBlocked::Redactor and the build refuses, a None pattern means only<br/>an empty marker set — ir/projection.rs:285, ir/projection.rs:313-315, ir/projection.rs:584, ir/projection.rs:308-309
```
- Step 1 (`PrivateLinks`) is structural: it rewrites wikilinks and frontmatter before `Corpus::build`; Step 2 (`Redactor`) is lexical: it masks identity strings after `Corpus::build`.
- `PrivateLinks` resolves its own vault first, then the other (`build/redact.rs:192-194`); a journal is always non-public (`build/redact.rs:170`).
- A definition naming a private class is withheld whole (`ir/projection.rs:411-414`), at both layers (`ir/projection.rs:632-636`).

## OM-02.5 One walk of the vault root — walk_pages feeding the census, the vault load and the held-page identities

```mermaid
flowchart TD
    DIR["options.vault_root — the vault root itself, no pages/ subfolder (ADR-OM-007)<br/>build/mod.rs:661"]
    W1["walk_pages — ONE listing: follow_links false, walk ERRORS propagate, hidden dirs become SkipReason::HiddenDirectory, a nested file under an excluded top-level folder (templates, bases, assets, journals, plus the profile's layout.excluded) becomes SkipReason::ExcludedDirectory, a symlink or device .md becomes SkipReason::NotARegularFile<br/>input/page.rs:607, input/page.rs:612-615, input/page.rs:623-633, input/page.rs:635, input/page.rs:642-649, build/mod.rs:661"]
    CEN["inspect_walk — census over that walk: hidden and excluded-folder skips counted as excluded_directory, non-regular files as excluded_not_regular_file, an unreadable file or an unterminated fence is invalid_input, only YAML true or false counts, any other public value is invalid_publication_flag and refuses the build<br/>ir/projection.rs:96, ir/projection.rs:104-110, ir/projection.rs:118-128, ir/projection.rs:139-143, ir/projection.rs:146-155, build/mod.rs:662"]
    LOAD["Vault::load_with_walk — parses the SAME file list, held pages included<br/>input/page.rs:506, build/mod.rs:663"]
    PUB["Page::is_public — only a real YAML boolean true grants publication, the string arm is gone<br/>input/page.rs:269-274, input/frontmatter.rs:247-251"]
    EXC["excluded_identities_of — held pages read from the loaded vault, no second walk: id, slug, resource, title<br/>ir/projection.rs:228, ir/projection.rs:238-245, build/mod.rs:752"]
    MIG["migrate compile: a SECOND copy of the same four-value set, from the in-memory vault<br/>migrate/compile.rs:105-113"]
    USE["extra_private into the collision check and the marker set<br/>ir/projection.rs:544, ir/projection.rs:570"]
    DIR --> W1
    W1 --> CEN
    W1 --> LOAD --> PUB
    LOAD --> EXC --> USE
    MIG --> USE
    n1["INVARIANT: ONE public rule. The census requires a real boolean ir/projection.rs:139-142, Page::is_public matches Bool(true) only input/page.rs:269-274, validate reports INVALID_PUBLIC_FLAG check/validate.rs:359, so the migrate and example paths read the same gate migrate/compile.rs:116."]
    n2["INVARIANT: one walk per vault and one error policy, an unreadable subtree fails the census and the load alike input/page.rs:612-615, build/mod.rs:661-663. Machine scope walks the layout's working vault root the same way build/mod.rs:692-697."]
    n4["DIVERGENCE: dead branch — the census sums keys starting conflicting_ ir/projection.rs:148 but nothing ever bumps one, and Census.rejected is the constant 0 ir/projection.rs:172."]
    n5["DIVERGENCE: held-back set mismatch — excluded_identities_of and the census use is_unpublished (_misc only, top level) input/page.rs:667, ir/projection.rs:114, ir/projection.rs:114; staging and the link path use held_back (_misc and misc, at any depth: UNPUBLISHED_DIRS or HELD_BACK_DIRS) build/publish.rs:134, build/publish.rs:138-148."]
```
- The census exists so the parser cannot silently drop a file and so a non-boolean flag is a refusal (`ir/projection.rs:6-9`); its counts are written to `api/census.json` (`build/mod.rs:926`).
- `excluded_identities_of` is what stops a link from publishing a held page: its values join `private_ids` (`ir/projection.rs:544`); the path-walking `excluded_identities` (`ir/projection.rs:202`) has no caller left in the workspace.
- **Changed at 07f4c1c, still true at 4b9d15d:** the walk is of the vault root, not of a `pages/` folder (ADR-OM-007), and a nested file under an excluded top-level folder is a reported `ExcludedDirectory` skip that the census counts as `excluded_directory` (`input/page.rs:642-649`, `ir/projection.rs:107`); staged page paths and held-back tests are vault-root-relative, so the old `strip_prefix("pages")` steps are gone (`build/publish.rs:220`, `ir/projection.rs:231`). The published tree names stay a URL contract: `stage_redacted` keys per-vault counts by `VaultKind::role_name`, not by the folder read (`build/publish.rs:211-213`, `input/page.rs:155`).
- A dot-directory is deliberately NOT a held page (`ir/projection.rs:190-197`), so it is absent from `excluded_identities_of` but counted as `excluded_directory` by the census (`ir/projection.rs:107`).

## OM-02.6 Register — parallel implementations and dead code, ranked

```mermaid
flowchart TD
    R1["1 three wikilink resolvers: Corpus resolve, PrivateLinks resolve, shacl Resolver<br/>ir/model.rs:338, build/redact.rs:210, shacl.rs:1201"]
    R2["2 two projection paths plus the machine fork, compat one kept for byte-parity, and held_back applied in the link path only<br/>build/mod.rs:755-770, ir/projection.rs:617"]
    R3["3 three held-back/identity derivations: is_unpublished, held_back, migrate copy<br/>input/page.rs:667, build/publish.rs:138, migrate/compile.rs:38-113"]
    R4["4 three Corpus::build per build over ONE walk<br/>build/mod.rs:661, build/mod.rs:671, ir/projection.rs:605, ir/projection.rs:607"]
    R5["5 slug and IRI derivations: Page::slug, Page::resource, inline format, target_slug<br/>input/page.rs:245, input/page.rs:255, ir/model.rs:344, ir/model.rs:427"]
    R6["6 by_class_slug built in two places, indexes re-derives a link's slug from IRI or label<br/>ir/model.rs:439-444, ir/projection.rs:510-516, build/indexes.rs:55-60"]
    D1["DEAD: parse_page has no caller in the workspace<br/>input/page.rs:683"]
    D2["DEAD: excluded_identities path form has no caller, excluded_identities_of replaced it<br/>ir/projection.rs:202"]
    D3["DEAD: Census.rejected constant 0 and the conflicting_ counter<br/>ir/projection.rs:172, ir/projection.rs:148"]
    D4["TEST-ONLY: project has no production caller in vault build, only migrate, an example and a test<br/>migrate/compile.rs:116, examples/shacl_export.rs:76, tests/jsonld_matches_turtle.rs:80"]
    X1["RESOLVED in 8d654d1: fail-open redactor, the coercing public gate, three separate directory walks, and the dead iri_for, page_files, has_secret and unsatisfiable_message are gone from the tree"]
    n1["EXTERNAL: parse_page documents VisionClaw CorpusSource::LocalDirectory as its caller input/page.rs:677 — no caller exists in the ontomark workspace, so this repo cannot verify it."]
    R1 --> R2 --> R3 --> R4 --> R5 --> R6
    R6 --> D1 --> D2 --> D3 --> D4
    D4 --> X1
```
- Ranking is by blast radius on the publication boundary, not by code size; R1 to R3 can mis-redact or mis-hold, R4 to R6 cost only duplication. The former top two (fail-open redactor, coercing public gate) were closed by the OM-00 reconcile.
- Grep scope was the whole ontomark workspace (`crates/*`) at 4b9d15d; the wasm, ingest, source, parse and model crates call none of these functions.
