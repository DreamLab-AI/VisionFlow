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
verified_commit: ffe06e45bfebd388a1d85849b24a8810e4d0d610
---

## OM-02.1 Corpus::build — Vault to ClassRecord, every field's source and every resolver

```mermaid
flowchart TD
    VAULT["Vault.pages — loaded by Vault::load<br/>input/page.rs:390"]
    NAMES["NameIndex::build — lowercased id, title, slug, aliases; first page wins<br/>ir/model.rs:329, ir/model.rs:226-234, ir/model.rs:224"]
    RES["resources[i] = Page::resource namespace<br/>ir/model.rs:330-334, input/page.rs:202"]
    RESOLVE["resolve target,label closure<br/>1 names.get target - resources[i]<br/>2 tail_iris by label, 3 tail_iris by target<br/>4 namespace + slugify target<br/>ir/model.rs:336-347, ir/model.rs:338-342"]
    LINKSOF["links_of page,key — wikilinks through resolve<br/>ir/model.rs:349-355"]
    REL["relations — 12 RELATION_KEYS plus disjointWith, empty lists absent<br/>ir/model.rs:363-373, ir/model.rs:36"]
    ISA["sub_class_of from is-a, instance_of from instance-of<br/>ir/model.rs:412-413"]
    CURATED["links — Page::outbound_links, slug-VERBATIM, bypasses NameIndex<br/>namespace + target_slug w.target<br/>ir/model.rs:420-427, input/page.rs:229"]
    DEFS["defines_as — definition_of through the SAME resolve<br/>ir/model.rs:430, ir/model.rs:297-323"]
    SCALARS["page_id, title, slug, public, domain, quality, legacy_term_id, body<br/>ir/model.rs:380-393, ir/model.rs:406-407, ir/model.rs:428"]
    SLUGSRC["slug = Page::slug declared else slugify title<br/>public = Page::is_public<br/>input/page.rs:192, input/page.rs:211"]
    IRIS["iri = resources[i]; page_iri = page_resource else resources[i]<br/>ir/model.rs:388-391"]
    TEXTS["label else title; definition else leading_paragraph; maturity else draft<br/>ir/model.rs:382-385, ir/model.rs:398-405, input/page.rs:260"]
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
    n1["DIVERGENCE: THREE wikilink-target resolvers. Corpus::build resolve ir/model.rs:336, PrivateLinks::resolve build/redact.rs:209, and shacl Resolver::page shacl.rs:1197 (no slug key, case-sensitive id match at shacl.rs:1199). The first two share NameIndex, the third does not."]
    n2["DIVERGENCE: PrivateLinks strips a #fragment and a .md suffix before lookup build/redact.rs:218, build/redact.rs:226; Corpus::build passes the target straight to names.get ir/model.rs:338, so a fragment link redacted as private is resolved here to namespace + slugify of the whole string."]
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
    C2["page_count = corpus.records.len<br/>build/mod.rs:406"]
    C3["class_count = public_classes.count<br/>build/mod.rs:405"]
    C4["OKF bundle classes and scaffold-index candidates<br/>build/okf.rs:74, build/indexes.rs:102"]
    C5["page API, mirror loop, search index<br/>build/page_api.rs:77, build/mod.rs:487, build/mod.rs:497, build/indexes.rs:255, build/indexes.rs:310"]
    C6["validate: parent label check on the RAW corpus<br/>check/validate.rs:496"]
    REC --> C1
    REC --> C2
    PCL --> C3
    PCL --> C4
    PUB --> C5
    BYS --> C6
    n1["INVARIANT: every consumer except validate receives the PROJECTED corpus, where records are already public only ir/projection.rs:426, so the public flag test in public and public_classes is a second, redundant public gate ir/model.rs:451, ir/model.rs:458."]
    n2["DIVERGENCE: closure and by_class_slug admit Individuals ir/closure.rs:99-102, ir/model.rs:438, while public_classes excludes them ir/model.rs:458; class_count and the closure therefore count different sets."]
    n3["INVARIANT: validate runs on the raw corpus build/mod.rs:356, before projection build/mod.rs:365, so by_slug there can resolve a PRIVATE record check/validate.rs:496."]
```
- Use `public_classes` for anything that is an OWL class count or class listing; `public` for per-page surfaces; `records` only before projection or for the closure.
- No selector is dead: `public` (5 non-test callers), `public_classes` (3), `by_slug` (1, `check/validate.rs:496`).

## OM-02.3 The two projection paths — project_with_redactor vs project_vault_with_links

```mermaid
flowchart TD
    SEL{"build::run selects by redact_private_links<br/>profile default true<br/>build/mod.rs:336-338, build/mod.rs:365-370, profile.rs:417"}
    subgraph LEGACY["Compat path — runs on the RAW corpus"]
      PWR["project_with_redactor profile, raw, excluded<br/>ir/projection.rs:383, build/mod.rs:369"]
    end
    subgraph LINKS["Link-aware path — rebuilds the corpus from a cleaned Vault clone"]
      L1["project profile, Corpus::build vault, extra — result DISCARDED, collision check only<br/>ir/projection.rs:460"]
      L2["vault.clone; held_back pages forced public:false<br/>ir/projection.rs:461-469, build/publish.rs:134"]
      L3["skip private pages<br/>ir/projection.rs:470"]
      L4["links pinned to non-private outbound_links, even when empty<br/>ir/projection.rs:475-480, ir/projection.rs:494-496"]
      L5["frontmatter.retain — defines-as dropped whole, other values through frontmatter_value<br/>ir/projection.rs:481-493, build/redact.rs:256"]
      L6["body = PrivateLinks::body, Tally thrown away<br/>ir/projection.rs:497, build/redact.rs:287"]
      L7["Corpus::build cleaned vault, THEN project_with_redactor<br/>ir/projection.rs:499"]
      L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7
    end
    CORE["project_with_redactor core: partition, collision, markers, clean_record, by_class_slug<br/>ir/projection.rs:388-440"]
    WRAP["project — thin wrapper dropping the redactor<br/>ir/projection.rs:356-362"]
    SEL -->|false| PWR --> CORE
    SEL -->|true| L1
    L7 --> CORE
    WRAP --> CORE
    PROD["non-test callers of project: migrate compile, shacl_export example, jsonld test<br/>migrate/compile.rs:101, examples/shacl_export.rs:76, tests/jsonld_matches_turtle.rs:80"]
    PROD --> WRAP
    n1["DIVERGENCE: the compat path never forces misc/ pages private (held_back is only applied in the link path ir/projection.rs:467); a public misc/ page stays public there, while stage_redacted skips it build/publish.rs:216."]
    n2["DIVERGENCE: the compat path exists only to keep the legacy projection bytes build/mod.rs:368, an opt-out of the default redact_private_links=true profile.rs:417. A projection path kept for byte-parity."]
    n3["DIVERGENCE: Corpus::build runs up to THREE times per build: raw build/mod.rs:313, collision probe ir/projection.rs:460, cleaned ir/projection.rs:499."]
```
- Both paths end in the same `project_with_redactor`; the link path only differs in WHAT corpus it is handed (links and frontmatter pre-redacted at the `Vault` level).
- `project_vault_with_links` is called once, `build/mod.rs:366`; `project_with_redactor` is called directly once, `build/mod.rs:369`.
- The `ProseRedactor` is the same `Redactor` that cleaned the records (`ir/projection.rs:439`), reused for the markdown mirror (`build/mod.rs:516`).

## OM-02.4 Redaction order — private-link rewrite, marker redaction, closure

```mermaid
sequenceDiagram
    autonumber
    participant RUN as build::run<br/>build/mod.rs:302
    participant PL as PrivateLinks<br/>build/redact.rs:153
    participant PUB as stage_redacted<br/>build/publish.rs:194
    participant VAL as validate<br/>check/validate.rs:496
    participant PRJ as projection<br/>ir/projection.rs:453
    participant RDX as Redactor<br/>ir/projection.rs:217
    participant CLO as closure::compute<br/>ir/closure.rs:91

    RUN->>RUN: inspect_inputs census (build/mod.rs:309)
    RUN->>RUN: load_vault then Corpus::build raw (build/mod.rs:311, build/mod.rs:313)
    RUN->>PL: PrivateLinks::new over knowledge, working, sibling vault (build/mod.rs:348-353)
    RUN->>PUB: stage_redacted rewrites publish/ markdown (build/mod.rs:354)
    RUN->>VAL: validate on the RAW corpus, errors abort (build/mod.rs:356-362)
    RUN->>PRJ: excluded_identities then project_vault_with_links (build/mod.rs:364-366)
    PRJ->>PL: is_private, frontmatter_value, body — Step 1 typed edges and body links (ir/projection.rs:478-497)
    PRJ->>RDX: Step 2 marker alternation over private ids, names, http projections (ir/projection.rs:410-425)
    RDX->>RDX: clean_record drops private refs, redacts every string, withholds defines-as whole (ir/projection.rs:426, ir/projection.rs:290-345)
    PRJ-->>RUN: Corpus and ProseRedactor (ir/projection.rs:434-440)
    RUN->>CLO: closure::compute over the PROJECTED corpus (build/mod.rs:372)
    Note over CLO: not filtered again, matches build.py — ir/closure.rs:87-89
    Note over RUN: INVARIANT: both redactions precede the closure, so a private intermediate class never enters inferred ancestry — ir/projection.rs:13-15
    Note over PL: DIVERGENCE: the mirror re-runs links.body on the RAW page.body and re-applies the prose redactor, a second body-cleaning pass whose only reason is the Tally — build/mod.rs:516, ir/projection.rs:497
    Note over RDX: DIVERGENCE: if Regex::new fails the pattern is None and redact returns the text UNCHANGED, failing open on the security boundary — ir/projection.rs:232, ir/projection.rs:243-245
```
- Step 1 (`PrivateLinks`) is structural: it rewrites wikilinks and frontmatter before `Corpus::build`; Step 2 (`Redactor`) is lexical: it masks identity strings after `Corpus::build`.
- `PrivateLinks` resolves its own vault first, then the other (`build/redact.rs:192-194`); a journal is always non-public (`build/redact.rs:170`).
- A definition naming a private class is withheld whole (`ir/projection.rs:323-328`), at both layers (`ir/projection.rs:482-484`).

## OM-02.5 The second and third reads of pages/ — inspect_inputs and excluded_identities beside Vault::load

```mermaid
flowchart TD
    DIR["options.vault_root/pages<br/>build/mod.rs:308"]
    W1["Vault::load — walk_pages: follow_links false, ERRORS propagate, only dot-dirs skipped<br/>input/page.rs:399, input/page.rs:492-530, input/page.rs:497-500, input/page.rs:512"]
    W2["inspect_inputs — OWN walkdir, filter_map ok SWALLOWS walk errors, re-reads every .md<br/>ir/projection.rs:77-83, ir/projection.rs:97"]
    W3["excluded_identities — a THIRD walkdir, re-reads only _misc files<br/>ir/projection.rs:172-184"]
    CEN["Census: strict boolean — only YAML true or false counts; any other value is invalid_publication_flag and refuses the build<br/>ir/projection.rs:112-116, ir/projection.rs:119-128"]
    LOAD["Page::is_public — frontmatter.boolean, ALSO accepts the strings true and false<br/>input/page.rs:211, input/frontmatter.rs:233-235"]
    EXC["identity set per held page: id, slug, resource, title; unparseable pages skipped silently<br/>ir/projection.rs:188-197, ir/projection.rs:189"]
    MIG["migrate compile: a FOURTH copy of the same four-value set, from the in-memory vault<br/>migrate/compile.rs:90-98"]
    USE["extra_private into the collision check and the marker set<br/>ir/projection.rs:393, ir/projection.rs:410"]
    DIR --> W1
    DIR --> W2 --> CEN
    DIR --> W3 --> EXC --> USE
    W1 --> LOAD
    MIG --> USE
    n1["DIVERGENCE: TWO public gates. Census requires a real boolean ir/projection.rs:112-115; Page::is_public coerces the string true input/frontmatter.rs:233-235. The census is what makes the coercion safe, and only build::run calls it build/mod.rs:309, so the migrate and example paths trust the looser gate migrate/compile.rs:101."]
    n2["DIVERGENCE: THREE walks of pages/ per build, one error policy each: propagate input/page.rs:497, swallow ir/projection.rs:79, swallow ir/projection.rs:174. An unreadable subtree fails Vault::load but not the census."]
    n3["DOC-DRIFT: excluded_identities documents a PublicationBlocked::Census error ir/projection.rs:168 but returns io::Error and always Ok ir/projection.rs:169, ir/projection.rs:199."]
    n4["DIVERGENCE: dead branch — the census sums keys starting conflicting_ ir/projection.rs:121 but nothing ever bumps one, and Census.rejected is the constant 0 ir/projection.rs:142."]
    n5["DIVERGENCE: held-back set mismatch — excluded_identities and the census use is_unpublished (_misc only, top level) input/page.rs:543, ir/projection.rs:93, ir/projection.rs:182; staging and the link path use held_back (_misc and misc, any depth) build/publish.rs:121-140."]
```
- The census exists so the parser cannot silently drop a file and so a non-boolean flag is a refusal (`ir/projection.rs:6-9`); its counts are written to `api/census.json` (`build/mod.rs:441`).
- `excluded_identities` is what stops a link from publishing a held page: its values join `private_ids` (`ir/projection.rs:393`).
- A dot-directory is deliberately NOT a held page (`ir/projection.rs:158-166`), so it is absent from `excluded_identities` but counted as `excluded_directory` by the census (`ir/projection.rs:93-94`).

## OM-02.6 Register — parallel implementations and dead code, ranked

```mermaid
flowchart TD
    R1["1 redactor fails open on regex compile error<br/>ir/projection.rs:232, ir/projection.rs:243-245"]
    R2["2 two public gates: strict census vs coercing Page::is_public<br/>ir/projection.rs:112-115, input/frontmatter.rs:233-235"]
    R3["3 three wikilink resolvers: Corpus resolve, PrivateLinks resolve, shacl Resolver<br/>ir/model.rs:336, build/redact.rs:209, shacl.rs:1197"]
    R4["4 two projection paths, compat one kept for byte-parity, and held_back applied in one only<br/>build/mod.rs:365-370, ir/projection.rs:467"]
    R5["5 three held-back/identity derivations: is_unpublished, held_back, migrate copy<br/>input/page.rs:543, build/publish.rs:134, migrate/compile.rs:38-98"]
    R6["6 three directory walks plus three Corpus::build per build<br/>input/page.rs:492, ir/projection.rs:77, ir/projection.rs:172, build/mod.rs:313, ir/projection.rs:460, ir/projection.rs:499"]
    R7["7 slug and IRI derivations: Page::slug, Page::resource, inline format, target_slug, iri_for<br/>input/page.rs:192, input/page.rs:202, ir/model.rs:342, ir/model.rs:424, input/slug.rs:88"]
    R8["8 by_class_slug built in two places; indexes re-derives a link's slug from IRI or label<br/>ir/model.rs:436-441, ir/projection.rs:428-433, build/indexes.rs:55-60"]
    D1["DEAD: slug::iri_for has no caller outside its own doctest<br/>input/slug.rs:88"]
    D2["DEAD: page_files only caller is a test; parse_page has none in the workspace<br/>input/page.rs:536, tests/publish_gate.rs:73, input/page.rs:557"]
    D3["DEAD: Census.rejected constant 0 and the conflicting_ counter<br/>ir/projection.rs:142, ir/projection.rs:121"]
    D4["TEST-ONLY: project has no production caller in vault build; only migrate, an example and a test<br/>migrate/compile.rs:101, examples/shacl_export.rs:76, tests/jsonld_matches_turtle.rs:80"]
    n1["EXTERNAL: parse_page documents VisionClaw CorpusSource::LocalDirectory as its caller input/page.rs:552; no caller exists in the ontomark workspace, so this repo cannot verify it."]
    R1 --> R2 --> R3 --> R4 --> R5 --> R6 --> R7 --> R8
    R8 --> D1 --> D2 --> D3 --> D4
```
- Ranking is by blast radius on the publication boundary, not by code size; R1 and R2 can leak, R3 to R5 can mis-redact or mis-hold, R6 to R8 cost only duplication.
- Grep scope was the whole ontomark workspace (`crates/*`); the wasm, ingest, parse and model crates call none of these functions.
