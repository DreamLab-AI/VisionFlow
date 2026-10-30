---
id: OM-03
title: vault build — run() gates, staging, promotion and generation identity
area: ontomark
governing:
  - docs/engineering/sovereign-corpus-contracts.md
adrs: [ADR-OM-001, ADR-OM-005, ADR-OM-007, ADR-OM-008, ADR-OM-009, ADR-2124, ADR-2125, ADR-2128]
sources:
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/build/publish.rs
  - ../ontomark/crates/ontomark-compile/src/build/generation.rs
  - ../ontomark/crates/ontomark-compile/src/build/attribution.rs
  - ../ontomark/crates/ontomark-compile/src/build/turtle.rs
  - ../ontomark/crates/ontomark-compile/src/input/page.rs
  - ../ontomark/crates/ontomark-compile/src/ir/projection.rs
  - ../ontomark/crates/ontomark-compile/src/reason.rs
  - ../ontomark/crates/ontomark-cli/src/main.rs
  - ../ontomark/crates/ontomark-cli/src/migrate_cmd.rs
  - ../ontomark/crates/ontomark-cli/src/embed.rs
  - ../ontomark/crates/ontomark-compile/src/build/records.rs
  - ../ontomark/crates/ontomark-compile/src/certificate.rs
  - ../ontomark/crates/ontomark-compile/src/profile.rs
  - ../ontomark/docs/adr/ADR-OM-009-every-published-closure-carries-a-derivation-certificate.md
  - docs/engineering/sovereign-corpus-contracts.md
verified_commit: {ontomark: 560e50c1a35bb9486b9f43ebf074b816c1e01349, VisionFlow: 4d361b58ce2b3ee17e1570361eaf78270b630012}
---

## OM-03.1 build::run_with — Options to promoted bundle, in the order the code runs

```mermaid
sequenceDiagram
    autonumber
    participant CLI as vault build CLI<br/>ontomark-cli/src/main.rs:825
    participant RUN as build::run_with<br/>build/mod.rs:545
    participant IN as walk and load<br/>input/page.rs:481
    participant PUB as publish staging<br/>build/publish.rs:200
    participant PRJ as validate + project<br/>ir/projection.rs:597
    participant RSN as closure + Whelk<br/>reason.rs:554
    participant STG as Staged<br/>build/mod.rs:340
    participant GEN as generation<br/>build/generation.rs:260

    CLI->>RUN: Options incl. scope + Profile + Vocabulary + WhelkEl classifier + optional Embedder (main.rs:801-831)
    RUN->>RUN: scope machine runs ensure_machine_scope first and refuses before any page is read, see OM-06.1 (mod.rs:554-557)
    RUN->>IN: walk_pages ONCE, then inspect_walk census refuses a non-boolean public (mod.rs:569-570)
    RUN->>IN: Vault::load_with_walk(vault_root) on that same walk, then alias_map then Corpus::build raw (mod.rs:571-577)
    RUN->>IN: working vault: loaded with its own walk and census under machine scope, else sibling_vault only if with_working_publish (mod.rs:596-608)
    RUN->>IN: sibling_vault(other kind) only if redact and no working (mod.rs:618-624)
    RUN->>PUB: stage_redacted(publish_vaults, private_links) = CREDENTIAL GATE (mod.rs:633)
    PUB-->>RUN: Err(SecretsFound) exits the CLI with 2, before validate (publish.rs:264-266)
    RUN->>PRJ: validate(profile, vault, raw, vocab) then ensure no errors, working too under machine (mod.rs:635-656)
    RUN->>PRJ: excluded_identities_of(vault) from the loaded pages, no further read of the vault root (mod.rs:658)
    RUN->>PRJ: project_machine, project_vault_with_links, or project_with_redactor when redaction is off, a redactor that cannot compile is PublicationBlocked Redactor and refuses (mod.rs:661-676)
    RUN->>RSN: closure::compute, backlink_index, Roster::discover, build_graph_attributed (mod.rs:678-685)
    RUN->>RSN: reason::reason(graph, classifier), ADR-2124 cap error returned unwrapped (mod.rs:689)
    RUN->>RSN: only with Options.certificate_out, certificate::certify over that same graph, then parity against the classifier's named closure, else CertificateParity refuses and nothing is written, ADR-OM-009 (mod.rs:692-731)
    RUN-->>CLI: not is_consistent returns the typed build::Unsatisfiable, ADR-2125 zero-unsatisfiable gate (mod.rs:743-749)
    RUN->>STG: add publish files if publish_out is None, else record_external, then every emitter (mod.rs:769-947)
    RUN->>STG: with an Embedder only, concept_records + embed_all + to_jsonl (mod.rs:949-954)
    RUN->>GEN: git_sha + git_dirty unless Options.source is given (mod.rs:925-942)
    RUN->>STG: okf bundle staged under okf/ using generation_id (mod.rs:943-947)
    RUN->>STG: ensure_single_claims, write-once check number one (mod.rs:957)
    RUN->>GEN: content_digest_of_hashes over page source_sha256, ontology_digest, generation(...) over staged.artifacts (mod.rs:963-1001)
    RUN->>STG: data marker first, root marker lists it and never itself, check number two (mod.rs:1008-1017)
    RUN->>STG: promote(out) via swap_in, then publish::write_to(publish_out) (mod.rs:1018-1021)
    RUN-->>CLI: Report (mod.rs:1023-1065)
    Note over CLI: SecretsFound, OutputCollision, ReasoningError, ScopeConflict, Unsatisfiable and CertificateParity exit 2, anything else is an anyhow error (main.rs:837-871)
```
- INVARIANT: nothing touches `out` until `promote` at `build/mod.rs:1018`; every earlier refusal (`mod.rs:633`, `:577`, `:642`) leaves the previous bundle intact.
- INVARIANT: the credential gate runs after the census, the load and up to two extra vault loads, and before validation (`build/mod.rs:569-633`); its comment now says the census runs earlier still (`mod.rs:588-589`).
- INVARIANT: an unsatisfiable corpus is the typed `build::Unsatisfiable` (`mod.rs:235`) and exits 2 (`main.rs:861-864`), the same refusal class as the ADR-2124 cap error (`main.rs:849-852`); it was an untyped error at exit 1 before the OM-00 reconcile.

## OM-03.2 What `--vault` means, and every place a vault is loaded

```mermaid
flowchart TD
    ARG["CLI --vault knowledge or working or all<br/>vault_kinds(args.vault)<br/>ontomark-cli/src/migrate_cmd.rs:72<br/>ontomark-cli/src/main.rs:791"]
    KIND["kind = Working only when the list is exactly Working and scope is not machine, else Knowledge<br/>ontomark-cli/src/main.rs:795-800"]
    ROOT["Options.vault_root = vault_root(root, profile, kind): the layout's root for the kind, ('.' for a flat vault)<br/>ONE root feeds every machine artefact<br/>ontomark-cli/src/main.rs:805<br/>ontomark-cli/src/main.rs:544-546"]
    WIDE["Options.with_working_publish = list contains Working AND kind is Knowledge<br/>ontomark-cli/src/main.rs:802-803"]
    MSC["scope machine: ensure_machine_scope refuses vault working or all, see OM-06.1<br/>build/mod.rs:476<br/>build/mod.rs:483-488"]
    LOAD["walk_pages once with the layout's excluded dirs, then Vault::load_with_walk: kind from profile.layout.kind_of(vault_root)<br/>build/mod.rs:569-575<br/>ontomark-compile/src/profile.rs:583"]
    MACH["Machine artefacts: Corpus raw, validate, project, closure, Turtle, reason,<br/>indexes, page API, graph tiers, okf, markdown mirror<br/>build/mod.rs:577,635,661,678,780"]
    SW["sibling_vault(Working): the layout's path for the kind under repo_root, loaded with the layout's excluded dirs only if that directory exists<br/>build/publish.rs:365-376<br/>build/mod.rs:604-605"]
    PUBV["publish_vaults = vault + working<br/>build/mod.rs:609-610"]
    STAGE["stage_redacted over publish_vaults: the ONLY consumer of working pages as content<br/>build/mod.rs:633"]
    OTHER["other_vault = sibling_vault(opposite kind)<br/>only if redact_links AND working is None<br/>build/mod.rs:618-626"]
    PL["PrivateLinks::new over vault + working + other_vault<br/>link resolution only, nothing is published from other_vault<br/>build/mod.rs:627-632"]
    GITD["git_dirty read_paths: vault_root, repo/ontology, plus the working root whenever working or other_vault was loaded<br/>build/mod.rs:933-940"]
    N1["INVARIANT: with_working_publish only decides whether working is loaded (build/mod.rs:604),<br/>the dirty paths follow from that load (build/mod.rs:937), never the projections<br/>build/mod.rs:76"]
    N2["INVARIANT: one loader, sibling_vault, serves the publish load and the redaction load,<br/>and other_vault is read only when working is None, so working is loaded at most once<br/>build/publish.rs:365<br/>build/mod.rs:618"]
    N3["DIVERGENCE: sibling_vault never compares its kind with vault.kind, so a library caller with<br/>vault_root = working plus with_working_publish stages the same vault twice<br/>build/publish.rs:365-376"]
    ARG --> KIND --> ROOT --> LOAD --> MACH
    KIND --> WIDE
    WIDE --> SW --> PUBV --> STAGE
    LOAD --> PUBV
    LOAD --> OTHER --> PL
    SW --> PL
    LOAD --> PL
    PL --> STAGE
    WIDE --> GITD
    KIND -.-> MSC
    MSC -.-> SW
    WIDE -.-> N1
    OTHER -.-> N2
    SW -.-> N3
```
- INVARIANT: in the public scope `working/` never enters `corpus`, so OWL membership stays `knowledge/`-only (`ontomark-cli/src/main.rs:786-790`, `build/mod.rs:76-77`); `--vault working` alone makes the WHOLE projection come from `working/` (`main.rs:796-800`).
- INVARIANT: `Options.repo_root` serves git, the vocabulary directory for dirtiness and, since the layout landed, the placement of every sibling vault (`build/mod.rs:928,933,940`, `build/mod.rs:468-472`, `build/publish.rs:370`), so the old `vault_root.parent()` derivations of the repository are gone from vault loading.
- **Debt:** the attribution roster is still read from `vault_root.parent()` (`build/mod.rs:680-684`), and a flat layout puts the vault root at `repo_root` itself (`ontomark-compile/src/profile.rs:514-518`, `ontomark-compile/src/profile.rs:549-550`), so for a flat vault the roster is looked up one directory above the repository while the dirtiness check reads `repo_root/ontology` (`build/mod.rs:933`).

## OM-03.3 Reads of the vault root — one walk, two file reads per page

```mermaid
flowchart TD
    DIR[("vault_root — the vault root itself, no pages/ subfolder (ADR-OM-007)<br/>build/mod.rs:569")]
    WK["0 walk_pages: ONE WalkDir listing, component-wise sort, hidden dirs, excluded top-level folders (templates, bases, assets, journals) and non-regular md reported as skips<br/>input/page.rs:581,609,616-623<br/>build/mod.rs:569"]
    W1["1 inspect_walk: reads each listed file, parses frontmatter, census counts incl. skips<br/>ir/projection.rs:96,118,104<br/>build/mod.rs:570"]
    W2["2 Vault::load_with_walk: reads each listed file again, Page::parse, records source_sha256<br/>input/page.rs:481,493,223<br/>build/mod.rs:571"]
    W3["3 excluded_identities_of: NO walk and NO read, iterates vault.pages for held pages<br/>ir/projection.rs:228<br/>build/mod.rs:658"]
    W4["4 content_digest_of_hashes: pairs of path and the parse-time sha256, no re-read<br/>build/generation.rs:234<br/>build/mod.rs:963-983"]
    J["journals/ is an excluded folder of the root walk and gets its own walk inside load_with_walk, loaded and hashed through all_pages, never published<br/>input/page.rs:499-509,506-508<br/>build/mod.rs:968"]
    C1["Corpus::build raw (mod.rs:577)"]
    C2["project(...) inside project_vault_with_links, result discarded, only the gate matters<br/>ir/projection.rs:604"]
    C3["Corpus::build(cleaned) after vault.clone() and per-page redaction<br/>ir/projection.rs:606,614"]
    R1["redaction of the same body text, three implementations: projection<br/>ir/projection.rs:646, publish render build/publish.rs:174, markdown mirror build/mod.rs:887"]
    N1["INVARIANT: census and load share one PageWalk, so the dot-directory rule is written once<br/>and a skip is counted as excluded_directory or excluded_not_regular_file, never silently lost<br/>input/page.rs:609<br/>ir/projection.rs:107"]
    N2["DIVERGENCE: each page is still read twice, census then load, so a file edited between the two<br/>is classified on one text and parsed from another, a time-of-check gap the digest no longer widens<br/>ir/projection.rs:118<br/>input/page.rs:493"]
    DIR --> WK
    WK --> W1
    WK --> W2
    W2 --> W3
    W2 --> C1
    W2 --> J
    W2 --> W4
    C1 --> C2
    W2 --> C3
    C3 --> R1
    W1 -.-> N1
    W2 -.-> N2
```
- INVARIANT: `excluded_identities(root)` with its own walk survives as public API (`ir/projection.rs:202`) but `run` calls only `excluded_identities_of` (`build/mod.rs:658`), which filters by `is_unpublished` alone (`projection.rs:114`).
- DIVERGENCE: the hold-back rule is still written in two shapes: `is_unpublished` covers `_misc` at top level only (`input/page.rs:47,641`), while `held_back` (`build/publish.rs:138`) applies `UNPUBLISHED_DIRS` at any depth and adds the site-only `misc` through `HELD_BACK_DIRS` (`publish.rs:134,144-146`); `clean_vault` and `hold_back` use the wider one (`projection.rs:616,657`). ontomark `95208e8` (2026-10-18, estate register ER-18) made the two lists disjoint, so they no longer overlap; the census and `excluded_identities_of` still use the top-level-only `is_unpublished` (`projection.rs:114,231`), which is the divergence that remains.

## OM-03.4 Credential gate and publish staging

```mermaid
flowchart TD
    IN["stage_redacted(vaults, links)<br/>build/publish.rs:200"]
    LOOP["for each vault, for each page in vault.pages<br/>build/publish.rs:210,219"]
    HB{"held_back(rel)?<br/>_misc top level, misc at any depth<br/>build/publish.rs:220"}
    CONS["counts.considered += 1<br/>build/publish.rs:223"]
    PUBQ{"page.is_public()?<br/>build/publish.rs:224"}
    SCAN["secrets::scan on the body AND on the frontmatter as rendered, before link redaction<br/>build/publish.rs:228<br/>build/publish.rs:240"]
    REND["render: frontmatter re-rendered, links redacted, body verbatim<br/>build/publish.rs:169-175"]
    PUSH["StagedPage path = publish/tree/id.md, tree = pages or working<br/>build/publish.rs:253-260<br/>build/publish.rs:157-161"]
    FIND{"any finding?<br/>build/publish.rs:264"}
    ERR["Err(SecretsFound): nothing staged, kind + file + line, never the value<br/>build/publish.rs:265<br/>build/publish.rs:36"]
    IDX["publish/index.md added: OKF section 8 home page, carries public: true<br/>build/publish.rs:267-271<br/>build/publish.rs:290"]
    RET["Staging: files, per_vault counts, redaction tally<br/>build/publish.rs:111-118"]
    BUN["publish_out is None: files go into Staged under their own path as owner<br/>build/mod.rs:769-772"]
    SEP["publish_out is Some: recorded as external artefacts, written by write_to after promote, gate still applied<br/>build/mod.rs:773-778<br/>build/mod.rs:1019-1021<br/>build/mod.rs:71-73"]
    N1["INVARIANT: the frontmatter is scanned as well as the body, so a credential in a frontmatter value<br/>is refused, its line being the line of the rendered block<br/>build/publish.rs:236-247"]
    N2["INVARIANT: the markdown mirror and page API are built from projected records, not from<br/>the scanned bytes, so only publish/ is covered by the gate by construction<br/>build/mod.rs:869"]
    IN --> LOOP --> HB
    HB -- yes --> LOOP
    HB -- no --> CONS --> PUBQ
    PUBQ -- no --> LOOP
    PUBQ -- yes --> SCAN --> REND --> PUSH --> LOOP
    LOOP --> FIND
    FIND -- yes --> ERR
    FIND -- no --> IDX --> RET
    RET --> BUN
    RET --> SEP
    SCAN -.-> N1
    BUN -.-> N2
```
- INVARIANT: the refusal is a typed error so the CLI exits 2 (`build/publish.rs:66`, `ontomark-cli/src/main.rs:837-840`); `stage` without links is only a wrapper (`publish.rs:191-193`) and `run_with` never calls it.
- DOC-DRIFT: `publish.rs` says `<out>/publish/` is the markdown that is published (`publish.rs:1`) but with `publish_out` it is not in `<out>` at all (`mod.rs:773-778`, `:1019-1021`).
- Under `--scope machine` the gate still scans `working/`'s public pages though nothing is staged into the bundle (`build/mod.rs:591-603`, `build/mod.rs:766-768`), see OM-06.

## OM-03.5 Staged write-once and the promote

```mermaid
flowchart TD
    ADD["Staged.add(path, bytes): owner is the build<br/>build/mod.rs:366-368<br/>build/mod.rs:333"]
    ADDFOR["Staged.add_for(path, bytes, owner): owners map entry<br/>build/mod.rs:372-388"]
    OCC{"path already owned?<br/>build/mod.rs:379"}
    CLASH["recorded in clashes, NOT applied: first writer wins the bytes<br/>build/mod.rs:381"]
    SLOT["owner stored, file pushed to files<br/>build/mod.rs:384-385"]
    CHK1["ensure_single_claims 1: after every emitter, before the stamp<br/>build/mod.rs:957"]
    ART["artifacts(): sha256 + bytes of every file staged so far, plus the external ones recorded for publish_out<br/>build/mod.rs:361-363, build/mod.rs:401-407<br/>build/generation.rs:40-46"]
    MARK["markers staged AFTER artifacts: data marker, then the root marker that lists it<br/>build/mod.rs:1015-1016"]
    CHK2["ensure_single_claims 2: the only new claims are the two markers<br/>build/mod.rs:1017"]
    ERRC["Err(OutputCollision): CLI exit 2, every clash named<br/>build/mod.rs:395<br/>ontomark-cli/src/main.rs:843-845"]
    PRO["promote(out): write all files into parent/.publication-PID<br/>build/mod.rs:415-430"]
    RET["swap_in: clear a stale retired directory, rename out to .publication-retired-PID if it exists<br/>build/mod.rs:306-312"]
    SWAP["rename stage to out; on error restore retired and delete stage<br/>build/mod.rs:313-327"]
    DEL["retired directory deleted, so no obsolete export survives<br/>build/mod.rs:315-317"]
    WT["write_to(publish_out): files written beside dir in .publish-staging-PID, then the same swap_in<br/>build/publish.rs:341-356<br/>build/mod.rs:1019-1021"]
    N1["INVARIANT: swap_in is two renames, not one: between them out does not exist, and any<br/>pre-existing content of out is replaced wholesale, not merged, the old tree being restored on failure<br/>build/mod.rs:310-313<br/>build/mod.rs:320-327"]
    N2["DIVERGENCE: write_to runs AFTER promote, so a failure there leaves the NEW bundle beside the PREVIOUS<br/>publication tree that swap_in kept, the two no longer one generation<br/>build/publish.rs:356<br/>build/mod.rs:1018-1020"]
    N3["INVARIANT: no marker lists itself, the root marker lists data/.generation.json,<br/>and publish_out files ARE stamped as external artefacts<br/>build/mod.rs:1009-1013<br/>build/mod.rs:361-363"]
    ADD --> ADDFOR --> OCC
    OCC -- yes --> CLASH
    OCC -- no --> SLOT
    SLOT --> CHK1
    CLASH --> CHK1
    CHK1 -- clashes --> ERRC
    CHK1 -- ok --> ART --> MARK --> CHK2
    CHK2 -- clashes --> ERRC
    CHK2 -- ok --> PRO --> RET --> SWAP --> DEL
    DEL --> WT
    RET -.-> N1
    WT -.-> N2
    MARK -.-> N3
```
- INVARIANT: `Staged` is the only writer path for the bundle and holds bytes in memory (`build/mod.rs:340-347`); no emitter can overwrite another silently, a clash names both claimants (`mod.rs:198-209`).
- DOC-DRIFT: the module doc calls step 6 an atomic promotion (`build/mod.rs:12`); the two-rename swap in `mod.rs:310-313` is rolled back on failure but is not a single atomic operation, and the `promote` doc no longer claims it (`mod.rs:409-410`).

## OM-03.6 Generation identity — what the stamp is made of

```mermaid
flowchart TD
    SRC{"Options.source given?<br/>build/mod.rs:925"}
    GIVEN["commit and dirty taken from the caller<br/>build/mod.rs:926<br/>ontomark-cli/src/main.rs:810-813"]
    SHA["git_sha(repo_root): git rev-parse HEAD, else unknown<br/>build/generation.rs:197-208<br/>build/mod.rs:928"]
    PATHS["read_paths: vault_root + repo_root/ontology + the working root whenever working or other_vault was loaded<br/>build/mod.rs:933-939"]
    DIRTY["git_dirty: git status --porcelain over those paths, canonicalised, false outside git<br/>build/generation.rs:175-191<br/>build/mod.rs:940"]
    ID["generation_id: prefix@sha, plus +dirty when dirty; prefix is the profile's GenerationNaming for the repo root, visionGraph for the estate, the directory name otherwise (ADR-OM-008)<br/>build/generation.rs:150-169<br/>build/mod.rs:943-944<br/>ontomark-compile/src/profile.rs:682"]
    CD["content_digest_of_hashes: SHA-256 over path, NUL, page source_sha256 hex, newline, sorted by path<br/>every page and journal of every loaded vault, plus the vocabulary<br/>build/generation.rs:234-243<br/>build/mod.rs:963-983"]
    OD["ontology_digest: sha256-12 over the asserted graph minus the header subject<br/>build/turtle.rs:611-640<br/>build/mod.rs:987"]
    GEN["Generation: id, commit, dirty, content_digest, ontology_digest, generated_at,<br/>class_count, page_count, vocabulary_version, stale_after, scope, artifacts<br/>build/generation.rs:90-124<br/>build/mod.rs:989-1002"]
    M1["data/.generation.json = gen.scoped(data): only data/ artefacts, names relative to data/<br/>build/generation.rs:131-147<br/>build/mod.rs:1008"]
    M2[".generation.json = gen plus the data marker entry: every artefact relative to out<br/>build/mod.rs:1009-1016"]
    GA["generated_at: Options value or now_iso, its first 10 chars also become today for graph tiers<br/>build/mod.rs:750-753,836"]
    N1["DIVERGENCE: the attribution roster shapes the Turtle and the page API but is not an input<br/>to content_digest, which hashes pages, journals and the vocabulary only<br/>build/mod.rs:680-685<br/>build/mod.rs:963-983"]
    N2["INVARIANT: C3 now stamps what the code writes: dirty, ontology_digest, scope? on a machine bundle, the data marker and the root marker that lists it<br/>sovereign-corpus-contracts.md:140-156<br/>build/mod.rs:1008-1016"]
    N3["DIVERGENCE: the vocabulary directory is hard-coded to repo_root/ontology for dirtiness,<br/>and the roster is read from vault_root.parent()/ontology, two roots for one directory<br/>build/mod.rs:933<br/>build/attribution.rs:32"]
    SRC -- yes --> GIVEN --> ID
    SRC -- no --> SHA --> ID
    SRC -- no --> PATHS --> DIRTY --> ID
    ID --> GEN
    CD --> GEN
    OD --> GEN
    GA --> GEN
    GEN --> M1
    GEN --> M2
    CD -.-> N1
    GEN -.-> N2
    PATHS -.-> N3
```
- INVARIANT: `commit` stays the bare sha even when dirty (`build/generation.rs:94-96`); the sha says which commit, the digest what was read, and the ontology digest what was emitted, so pages alone do not fix the Turtle (`build/mod.rs:984-986`).
- INVARIANT: `Roster::discover` is the only reader of `attribution.yaml`, an absent file is an empty roster (`build/attribution.rs:29-37`), and it is called with `vault_root.parent()` (`build/mod.rs:680-683`), not `repo_root`.

## OM-03.7 Report, and C3 artefact map — what run() writes versus what C3 lists

```mermaid
flowchart LR
    subgraph C3["C3 artefacts, sovereign-corpus-contracts.md:110-157"]
        A1["data/ontology.ttl, ontology-inferred.ttl<br/>sovereign-corpus-contracts.md:111-112"]
        A2["data/scaffold-index.json, prose-index.json<br/>sovereign-corpus-contracts.md:113-114"]
        A3["ontology-corpus.records.jsonl, --with-rvdb only<br/>sovereign-corpus-contracts.md:115-122"]
        A4["api/search-index, pages, census, validation-report<br/>sovereign-corpus-contracts.md:123-124"]
        A5["ns/v2.jsonld, api/schema/context.jsonld, context/v1<br/>sovereign-corpus-contracts.md:125-126"]
        A6["data/ontology.json and data/graph tiers<br/>sovereign-corpus-contracts.md:127-128"]
        A7["api/markdown only with the flag<br/>sovereign-corpus-contracts.md:129"]
        A8["publish/ pages, working, index.md<br/>sovereign-corpus-contracts.md:130-138"]
        A9["okf/ and .generation.json<br/>sovereign-corpus-contracts.md:139-156"]
    end
    subgraph RUN["written by run_with()"]
        W1["staged.add ttl and inferred<br/>build/mod.rs:780-784"]
        W2["scaffold, prose<br/>build/mod.rs:787,789"]
        W3["records.jsonl only when an Embedder is handed in: concept_records, embed_all, to_jsonl<br/>build/mod.rs:949-954<br/>build/records.rs:173"]
        W4["search-index, api/pages/slug, _domain-index, census, validation-report<br/>build/mod.rs:793,805-816"]
        W5["context/v1, api/schema/context, ns/v2, one byte-identical document<br/>build/mod.rs:824-826"]
        W6["data/ontology.json, data/graph/name for each tier file<br/>build/mod.rs:830,838"]
        W7["api/markdown/name.md gated by with_markdown_mirror<br/>build/mod.rs:842,917"]
        W8["publish/ via Staged, or write_to publish_out, and none at all under scope machine<br/>build/mod.rs:766-779,1019-1021"]
        W9["okf/path, then both markers<br/>build/mod.rs:946,1015-1016"]
    end
    RPT["Report: generation, counts, reasoner mode, published per vault, written, out, redaction, machine<br/>build/mod.rs:140-186"]
    CLI["CLI prints id, out, written; fails when written is 0<br/>ontomark-cli/src/main.rs:875-895"]
    N1["INVARIANT: the records path exists in run_with: the compile crate opens no socket, the host hands<br/>in an Embedder and the CLI supplies the Xinference transport<br/>build/records.rs:61<br/>ontomark-cli/src/embed.rs:30<br/>ontomark-cli/src/main.rs:821-824"]
    N2["INVARIANT: api/pages/_domain-index.json is written and C3 now lists it<br/>build/mod.rs:811<br/>sovereign-corpus-contracts.md:123"]
    N3["INVARIANT: Report.unsatisfiable is always empty on success, because a non-empty list already<br/>refused the build with build::Unsatisfiable, so the field cannot carry its documented meaning<br/>build/mod.rs:157-161,1035<br/>build/mod.rs:743-749"]
    N4["INVARIANT: reason already returns ReasoningError UnsupportedProfile on classifier refusals,<br/>so is_consistent at the gate can fail only on unsatisfiable classes, and Unsatisfiable names root causes<br/>reason.rs:590-598<br/>build/mod.rs:243-268"]
    A1 --- W1
    A2 --- W2
    A3 --- W3
    A4 --- W4
    A5 --- W5
    A6 --- W6
    A7 --- W7
    A8 --- W8
    A9 --- W9
    W9 --> RPT --> CLI
    A3 -.-> N1
    W4 -.-> N2
    RPT -.-> N3
    RPT -.-> N4
```
- INVARIANT: `Report.written` counts both destinations (`build/mod.rs:1042-1049`) and the CLI refuses a zero (`ontomark-cli/src/main.rs:889-895`); `mod.rs:1044` sums `staged.files` plus the separate publish staging.
- INVARIANT: C3 now says `content_digest` is over sorted `(path, sha256(bytes))` pairs of every byte the build read, pages and journals of every loaded vault plus the vocabulary, not comparable with vault's one-vault digest (`sovereign-corpus-contracts.md:159-163`), which is what `run_with` hashes (`build/mod.rs:963-983`, `build/generation.rs:234`); the OM-03 divergence recorded against the earlier C3 text is closed at this stamp.
- **Drift:** C3 still illustrates page identity as `knowledge/pages/Knowledge Graph` (`sovereign-corpus-contracts.md:106`) while the loader derives the id from the vault-root-relative path with no `pages/` folder (`ontomark-compile/src/input/page.rs:489-492`).

## OM-03.8 Derivation certificate — the optional warrant for the published closure (ADR-OM-009)

```mermaid
flowchart TD
    FLAG["ontomark build --certificate-out DIR → Options.certificate_out<br/>ontomark-cli/src/main.rs:401, ontomark-cli/src/main.rs:817<br/>build/mod.rs:89-95"]
    GRAPH["the asserted graph that is reasoned and published<br/>build/mod.rs:685"]
    REASON["reason::reason graph, classifier → Reasoning.inferred<br/>build/mod.rs:689"]
    SPELL["certificate::spell — every triple as N-Triples terms<br/>ontomark-compile/src/certificate.rs:27"]
    CERT["certificate::certify — rdfs-certificate over the spelled triples, oo-cert/1<br/>ontomark-compile/src/certificate.rs:51"]
    PAR{"certificate::parity — certified named subClassOf closure equals the classifier's?<br/>ontomark-compile/src/certificate.rs:63"}
    REFUSE["Err CertificateParity, CLI exit 2, no bundle written<br/>build/mod.rs:98-118<br/>ontomark-cli/src/main.rs:867-870"]
    WRITE["cert.write DIR, generation label from the same prefix and commit as the stamp<br/>build/mod.rs:704-717"]
    REP["Report.certificate: files, digest, certify_ms, counts, the oo-cert command that re-checks it<br/>build/mod.rs:183-185, build/mod.rs:1063"]
    NONE["certificate_out is None → nothing written, nothing checked<br/>build/mod.rs:730"]
    FLAG --> GRAPH --> REASON --> SPELL --> CERT --> PAR
    PAR -- no --> REFUSE
    PAR -- yes --> WRITE --> REP
    FLAG -. unset .-> NONE
    n1["INVARIANT: a generation is never published with a closure its certificate does not cover — parity is enforced inside the build,<br/>ontomark-compile/src/certificate.rs:1-12<br/>build/mod.rs:696-701"]
    n2["INVARIANT: the certificate is an RDFS warrant for the named closure only, and parity is structural because restrictions are emitted on the superclass side of subClassOf; a future emitter that puts one on the subclass side would trip the refusal rather than weaken it,<br/>ontomark-compile/src/certificate.rs:7-12"]
    PAR --- n1
    CERT --- n2
```
**What it shows.** The one new gate between reasoning and the unsatisfiable check: with `--certificate-out`, the build spells the graph it is about to publish, certifies it under the RDFS rule table, compares the certified named subsumptions with the classifier's, and refuses on any difference before writing anything.

**Why it is this way.** ADR-OM-009 (Accepted, owner request of 20 October 2026, active from v0.6.0) makes ontomark, the sole producer of published bundles (ADR-OM-005), also the writer of the warrant a Lean 4 checker can re-verify, instead of leaving that to the `vault` path (`docs/adr/ADR-OM-009-every-published-closure-carries-a-derivation-certificate.md:3-5`).

**Open:** is a bundle published without `--certificate-out` meant to carry no warrant at all? `certificate_out` defaults to absent and the build writes and checks nothing then (`ontomark-compile/src/build/mod.rs:730`, `ontomark-cli/src/main.rs:401`).
