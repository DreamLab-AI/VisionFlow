---
id: OM-03
title: vault build — run() gates, staging, promotion and generation identity
area: ontomark
governing:
  - docs/engineering/sovereign-corpus-contracts.md
adrs: [ADR-OM-001, ADR-OM-005, ADR-2124, ADR-2125, ADR-2128]
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
  - docs/engineering/sovereign-corpus-contracts.md
verified_commit: {ontomark: ee9e04a25715e70d3de211c17a012e878d14f339, VisionFlow: 6443781fe6ed5f65d980510c86d92f8b373f0d3a}
---

## OM-03.1 build::run_with — Options to promoted bundle, in the order the code runs

```mermaid
sequenceDiagram
    autonumber
    participant CLI as vault build CLI<br/>ontomark-cli/src/main.rs:512
    participant RUN as build::run_with<br/>build/mod.rs:494
    participant IN as walk and load<br/>input/page.rs:436
    participant PUB as publish staging<br/>build/publish.rs:194
    participant PRJ as validate + project<br/>ir/projection.rs:600
    participant RSN as closure + Whelk<br/>reason.rs:554
    participant STG as Staged<br/>build/mod.rs:290
    participant GEN as generation<br/>build/generation.rs:246

    CLI->>RUN: Options incl. scope + Profile + Vocabulary + WhelkEl classifier + optional Embedder (main.rs:489-518)
    RUN->>RUN: scope machine runs ensure_machine_scope first and refuses before any page is read, see OM-06.1 (mod.rs:503-506)
    RUN->>IN: walk_pages ONCE, then inspect_walk census refuses a non-boolean public (mod.rs:512-513)
    RUN->>IN: Vault::load_with_walk(vault_root) on that same walk, then alias_map then Corpus::build raw (mod.rs:514-520)
    RUN->>IN: working vault: loaded with its own walk and census under machine scope, else sibling_vault only if with_working_publish (mod.rs:539-549)
    RUN->>IN: sibling_vault(other kind) only if redact and no working (mod.rs:559-565)
    RUN->>PUB: stage_redacted(publish_vaults, private_links) = CREDENTIAL GATE (mod.rs:574)
    PUB-->>RUN: Err(SecretsFound) exits the CLI with 2, before validate (publish.rs:258-260)
    RUN->>PRJ: validate(profile, vault, raw, vocab) then ensure no errors, working too under machine (mod.rs:576-597)
    RUN->>PRJ: excluded_identities_of(vault) from the loaded pages, no further read of the pages dir (mod.rs:599)
    RUN->>PRJ: project_machine, project_vault_with_links, or project_with_redactor when redaction is off, a redactor that cannot compile is PublicationBlocked Redactor and refuses (mod.rs:602-617)
    RUN->>RSN: closure::compute, backlink_index, Roster::discover, build_graph_attributed (mod.rs:619-626)
    RUN->>RSN: reason::reason(graph, classifier), ADR-2124 cap error returned unwrapped (mod.rs:630)
    RUN-->>CLI: not is_consistent returns the typed build::Unsatisfiable, ADR-2125 zero-unsatisfiable gate (mod.rs:642-648)
    RUN->>STG: add publish files if publish_out is None, else record_external, then every emitter (mod.rs:668-845)
    RUN->>STG: with an Embedder only, concept_records + embed_all + to_jsonl (mod.rs:847-852)
    RUN->>GEN: git_sha + git_dirty unless Options.source is given (mod.rs:824-841)
    RUN->>STG: okf bundle staged under okf/ using generation_id (mod.rs:842-845)
    RUN->>STG: ensure_single_claims, write-once check number one (mod.rs:855)
    RUN->>GEN: content_digest_of_hashes over page source_sha256, ontology_digest, generation(...) over staged.artifacts (mod.rs:861-896)
    RUN->>STG: data marker first, root marker lists it and never itself, check number two (mod.rs:903-912)
    RUN->>STG: promote(out) via swap_in, then publish::write_to(publish_out) (mod.rs:913-916)
    RUN-->>CLI: Report (mod.rs:918-958)
    Note over CLI: SecretsFound, OutputCollision, ReasoningError, ScopeConflict and Unsatisfiable exit 2, anything else is an anyhow error (main.rs:524-552)
```
- INVARIANT: nothing touches `out` until `promote` at `build/mod.rs:913`; every earlier refusal (`mod.rs:574`, `:577`, `:642`) leaves the previous bundle intact.
- INVARIANT: the credential gate runs after the census, the load and up to two extra vault loads, and before validation (`build/mod.rs:512-576`); its comment now says the census runs earlier still (`mod.rs:531-532`).
- INVARIANT: an unsatisfiable corpus is the typed `build::Unsatisfiable` (`mod.rs:185`) and exits 2 (`main.rs:548-551`), the same refusal class as the ADR-2124 cap error (`main.rs:536-539`); it was an untyped error at exit 1 before the OM-00 reconcile.

## OM-03.2 What `--vault` means, and every place a vault is loaded

```mermaid
flowchart TD
    ARG["CLI --vault knowledge or working or all<br/>vault_kinds(args.vault)<br/>ontomark-cli/src/migrate_cmd.rs:72<br/>ontomark-cli/src/main.rs:479"]
    KIND["kind = Working only when the list is exactly Working and scope is not machine, else Knowledge<br/>ontomark-cli/src/main.rs:483-488"]
    ROOT["Options.vault_root = root/kind.dir_name()<br/>ONE root feeds every machine artefact<br/>ontomark-cli/src/main.rs:493"]
    WIDE["Options.with_working_publish = list contains Working AND kind is Knowledge<br/>ontomark-cli/src/main.rs:490-491"]
    MSC["scope machine: ensure_machine_scope refuses vault working or all, see OM-06.1<br/>build/mod.rs:426<br/>build/mod.rs:433-438"]
    LOAD["walk_pages once, then Vault::load_with_walk: kind inferred from the directory name<br/>build/mod.rs:512-518<br/>input/page.rs:617"]
    MACH["Machine artefacts: Corpus raw, validate, project, closure, Turtle, reason,<br/>indexes, page API, graph tiers, okf, markdown mirror<br/>build/mod.rs:520,576,602,619,679"]
    SW["sibling_vault(Working): repo.join of the kind dir, loaded only if its pages dir exists<br/>build/publish.rs:356-367<br/>build/mod.rs:545-546"]
    PUBV["publish_vaults = vault + working<br/>build/mod.rs:550-551"]
    STAGE["stage_redacted over publish_vaults: the ONLY consumer of working pages as content<br/>build/mod.rs:574"]
    OTHER["other_vault = sibling_vault(opposite kind)<br/>only if redact_links AND working is None<br/>build/mod.rs:559-567"]
    PL["PrivateLinks::new over vault + working + other_vault<br/>link resolution only, nothing is published from other_vault<br/>build/mod.rs:568-573"]
    GITD["git_dirty read_paths: vault_root, repo/ontology, plus the working root whenever working or other_vault was loaded<br/>build/mod.rs:832-839"]
    N1["INVARIANT: with_working_publish only decides whether working is loaded (build/mod.rs:545),<br/>the dirty paths follow from that load (build/mod.rs:836), never the projections<br/>build/mod.rs:76"]
    N2["INVARIANT: one loader, sibling_vault, serves the publish load and the redaction load,<br/>and other_vault is read only when working is None, so working is loaded at most once<br/>build/publish.rs:356<br/>build/mod.rs:559"]
    N3["DIVERGENCE: sibling_vault never compares its kind with vault.kind, so a library caller with<br/>vault_root = working plus with_working_publish stages the same vault twice<br/>build/publish.rs:363"]
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
- INVARIANT: in the public scope `working/` never enters `corpus`, so OWL membership stays `knowledge/`-only (`ontomark-cli/src/main.rs:474-478`, `build/mod.rs:76-77`); `--vault working` alone makes the WHOLE projection come from `working/` (`main.rs:484-488`).
- INVARIANT: `Options.repo_root` serves git and the vocabulary directory for dirtiness (`build/mod.rs:827,832,839`); every other "repo" is `vault_root.parent()` (`mod.rs:419,623,833`, `publish.rs:360`), two derivations of one directory.

## OM-03.3 Reads of the pages dir — one walk, two file reads per page

```mermaid
flowchart TD
    DIR[("vault_root/pages<br/>build/mod.rs:511")]
    WK["0 walk_pages: ONE WalkDir listing, component-wise sort, hidden dirs and non-regular md reported as skips<br/>input/page.rs:532,560<br/>build/mod.rs:512"]
    W1["1 inspect_walk: reads each listed file, parses frontmatter, census counts incl. skips<br/>ir/projection.rs:95,117,103<br/>build/mod.rs:513"]
    W2["2 Vault::load_with_walk: reads each listed file again, Page::parse, records source_sha256<br/>input/page.rs:436,452,186<br/>build/mod.rs:514"]
    W3["3 excluded_identities_of: NO walk and NO read, iterates vault.pages for held pages<br/>ir/projection.rs:227<br/>build/mod.rs:599"]
    W4["4 content_digest_of_hashes: pairs of path and the parse-time sha256, no re-read<br/>build/generation.rs:221<br/>build/mod.rs:861-879"]
    J["journals/ get their own walk inside load_with_walk, loaded and hashed through all_pages, never published<br/>input/page.rs:457-461,507<br/>build/mod.rs:866"]
    C1["Corpus::build raw (mod.rs:520)"]
    C2["project(...) inside project_vault_with_links, result discarded, only the gate matters<br/>ir/projection.rs:607"]
    C3["Corpus::build(cleaned) after vault.clone() and per-page redaction<br/>ir/projection.rs:609,617"]
    R1["redaction of the same body text, three implementations: projection<br/>ir/projection.rs:653, publish render build/publish.rs:168, markdown mirror build/mod.rs:786"]
    N1["INVARIANT: census and load share one PageWalk, so the dot-directory rule is written once<br/>and a skip is counted as excluded_directory or excluded_not_regular_file, never silently lost<br/>input/page.rs:560<br/>ir/projection.rs:106"]
    N2["DIVERGENCE: each page is still read twice, census then load, so a file edited between the two<br/>is classified on one text and parsed from another, a time-of-check gap the digest no longer widens<br/>ir/projection.rs:117<br/>input/page.rs:452"]
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
- INVARIANT: `excluded_identities(pages_dir)` with its own walk survives as public API (`ir/projection.rs:201`) but `run` calls only `excluded_identities_of` (`build/mod.rs:599`), which filters by `is_unpublished` alone (`projection.rs:234`).
- DIVERGENCE: the hold-back rule is still written twice: `is_unpublished` covers `_misc` at top level only (`input/page.rs:38,583`), `held_back` (`build/publish.rs:134`) adds `misc` at any depth through `HELD_BACK_DIRS` (`publish.rs:130`), and `clean_vault` and `hold_back` use the wider one (`projection.rs:623,668`).

## OM-03.4 Credential gate and publish staging

```mermaid
flowchart TD
    IN["stage_redacted(vaults, links)<br/>build/publish.rs:194"]
    LOOP["for each vault, for each page in vault.pages<br/>build/publish.rs:204,211"]
    HB{"held_back(rel)?<br/>_misc top level, misc at any depth<br/>build/publish.rs:216"}
    CONS["counts.considered += 1<br/>build/publish.rs:219"]
    PUBQ{"page.is_public()?<br/>build/publish.rs:220"}
    SCAN["secrets::scan on the body AND on the frontmatter as rendered, before link redaction<br/>build/publish.rs:224<br/>build/publish.rs:236"]
    REND["render: frontmatter re-rendered, links redacted, body verbatim<br/>build/publish.rs:163-169"]
    PUSH["StagedPage path = publish/tree/id.md, tree = pages or working<br/>build/publish.rs:249-254<br/>build/publish.rs:151-155"]
    FIND{"any finding?<br/>build/publish.rs:258"}
    ERR["Err(SecretsFound): nothing staged, kind + file + line, never the value<br/>build/publish.rs:259<br/>build/publish.rs:36"]
    IDX["publish/index.md added: OKF section 8 home page, carries public: true<br/>build/publish.rs:261-265<br/>build/publish.rs:284"]
    RET["Staging: files, per_vault counts, redaction tally<br/>build/publish.rs:111-118"]
    BUN["publish_out is None: files go into Staged under their own path as owner<br/>build/mod.rs:668-671"]
    SEP["publish_out is Some: recorded as external artefacts, written by write_to after promote, gate still applied<br/>build/mod.rs:672-678<br/>build/mod.rs:914-916<br/>build/mod.rs:71-73"]
    N1["INVARIANT: the frontmatter is scanned as well as the body, so a credential in a frontmatter value<br/>is refused, its line being the line of the rendered block<br/>build/publish.rs:232-243"]
    N2["INVARIANT: the markdown mirror and page API are built from projected records, not from<br/>the scanned bytes, so only publish/ is covered by the gate by construction<br/>build/mod.rs:768"]
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
- INVARIANT: the refusal is a typed error so the CLI exits 2 (`build/publish.rs:66`, `ontomark-cli/src/main.rs:524-527`); `stage` without links is only a wrapper (`publish.rs:185-187`) and `run_with` never calls it.
- DOC-DRIFT: `publish.rs` says `<out>/publish/` is the markdown that is published (`publish.rs:1`) but with `publish_out` it is not in `<out>` at all (`mod.rs:672-678`, `:914`).
- Under `--scope machine` the gate still scans `working/`'s public pages though nothing is staged into the bundle (`build/mod.rs:665-667`), see OM-06.

## OM-03.5 Staged write-once and the promote

```mermaid
flowchart TD
    ADD["Staged.add(path, bytes): owner is the build<br/>build/mod.rs:316-318<br/>build/mod.rs:283"]
    ADDFOR["Staged.add_for(path, bytes, owner): owners map entry<br/>build/mod.rs:322-328"]
    OCC{"path already owned?<br/>build/mod.rs:329"}
    CLASH["recorded in clashes, NOT applied: first writer wins the bytes<br/>build/mod.rs:331"]
    SLOT["owner stored, file pushed to files<br/>build/mod.rs:334-335"]
    CHK1["ensure_single_claims 1: after every emitter, before the stamp<br/>build/mod.rs:855"]
    ART["artifacts(): sha256 + bytes of every file staged so far, plus the external ones recorded for publish_out<br/>build/mod.rs:351-357<br/>build/generation.rs:38-44"]
    MARK["markers staged AFTER artifacts: data marker, then the root marker that lists it<br/>build/mod.rs:910-911"]
    CHK2["ensure_single_claims 2: the only new claims are the two markers<br/>build/mod.rs:912"]
    ERRC["Err(OutputCollision): CLI exit 2, every clash named<br/>build/mod.rs:345<br/>ontomark-cli/src/main.rs:530-532"]
    PRO["promote(out): write all files into parent/.publication-PID<br/>build/mod.rs:365-380"]
    RET["swap_in: clear a stale retired directory, rename out to .publication-retired-PID if it exists<br/>build/mod.rs:256-262"]
    SWAP["rename stage to out; on error restore retired and delete stage<br/>build/mod.rs:263-277"]
    DEL["retired directory deleted, so no obsolete export survives<br/>build/mod.rs:265-267"]
    WT["write_to(publish_out): files written beside dir in .publish-staging-PID, then the same swap_in<br/>build/publish.rs:334-349<br/>build/mod.rs:914-916"]
    N1["INVARIANT: swap_in is two renames, not one: between them out does not exist, and any<br/>pre-existing content of out is replaced wholesale, not merged, the old tree being restored on failure<br/>build/mod.rs:260-263<br/>build/mod.rs:270-274"]
    N2["DIVERGENCE: write_to runs AFTER promote, so a failure there leaves the NEW bundle beside the PREVIOUS<br/>publication tree that swap_in kept, the two no longer one generation<br/>build/publish.rs:349<br/>build/mod.rs:913-915"]
    N3["INVARIANT: no marker lists itself, the root marker lists data/.generation.json,<br/>and publish_out files ARE stamped as external artefacts<br/>build/mod.rs:904-908<br/>build/mod.rs:311-313"]
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
- INVARIANT: `Staged` is the only writer path for the bundle and holds bytes in memory (`build/mod.rs:290-297`); no emitter can overwrite another silently, a clash names both claimants (`mod.rs:148-159`).
- DOC-DRIFT: the module doc calls step 6 an atomic promotion (`build/mod.rs:12`); the two-rename swap in `mod.rs:260-263` is rolled back on failure but is not a single atomic operation, and the `promote` doc no longer claims it (`mod.rs:359-360`).

## OM-03.6 Generation identity — what the stamp is made of

```mermaid
flowchart TD
    SRC{"Options.source given?<br/>build/mod.rs:824"}
    GIVEN["commit and dirty taken from the caller<br/>build/mod.rs:825<br/>ontomark-cli/src/main.rs:498-501"]
    SHA["git_sha(repo_root): git rev-parse HEAD, else unknown<br/>build/generation.rs:184-195<br/>build/mod.rs:827"]
    PATHS["read_paths: vault_root + repo_root/ontology + the working root whenever working or other_vault was loaded<br/>build/mod.rs:832-838"]
    DIRTY["git_dirty: git status --porcelain over those paths, canonicalised, false outside git<br/>build/generation.rs:162-178<br/>build/mod.rs:839"]
    ID["generation_id: visionGraph@sha, plus +dirty when dirty<br/>build/generation.rs:150-156<br/>build/mod.rs:842"]
    CD["content_digest_of_hashes: SHA-256 over path, NUL, page source_sha256 hex, newline, sorted by path<br/>every page and journal of every loaded vault, plus the vocabulary<br/>build/generation.rs:221-230<br/>build/mod.rs:861-879"]
    OD["ontology_digest: sha256-12 over the asserted graph minus the header subject<br/>build/turtle.rs:611-640<br/>build/mod.rs:883"]
    GEN["Generation: id, commit, dirty, content_digest, ontology_digest, generated_at,<br/>class_count, page_count, vocabulary_version, stale_after, scope, artifacts<br/>build/generation.rs:88-121<br/>build/mod.rs:885-897"]
    M1["data/.generation.json = gen.scoped(data): only data/ artefacts, names relative to data/<br/>build/generation.rs:128-144<br/>build/mod.rs:903"]
    M2[".generation.json = gen plus the data marker entry: every artefact relative to out<br/>build/mod.rs:904-911"]
    GA["generated_at: Options value or now_iso, its first 10 chars also become today for graph tiers<br/>build/mod.rs:649-652,735"]
    N1["DIVERGENCE: the attribution roster shapes the Turtle and the page API but is not an input<br/>to content_digest, which hashes pages, journals and the vocabulary only<br/>build/mod.rs:621-626<br/>build/mod.rs:861-879"]
    N2["DIVERGENCE: C3 as stamped shows one marker and no dirty or ontology_digest field, code writes two<br/>markers, both extra fields and scope on a machine bundle<br/>sovereign-corpus-contracts.md:137-138"]
    N3["DIVERGENCE: the vocabulary directory is hard-coded to repo_root/ontology for dirtiness,<br/>and the roster is read from vault_root.parent()/ontology, two roots for one directory<br/>build/mod.rs:832<br/>build/attribution.rs:32"]
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
- INVARIANT: `commit` stays the bare sha even when dirty (`build/generation.rs:91-93`); the sha says which commit, the digest what was read, and the ontology digest what was emitted, so pages alone do not fix the Turtle (`build/mod.rs:880-882`).
- INVARIANT: `Roster::discover` is the only reader of `attribution.yaml`, an absent file is an empty roster (`build/attribution.rs:29-37`), and it is called with `vault_root.parent()` (`build/mod.rs:621-624`), not `repo_root`.

## OM-03.7 Report, and C3 artefact map — what run() writes versus what C3 lists

```mermaid
flowchart LR
    subgraph C3["C3 artefacts, sovereign-corpus-contracts.md:110-139"]
        A1["data/ontology.ttl, ontology-inferred.ttl<br/>sovereign-corpus-contracts.md:111-112"]
        A2["data/scaffold-index.json, prose-index.json<br/>sovereign-corpus-contracts.md:113-114"]
        A3["ontology-corpus.records.jsonl, --with-rvdb only<br/>sovereign-corpus-contracts.md:115"]
        A4["api/search-index, pages, census, validation-report<br/>sovereign-corpus-contracts.md:121"]
        A5["ns/v2.jsonld, api/schema/context.jsonld, context/v1<br/>sovereign-corpus-contracts.md:122-123"]
        A6["data/ontology.json and data/graph tiers<br/>sovereign-corpus-contracts.md:124-125"]
        A7["api/markdown only with the flag<br/>sovereign-corpus-contracts.md:126"]
        A8["publish/ pages, working, index.md<br/>sovereign-corpus-contracts.md:127-133"]
        A9["okf/ and .generation.json<br/>sovereign-corpus-contracts.md:136-138"]
    end
    subgraph RUN["written by run_with()"]
        W1["staged.add ttl and inferred<br/>build/mod.rs:679-683"]
        W2["scaffold, prose<br/>build/mod.rs:686,688"]
        W3["records.jsonl only when an Embedder is handed in: concept_records, embed_all, to_jsonl<br/>build/mod.rs:847-852<br/>build/records.rs:173"]
        W4["search-index, api/pages/slug, _domain-index, census, validation-report<br/>build/mod.rs:692,704-715"]
        W5["context/v1, api/schema/context, ns/v2, one byte-identical document<br/>build/mod.rs:723-725"]
        W6["data/ontology.json, data/graph/name for each tier file<br/>build/mod.rs:729,737"]
        W7["api/markdown/name.md gated by with_markdown_mirror<br/>build/mod.rs:741,816"]
        W8["publish/ via Staged, or write_to publish_out, and none at all under scope machine<br/>build/mod.rs:665-678,914-916"]
        W9["okf/path, then both markers<br/>build/mod.rs:844,910-911"]
    end
    RPT["Report: generation, counts, reasoner mode, published per vault, written, out, redaction, machine<br/>build/mod.rs:93-136"]
    CLI["CLI prints id, out, written; fails when written is 0<br/>ontomark-cli/src/main.rs:556-576"]
    N1["INVARIANT: the records path exists in run_with: the compile crate opens no socket, the host hands<br/>in an Embedder and the CLI supplies the Xinference transport<br/>build/records.rs:61<br/>ontomark-cli/src/embed.rs:30<br/>ontomark-cli/src/main.rs:508-511"]
    N2["DIVERGENCE: api/pages/_domain-index.json is written but not listed in C3 as stamped<br/>build/mod.rs:710"]
    N3["INVARIANT: Report.unsatisfiable is always empty on success, because a non-empty list already<br/>refused the build with build::Unsatisfiable, so the field cannot carry its documented meaning<br/>build/mod.rs:110-114,930<br/>build/mod.rs:642-648"]
    N4["INVARIANT: reason already returns ReasoningError UnsupportedProfile on classifier refusals,<br/>so is_consistent at the gate can fail only on unsatisfiable classes, and Unsatisfiable names root causes<br/>reason.rs:590-598<br/>build/mod.rs:193-218"]
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
- INVARIANT: `Report.written` counts both destinations (`build/mod.rs:937-944`) and the CLI refuses a zero (`ontomark-cli/src/main.rs:570-576`); `mod.rs:939` sums `staged.files` plus the separate publish staging.
- INVARIANT: C3 as stamped says `content_digest` is over pairs "the build read" (`sovereign-corpus-contracts.md:141`) and `run_with` now hashes every page and journal it loaded plus the vocabulary, see OM-03.6.
