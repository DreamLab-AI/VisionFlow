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
  - docs/engineering/sovereign-corpus-contracts.md
verified_commit: {ontomark: ffe06e45bfebd388a1d85849b24a8810e4d0d610, VisionFlow: 6443781fe6ed5f65d980510c86d92f8b373f0d3a}
---

## OM-03.1 build::run — Options to promoted bundle, in the order the code runs

```mermaid
sequenceDiagram
    autonumber
    participant CLI as vault build CLI<br/>ontomark-cli/src/main.rs:418
    participant RUN as build::run<br/>build/mod.rs:302
    participant IN as input loaders<br/>input/page.rs:568
    participant PUB as publish staging<br/>build/publish.rs:194
    participant PRJ as validate + project<br/>ir/projection.rs:453
    participant RSN as closure + Whelk<br/>reason.rs:554
    participant STG as Staged<br/>build/mod.rs:163
    participant GEN as generation<br/>build/generation.rs:184

    CLI->>RUN: Options + Profile + Vocabulary + WhelkEl classifier (main.rs:418-423)
    RUN->>IN: inspect_inputs(pages_dir) census, refuses a non-boolean public (mod.rs:309)
    RUN->>IN: load_vault(vault_root) then alias_map then Corpus::build raw (mod.rs:311-313)
    RUN->>IN: sibling_working only if with_working_publish (mod.rs:325-326)
    RUN->>IN: sibling_vault(other kind) only if redact and no working (mod.rs:339-345)
    RUN->>PUB: stage_redacted(publish_vaults, private_links) = CREDENTIAL GATE (mod.rs:354)
    PUB-->>RUN: Err(SecretsFound) exits the CLI with 2, before validate (publish.rs:246-248)
    RUN->>PRJ: validate(profile, vault, raw, vocab) then ensure no errors (mod.rs:356-362)
    RUN->>PRJ: excluded_identities(pages_dir), third read of the pages dir (mod.rs:364)
    RUN->>PRJ: project_vault_with_links, or project_with_redactor when redaction is off (mod.rs:365-370)
    RUN->>RSN: closure::compute, backlink_index, Roster::discover, build_graph_attributed (mod.rs:372-379)
    RUN->>RSN: reason::reason(graph, classifier), ADR-2124 cap error returned unwrapped (mod.rs:383)
    RUN->>RSN: ensure is_consistent, ADR-2125 zero-unsatisfiable gate (mod.rs:395-399)
    RUN->>STG: add publish files if publish_out is None, then every emitter (mod.rs:411-552)
    RUN->>GEN: git_sha + git_dirty unless Options.source is given (mod.rs:554-569)
    RUN->>STG: okf bundle staged under okf/ using generation_id (mod.rs:570-573)
    RUN->>STG: ensure_single_claims, write-once check number one (mod.rs:576)
    RUN->>GEN: content_digest, ontology_digest, generation(...) over staged.artifacts (mod.rs:578-603)
    RUN->>STG: add data/.generation.json and .generation.json, check number two (mod.rs:607-609)
    RUN->>STG: promote(out) atomic swap, then write_to(publish_out) (mod.rs:610-613)
    RUN-->>CLI: Report (mod.rs:615-645)
    Note over CLI: SecretsFound, OutputCollision and ReasoningError exit 2, anything else is an anyhow error (main.rs:429-445)
```
- INVARIANT: nothing touches `out` until `promote` at `build/mod.rs:610`; every earlier refusal (`mod.rs:357`, `:395`, `:576`) leaves the previous bundle intact.
- DOC-DRIFT: the gate comment says it runs FIRST (`build/mod.rs:315`), yet census, vault load and `Corpus::build` precede it (`mod.rs:309-313`) and `stage_redacted` runs after up to two extra vault loads (`mod.rs:325-347`).
- DIVERGENCE: an unsatisfiable corpus is a plain `anyhow::ensure!` (`mod.rs:395`), so it exits 1 not 2 (`main.rs:445`); only the ADR-2124 cap error is downcast to exit 2 (`main.rs:441`).

## OM-03.2 What `--vault` means, and every place a vault is loaded

```mermaid
flowchart TD
    ARG["CLI --vault knowledge or working or all<br/>vault_kinds(args.vault)<br/>ontomark-cli/src/main.rs:396"]
    KIND["kind = Working only when the list is exactly Working, else Knowledge<br/>ontomark-cli/src/main.rs:397-401"]
    ROOT["Options.vault_root = root/kind.dir_name()<br/>ONE root feeds every machine artefact<br/>ontomark-cli/src/main.rs:406"]
    WIDE["Options.with_working_publish = list contains Working AND kind is Knowledge<br/>ontomark-cli/src/main.rs:403-404"]
    LOAD["load_vault(vault_root): kind inferred from the directory name<br/>build/mod.rs:311<br/>input/page.rs:570"]
    MACH["Machine artefacts: Corpus raw, validate, project, closure, Turtle, reason,<br/>indexes, page API, graph tiers, okf, markdown mirror<br/>build/mod.rs:313,356,365,372,416"]
    SW["sibling_working: repo.join working, loaded only if pages dir exists<br/>build/publish.rs:342-346<br/>build/mod.rs:325-326"]
    PUBV["publish_vaults = vault + working<br/>build/mod.rs:330-331"]
    STAGE["stage_redacted over publish_vaults: the ONLY consumer of working pages as content<br/>build/mod.rs:354"]
    OTHER["other_vault = sibling_vault(opposite kind)<br/>only if redact_links AND working is None<br/>build/mod.rs:339-347"]
    PL["PrivateLinks::new over vault + working + other_vault<br/>link resolution only, nothing is published from other_vault<br/>build/mod.rs:348-353"]
    GITD["git_dirty read_paths: vault_root, repo/ontology, plus working only if with_working_publish<br/>build/mod.rs:560-567"]
    N1["DIVERGENCE: with_working_publish changes behaviour in two places (loads working at<br/>build/mod.rs:325 and widens the dirty paths at build/mod.rs:564), never the projections<br/>build/mod.rs:73"]
    N2["DIVERGENCE: when redaction is on, working is loaded exactly once either way,<br/>but under a different variable (working vs other_vault), so two near-identical loaders exist<br/>build/publish.rs:336,353"]
    N3["DIVERGENCE: sibling_working never checks vault.kind, so a library caller with<br/>vault_root = working plus with_working_publish stages the same vault twice<br/>build/publish.rs:342"]
    ARG --> KIND --> ROOT --> LOAD --> MACH
    KIND --> WIDE
    WIDE --> SW --> PUBV --> STAGE
    LOAD --> PUBV
    LOAD --> OTHER --> PL
    SW --> PL
    LOAD --> PL
    PL --> STAGE
    WIDE --> GITD
    WIDE -.-> N1
    OTHER -.-> N2
    SW -.-> N3
```
- INVARIANT: `working/` never enters `corpus`, so OWL membership stays `knowledge/`-only (`ontomark-cli/src/main.rs:391-395`, `build/mod.rs:73`); `--vault working` alone makes the WHOLE machine corpus come from `working/` (`main.rs:397-399`).
- INVARIANT: `Options.repo_root` is used only for git (`build/mod.rs:557,560,567`); every other "repo" is `vault_root.parent()` (`mod.rs:326,344,374,561`), two derivations of one directory.

## OM-03.3 Repeated reads — three independent walkers over the pages dir

```mermaid
flowchart TD
    DIR[("vault_root/pages<br/>build/mod.rs:308")]
    W1["1 inspect_inputs: WalkDir, read_to_string, parse frontmatter, census counts<br/>ir/projection.rs:72,77,97<br/>build/mod.rs:309"]
    W2["2 load_vault to Vault::load to walk_pages: WalkDir, read_to_string, Page::parse<br/>input/page.rs:568,399,496,413<br/>build/mod.rs:311"]
    W3["3 excluded_identities: WalkDir again, reads ONLY held pages and re-parses them<br/>ir/projection.rs:169,172,185,189<br/>build/mod.rs:364"]
    W4["4 content_digest sources: std::fs::read every vault.pages path a second time, unwrap_or_default<br/>build/mod.rs:578-586"]
    J["journals/ walked inside Vault::load, loaded but never hashed or published<br/>input/page.rs:418-422"]
    C1["Corpus::build raw (mod.rs:313)"]
    C2["project(...) inside project_vault_with_links, result discarded, only the gate matters<br/>ir/projection.rs:460"]
    C3["Corpus::build(cleaned) after vault.clone() and per-page redaction<br/>ir/projection.rs:461,499"]
    R1["redaction of the same body text, three implementations: projection<br/>ir/projection.rs:497, publish render build/publish.rs:168, markdown mirror build/mod.rs:516"]
    N1["DIVERGENCE: every public page is read 3 times (census, load, digest) and a held page 4 times,<br/>from 3 independent WalkDir passes with 3 different filters<br/>input/page.rs:512, ir/projection.rs:93, ir/projection.rs:182"]
    N2["INVARIANT: the digest re-read is a time-of-check gap, the digest can describe bytes that<br/>differ from the bytes parsed at input/page.rs:413, and a read error silently hashes empty bytes<br/>build/mod.rs:582"]
    DIR --> W1
    DIR --> W2
    DIR --> W3
    W2 --> C1
    W2 --> J
    W2 --> W4
    C1 --> C2
    W2 --> C3
    C3 --> R1
    W1 -.-> N1
    W4 -.-> N2
```
- DIVERGENCE: the dot-directory rule is written twice, in `walk_pages` (`input/page.rs:512`) and in `inspect_inputs` (`ir/projection.rs:93`), while `excluded_identities` has no dot rule at all and filters by `is_unpublished` only (`projection.rs:182`).
- DIVERGENCE: `held_back` (`build/publish.rs:134`) is a fourth copy of the hold-back rule: `is_unpublished` covers `_misc` at top level only (`input/page.rs:38,543`), `HELD_BACK_DIRS` adds `misc` at any depth (`publish.rs:130`).

## OM-03.4 Credential gate and publish staging

```mermaid
flowchart TD
    IN["stage_redacted(vaults, links)<br/>build/publish.rs:194"]
    LOOP["for each vault, for each page in vault.pages<br/>build/publish.rs:204,211"]
    HB{"held_back(rel)?<br/>_misc top level, misc at any depth<br/>build/publish.rs:216"}
    CONS["counts.considered += 1<br/>build/publish.rs:219"]
    PUBQ{"page.is_public()?<br/>build/publish.rs:220"}
    SCAN["secrets::scan(page.body) ONLY the body, never the frontmatter<br/>build/publish.rs:224"]
    REND["render: frontmatter re-rendered, links redacted, body verbatim<br/>build/publish.rs:163-169"]
    PUSH["StagedPage path = publish/tree/id.md, tree = pages or working<br/>build/publish.rs:237-242<br/>build/publish.rs:151-155"]
    FIND{"any finding?<br/>build/publish.rs:246"}
    ERR["Err(SecretsFound): nothing staged, kind + file + line, never the value<br/>build/publish.rs:247<br/>build/publish.rs:36"]
    IDX["publish/index.md added: OKF section 8 home page, carries public: true<br/>build/publish.rs:249-253<br/>build/publish.rs:272"]
    RET["Staging: files, per_vault counts, redaction tally<br/>build/publish.rs:111-118"]
    BUN["publish_out is None: files go into Staged under their own path as owner<br/>build/mod.rs:411-415"]
    SEP["publish_out is Some: write_to after promote, gate still applied<br/>build/mod.rs:611-613<br/>build/mod.rs:68-70"]
    N1["DIVERGENCE: the gate scans page.body, but render publishes the re-rendered frontmatter too,<br/>so a credential in a frontmatter value is published unscanned<br/>build/publish.rs:164"]
    N2["INVARIANT: the markdown mirror and page API are built from projected records, not from<br/>the scanned bytes, so only publish/ is covered by the gate by construction<br/>build/mod.rs:505-535"]
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
- INVARIANT: the refusal is a typed error so the CLI exits 2 (`build/publish.rs:66`, `ontomark-cli/src/main.rs:429-431`); `stage` without links is only a wrapper (`publish.rs:185-187`) and `run` never calls it.
- DOC-DRIFT: `publish.rs` says `<out>/publish/` is the markdown that is published (`publish.rs:1`) but with `publish_out` it is not in `<out>` at all (`mod.rs:411`, `:611`).

## OM-03.5 Staged write-once and the promote

```mermaid
flowchart TD
    ADD["Staged.add(path, bytes): owner is the build<br/>build/mod.rs:179-181<br/>build/mod.rs:156"]
    ADDFOR["Staged.add_for(path, bytes, owner): owners map entry<br/>build/mod.rs:185-191"]
    OCC{"path already owned?<br/>build/mod.rs:192"}
    CLASH["recorded in clashes, NOT applied: first writer wins the bytes<br/>build/mod.rs:194"]
    SLOT["owner stored, file pushed to files<br/>build/mod.rs:197-198"]
    CHK1["ensure_single_claims 1: after every emitter, before the stamp<br/>build/mod.rs:576"]
    ART["artifacts(): sha256 + bytes of every file staged so far<br/>build/mod.rs:214-218<br/>build/generation.rs:38-44"]
    MARK["markers staged AFTER artifacts: data/.generation.json then .generation.json<br/>build/mod.rs:607-608"]
    CHK2["ensure_single_claims 2: the only new claims are the two markers<br/>build/mod.rs:609"]
    ERRC["Err(OutputCollision): CLI exit 2, every clash named<br/>build/mod.rs:208<br/>ontomark-cli/src/main.rs:435-437"]
    PRO["promote(out): write all files into parent/.publication-PID<br/>build/mod.rs:226-239"]
    RET["rename out to .publication-retired-PID if it exists<br/>build/mod.rs:242-245"]
    SWAP["rename stage to out; on error restore retired and delete stage<br/>build/mod.rs:246-261"]
    DEL["retired directory deleted, so no obsolete export survives<br/>build/mod.rs:248-250"]
    WT["write_to(publish_out): remove_dir_all(dir) FIRST, then write each file<br/>build/publish.rs:316-328<br/>build/mod.rs:611-613"]
    N1["DIVERGENCE: promote is two renames, not one: between them out does not exist, and any<br/>pre-existing content of out is deleted wholesale, not merged<br/>build/mod.rs:243-246"]
    N2["DIVERGENCE: write_to is neither atomic nor write-once checked, and runs AFTER promote,<br/>so a failure there leaves the new bundle beside a deleted or partial publish directory<br/>build/publish.rs:316"]
    N3["INVARIANT: the markers are not listed in their own artifacts array, and nothing hashes<br/>publish/ when publish_out is Some<br/>build/mod.rs:602"]
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
- INVARIANT: `Staged` is the only writer path for the bundle and holds bytes in memory (`build/mod.rs:158-167`); no emitter can overwrite another silently, a clash names both claimants (`mod.rs:125-136`).
- DOC-DRIFT: the module doc calls step 6 an atomic promotion (`build/mod.rs:12`); the two-rename swap in `mod.rs:244-246` is rolled back on failure but is not a single atomic operation.

## OM-03.6 Generation identity — what the stamp is made of

```mermaid
flowchart TD
    SRC{"Options.source given?<br/>build/mod.rs:554"}
    GIVEN["commit and dirty taken from the caller<br/>build/mod.rs:555<br/>ontomark-cli/src/main.rs:411-414"]
    SHA["git_sha(repo_root): git rev-parse HEAD, else unknown<br/>build/generation.rs:140-151<br/>build/mod.rs:557"]
    PATHS["read_paths: vault_root + repo_root/ontology + working if with_working_publish<br/>build/mod.rs:560-566"]
    DIRTY["git_dirty: git status --porcelain over those paths, canonicalised, false outside git<br/>build/generation.rs:118-134<br/>build/mod.rs:567"]
    ID["generation_id: visionGraph@sha, plus +dirty when dirty<br/>build/generation.rs:106-112<br/>build/mod.rs:570"]
    CD["content_digest: SHA-256 over path, NUL, bytes, newline of vault.pages sorted by path<br/>build/generation.rs:159-168<br/>build/mod.rs:578-586"]
    OD["ontology_digest: sha256-12 over the asserted graph minus the header subject<br/>build/turtle.rs:611-640<br/>build/mod.rs:590"]
    GEN["Generation: id, commit, dirty, content_digest, ontology_digest, generated_at,<br/>class_count, page_count, vocabulary_version, stale_after, artifacts<br/>build/generation.rs:49-77<br/>build/mod.rs:592-603"]
    M1["data/.generation.json = gen.scoped(data): only data/ artefacts, names relative to data/<br/>build/generation.rs:84-100<br/>build/mod.rs:607"]
    M2[".generation.json = gen: every artefact relative to out<br/>build/mod.rs:608"]
    GA["generated_at: Options value or now_iso, its first 10 chars also become today for graph tiers<br/>build/mod.rs:400-403,465"]
    N1["DIVERGENCE: content_digest covers vault.pages of ONE vault_root only, not journals,<br/>working pages that are published, vocabulary or attribution roster<br/>build/mod.rs:578-585"]
    N2["DIVERGENCE: C3 shows one marker and no dirty or ontology_digest field, code writes two<br/>markers and both extra fields<br/>sovereign-corpus-contracts.md:137-138"]
    N3["DIVERGENCE: the vocabulary directory is hard-coded to repo_root/ontology for dirtiness,<br/>and the roster is read from vault_root.parent()/ontology, two roots for one directory<br/>build/mod.rs:560<br/>build/attribution.rs:32"]
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
- INVARIANT: `commit` stays the bare sha even when dirty (`build/generation.rs:52-54`); the sha says which commit, the digest what was read, and the ontology digest what was emitted, so pages alone do not fix the Turtle (`build/mod.rs:587-589`).
- INVARIANT: `Roster::discover` is the only reader of `attribution.yaml`, an absent file is an empty roster (`build/attribution.rs:29-37`), and it is called with `vault_root.parent()` (`build/mod.rs:374-378`), not `repo_root`.

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
    subgraph RUN["written by run()"]
        W1["staged.add ttl and inferred<br/>build/mod.rs:416-420"]
        W2["scaffold, prose<br/>build/mod.rs:423,425"]
        W3["NOT WRITTEN: no rvdb, jsonl or with-rvdb option exists in Options<br/>build/mod.rs:48-82"]
        W4["search-index, api/pages/slug, _domain-index, census, validation-report<br/>build/mod.rs:429,434-445"]
        W5["context/v1, api/schema/context, ns/v2, one byte-identical document<br/>build/mod.rs:453-455"]
        W6["data/ontology.json, data/graph/name for each tier file<br/>build/mod.rs:459,467"]
        W7["api/markdown/name.md gated by with_markdown_mirror<br/>build/mod.rs:471,546"]
        W8["publish/ via Staged, or write_to publish_out<br/>build/mod.rs:411-415,611-613"]
        W9["okf/path, then both markers<br/>build/mod.rs:572,607-608"]
    end
    RPT["Report: generation, counts, reasoner mode, published per vault, written, out, redaction<br/>build/mod.rs:86-123"]
    CLI["CLI prints id, out, written; fails when written is 0<br/>ontomark-cli/src/main.rs:449-469"]
    N1["DIVERGENCE: C3 lists ontology-corpus.records.jsonl and says vault once wrote it under the rvdb name,<br/>but run() has no code path for it at all<br/>sovereign-corpus-contracts.md:115-120"]
    N2["DIVERGENCE: api/pages/_domain-index.json is written but not listed in C3<br/>build/mod.rs:440"]
    N3["INVARIANT: Report.unsatisfiable is always empty on success, because a non-empty list already<br/>bailed at mod.rs:395, so the field cannot ever carry its documented meaning<br/>build/mod.rs:103-104,627"]
    N4["DIVERGENCE: is_consistent also fails on classifier refusals, but unsatisfiable_message names<br/>only unsatisfiable classes, so a refusal-only failure reads 0 unsatisfiable, 0 root causes<br/>reason.rs:78-79<br/>build/mod.rs:652-673"]
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
- INVARIANT: `Report.written` counts both destinations (`build/mod.rs:629-636`) and the CLI refuses a zero (`ontomark-cli/src/main.rs:465-469`); `mod.rs:631` sums `staged.files` plus the separate publish staging.
- DOC-DRIFT: C3 says `content_digest` is over pairs "the build read" (`sovereign-corpus-contracts.md:141`), but `run()` hashes only the one vault's pages, see OM-03.6.
