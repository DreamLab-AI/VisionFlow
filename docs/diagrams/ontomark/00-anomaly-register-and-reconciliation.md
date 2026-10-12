---
id: OM-00
title: Anomaly register — duplicated rules, ranked faults, revert-or-reconcile, canonical run()
area: ontomark
governing:
  - docs/engineering/sovereign-corpus-contracts.md
adrs: [ADR-OM-001, ADR-OM-005, ADR-2137]
sources:
  - ../ontomark/crates/ontomark-compile/src/input/page.rs
  - ../ontomark/crates/ontomark-compile/src/input/frontmatter.rs
  - ../ontomark/crates/ontomark-compile/src/input/slug.rs
  - ../ontomark/crates/ontomark-compile/src/input/secrets.rs
  - ../ontomark/crates/ontomark-compile/src/input/consistency.rs
  - ../ontomark/crates/ontomark-compile/src/input/promotion.rs
  - ../ontomark/crates/ontomark-compile/src/ir/model.rs
  - ../ontomark/crates/ontomark-compile/src/ir/projection.rs
  - ../ontomark/crates/ontomark-compile/src/ir/closure.rs
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/build/publish.rs
  - ../ontomark/crates/ontomark-compile/src/build/redact.rs
  - ../ontomark/crates/ontomark-compile/src/build/turtle.rs
  - ../ontomark/crates/ontomark-compile/src/build/indexes.rs
  - ../ontomark/crates/ontomark-compile/src/build/page_api.rs
  - ../ontomark/crates/ontomark-compile/src/build/webvowl.rs
  - ../ontomark/crates/ontomark-compile/src/build/graph_tiers.rs
  - ../ontomark/crates/ontomark-compile/src/check/validate.rs
  - ../ontomark/crates/ontomark-compile/src/check/gate.rs
  - ../ontomark/crates/ontomark-compile/src/check/conflicts.rs
  - ../ontomark/crates/ontomark-compile/src/shacl.rs
  - ../ontomark/crates/ontomark-compile/src/reason.rs
  - ../ontomark/crates/ontomark-compile/src/migrate/compile.rs
  - ../ontomark/crates/ontomark-model/src/profile.rs
  - ../ontomark/crates/ontomark-model/src/mint.rs
  - ../ontomark/crates/ontomark-model/tests/adr2015.rs
  - ../ontomark/crates/ontomark-wasm/src/profile.rs
  - ../ontomark/crates/ontomark-ngg1/src/lib.rs
  - ../ontomark/crates/ontomark-cli/src/main.rs
  - ../ontomark/crates/ontomark-cli/src/migrate_cmd.rs
  - ../ontomark/docs/HANDOFF.md
  - ../ontomark/docs/DEPRECATION.md
  - ../ontomark/docs/adr/ADR-OM-005-estate-alignment-one-producer-explicit-projections.md
  - ../ontomark/conformance/evidence/standards.md
  - ../project/crates/vault/src/main.rs
  - ../project/crates/vault/src/projection.rs
  - ../project/crates/vault/src/build/publish.rs
  - ../project/crates/vault/src/build/mod.rs
  - ../project/crates/vault/src/build/graph_tiers.rs
  - ../project/crates/vault/src/gate.rs
  - ../project/crates/vault/src/conflicts.rs
  - ../project/crates/vault-core/src/frontmatter.rs
  - ../project/crates/vault-core/src/page.rs
  - docs/engineering/sovereign-corpus-contracts.md
verified_commit: {ontomark: ffe06e45bfebd388a1d85849b24a8810e4d0d610, visionclaw: 4806604edacf68179877e8db995299dceba46e75, visionflow: 3abc1afcde9cf73e3488351d0955bf1bb53272a5}
---

## OM-00.1 Duplicated rules — every copy, whether it diverges, and the one home each should collapse into

```mermaid
flowchart LR
    subgraph PUB["public gate x4"]
        P1["lenient: strings true/false accepted<br/>input/page.rs:211-213 via input/frontmatter.rs:233-236"]
        P2["strict: only a YAML bool, anything else refuses the build<br/>ir/projection.rs:112-116"]
        P3["profile-configurable lenient_booleans<br/>ontomark-model/src/profile.rs:604-611"]
        P4["strict, gate key optional<br/>ontomark-wasm/src/profile.rs:316-321"]
    end
    PUBC["CANONICAL: one is_public on the model Profile,<br/>estate profile with lenient_booleans false<br/>ontomark-model/src/profile.rs:607"]
    P1 -->|DIVERGENCE| PUBC
    P2 -->|same rule| PUBC
    P3 --> PUBC
    P4 -->|same rule| PUBC

    subgraph WALK["directory walk x3"]
        W1["walk_pages: errors surface, skips reported, dot rule<br/>input/page.rs:492-515"]
        W2["inspect_inputs census: own dot + unpublished filter<br/>ir/projection.rs:77, ir/projection.rs:93"]
        W3["excluded_identities: filter_map ok swallows walk errors, no dot rule<br/>ir/projection.rs:172-175"]
    end
    WALKC["CANONICAL: walk_pages once, PageWalk feeds census and holds<br/>input/page.rs:492"]
    W2 -->|DIVERGENCE| WALKC
    W3 -->|DIVERGENCE| WALKC
    W1 --> WALKC

    subgraph LINK["wikilink resolve x4"]
        L1["Corpus::build resolve, fragment and .md kept<br/>ir/model.rs:336-342"]
        L2["PrivateLinks::resolve strips fragment and .md<br/>build/redact.rs:209-226"]
        L3["shacl Resolver::page, no slug key, case-sensitive<br/>shacl.rs:1197-1199"]
        L4["curated links slug verbatim via target_slug<br/>ir/model.rs:424"]
    end
    LINKC["CANONICAL: one Resolver in ir/, rules of build/redact.rs:209"]
    L1 -->|DIVERGENCE| LINKC
    L2 --> LINKC
    L3 -->|DIVERGENCE| LINKC
    L4 -->|DIVERGENCE| LINKC

    subgraph HOLD["held-back list x4"]
        H1["UNPUBLISHED_DIRS _misc, first component only<br/>input/page.rs:38, input/page.rs:543-547"]
        H2["HELD_BACK_DIRS _misc + misc at any depth<br/>ontomark-compile/src/build/publish.rs:130-142"]
        H3["copy of the excluded_identities loop<br/>migrate/compile.rs:90-99"]
        H4["held_back applied only on the redacted projection path<br/>ir/projection.rs:467"]
    end
    HOLDC["CANONICAL: one held_back(rel) in input/, one list<br/>ontomark-compile/src/build/publish.rs:134"]
    H1 -->|DIVERGENCE| HOLDC
    H2 --> HOLDC
    H3 -->|dup only| HOLDC
    H4 -->|DIVERGENCE| HOLDC

    subgraph SLUG["slugify x3"]
        S1["regex non-alnum to dash, trimmed<br/>input/slug.rs:15-32"]
        S2["hand-rolled, same output, parity-tallied<br/>ontomark-model/src/mint.rs:73-88, adr2015.rs:727"]
        S3["target_slug: kebab target taken verbatim, keeps leading/doubled dashes<br/>input/slug.rs:66-75"]
    end
    SLUGC["CANONICAL: the model minter<br/>ontomark-model/src/mint.rs:73"]
    S1 -->|dup only| SLUGC
    S2 --> SLUGC
    S3 -->|DIVERGENCE| SLUGC

    subgraph REL["relation-order table x4 + stub list"]
        R1["RELATION_KEYS<br/>ir/model.rs:36"]
        R2["SCAFFOLD_REL_ORDER<br/>indexes.rs:30"]
        R3["PAGE_API_REL_ORDER<br/>page_api.rs:18"]
        R4["REL_MAP related-to to skos:related<br/>webvowl.rs:30"]
        R5["hard-coded 13 stub names, misses vocabulary-added relations<br/>turtle.rs:446-460"]
    end
    RELC["CANONICAL: RELATION_KEYS + vocabulary<br/>ir/model.rs:36"]
    R2 -->|dup only| RELC
    R3 -->|dup only| RELC
    R4 -->|DIVERGENCE| RELC
    R5 -->|DIVERGENCE| RELC
    R1 --> RELC

    subgraph LBL["label fallback x3"]
        B1["label else title<br/>indexes.rs:129-133"]
        B2["label else title else slug<br/>ir/closure.rs:109-113"]
        B3["label else slug, title skipped<br/>ontomark-compile/src/build/graph_tiers.rs:254-256"]
    end
    LBLC["CANONICAL: one display_label on the record, closure rule"]
    B1 -->|DIVERGENCE| LBLC
    B2 --> LBLC
    B3 -->|DIVERGENCE| LBLC

    subgraph DOM["domain normalisation x5"]
        D1["verbatim<br/>turtle.rs:302-304"]
        D2["alias fold via profile<br/>ontomark-compile/src/build/graph_tiers.rs:81-90"]
        D3["raw slug match<br/>webvowl.rs:91"]
        D4["empty to unclassified<br/>page_api.rs:147-148"]
        D5["title-case for display<br/>indexes.rs:328"]
    end
    DOMC["CANONICAL: profile resolver, then per-surface formatting<br/>ontomark-compile/src/build/graph_tiers.rs:81"]
    D1 -->|DIVERGENCE| DOMC
    D3 -->|DIVERGENCE| DOMC
    D4 -->|DIVERGENCE| DOMC
    D5 -->|presentation| DOMC
    D2 --> DOMC

    subgraph VK["vault_kinds x3"]
        K1["unknown to knowledge, clap-guarded<br/>ontomark-cli/src/main.rs:141-147"]
        K2["unknown to BOTH vaults<br/>migrate_cmd.rs:59-65"]
        K3["VisionClaw legacy copy<br/>vault/src/main.rs:165"]
    end
    VKC["CANONICAL: one VaultSelector beside VAULT_VALUES<br/>ontomark-cli/src/main.rs:131"]
    K1 --> VKC
    K2 -->|DIVERGENCE| VKC
    K3 -->|"EXTERNAL: dup until VC-60 retires"| VKC

    subgraph RED["body redaction x3, Corpus::build x3, sibling loaders x2, secret scan x2"]
        Y1["links.body on cleaned vault<br/>ir/projection.rs:497"]
        Y2["links.body in render<br/>ontomark-compile/src/build/publish.rs:168"]
        Y3["links.body + prose redactor for the mirror tally<br/>ontomark-compile/src/build/mod.rs:516"]
        C1["Corpus::build raw<br/>ontomark-compile/src/build/mod.rs:313"]
        C2["Corpus::build collision probe, result discarded<br/>ir/projection.rs:460"]
        C3["Corpus::build cleaned<br/>ir/projection.rs:499"]
        G1["sibling_working ignores vault.kind<br/>ontomark-compile/src/build/publish.rs:336-346"]
        G2["sibling_vault(kind)<br/>ontomark-compile/src/build/publish.rs:353"]
        X1["validate scans page.body only<br/>check/validate.rs:305"]
        X2["staging scans page.body only<br/>ontomark-compile/src/build/publish.rs:224"]
    end
    REDC["CANONICAL: one redacted page view, one Corpus::build,<br/>sibling_vault only, one census scan of rendered bytes"]
    Y1 -->|dup only| REDC
    Y2 -->|dup only| REDC
    Y3 -->|dup only| REDC
    C1 -->|dup only| REDC
    C2 -->|dup only| REDC
    C3 -->|dup only| REDC
    G1 -->|DIVERGENCE| REDC
    G2 --> REDC
    X1 -->|dup only| REDC
    X2 -->|dup only| REDC
```
- DIVERGENCE: an edge labelled DIVERGENCE joins a copy whose output differs from the canonical rule for some input; `dup only` copies agree today and differ only in cost.
- DIVERGENCE: slug.rs vs mint.rs are equivalent (`[^a-z0-9]+` at `input/slug.rs:17` vs ASCII-alnum at `ontomark-model/src/mint.rs:77`), overruling OM-01's "two slugify" as a divergence; the real fork is `target_slug` at `input/slug.rs:66-75`.

## OM-00.2 Ranked anomalies, grouped by severity

```mermaid
flowchart TB
    subgraph SEV1["SEV-1 publication boundary"]
        A1["S1.1 prose redactor fails OPEN: Regex::new(..).ok() leaves pattern None, redact returns text unchanged<br/>ir/projection.rs:232, ir/projection.rs:243-245"]
        A2["S1.2 credential scan is body-only, but render publishes re-rendered frontmatter<br/>ontomark-compile/src/build/publish.rs:224, ontomark-compile/src/build/publish.rs:164"]
        A3["S1.3 public: #quot;true#quot; string passes validate and stage, refused only by the build census<br/>input/page.rs:211-213, ir/projection.rs:112-116, ontomark-compile/src/build/mod.rs:309"]
        A4["S1.4 C3 names data/ontology-corpus.records.jsonl, no Options field or emitter writes it<br/>sovereign-corpus-contracts.md:115, ontomark-compile/src/build/mod.rs:48-82"]
        A5["S1.5 gate and conflicts take only the first vault under --vault all<br/>ontomark-cli/src/main.rs:334-337, ontomark-cli/src/main.rs:369-372"]
    end
    subgraph SEV2["SEV-2 contract and exit code"]
        B1["S2.1 unsatisfiable corpus is anyhow::ensure!, exits 1<br/>ontomark-compile/src/build/mod.rs:395-399, ontomark-cli/src/main.rs:441-445"]
        B2["S2.2 content_digest covers one vault's pages, re-read, read error hashes empty bytes<br/>ontomark-compile/src/build/mod.rs:578-586"]
        B3["S2.3 promote is two renames; publish_out is remove_dir_all then write, after promote<br/>ontomark-compile/src/build/mod.rs:241-246, ontomark-compile/src/build/publish.rs:316-317"]
        B4["S2.4 --tier and --severity accept anything, unknown becomes quick or high<br/>check/gate.rs:42-48, check/conflicts.rs:48-55"]
        B5["S2.5 generation markers are not in the artefact list they stamp, and publish/ is unhashed when publish_out is set<br/>ontomark-compile/src/build/mod.rs:602-608, ontomark-compile/src/build/mod.rs:411"]
    end
    subgraph SEV3["SEV-3 silent data loss into consumers"]
        C1["S3.1 NGG1: 12 relations collapse to EDGE_RELATION, dedup key merges distinct relations<br/>ontomark-compile/src/build/graph_tiers.rs:283, ontomark-compile/src/build/graph_tiers.rs:313-315"]
        C2["S3.2 a symlinked .md is skipped with no Skip record<br/>input/page.rs:496, input/page.rs:501-503"]
        C3["S3.3 unclosed opening --- is read as no frontmatter, page silently private<br/>input/frontmatter.rs:120-130, input/frontmatter.rs:153-157"]
        C4["S3.4 compat projection never applies held_back, a public misc/ page is projected<br/>ontomark-compile/src/build/mod.rs:368-369, ir/projection.rs:467"]
    end
    subgraph SEV4["SEV-4 duplication and cost"]
        E1["three WalkDir passes, three filters<br/>input/page.rs:496, ir/projection.rs:77, ir/projection.rs:172"]
        E2["three Corpus::build<br/>ontomark-compile/src/build/mod.rs:313, ir/projection.rs:460, ir/projection.rs:499"]
        E3["each public page read 3x, held page 4x; digest re-reads<br/>input/page.rs:413, ontomark-compile/src/build/mod.rs:582"]
        E4["by_class_slug built twice<br/>ir/model.rs:436, ir/projection.rs:428"]
        E5["ontology_digest computed twice, same value<br/>turtle.rs:488, ontomark-compile/src/build/mod.rs:590"]
    end
    subgraph SEV5["SEV-5 dead code and doc drift"]
        F1["no production caller: page_files input/page.rs:536, parse_page input/page.rs:557,<br/>Vault::get input/page.rs:454, has_secret input/secrets.rs:191, iri_for input/slug.rs:88"]
        F2["no production caller: whelk_inconsistent input/consistency.rs:144,<br/>PromotionState input/promotion.rs:30, publish::stage ontomark-compile/src/build/publish.rs:185"]
        F3["test-only pub: version_iri turtle.rs:645, reason_with reason.rs:604, page_api::build page_api.rs:55"]
        F4["constant or unreachable: Census.rejected ir/projection.rs:142,<br/>Report.unsatisfiable ontomark-compile/src/build/mod.rs:103-104"]
        F5["DOC-DRIFT: gate FIRST ontomark-compile/src/build/mod.rs:315; atomic promote ontomark-compile/src/build/mod.rs:12;<br/>one shared reader input/page.rs:1; --stats prints counts ontomark-cli/src/main.rs:189"]
        F6["latent: unsatisfiable_message ignores refusals<br/>ontomark-compile/src/build/mod.rs:652-660"]
        F7["no production caller: Vault::get_mut input/page.rs:459, all_pages input/page.rs:468, public input/page.rs:473;<br/>let _ = line input/frontmatter.rs:135"]
        F8["unreachable: generic-profile auto-detect needs ontology/vocabulary.yaml<br/>ontomark-cli/src/main.rs:265-268"]
        F9["DOC-DRIFT: projections are knowledge's by definition ontomark-cli/src/main.rs:391-392, yet --vault working builds them from working/;<br/>Census error documented ir/projection.rs:168 but io::Error returned ir/projection.rs:169"]
        F10["DOC-DRIFT: emitted but not in C3: api/pages/_domain-index.json ontomark-compile/src/build/mod.rs:440,<br/>context/v1.jsonld ontomark-compile/src/build/mod.rs:453, data/.generation.json ontomark-compile/src/build/mod.rs:607"]
    end
    SEV1 --> SEV2 --> SEV3 --> SEV4 --> SEV5
    N1["DIVERGENCE: refusal-only reads 0 unsatisfiable is NOT reachable from run(): reason() wraps every path in checked,<br/>refusals become ReasoningError::UnsupportedProfile and exit 2; overrules OM-03 finding 3, demoted to SEV-5<br/>reason.rs:571, reason.rs:587-597, ontomark-cli/src/main.rs:441"]
    N2["DIVERGENCE: S1.1 is reachable only when the escaped-literal alternation exceeds the regex size limit;<br/>the Phase 3 repro must measure the private-title count that trips it<br/>ir/projection.rs:231-232"]
    N3["INVARIANT: S1.3 never leaks through build: the census at ontomark-compile/src/build/mod.rs:309 runs before staging,<br/>but publish::stage and migrate trust the lenient gate<br/>ontomark-compile/src/build/publish.rs:185, migrate/compile.rs:101"]
    F6 -.- N1
    A1 -.- N2
    A3 -.- N3
```
- DIVERGENCE: S2.1 conforms to the crate doc, which lists only credential, collision and reasoner-cap refusals as exit 2 (`ontomark-cli/src/main.rs:6-9`); making it 2 is a contract decision recorded in an ADR, not a bug fix.

## OM-00.3 Revert, reconcile or keep — origin of each SEV-1..3 divergence

```mermaid
flowchart LR
    VC["VisionClaw crates/vault, the legacy producer,<br/>relocated by 2bb20ec WIP relocate legacy pipeline into ontomark-compile<br/>and d7863ff Relocate the vault pipeline: byte parity"]
    S11["S1.1 fail-open redactor<br/>INHERITED vault/src/projection.rs:229"]
    S12["S1.2 body-only scan, frontmatter published<br/>INHERITED vault/src/build/publish.rs:159-160, vault/src/build/publish.rs:200;<br/>d5c1188 EXP-04 wrapped link redaction round render, not the scan<br/>ontomark-compile/src/build/publish.rs:162-168"]
    S13["S1.3 lenient vs strict public<br/>INHERITED vault-core/src/page.rs:211-213, vault/src/projection.rs:112<br/>lenient pinned as expected by standards.md:179"]
    S14["S1.4 records.jsonl absent<br/>VisionClaw has --with-rvdb vault/src/main.rs:389-392; git log --all -S with_rvdb finds no ontomark commit, never ported;<br/>deferral DEPRECATION.md:145; C3 line added by VisionFlow 6443781 sovereign-corpus-contracts.md:115"]
    S15["S1.5 gate/conflicts first vault only<br/>INHERITED vault/src/main.rs:855-858 via d7863ff"]
    S21["S2.1 unsatisfiable exit 1<br/>INHERITED vault/src/build/mod.rs:348 via 2bb20ec"]
    S22["S2.2 narrow digest<br/>INHERITED vault/src/build/mod.rs:514 via 2bb20ec"]
    S23["S2.3 non-atomic publish_out<br/>INHERITED vault/src/build/publish.rs:288"]
    S24["S2.4 free-text --tier/--severity<br/>INHERITED vault/src/gate.rs:42, vault/src/conflicts.rs:48 via 2bb20ec"]
    S31["S3.1 NGG1 relation collapse<br/>INHERITED vault/src/build/graph_tiers.rs:426-428;<br/>format published as ontomark-ngg1 0.1.0, 6c71ebc"]
    S32["S3.2 silent symlink skip<br/>INHERITED vault-core/src/page.rs:501-503 via 2bb20ec"]
    S33["S3.3 unclosed fence<br/>INHERITED vault-core/src/frontmatter.rs:153-157 via 2bb20ec"]
    S34["S3.4 compat path without held_back<br/>d5c1188 EXP-04 added the compat branch to keep legacy bytes; e96f726 merge put held_back on the redacted path only<br/>ontomark-compile/src/build/mod.rs:368-369, ir/projection.rs:467"]
    VC --> S11 & S12 & S13 & S14 & S15 & S21 & S22 & S23 & S24 & S31 & S32 & S33
    S11 --> RC1["RECONCILE: ProseRedactor::new returns Result, build refuses on failure (fail closed)<br/>ir/projection.rs:232"]
    S12 --> RC2["RECONCILE: scan the rendered page bytes, frontmatter + body, once in the census<br/>ontomark-compile/src/build/publish.rs:164, check/validate.rs:305"]
    S13 --> RC3["RECONCILE: strict rule canonical per its own comment ir/projection.rs:110-111;<br/>validate reports the string flag; rewrite standards.md:179 C03 row"]
    S14 --> RC4["RECONCILE contract first: C3 marks records.jsonl VisionClaw-only until the optional<br/>capability of DEPRECATION.md:145 lands; never fake an empty file"]
    S15 --> RC5["RECONCILE: loop every selected vault as validate does<br/>ontomark-cli/src/main.rs:304"]
    S21 --> RC6["RECONCILE by ADR: typed Unsatisfiable error, exit 2 beside ReasoningError,<br/>then amend ontomark-cli/src/main.rs:6-9"]
    S22 --> RC7["RECONCILE: digest the bytes parsed at input/page.rs:413 for every vault + vocabulary; read error refuses"]
    S23 --> RC8["RECONCILE: stage publish_out as a sibling and swap; promote doc says two renames<br/>ontomark-compile/src/build/publish.rs:316"]
    S24 --> RC9["RECONCILE: clap value_parser as VAULT_VALUES does<br/>ontomark-cli/src/main.rs:131"]
    S31 --> KP1["KEEP in NGG1 v1, DIVERGENCE recorded: EDGE_RELATION ontomark-ngg1/src/lib.rs:144 is a published<br/>wire value; per-relation edge types need an NGG2 format bump; TTL keeps them turtle.rs:337-341"]
    S32 --> RC10["RECONCILE: SkipReason for symlinks, honouring input/page.rs:79-81"]
    S33 --> RC11["RECONCILE: unclosed fence is an invalid_input census count, not a private page"]
    S34 --> KP2["KEEP: the compat flag is the parity comparison mechanism, docs/HANDOFF.md:28-29;<br/>default build takes the redacted path ontomark-compile/src/build/mod.rs:366"]
    NOREV["DIVERGENCE: no SEV-1..3 item is a regression of an earlier ontomark copy, so REVERT applies to none;<br/>reverting 2bb20ec or d7863ff would remove the whole port"]
    RC1 -.- NOREV
    EXT["EXTERNAL: VisionClaw's vault build job ships S1.1, S1.2, S1.5, S2.3 and S3.1 unchanged<br/>until VC-60 retires under its four gates<br/>ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:11"]
    VC -.- EXT
```
- INVARIANT: the closure-vs-Whelk split is KEEP, pinned by `the_closure_and_the_reasoner_agree_on_the_taxonomy` (`ontomark-compile/src/build/mod.rs:1254`); the NGG1 category ancestry walk at `ontomark-compile/src/build/graph_tiers.rs:154` has no such test.
- DIVERGENCE: `reason.rs:79` gained `refusals.is_empty()` in 6c040a5 together with `checked` (`reason.rs:590-597`); reverting it would let refusals pass, so it stays.

## OM-00.5 Resolution at ontomark 8d654d1 (2026-10-12)

```mermaid
flowchart TB
    PH3["Phase 3: 19 red reproduction tests, commit 93c78a8<br/>register_sev1/2/3.rs and register_sev1_cli/2_cli.rs;<br/>two passing controls; S1.1 driven through the redactor API"]
    PH4["Phase 4: one fix commit 8d654d1<br/>829 tests pass; clippy, strict rustdoc, wasm32, fmt clean"]
    PH3 --> PH4
    subgraph SEV1["SEV-1 publication boundary"]
        direction TB
        F11["S1.1 FIXED: Redactor::new fallible, RegexBuilder size limit;<br/>PublicationBlocked::Redactor refuses;<br/>ProseRedactor::{new,with_size_limit} public"]
        F12["S1.2 FIXED: rendered frontmatter scanned beside the body<br/>in the publish gate and in validate (SECRET_DETECTED)"]
        F13["S1.3 FIXED: Frontmatter::boolean and Page::is_public take a YAML bool only;<br/>INVALID_PUBLIC_FLAG; SHACL emits a quoted true as a plain literal;<br/>conformance row C03 rewritten"]
        F14["S1.4 FIXED as an emitter: build::records ports VisionClaw rvdb.rs (pure);<br/>CLI embed.rs holds the Xinference transport<br/>behind --with-rvdb / --embed-endpoint; build::run_with(embedder)"]
        F15["S1.5 FIXED: gate and conflicts iterate every selected vault;<br/>one vault = legacy bytes, all = per-vault + JSON map;<br/>golden gate-full re-captured: it had recorded the bug"]
        F11 ~~~ F12 ~~~ F13 ~~~ F14 ~~~ F15
    end
    subgraph SEV2["SEV-2 contract and exit codes"]
        direction TB
        F21["S2.1 FIXED, CONTRACT CHANGED: typed build::Unsatisfiable exits 2<br/>(same refusal class as OutputCollision); CLI crate doc amended;<br/>recorded in the ADR-OM-005 amendment"]
        F22["S2.2 FIXED: content_digest_of_hashes over Page::source_sha256<br/>for every loaded vault, journals, sibling vault and vocabulary;<br/>no re-read; legacy value not reproducible"]
        F23["S2.3 FIXED: publish::write_to stages beside the target<br/>and swaps via the shared swap_in; promote doc no longer<br/>claims atomicity (two renames, restore on failure)"]
        F24["S2.4 FIXED: --tier and --severity are clap value_parsers;<br/>unknown values exit 2"]
        F25["S2.5 FIXED, WORDING CORRECTED: root marker lists data/.generation.json;<br/>--publish-out files recorded as external artefacts;<br/>RULE: a marker never lists itself"]
        F21 ~~~ F22 ~~~ F23 ~~~ F24 ~~~ F25
    end
    subgraph SEV3["SEV-3, structure, parity"]
        direction TB
        F32["S3.2 FIXED: SkipReason::NotARegularFile;<br/>census counts excluded_not_regular_file"]
        F33["S3.3 FIXED, WORDING CORRECTED: Split::unterminated /<br/>Page::unterminated_frontmatter; census refuses as invalid_input;<br/>validate reports UNCLOSED_FRONTMATTER (was counted excluded)"]
        ST1["Step 1 DONE: run() walks pages/ once; inspect_walk,<br/>Vault::load_with_walk and excluded_identities_of share the listing;<br/>sibling_working folded; one vault_kinds for every verb"]
        KP["KEEP unchanged: S3.1 NGG1 relation byte (NGG2 item);<br/>S3.4 compat projection (the parity mechanism)"]
        PAR["Parity: estate-small golden byte-identical except the two<br/>generation markers, compared structurally in conformance.rs<br/>(identity fields equal; every legacy artefact present, same hash)"]
        WD["DIVERGENCE withdrawn: ADR-OM-005 decision 4 said the machine bundle<br/>is --vault all, unredacted; --vault all is the public publish value and<br/>redaction applies to every bundle. The mode needs its own flag: next item"]
        F32 ~~~ F33 ~~~ ST1 ~~~ KP ~~~ PAR ~~~ WD
    end
    PH4 --> SEV1 & SEV2 & SEV3
```
- Every citation in OM-00.1 to OM-00.4 is at `ffe06e4` and describes the state the fixes removed; this section carries commit shas only.
- Dead code removed with the fixes: `page_files`, `iri_for`, `has_secret`, `unsatisfiable_message`; docs corrected for `parse_page`, `excluded_identities` and `Census.rejected`.

## OM-00.4 Canonical run() after reconciliation

```mermaid
flowchart TD
    W["1 WALK once: walk_pages per selected vault, keep bytes in memory<br/>replaces ir/projection.rs:77, ir/projection.rs:172, ontomark-compile/src/build/mod.rs:311<br/>input/page.rs:492, input/page.rs:413"]
    V["2 VAULT SET: knowledge + optional working from one VaultSelector<br/>replaces ontomark-compile/src/build/publish.rs:336-353, ontomark-cli/src/main.rs:396-406"]
    CEN["3 CENSUS: strict public gate, one held_back, secrets over rendered frontmatter + body; refusal exits 2<br/>replaces ir/projection.rs:112-116, input/page.rs:211, ontomark-compile/src/build/publish.rs:224, check/validate.rs:305"]
    CB["4 Corpus::build ONCE on the link-cleaned vault<br/>replaces ontomark-compile/src/build/mod.rs:313, ir/projection.rs:460, ir/projection.rs:499"]
    VAL["5 VALIDATE incl. DUPLICATE_IRI<br/>ontomark-compile/src/build/mod.rs:356-362, check/validate.rs:561"]
    PR["6 PROJECT + REDACT once: one Resolver, fallible ProseRedactor, one redacted page view<br/>replaces ir/projection.rs:232, build/redact.rs:209, ir/projection.rs:497, ontomark-compile/src/build/mod.rs:516"]
    RS["7 REASON: closure + Whelk, typed Unsatisfiable refusal<br/>ontomark-compile/src/build/mod.rs:372-399"]
    EM["8 EMIT over a public-only view: one relation table, one domain resolver, one label rule<br/>ir/model.rs:456-460, ir/model.rs:36"]
    ST["9 STAGE write-once: bundle, publish/, markers, optional records.jsonl capability<br/>ontomark-compile/src/build/mod.rs:160-167, ontomark-compile/src/build/mod.rs:576"]
    DG["10 DIGEST over every byte read in step 1 + vocabulary, no re-read<br/>replaces ontomark-compile/src/build/mod.rs:578-586"]
    PM["11 PROMOTE atomic: out and publish_out staged as siblings, swapped<br/>replaces ontomark-compile/src/build/mod.rs:241-246, ontomark-compile/src/build/publish.rs:316"]
    RP["12 REPORT, without the dead unsatisfiable field<br/>ontomark-compile/src/build/mod.rs:615-645"]
    W --> V --> CEN --> CB --> VAL --> PR --> RS --> EM --> ST --> DG --> PM --> RP
    PRIV["FLAG private machine bundle, ADR-OM-005 decision 4: steps 1-2 select knowledge + working for the machine artefacts,<br/>step 6 skips redaction, steps 9-11 write a separate out, never publish/ or the site<br/>ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:13"]
    PRIV -.-> V
    PRIV -.-> PR
    PRIV -.-> ST
    NOW["DIVERGENCE: today --vault all only widens publish/ staging; every machine artefact is knowledge's<br/>ontomark-cli/src/main.rs:391-406; the private mode is a new flag on this shape, not the current run()"]
    PRIV -.- NOW
    NAME["DIVERGENCE: the ADR names the private mode --vault all, the value visionGraph's public publish passes today;<br/>implemented literally it would move the public site build onto the unredacted bundle, so the mode needs its own flag<br/>ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:13, ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:53"]
    PRIV -.- NAME
    I1["INVARIANT: zero differences over 32 artefact lines and 12 command outputs<br/>docs/HANDOFF.md:27-29"]
    I2["INVARIANT: write-once Staged refuses a second claim before stamp or promote<br/>ontomark-compile/src/build/mod.rs:160-162, ontomark-compile/src/build/mod.rs:609"]
    I3["INVARIANT: DUPLICATE_IRI refusal stays an error<br/>check/validate.rs:561"]
    I4["INVARIANT: structural closure == Whelk taxonomy<br/>ontomark-compile/src/build/mod.rs:1254"]
    ST -.- I2
    VAL -.- I3
    RS -.- I4
    RP -.- I1
```
- INVARIANT: the parity receipt is over the complete committed corpus through the compat flag (`docs/HANDOFF.md:27-29`), so S3.4 stays byte-identical; a strict validate cannot move it, because the census at `ontomark-compile/src/build/mod.rs:309` already runs on every build, compat included, so a corpus with a passing receipt holds only real booleans.
- INVARIANT: the census refusal must keep exit 2 and stage nothing (`ontomark-compile/src/build/publish.rs:246-248`, `ontomark-cli/src/main.rs:429-431`).
