---
id: OM-05
title: ontomark CLI — commands, vault selector, exit codes and consumers
area: ontomark
governing:
  - docs/engineering/sovereign-corpus-contracts.md
adrs: [ADR-OM-001, ADR-OM-005, ADR-2137]
sources:
  - ../ontomark/crates/ontomark-cli/src/main.rs
  - ../ontomark/crates/ontomark-cli/src/migrate_cmd.rs
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/build/publish.rs
  - ../ontomark/crates/ontomark-compile/src/reason.rs
  - ../ontomark/crates/ontomark-compile/src/check/gate.rs
  - ../ontomark/crates/ontomark-compile/src/check/conflicts.rs
  - ../ontomark/crates/ontomark-compile/src/check/repair.rs
  - ../ontomark/crates/ontomark-compile/src/input/vocabulary.rs
  - ../ontomark/crates/ontomark-compile/src/input/page.rs
  - ../ontomark/crates/ontomark-compile/src/profile.rs
  - ../ontomark/profiles/estate/profile.yaml
  - ../ontomark/docs/adr/ADR-OM-005-estate-alignment-one-producer-explicit-projections.md
  - ../project/crates/vault/src/main.rs
  - ../visionGraph/.github/workflows/publish.yml
  - ../project/.github/workflows/ontology-publish.yml
  - docs/engineering/sovereign-corpus-contracts.md
verified_commit: {ontomark: ee9e04a25715e70d3de211c17a012e878d14f339, visiongraph: 360bc9909aa1ff6ff8b249920c90d1ea90719953, visionclaw: 4806604edacf68179877e8db995299dceba46e75, visionflow: fa374b0282fe80849bc1fd4691f23aed3532fad7}
---

## OM-05.1 Command tree — six verbs, three global flags, and which vault each one reads

```mermaid
flowchart TD
    CLI["ontomark<br/>global --repo --json --profile<br/>ontomark-cli/src/main.rs:47-65"]
    VAL["validate<br/>VaultSelector default knowledge, --strict<br/>loops over EVERY selected vault<br/>ontomark-cli/src/main.rs:165-172,338"]
    GATE["gate<br/>VaultSelector default knowledge, --tier quick or full via value_parser<br/>gates EVERY selected vault, exit is the worst verdict<br/>ontomark-cli/src/main.rs:174-181,374"]
    CONF["conflicts<br/>VaultSelector default knowledge, --severity via value_parser<br/>analyses EVERY selected vault, exit is the worst report<br/>ontomark-cli/src/main.rs:183-191,433"]
    BUILD["build<br/>--out required, --vault default knowledge, --scope public or machine<br/>one projection vault plus optional working publish, see OM-06.1<br/>ontomark-cli/src/main.rs:195-213,479-505"]
    REPAIR["repair<br/>subcommands fences and bodies<br/>--vault default all<br/>ontomark-cli/src/main.rs:77-79,88-95,117"]
    MIG["migrate<br/>list plan apply rollback status<br/>plan --vault default all<br/>ontomark-cli/src/main.rs:80-85, migrate_cmd.rs:22-57"]
    CLI --> VAL
    CLI --> GATE
    CLI --> CONF
    CLI --> BUILD
    CLI --> REPAIR
    CLI --> MIG
    n1["INVARIANT: every vault flag is validated by value_parser VAULT_VALUES, so<br/>an unknown value is a clap error rather than a silent knowledge,<br/>ontomark-cli/src/main.rs:130, migrate_cmd.rs:59-62"]
    n2["INVARIANT: all means BOTH for every verb: the vault_kinds doc says so and gate and<br/>conflicts now loop over every selected vault like validate always did —<br/>migrate_cmd.rs:64-68, ontomark-cli/src/main.rs:338,374,433"]
    n3["INVARIANT: --tier and --severity are clap value_parsers, so an unknown value is a clap<br/>error and exit 2 as the crate doc promises; the lenient parse fallbacks are unreachable —<br/>ontomark-cli/src/main.rs:161-163,179,189, gate.rs:42-48, conflicts.rs:48-55"]
    GATE --- n2
    CONF --- n3
    CLI --- n1
```
- `repair` defaults to `all` while `validate`, `gate`, `conflicts` and `build` default to `knowledge`, so the same word means a different scope per verb (`ontomark-cli/src/main.rs:100,117,130,204`).
- DIVERGENCE: `--stats` is documented as "print the counts" but only forces the JSON report, a build-only alias for `--json` (`ontomark-cli/src/main.rs:214-216,556`).
- DIVERGENCE: global `--profile` is resolved for every verb (`ontomark-cli/src/main.rs:331`) but `conflicts`, `repair bodies` and `repair fences` never consume it (`ontomark-cli/src/main.rs:437,585,629`).

## OM-05.2 `--vault knowledge|working|all` — resolution into vault_root and with_working_publish

```mermaid
flowchart TD
    FLAG["--vault string<br/>parser accepts knowledge, working, all only<br/>ontomark-cli/src/main.rs:130,204"]
    VK["vault_kinds value, the one parser for every verb<br/>trim + lowercase, then match<br/>ontomark-cli/src/migrate_cmd.rs:72"]
    W["working -> Working only<br/>ontomark-cli/src/migrate_cmd.rs:74"]
    A["all -> Knowledge AND Working<br/>ontomark-cli/src/migrate_cmd.rs:75"]
    K["anything else -> Knowledge<br/>unreachable after the parser, ontomark-cli/src/migrate_cmd.rs:76"]
    KIND["kind = Working only when kinds == Working and scope is not machine,<br/>else Knowledge<br/>ontomark-cli/src/main.rs:483-488"]
    ROOT["vault_root = repo root joined with kind.dir_name()<br/>ontomark-cli/src/main.rs:493, page.rs:118-123"]
    WWP["with_working_publish =<br/>kinds contains Working AND kind is Knowledge<br/>true ONLY for all<br/>ontomark-cli/src/main.rs:490-491"]
    PUB["build: sibling_vault(Working) loaded and added to publish_vaults<br/>build/mod.rs:545-551"]
    MS["scope machine reads knowledge and working itself and refuses vault working or all, see OM-06.1<br/>ontomark-cli/src/main.rs:206-213,480-483"]
    ART["ontology projections stay on vault_root alone:<br/>OWL membership is not a per-page decision<br/>ontomark-cli/src/main.rs:474-478, build/mod.rs:76-77"]
    FLAG --> VK
    VK --> W
    VK --> A
    VK --> K
    W --> KIND
    A --> KIND
    K --> KIND
    KIND --> ROOT
    KIND --> WWP
    WWP --> PUB
    ROOT --> ART
    n1["INVARIANT: all widens ONLY publish/ staging; machine artefacts<br/>stay knowledge-only — ontomark-cli/src/main.rs:201-203, build/mod.rs:76-78"]
    n2["INVARIANT: working alone sets vault_root to working/ and<br/>with_working_publish false — ontomark-cli/src/main.rs:484-491"]
    PUB --- n1
    ROOT --- n2
    KIND -.-> MS
```
- Every verb, `migrate` included, reads `--vault` through the one `vault_kinds` (`migrate_cmd.rs:72-83`): `validate`, `gate` and `conflicts` loop over every kind via `load` (`ontomark-cli/src/main.rs:306-314,338,374,433`), `repair` maps them to directory names for `scopes_for` (`main.rs:581-586,625-630`, `repair.rs:424-435`).
- DOC-DRIFT: comments say the projections are `knowledge/`'s by definition (`ontomark-cli/src/main.rs:474-475`, `build/mod.rs:76-77`), yet `--vault working` roots the build at `working/` (`ontomark-cli/src/main.rs:484-488`).

## OM-05.3 Exit codes — which error type maps to 0, 1 or 2

```mermaid
flowchart TD
    MAIN["main: run Ok(code) returns code;<br/>Err prints ontomark: and exits 1<br/>ontomark-cli/src/main.rs:259-267"]
    ZERO["0 — success<br/>ontomark-cli/src/main.rs:577,621,671"]
    ONE["1 — failed check or execution error"]
    TWO["2 — refused operation or invalid argument<br/>ontomark-cli/src/main.rs:5-13"]
    V1["validate: any error, or warning with --strict<br/>ontomark-cli/src/main.rs:341-342,363"]
    G1["gate: worst verdict.exit_code 0 or 1 over the selected vaults<br/>gate.rs:110-112, ontomark-cli/src/main.rs:389-393,426"]
    C1["conflicts: worst report.exit_code 0 or 1 over the selected vaults<br/>conflicts.rs:119-121, ontomark-cli/src/main.rs:440-444,470"]
    B1["build: written == 0 is an anyhow error<br/>ontomark-cli/src/main.rs:572-576"]
    R1["repair bodies --check with residue<br/>ExitCode::FAILURE, ontomark-cli/src/main.rs:618-620"]
    ANY["any other Err from build::run_with, load, resolve_repo<br/>ontomark-cli/src/main.rs:552, 330-331"]
    S2["SecretsFound — credential on a published page<br/>build/publish.rs:66, ontomark-cli/src/main.rs:524-527"]
    O2["OutputCollision — two artefacts claim one path<br/>build/mod.rs:156, ontomark-cli/src/main.rs:530-533"]
    R2["ReasoningError — definition reasoner would not finish<br/>reason.rs:481, ontomark-cli/src/main.rs:536-539"]
    SC2["ScopeConflict or Unsatisfiable — machine scope misuse, or an unsatisfiable class<br/>build/mod.rs:232,185<br/>ontomark-cli/src/main.rs:542-551"]
    CL["clap parse failure: bad flag, or a vault, tier,<br/>severity or scope value outside its list<br/>clap default exit 2, ontomark-cli/src/main.rs:130,179,189,212"]
    M2["migrate plan not Ready -> 2<br/>migrate_cmd.rs:152-157"]
    M1["migrate apply incomplete, rollback with conflicts -> 1<br/>migrate_cmd.rs:186,212"]
    MAIN --> ZERO
    MAIN --> ONE
    MAIN --> TWO
    V1 --> ONE
    G1 --> ONE
    C1 --> ONE
    B1 --> ONE
    R1 --> ONE
    ANY --> ONE
    M1 --> ONE
    S2 --> TWO
    O2 --> TWO
    R2 --> TWO
    SC2 --> TWO
    CL --> TWO
    M2 --> TWO
    n1["INVARIANT: only build and migrate plan return 2 from code; the five build<br/>refusals are downcast_ref matches on the anyhow error, ontomark-cli/src/main.rs:524-551"]
    n2["DIVERGENCE: validate, gate and conflicts never return 2 from their own<br/>code (only clap does), so refusal is distinguishable only on build —<br/>ontomark-cli/src/main.rs:363,426,470"]
    TWO --- n1
    ONE --- n2
```
- `SecretsFound` is checked before validation on purpose so a credential is a code-2 refusal and never swallowed by the generic code-1 validation message (`build/mod.rs:522-532`).
- The `0/1/2` contract is declared in the crate doc, with `1` also covering execution errors, and the doc lists the unsatisfiable class and the machine-scope refusals under 2 (`ontomark-cli/src/main.rs:5-13`).
- An unsatisfiable corpus exits 2 through the typed `build::Unsatisfiable`; before the OM-00 reconcile it was an untyped error at exit 1 (`build/mod.rs:642-648`, `ontomark-cli/src/main.rs:548-551`).

## OM-05.4 Repo-root discovery and profile resolution — before any verb runs

```mermaid
flowchart TD
    START["run: Cli::parse<br/>ontomark-cli/src/main.rs:329"]
    EXPL{"--repo given?<br/>ontomark-cli/src/main.rs:271-274"}
    P1["start = the explicit path"]
    P2["start = current_dir"]
    DISC["Vocabulary::discover start<br/>walk start and every ancestor for<br/>ontology/vocabulary.yaml<br/>vocabulary.rs:470-480"]
    NOTV["Err NotAVault no vocabulary in this directory or any ancestor<br/>vocabulary.rs:481-484"]
    ROOT["root = vocabulary path .parent().parent()<br/>the directory ABOVE ontology/<br/>ontomark-cli/src/main.rs:281-286"]
    PROF{"--profile?<br/>ontomark-cli/src/main.rs:290-303"}
    EST["estate -> ontomark_profile_estate<br/>ontomark-cli/src/main.rs:292"]
    GEN["generic -> Profile::generic<br/>ontomark-cli/src/main.rs:293"]
    YAML["other string -> read and parse YAML file<br/>ontomark-cli/src/main.rs:294-298"]
    AUTO["unset: estate if root has ontology/vocabulary.yaml, else generic<br/>ontomark-cli/src/main.rs:299-302"]
    LOAD["load: root joined with kind.dir_name(), load_vault each<br/>ontomark-cli/src/main.rs:306-314"]
    START --> EXPL
    EXPL -->|yes| P1
    EXPL -->|no| P2
    P1 --> DISC
    P2 --> DISC
    DISC -->|found| ROOT
    DISC -->|none| NOTV
    ROOT --> PROF
    PROF --> EST
    PROF --> GEN
    PROF --> YAML
    PROF --> AUTO
    ROOT --> LOAD
    n1["INVARIANT: discovery is mandatory for every verb including migrate,<br/>so a repo without ontology/vocabulary.yaml cannot run at all<br/>ontomark-cli/src/main.rs:330 — then exit 1 via ontomark-cli/src/main.rs:262-265"]
    n2["DIVERGENCE: the generic profile branch at ontomark-cli/src/main.rs:302 is unreachable via<br/>auto-detection, because discovery already required vocabulary.yaml;<br/>only an explicit --profile generic reaches it — ontomark-cli/src/main.rs:299-302,330"]
    NOTV --- n1
    AUTO --- n2
```
- The `--repo` argument is the START of an upward search, not necessarily the root, so `--repo a/b/c` can resolve to an ancestor (`ontomark-cli/src/main.rs:271-275`).

## OM-05.5 Two consumers — visionGraph publishes both vaults from a pinned release, VisionClaw builds knowledge in-tree

```mermaid
flowchart LR
    subgraph VG["visionGraph publish.yml — narrativegoldmine.com"]
        direction TB
        VGPIN["Pinned release v0.1.0-rc.1 AND sha256 digest<br/>no build-from-source fallback<br/>publish.yml:42-51"]
        VGFETCH["gh release download, sha256sum --check --strict,<br/>install as ontomark<br/>publish.yml:106-115"]
        VGVAL["ontomark validate --vault all<br/>publish.yml:130"]
        VGBUILD["ontomark build --vault all --out site-data<br/>--publish-out quartz/content --with-markdown-mirror --stats<br/>publish.yml:158-163"]
        VGREP["ontomark repair bodies --vault all --check<br/>warning only<br/>publish.yml:196"]
        VGPIN --> VGFETCH --> VGVAL --> VGBUILD --> VGREP
    end
    subgraph VC["VisionClaw ontology-publish.yml — Pod resources"]
        direction TB
        VCCACHE["Cache Cargo registry and target<br/>ontology-publish.yml:156-165"]
        VCBUILD["cargo build --release --locked -p vault<br/>in-tree crate, NOT ontomark<br/>ontology-publish.yml:168"]
        VCRUN["vault --repo vault-source build --vault knowledge<br/>--out output/vault — no publish-out, no mirror<br/>ontology-publish.yml:172-174"]
        VCCACHE --> VCBUILD --> VCRUN
    end
    BIN["ontomark build — ontomark-cli/src/main.rs:473-518"]
    VGBUILD -->|"vault_kinds all -> with_working_publish true"| BIN
    VCRUN -.->|"different binary"| BIN
    n1["EXTERNAL: the two consumers differ — visionGraph uses ontomark with<br/>--vault all, VisionClaw still uses its own vault crate with<br/>--vault knowledge; ADR-OM-005 records this and routes VisionClaw to build --scope machine, see OM-06.1,<br/>publish.yml:158-159, ontology-publish.yml:168-173, ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:13"]
    n2["EXTERNAL: this VisionClaw workflow never invokes ontomark; vault build output<br/>feeds pack-pod-resources.py, ontology-publish.yml:194. ADR-OM-005 decision 2<br/>retires this job, gated, so it keeps running — ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:11"]
    n3["INVARIANT: visionGraph treats the staging tree as wrong unless both<br/>pages/ and working/ exist, because ontomark silently accepts a<br/>knowledge-only build — publish.yml:180-186"]
    VG --- n1
    VC --- n2
    VGBUILD --- n3
```
- Redaction is not a flag in either workflow: visionGraph gets it from the estate profile (`estate/profile.yaml:198`, `build/mod.rs:556-558`) and never passes `--no-redact-private-links`; the legacy `vault` build kept private links (`ontomark-cli/src/main.rs:249-254`).
- The consumer does not pass `--generated-at`, `--source-commit` or `--stale-after`, so `.generation.json` takes the clock and git (`build/mod.rs:63-67`, `publish.yml:158-163`).

## OM-05.6 C2 contract versus the shipped CLI — what C2 lists that ontomark lacks and the reverse

```mermaid
flowchart LR
    subgraph C2["C2 vault CLI — sovereign-corpus-contracts.md"]
        direction TB
        C2V["validate --vault --strict<br/>sovereign-corpus-contracts.md:88"]
        C2G["gate --tier<br/>sovereign-corpus-contracts.md:99"]
        C2C["conflicts --severity high|all<br/>sovereign-corpus-contracts.md:100"]
        C2B["build --out --vault --stats --with-rvdb<br/>sovereign-corpus-contracts.md:101-102"]
        C2M["migrate --fences-to-properties<br/>sovereign-corpus-contracts.md:103-104"]
        C2Q["find retrieve tree edit create propose<br/>sovereign-corpus-contracts.md:89-98"]
    end
    subgraph OM["ontomark CLI — main.rs"]
        direction TB
        OMV["validate"]
        OMG["gate"]
        OMC["conflicts, severity also medium and low<br/>conflicts.rs:48-55"]
        OMB["build with --with-rvdb and --embed-endpoint<br/>plus --scope --publish-out --with-markdown-mirror<br/>--generated-at --source-commit --stale-after<br/>--no-redact-private-links<br/>ontomark-cli/src/main.rs:195-257"]
        OMM["migrate list plan apply rollback status<br/>migrate_cmd.rs:22-57"]
        OMR["repair fences, repair bodies — NOT in C2<br/>ontomark-cli/src/main.rs:88-95"]
        OMX["not shipped: the governance and query verbs stay in VisionClaw<br/>ontomark-cli/src/main.rs:21-23"]
    end
    C2V --- OMV
    C2G --- OMG
    C2C --- OMC
    C2B --- OMB
    C2M -.-|"shape differs"| OMM
    C2Q -.-|"stay in VisionClaw"| OMX
    n1["DIVERGENCE: C2 lists find retrieve tree edit create propose; the ontomark binary has none of them,<br/>while build --with-rvdb now exists in it, see OM-03.7<br/>sovereign-corpus-contracts.md:89-98 vs ontomark-cli/src/main.rs:67-86,230-231"]
    n2["DIVERGENCE: C2 defines migrate --fences-to-properties; ontomark ships a<br/>registry-driven migrate with five subcommands and a repair verb C2 lacks<br/>sovereign-corpus-contracts.md:103 vs ontomark-cli/src/main.rs:80-95"]
    n3["INVARIANT: one vault parser in ontomark: VAULT_VALUES and vault_kinds in migrate_cmd.rs serve every<br/>verb, and its own kinds() only delegates; the VisionClaw vault crate keeps a separate copy<br/>migrate_cmd.rs:31,62,72-83 vs crates/vault/src/main.rs:155,165-168"]
    OM --- n1
    OM --- n2
    OM --- n3
```
- The missing C2 verbs are a deliberate subset: governance and graph queries stay in VisionClaw (`ontomark-cli/src/main.rs:21-23`).
- No `BuildArgs` option is parsed and left unused: all thirteen reach `build::Options`, the embedder or the report (`ontomark-cli/src/main.rs:195-257,489-512,556`).
- DIVERGENCE: `--no-redact-private-links` overlays the profile setting `publication.redact_private_links` as a `Some(false)` override rather than a separate behaviour (`ontomark-cli/src/main.rs:503`, `build/mod.rs:84`, `profile.rs:411`).
