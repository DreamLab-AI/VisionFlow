---
id: OM-05
title: ontomark CLI — commands, vault selector, layouts, exit codes and consumers
area: ontomark
governing:
  - docs/engineering/sovereign-corpus-contracts.md
adrs: [ADR-OM-001, ADR-OM-005, ADR-OM-007, ADR-OM-008, ADR-OM-009, ADR-2137, ADR-2147]
sources:
  - ../ontomark/crates/ontomark-cli/src/main.rs
  - ../ontomark/crates/ontomark-cli/src/migrate_cmd.rs
  - ../ontomark/docs/DEPRECATION.md
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
verified_commit: {ontomark: 13af42340448022d4d43ac349a8bb5f3ae6d1d47, visiongraph: fd5269a55e52b8b1d1e9d7d59cb5f0c7835a5a2e, visionclaw: e4ba5eb35ee6715cfd7f2daf1cd9af05e8a18cce, visionflow: ff15faf83d82fef7f23d96da381c8ab78d0206b0}
---

## OM-05.1 Command tree — ten verbs, repair and migrate, three global flags, and which vault each one reads

```mermaid
flowchart TD
    CLI["ontomark<br/>global --repo --json --profile<br/>ontomark-cli/src/main.rs:81-106"]
    VAL["validate<br/>VaultSelector default knowledge, --strict<br/>loops over EVERY selected vault<br/>ontomark-cli/src/main.rs:218-224,650"]
    GATE["gate<br/>VaultSelector default knowledge, --tier quick or full via value_parser<br/>gates EVERY selected vault, exit is the worst verdict<br/>ontomark-cli/src/main.rs:353-359,686"]
    CONF["conflicts<br/>VaultSelector default knowledge, --severity via value_parser<br/>analyses EVERY selected vault, exit is the worst report<br/>ontomark-cli/src/main.rs:362-369,745"]
    BUILD["build<br/>--out required, --vault default knowledge, --scope public or machine<br/>one projection vault plus optional working publish, see OM-06.1<br/>ontomark-cli/src/main.rs:373-442,785-897"]
    QRY["find, retrieve, tree<br/>VaultSelector default knowledge; find loops over every selected vault,<br/>retrieve and tree take the FIRST<br/>ontomark-cli/src/main.rs:227-273,899-959"]
    WR["edit, create, apply<br/>guarded writes, each refused without --expect; create and apply always load knowledge<br/>ontomark-cli/src/main.rs:275-351,961-1126"]
    REPAIR["repair<br/>subcommands fences and bodies<br/>--vault default all<br/>ontomark-cli/src/main.rs:141-175,1128-1214"]
    MIG["migrate<br/>list plan apply rollback status<br/>plan --vault default all<br/>ontomark-cli/src/main.rs:134-137, migrate_cmd.rs:22-57"]
    CLI --> VAL
    CLI --> GATE
    CLI --> CONF
    CLI --> BUILD
    CLI --> QRY
    CLI --> WR
    CLI --> REPAIR
    CLI --> MIG
    n1["INVARIANT: every vault flag is validated by value_parser VAULT_VALUES, so<br/>an unknown value is a clap error rather than a silent knowledge,<br/>ontomark-cli/src/main.rs:182, migrate_cmd.rs:59-62"]
    n2["INVARIANT: all means BOTH for validate, gate, conflicts, find and build: the vault_kinds doc says so and each<br/>loops over every selected vault —<br/>migrate_cmd.rs:64-68, ontomark-cli/src/main.rs:650,686,745,901<br/>DIVERGENCE: retrieve, tree and edit call .next() on the loaded vaults, so under all they read knowledge alone —<br/>ontomark-cli/src/main.rs:923-926,948-951,971-974"]
    n3["INVARIANT: --tier and --severity are clap value_parsers, so an unknown value is a clap<br/>error and exit 2 as the crate doc promises; the lenient parse fallbacks are unreachable —<br/>ontomark-cli/src/main.rs:213-215,357,367, gate.rs:42-48, conflicts.rs:48-55"]
    GATE --- n2
    CONF --- n3
    CLI --- n1
```
- `repair` defaults to `all` while `validate`, `gate`, `conflicts`, `build` and the query verbs default to `knowledge`, so the same word means a different scope per verb (`ontomark-cli/src/main.rs:152,169,182,382`).
- DIVERGENCE: `--stats` is documented as "print the counts" but only forces the JSON report, a build-only alias for `--json` (`ontomark-cli/src/main.rs:394,875`).
- INVARIANT: `--profile` reaches every verb through the loaded vaults' layout (`ontomark-cli/src/main.rs:548-558,561-575`), but the conflict analysis itself takes no profile (`ontomark-cli/src/main.rs:749`).

## OM-05.2 `--vault knowledge|working|all` — resolution into vault_root and with_working_publish

```mermaid
flowchart TD
    FLAG["--vault string<br/>parser accepts knowledge, working, all only<br/>ontomark-cli/src/main.rs:182,382"]
    VK["vault_kinds value, the one parser for every verb<br/>trim + lowercase, then match<br/>ontomark-cli/src/migrate_cmd.rs:72"]
    W["working -> Working only<br/>ontomark-cli/src/migrate_cmd.rs:74"]
    A["all -> Knowledge AND Working<br/>ontomark-cli/src/migrate_cmd.rs:75"]
    K["anything else -> Knowledge<br/>unreachable after the parser, ontomark-cli/src/migrate_cmd.rs:76"]
    KIND["kind = Working only when kinds == Working and scope is not machine,<br/>else Knowledge<br/>ontomark-cli/src/main.rs:795-800"]
    ROOT["vault_root = the profile layout places the vault under the repo root:<br/>root_of refuses a role the layout lacks or a missing folder<br/>ontomark-cli/src/main.rs:544-546, profile.rs:561-576"]
    MISS["MissingVault, a refused request: exit 2 from main<br/>ontomark-cli/src/main.rs:451-452, profile.rs:622"]
    WWP["with_working_publish =<br/>kinds contains Working AND kind is Knowledge<br/>true ONLY for all<br/>ontomark-cli/src/main.rs:802-803"]
    PUB["build: sibling_vault(Working) loaded and added to publish_vaults<br/>build/mod.rs:604-610"]
    MS["scope machine reads knowledge and working itself and refuses vault working or all, see OM-06.1<br/>ontomark-cli/src/main.rs:384-392, build/mod.rs:476-518"]
    ART["ontology projections stay on vault_root alone:<br/>OWL membership is not a per-page decision<br/>ontomark-cli/src/main.rs:786-790, build/mod.rs:76-77"]
    FLAG --> VK
    VK --> W
    VK --> A
    VK --> K
    W --> KIND
    A --> KIND
    K --> KIND
    KIND --> ROOT
    ROOT -.-> MISS
    KIND --> WWP
    WWP --> PUB
    ROOT --> ART
    n1["INVARIANT: all widens ONLY publish/ staging; machine artefacts<br/>stay knowledge-only — ontomark-cli/src/main.rs:377-381, build/mod.rs:76-78"]
    n2["INVARIANT: working alone sets vault_root to the working folder and<br/>with_working_publish false — ontomark-cli/src/main.rs:796-803"]
    PUB --- n1
    ROOT --- n2
    KIND -.-> MS
```
- Every verb, `migrate` included, reads `--vault` through the one `vault_kinds` (`migrate_cmd.rs:72-83`): `validate`, `gate`, `conflicts` and `find` loop over every kind via `load` (`ontomark-cli/src/main.rs:548-558,650,686,745,901`), `repair` maps them to layout-placed roots for `scopes_for` (`main.rs:561-575,1129,1170`, `repair.rs:444`).
- DOC-DRIFT: comments say the projections are `knowledge/`'s by definition (`ontomark-cli/src/main.rs:786-787`, `build/mod.rs:76-77`), yet `--vault working` roots the build at the working vault (`ontomark-cli/src/main.rs:796-800`).

## OM-05.3 Exit codes — which error type maps to 0, 1 or 2

```mermaid
flowchart LR
    MAIN["main: run Ok(code) returns code;<br/>Err prints ontomark: and exits 1, except MissingVault which exits 2<br/>ontomark-cli/src/main.rs:444-456"]
    ZERO["0 — success<br/>ontomark-cli/src/main.rs:896,918,1064,1125"]
    ONE["1 — failed check or execution error"]
    TWO["2 — refused operation or invalid argument<br/>ontomark-cli/src/main.rs:5-13,32-36"]
    V1["validate: any error, or warning with --strict<br/>ontomark-cli/src/main.rs:653-654,675"]
    G1["gate: worst verdict.exit_code 0 or 1 over the selected vaults<br/>gate.rs:110-112, ontomark-cli/src/main.rs:701-705,738"]
    C1["conflicts: worst report.exit_code 0 or 1 over the selected vaults<br/>conflicts.rs:119-121, ontomark-cli/src/main.rs:752-756,782"]
    B1["build: written == 0 is an anyhow<br/>error<br/>ontomark-cli/src/main.rs:891-895"]
    R1["repair bodies --check with residue<br/>ExitCode::FAILURE,<br/>ontomark-cli/src/main.rs:1163-1165"]
    ANY["any other Err from<br/>build::run_with, load,<br/>resolve_repo<br/>ontomark-cli/src/main.rs:871,<br/>641-642"]
    S2["SecretsFound — credential on a<br/>published page<br/>build/publish.rs:66,<br/>ontomark-cli/src/main.rs:837-840"]
    O2["OutputCollision — two artefacts<br/>claim one path build/mod.rs:206,<br/>ontomark-cli/src/main.rs:843-846"]
    R2["ReasoningError — definition<br/>reasoner would not finish<br/>reason.rs:481,<br/>ontomark-cli/src/main.rs:849-852"]
    SC2["ScopeConflict or Unsatisfiable —<br/>machine scope misuse, or an<br/>unsatisfiable class<br/>build/mod.rs:282,235<br/>ontomark-cli/src/main.rs:855-864"]
    CP2["CertificateParity — the derivation<br/>certificate does not cover the<br/>classifier's closure (ADR-OM-009)<br/>build/mod.rs:101,<br/>ontomark-cli/src/main.rs:867-870"]
    MV2["MissingVault — a vault role the<br/>layout does not place, for example<br/>vault working on a plain vault<br/>profile.rs:622,<br/>ontomark-cli/src/main.rs:451-452"]
    GW2["create or apply refused by its<br/>guards, nothing written<br/>ontomark-cli/src/main.rs:1010-1021,1070-1081"]
    CL["clap parse failure: bad flag, or a<br/>vault, tier, severity or scope<br/>value outside its list clap<br/>default exit 2,<br/>ontomark-cli/src/main.rs:182,357,367,213"]
    M2["migrate plan not Ready -> 2<br/>migrate_cmd.rs:152-157"]
    M1["migrate apply incomplete, rollback<br/>with conflicts -> 1<br/>migrate_cmd.rs:186,212"]
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
    CP2 --> TWO
    MV2 --> TWO
    GW2 --> TWO
    CL --> TWO
    M2 --> TWO
    n1["INVARIANT: build maps six typed errors to 2 by downcast_ref<br/>on the anyhow error, and create, apply, migrate plan and<br/>MissingVault refuse with 2 too,<br/>ontomark-cli/src/main.rs:837-870,1010-1021,1070-1081,451-452"]
    n2["DIVERGENCE: validate, gate and conflicts never return 2 from their<br/>own code (only clap and MissingVault do), so a refusal is<br/>distinguishable only on the writing verbs —<br/>ontomark-cli/src/main.rs:675,738,782"]
    TWO --- n1
    ONE --- n2
```
- `SecretsFound` is checked before validation on purpose so a credential is a code-2 refusal and never swallowed by the generic code-1 validation message (`build/mod.rs:579-589`).
- The `0/1/2` contract is declared in the crate doc, with `1` also covering execution errors, and the doc lists the unsatisfiable class, the machine-scope refusals, the layout refusals and the `create`/`apply` guards under 2 (`ontomark-cli/src/main.rs:5-13,32-36`).
- An unsatisfiable corpus exits 2 through the typed `build::Unsatisfiable`; before the OM-00 reconcile it was an untyped error at exit 1 (`build/mod.rs:743-749`, `ontomark-cli/src/main.rs:861-864`).

## OM-05.4 Repo-root discovery and profile resolution — before any verb runs

```mermaid
flowchart TD
    START["run: Cli::parse<br/>ontomark-cli/src/main.rs:640"]
    EXPL{"--repo given?<br/>ontomark-cli/src/main.rs:474-478"}
    P1["start = the explicit path"]
    P2["start = current_dir"]
    DISC["Vocabulary::discover start<br/>walk start and every ancestor for<br/>ontology/vocabulary.yaml<br/>vocabulary.rs:491-501"]
    ESTATE["found: root = vocabulary path .parent().parent()<br/>the directory ABOVE ontology/, estate_shape true<br/>ontomark-cli/src/main.rs:480-491"]
    PLAIN["NotAVault: start is a plain Obsidian vault and is the root,<br/>built-in vocabulary, estate_shape false; a start that is not a directory is an error<br/>ontomark-cli/src/main.rs:492-503, vocabulary.rs:502-505"]
    BADV["a vocabulary file that exists but does not parse is still an error, exit 1<br/>ontomark-cli/src/main.rs:504-511"]
    PROF{"--profile?<br/>ontomark-cli/src/main.rs:518-538"}
    EST["estate -> ontomark_profile_estate<br/>ontomark-cli/src/main.rs:520"]
    GEN["generic -> Profile::generic<br/>ontomark-cli/src/main.rs:521"]
    YAML["other string -> read and parse YAML file, layout kept as written<br/>ontomark-cli/src/main.rs:522-526"]
    AUTO["unset: estate if estate_shape, else generic<br/>ontomark-cli/src/main.rs:527-528"]
    FLAT["built-in profile on a plain vault takes Layout::flat:<br/>knowledge = the root, no working vault, layout.excluded kept<br/>ontomark-cli/src/main.rs:530-536, profile.rs:514-520"]
    LOAD["load: layout.root_of per kind, load_excluding each<br/>ontomark-cli/src/main.rs:548-558"]
    START --> EXPL
    EXPL -->|yes| P1
    EXPL -->|no| P2
    P1 --> DISC
    P2 --> DISC
    DISC -->|found| ESTATE
    DISC -->|none| PLAIN
    DISC -.->|parse error| BADV
    ESTATE --> PROF
    PLAIN --> PROF
    PROF --> EST
    PROF --> GEN
    PROF --> YAML
    PROF --> AUTO
    PLAIN --> FLAT
    ESTATE --> LOAD
    FLAT --> LOAD
    n1["INVARIANT: discovery no longer needs a vocabulary file: a directory with none above it is a plain<br/>Obsidian vault (ADR-OM-008), and vault working or all, or scope machine, are then refused with exit 2 naming the layout —<br/>ontomark-cli/src/main.rs:492-503,451-452, profile.rs:561-576"]
    n2["INVARIANT: the generic profile is reachable by auto-detection, because only a plain vault<br/>lacks the vocabulary file; an explicit --profile estate on a plain vault still gets the flat layout —<br/>ontomark-cli/src/main.rs:528-536"]
    PLAIN --- n1
    AUTO --- n2
```
- The `--repo` argument is the START of an upward search, not necessarily the root, so `--repo a/b/c` can resolve to an ancestor holding `ontology/vocabulary.yaml` (`ontomark-cli/src/main.rs:474-503`).

## OM-05.5 Two consumers — visionGraph publishes both vaults from a pinned release, VisionClaw builds knowledge in-tree

```mermaid
flowchart LR
    subgraph VG["visionGraph publish.yml — narrativegoldmine.com"]
        direction TB
        VGPIN["Pinned release v0.6.0 AND sha256 digest<br/>no build-from-source fallback<br/>publish.yml:48-51"]
        VGFETCH["gh release download, sha256sum --check --strict,<br/>install as ontomark<br/>publish.yml:107-112"]
        VGVAL["ontomark validate --vault all<br/>publish.yml:130"]
        VGBUILD["ontomark build --vault all --out site-data<br/>--publish-out quartz/content --with-markdown-mirror<br/>--certificate-out --stats<br/>publish.yml:165-171"]
        VGCERT["oo-cert v2.0.1 re-derives the certificate from its premises, digest-pinned<br/>publish.yml:210-227"]
        VGREP["ontomark repair bodies --vault all --check<br/>warning only<br/>publish.yml:239"]
        VGPIN --> VGFETCH --> VGVAL --> VGBUILD --> VGCERT --> VGREP
    end
    subgraph VC["VisionClaw ontology-publish.yml — Pod resources"]
        direction TB
        VCCACHE["Cache Cargo registry and target<br/>ontology-publish.yml:156-165"]
        VCBUILD["cargo build --release --locked -p vault<br/>in-tree crate, NOT ontomark<br/>ontology-publish.yml:167-168"]
        VCRUN["vault --repo vault-source build --vault knowledge<br/>--out output/vault — no publish-out, no mirror<br/>ontology-publish.yml:172-175"]
        VCCERT["vault certify then the same oo-cert re-check (ADR-2147)<br/>ontology-publish.yml:180-206"]
        VCCACHE --> VCBUILD --> VCRUN --> VCCERT
    end
    BIN["ontomark build — ontomark-cli/src/main.rs:785-897"]
    VGBUILD -->|"vault_kinds all -> with_working_publish true"| BIN
    VCRUN -.->|"different binary"| BIN
    n1["EXTERNAL: the two consumers differ — visionGraph uses ontomark with<br/>--vault all, VisionClaw still uses its own vault crate with<br/>--vault knowledge; ADR-OM-005 records this and routes VisionClaw to build --scope machine, see OM-06.1,<br/>publish.yml:165-166, ontology-publish.yml:168-173, ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:13"]
    n2["EXTERNAL: this VisionClaw workflow never invokes ontomark; vault build output<br/>feeds pack-pod-resources.py, ontology-publish.yml:247. ADR-OM-005 decision 2<br/>retires this job, gated, so it keeps running — ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:11"]
    n3["INVARIANT: visionGraph treats the staging tree as wrong unless both<br/>pages/ and working/ exist, because ontomark silently accepts a<br/>knowledge-only build — publish.yml:189-195"]
    VG --- n1
    VC --- n2
    VGBUILD --- n3
```
- Redaction is not a flag in either workflow: visionGraph gets it from the estate profile (`estate/profile.yaml:226`, `build/mod.rs:615-617`) and never passes `--no-redact-private-links`; the legacy `vault` build kept private links (`ontomark-cli/src/main.rs:434-441`).
- The consumer does not pass `--generated-at`, `--source-commit` or `--stale-after`, so `.generation.json` takes the clock and git (`build/mod.rs:63-67`, `publish.yml:165-171`).

## OM-05.6 C2 contract versus the shipped CLI — what C2 lists that ontomark and vault now split between them

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
        OMB["build with --with-rvdb, --embed-endpoint and --certificate-out<br/>plus --scope --publish-out --with-markdown-mirror<br/>--generated-at --source-commit --stale-after<br/>--no-redact-private-links<br/>ontomark-cli/src/main.rs:373-442"]
        OMM["migrate list plan apply rollback status<br/>migrate_cmd.rs:22-57"]
        OMR["repair fences, repair bodies — NOT in C2<br/>ontomark-cli/src/main.rs:141-148"]
        OMX["find retrieve tree edit create apply — ported from vault (ADR-OM-007)<br/>ontomark-cli/src/main.rs:117-129"]
    end
    subgraph VCV["VisionClaw vault crate — crates/vault/src/main.rs"]
        direction TB
        VCM["help text: find, retrieve, tree, edit, create and apply now live in ontomark<br/>crates/vault/src/main.rs:28-29"]
        VCP["still in vault: propose, panel, rank, certify<br/>crates/vault/src/main.rs:48-80"]
    end
    C2V --- OMV
    C2G --- OMG
    C2C --- OMC
    C2B --- OMB
    C2M -.-|"shape differs"| OMM
    C2Q -.-|"moved"| OMX
    C2Q -.-|"propose stays"| VCP
    n1["DRIFT: C2 lists find retrieve tree edit create under vault, but the vault binary's own help says they moved to ontomark and DEPRECATION records them as satisfied;<br/>only propose and panel publish remain, and certify is in no C2 row —<br/>sovereign-corpus-contracts.md:89-98 vs crates/vault/src/main.rs:28-29,76, DEPRECATION.md:95"]
    n2["DIVERGENCE: C2 defines migrate --fences-to-properties; ontomark ships a<br/>registry-driven migrate with five subcommands and a repair verb C2 lacks<br/>sovereign-corpus-contracts.md:103 vs ontomark-cli/src/main.rs:134-137,141-147"]
    n3["INVARIANT: one vault parser in ontomark: VAULT_VALUES and vault_kinds in migrate_cmd.rs serve every<br/>verb, and its own kinds() only delegates; the VisionClaw vault crate keeps a separate copy<br/>migrate_cmd.rs:31,62,72-83 vs crates/vault/src/main.rs:171,181"]
    OM --- n1
    OM --- n2
    OM --- n3
```
- The verbs ontomark leaves out are the forum transport only: `propose` and `panel publish` stay in VisionClaw (`ontomark-cli/src/main.rs:49-50`, `DEPRECATION.md:95`).
- No `BuildArgs` option is parsed and left unused: all fourteen reach `build::Options`, the embedder or the report (`ontomark-cli/src/main.rs:373-442,801-824,875`).
- DIVERGENCE: `--no-redact-private-links` overlays the profile setting `publication.redact_private_links` as a `Some(false)` override rather than a separate behaviour (`ontomark-cli/src/main.rs:815`, `build/mod.rs:84`, `profile.rs:422`).
