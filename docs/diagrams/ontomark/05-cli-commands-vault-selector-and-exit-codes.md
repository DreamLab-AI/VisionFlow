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
  - ../ontomark/crates/ontomark-compile/src/profile/layout.rs
  - ../ontomark/profiles/estate/profile.yaml
  - ../ontomark/docs/adr/ADR-OM-005-estate-alignment-one-producer-explicit-projections.md
  - ../project/crates/vault/src/main.rs
  - ../visionGraph/.github/workflows/publish.yml
  - ../project/.github/workflows/ontology-publish.yml
  - docs/engineering/sovereign-corpus-contracts.md
verified_commit: {ontomark: 6c3f693b36301ee95c8b684a000b5aeec3ac9bbd, visiongraph: 556a0156b5c326fb61141ed2ab1760dd8d9c4dd6, visionclaw: 86cdd76355823a51bb64d503c005fe277726e724, visionflow: 5f8b1da92ce5d9e6b32a0bd90ec3630c66619083}
---

Restamped at ontomark 4b9d15d on 2026-11-02. The `ingest` verb is now in the command tree (`ontomark-cli/src/main.rs:137-140`) and is shown in OM-05.1; its pipeline is mapped in [OM-08](08-external-source-ingest.md).

## OM-05.1 Command tree — ten verbs, ingest, repair and migrate, three global flags, and which vault each one reads

```mermaid
flowchart TD
    CLI["ontomark<br/>global --repo --json --profile<br/>ontomark-cli/src/main.rs:86-111"]
    VAL["validate<br/>VaultSelector default knowledge, --strict<br/>loops over EVERY selected vault<br/>ontomark-cli/src/main.rs:227-233,663"]
    GATE["gate<br/>VaultSelector default knowledge, --tier quick or full via value_parser<br/>gates EVERY selected vault, exit is the worst verdict<br/>ontomark-cli/src/main.rs:362-368,699"]
    CONF["conflicts<br/>VaultSelector default knowledge, --severity via value_parser<br/>analyses EVERY selected vault, exit is the worst report<br/>ontomark-cli/src/main.rs:371-378,758"]
    BUILD["build<br/>--out required, --vault default knowledge, --scope public or machine<br/>one projection vault plus optional working publish, see OM-06.1<br/>ontomark-cli/src/main.rs:382-451,798-910"]
    QRY["find, retrieve, tree<br/>VaultSelector default knowledge; find loops over every selected vault,<br/>retrieve and tree take the FIRST<br/>ontomark-cli/src/main.rs:236-282,912-972"]
    WR["edit, create, apply<br/>guarded writes, each refused without --expect; create and apply always load knowledge<br/>ontomark-cli/src/main.rs:284-360,974-1139"]
    REPAIR["repair<br/>subcommands fences and bodies<br/>--vault default all<br/>ontomark-cli/src/main.rs:150-184,1141-1227"]
    MIG["migrate<br/>list plan apply rollback status<br/>plan --vault default all<br/>ontomark-cli/src/main.rs:143-146, migrate_cmd.rs:22-57"]
    ING["ingest<br/>external sources to a vault propose bundle, never writes the vault;<br/>refuses --vault working or all, see OM-08<br/>ontomark-cli/src/main.rs:137-140,659, ontomark-cli/src/main.rs:39-41"]
    CLI --> VAL
    CLI --> GATE
    CLI --> CONF
    CLI --> BUILD
    CLI --> QRY
    CLI --> WR
    CLI --> REPAIR
    CLI --> MIG
    CLI --> ING
    n1["INVARIANT: every vault flag is validated by value_parser VAULT_VALUES, so<br/>an unknown value is a clap error rather than a silent knowledge,<br/>ontomark-cli/src/main.rs:191, migrate_cmd.rs:59-62"]
    n2["INVARIANT: all means BOTH for validate, gate, conflicts, find and build: the vault_kinds doc says so and each<br/>loops over every selected vault —<br/>migrate_cmd.rs:64-68, ontomark-cli/src/main.rs:663,699,758,914<br/>DIVERGENCE: retrieve, tree and edit call .next() on the loaded vaults, so under all they read knowledge alone —<br/>ontomark-cli/src/main.rs:936-939,961-964,984-987"]
    n3["INVARIANT: --tier and --severity are clap value_parsers, so an unknown value is a clap<br/>error and exit 2 as the crate doc promises; the lenient parse fallbacks are unreachable —<br/>ontomark-cli/src/main.rs:222-224,366,376, gate.rs:42-48, conflicts.rs:48-55"]
    GATE --- n2
    CONF --- n3
    CLI --- n1
```
- `repair` defaults to `all` while `validate`, `gate`, `conflicts`, `build` and the query verbs default to `knowledge`, so the same word means a different scope per verb (`ontomark-cli/src/main.rs:161,178,191,391`).
- DIVERGENCE: `--stats` is documented as "print the counts" but only forces the JSON report, a build-only alias for `--json` (`ontomark-cli/src/main.rs:403,888`).
- INVARIANT: `--profile` reaches every verb through the loaded vaults' layout (`ontomark-cli/src/main.rs:559-569,572-586`), but the conflict analysis itself takes no profile (`ontomark-cli/src/main.rs:762`).

## OM-05.2 `--vault knowledge|working|all` — resolution into vault_root and with_working_publish

```mermaid
flowchart TD
    FLAG["--vault string<br/>parser accepts knowledge, working, all only<br/>ontomark-cli/src/main.rs:191,391"]
    VK["vault_kinds value, the one parser for every verb<br/>trim + lowercase, then match<br/>ontomark-cli/src/migrate_cmd.rs:72"]
    W["working -> Working only<br/>ontomark-cli/src/migrate_cmd.rs:74"]
    A["all -> Knowledge AND Working<br/>ontomark-cli/src/migrate_cmd.rs:75"]
    K["anything else -> Knowledge<br/>unreachable after the parser, ontomark-cli/src/migrate_cmd.rs:76"]
    KIND["kind = Working only when kinds == Working and scope is not machine,<br/>else Knowledge<br/>ontomark-cli/src/main.rs:808-813"]
    ROOT["vault_root = the profile layout places the vault under the repo root:<br/>root_of refuses a role the layout lacks or a missing folder<br/>ontomark-cli/src/main.rs:555-557, profile.rs:573-588"]
    MISS["MissingVault, or a LayoutError from the layout check, a refused request: exit 2 from main<br/>ontomark-cli/src/main.rs:457-466, profile.rs:635, profile/layout.rs:11"]
    WWP["with_working_publish =<br/>kinds contains Working AND kind is Knowledge<br/>true ONLY for all<br/>ontomark-cli/src/main.rs:815-816"]
    PUB["build: sibling_vault(Working) loaded and added to publish_vaults<br/>build/mod.rs:698-704"]
    MS["scope machine reads knowledge and working itself and refuses vault working or all, see OM-06.1<br/>ontomark-cli/src/main.rs:393-401, build/mod.rs:557-604"]
    ART["ontology projections stay on vault_root alone:<br/>OWL membership is not a per-page decision<br/>ontomark-cli/src/main.rs:799-803, build/mod.rs:76-77"]
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
    n1["INVARIANT: all widens ONLY publish/ staging; machine artefacts<br/>stay knowledge-only — ontomark-cli/src/main.rs:386-390, build/mod.rs:76-78"]
    n2["INVARIANT: working alone sets vault_root to the working folder and<br/>with_working_publish false — ontomark-cli/src/main.rs:809-816"]
    PUB --- n1
    ROOT --- n2
    KIND -.-> MS
```
- Every verb, `migrate` included, reads `--vault` through the one `vault_kinds` (`migrate_cmd.rs:72-83`): `validate`, `gate`, `conflicts` and `find` loop over every kind via `load` (`ontomark-cli/src/main.rs:559-569,663,699,758,914`), `repair` maps them to layout-placed roots for `scopes_for` (`main.rs:561-575,1129,1170`, `repair.rs:445`).
- DOC-DRIFT: comments say the projections are `knowledge/`'s by definition (`ontomark-cli/src/main.rs:799-800`, `build/mod.rs:76-77`), yet `--vault working` roots the build at the working vault (`ontomark-cli/src/main.rs:809-813`).

## OM-05.3 Exit codes — which error type maps to 0, 1 or 2

```mermaid
flowchart LR
    MAIN["main: run Ok(code) returns code;<br/>Err prints ontomark: and exits 1, except MissingVault and LayoutError which exit 2<br/>ontomark-cli/src/main.rs:453-467"]
    ZERO["0 — success<br/>ontomark-cli/src/main.rs:909,931,1077,1138"]
    ONE["1 — failed check or execution error"]
    TWO["2 — refused operation or invalid argument<br/>ontomark-cli/src/main.rs:31-42"]
    V1["validate: any error, or warning with --strict<br/>ontomark-cli/src/main.rs:666-667,688"]
    G1["gate: worst verdict.exit_code 0 or 1 over the selected vaults<br/>gate.rs:110-112, ontomark-cli/src/main.rs:714-718,751"]
    C1["conflicts: worst report.exit_code 0 or 1 over the selected vaults<br/>conflicts.rs:119-121, ontomark-cli/src/main.rs:765-769,795"]
    B1["build: written == 0 is an anyhow<br/>error<br/>ontomark-cli/src/main.rs:904-908"]
    R1["repair bodies --check with residue<br/>ExitCode::FAILURE,<br/>ontomark-cli/src/main.rs:1176-1178"]
    ANY["any other Err from<br/>build::run_with, load,<br/>resolve_repo<br/>ontomark-cli/src/main.rs:884,<br/>641-642"]
    S2["SecretsFound — credential on a<br/>published page<br/>build/publish.rs:66,<br/>ontomark-cli/src/main.rs:850-853"]
    O2["OutputCollision — two artefacts<br/>claim one path build/mod.rs:206,<br/>ontomark-cli/src/main.rs:856-859"]
    R2["ReasoningError — definition<br/>reasoner would not finish<br/>reason.rs:481,<br/>ontomark-cli/src/main.rs:862-865"]
    SC2["ScopeConflict or Unsatisfiable —<br/>machine scope misuse, or an<br/>unsatisfiable class<br/>build/mod.rs:282,235<br/>ontomark-cli/src/main.rs:868-877"]
    CP2["CertificateParity — the derivation<br/>certificate does not cover the<br/>classifier's closure (ADR-OM-009)<br/>build/mod.rs:101,<br/>ontomark-cli/src/main.rs:880-883"]
    MV2["MissingVault or LayoutError — a vault role<br/>the layout does not place, or role roots that<br/>escape the repository or overlap<br/>profile.rs:635, profile/layout.rs:11,<br/>ontomark-cli/src/main.rs:457-466"]
    GW2["create or apply refused by its<br/>guards, nothing written<br/>ontomark-cli/src/main.rs:1023-1034,1083-1094"]
    CL["clap parse failure: bad flag, or a<br/>vault, tier, severity or scope<br/>value outside its list clap<br/>default exit 2,<br/>ontomark-cli/src/main.rs:191,366,376,222"]
    M2["migrate plan not Ready -> 2<br/>migrate_cmd.rs:155-160"]
    M1["migrate apply incomplete, rollback<br/>with conflicts -> 1<br/>migrate_cmd.rs:189,216"]
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
    n1["INVARIANT: build maps six typed errors to 2 by downcast_ref<br/>on the anyhow error, and create, apply, migrate plan and<br/>MissingVault and LayoutError refuse with 2 too,<br/>ontomark-cli/src/main.rs:850-883,1023-1034,1083-1094,457-466"]
    n2["DIVERGENCE: validate, gate and conflicts never return 2 from their<br/>own code (only clap and MissingVault do), so a refusal is<br/>distinguishable only on the writing verbs —<br/>ontomark-cli/src/main.rs:688,751,795"]
    TWO --- n1
    ONE --- n2
```
- `SecretsFound` is checked before validation on purpose so a credential is a code-2 refusal and never swallowed by the generic code-1 validation message (`build/mod.rs:673-683`).
- The `0/1/2` contract is declared in the crate doc, with `1` also covering execution errors, and the doc lists the unsatisfiable class, the machine-scope refusals, a `--vault working|all` on a layout without that vault, the `create`/`apply` guards and the `ingest` refusals under 2 (`ontomark-cli/src/main.rs:31-42`). **Drift:** the doc does not name the `LayoutError` refusals (escaping or overlapping role roots) that `main` also maps to 2 (`ontomark-cli/src/main.rs:460-463`).
- An unsatisfiable corpus exits 2 through the typed `build::Unsatisfiable`; before the OM-00 reconcile it was an untyped error at exit 1 (`build/mod.rs:858-864`, `ontomark-cli/src/main.rs:874-877`).

## OM-05.4 Repo-root discovery and profile resolution — before any verb runs

```mermaid
flowchart TD
    START["run: Cli::parse<br/>ontomark-cli/src/main.rs:651"]
    EXPL{"--repo given?<br/>ontomark-cli/src/main.rs:485-489"}
    P1["start = the explicit path"]
    P2["start = current_dir"]
    DISC["Vocabulary::discover start<br/>canonicalise start, then walk it and its ancestors for<br/>ontology/vocabulary.yaml; a start that cannot be resolved is an io error<br/>vocabulary.rs:491-495, vocabulary.rs:507-523"]
    ESTATE["found: root = vocabulary path .parent().parent()<br/>the directory ABOVE ontology/, estate_shape true<br/>ontomark-cli/src/main.rs:491-502"]
    PLAIN["NotAVault: start is a plain Obsidian vault and is the root,<br/>built-in vocabulary, estate_shape false; a start that is not a directory is an error<br/>ontomark-cli/src/main.rs:503-514, vocabulary.rs:519-522"]
    BADV["a vocabulary file that exists but does not parse is still an error, exit 1<br/>ontomark-cli/src/main.rs:515-522"]
    PROF{"--profile?<br/>ontomark-cli/src/main.rs:529-549"}
    EST["estate -> ontomark_profile_estate<br/>ontomark-cli/src/main.rs:531"]
    GEN["generic -> Profile::generic<br/>ontomark-cli/src/main.rs:532"]
    YAML["other string -> read and parse YAML file, layout kept as written<br/>ontomark-cli/src/main.rs:533-537"]
    AUTO["unset: estate if estate_shape, else generic<br/>ontomark-cli/src/main.rs:538-539"]
    FLAT["built-in profile on a plain vault takes Layout::flat:<br/>knowledge = the root, no working vault, layout.excluded kept<br/>ontomark-cli/src/main.rs:541-547, profile.rs:520-526"]
    LCHK["profile.layout.validate before any verb: invalid, escaping or overlapping role roots refuse, exit 2<br/>ontomark-cli/src/main.rs:654, profile/layout.rs:190-192"]
    LOAD["load: layout.root_of per kind, load_excluding each<br/>ontomark-cli/src/main.rs:559-569"]
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
    ESTATE --> LCHK
    FLAT --> LCHK
    LCHK --> LOAD
    n1["INVARIANT: no discovered vocabulary means a plain vault;<br/>selected absent roles refuse, migrate plan included, which checks root_of for every selected kind before planning —<br/>migrate_cmd.rs:144-146<br/>profile.rs:573-588"]
    n2["INVARIANT: the generic profile is reachable by auto-detection, because only a plain vault<br/>lacks the vocabulary file; an explicit --profile estate on a plain vault still gets the flat layout —<br/>ontomark-cli/src/main.rs:539-547"]
    PLAIN --- n1
    AUTO --- n2
```
- The `--repo` argument starts an upward search (`ontomark-cli/src/main.rs:485-514`). **Changed by 4b9d15d:** the search starts from the canonical path, so implicit cwd and explicit `--repo .` agree (`vocabulary.rs:507-510`), and `migrate plan` refuses an absent selected role (`migrate_cmd.rs:144-146`); both faults were reproduced by the [layout audit](../../estate-review/2026-11-01-ontomark-layout-and-shared-checks.md), and [OM-12](12-shared-layout-checks.md) owns the shared check.

## OM-05.5 Two consumers — visionGraph publishes both vaults from a pinned release, VisionClaw builds knowledge in-tree

```mermaid
flowchart LR
    subgraph VG["visionGraph publish.yml — narrativegoldmine.com"]
        direction TB
        VGPIN["Pinned release v0.7.0 AND sha256 digest<br/>no build-from-source fallback<br/>publish.yml:48-51"]
        VGFETCH["gh release download, sha256sum --check --strict,<br/>install as ontomark<br/>publish.yml:107-112"]
        VGVAL["ontomark validate --vault all<br/>publish.yml:130"]
        VGBUILD["ontomark build --vault all --out site-data<br/>--publish-out quartz/content --with-markdown-mirror<br/>--certificate-out --stats<br/>publish.yml:175-181"]
        VGCERT["oo-cert v2.0.1 re-derives the certificate from its premises, digest-pinned<br/>publish.yml:217-234"]
        VGREP["ontomark repair bodies --vault all --check<br/>outline residue now fails the publish<br/>publish.yml:244-254"]
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
    BIN["ontomark build — ontomark-cli/src/main.rs:798-910"]
    VGBUILD -->|"vault_kinds all -> with_working_publish true"| BIN
    VCRUN -.->|"different binary"| BIN
    n1["EXTERNAL: the two consumers differ — visionGraph uses ontomark with<br/>--vault all, VisionClaw still uses its own vault crate with<br/>--vault knowledge; ADR-OM-005 records this and routes VisionClaw to build --scope machine, see OM-06.1,<br/>publish.yml:175-176, ontology-publish.yml:168-173, ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:40"]
    n2["EXTERNAL: this VisionClaw workflow never invokes ontomark; vault build output<br/>feeds pack-pod-resources.py, ontology-publish.yml:247. ADR-OM-005 decision 2<br/>retires this job, gated, so it keeps running — ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:38"]
    n3["INVARIANT: visionGraph treats the staging tree as wrong unless both<br/>pages/ and working/ exist, because ontomark silently accepts a<br/>knowledge-only build — publish.yml:199-205"]
    VG --- n1
    VC --- n2
    VGBUILD --- n3
```
- Redaction is not a flag in either workflow: visionGraph gets it from the estate profile (`estate/profile.yaml:226`, `build/mod.rs:709-711`) and never passes `--no-redact-private-links`; the legacy `vault` build kept private links (`ontomark-cli/src/main.rs:443-450`).
- The consumer does not pass `--generated-at`, `--source-commit` or `--stale-after`, so `.generation.json` takes the clock and git (`build/mod.rs:63-67`, `publish.yml:175-181`).

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
        OMB["build with --with-rvdb, --embed-endpoint and --certificate-out<br/>plus --scope --publish-out --with-markdown-mirror<br/>--generated-at --source-commit --stale-after<br/>--no-redact-private-links<br/>ontomark-cli/src/main.rs:382-451"]
        OMM["migrate list plan apply rollback status<br/>migrate_cmd.rs:22-57"]
        OMR["repair fences, repair bodies — NOT in C2<br/>ontomark-cli/src/main.rs:150-157"]
        OMX["find retrieve tree edit create apply — ported from vault (ADR-OM-007)<br/>ontomark-cli/src/main.rs:122-134"]
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
    n2["DIVERGENCE: C2 defines migrate --fences-to-properties; ontomark ships a<br/>registry-driven migrate with five subcommands, and repair and ingest verbs C2 lacks<br/>sovereign-corpus-contracts.md:103 vs ontomark-cli/src/main.rs:137-146,150-156"]
    n3["INVARIANT: one vault parser in ontomark: VAULT_VALUES and vault_kinds in migrate_cmd.rs serve every<br/>verb, and its own kinds() only delegates; the VisionClaw vault crate keeps a separate copy<br/>migrate_cmd.rs:31,62,72-83 vs crates/vault/src/main.rs:171,181"]
    OM --- n1
    OM --- n2
    OM --- n3
```
- The verbs ontomark leaves out are the forum transport only: `propose` and `panel publish` stay in VisionClaw (`ontomark-cli/src/main.rs:52-53`, `DEPRECATION.md:95`).
- No `BuildArgs` option is parsed and left unused: all fourteen reach `build::Options`, the embedder or the report (`ontomark-cli/src/main.rs:382-451,814-837,888`).
- DIVERGENCE: `--no-redact-private-links` overlays the profile setting `publication.redact_private_links` as a `Some(false)` override rather than a separate behaviour (`ontomark-cli/src/main.rs:828`, `build/mod.rs:84`, `profile.rs:427`).

**Tension (ontomark 6c3f693 vs this topic's visionGraph stamp):** since 6c3f693, released in v0.7.3, `build` refuses a `--certificate-out` that is not a directory inside `--out`, because certificate files are now staged into the bundle and share its commit point (`../ontomark/crates/ontomark-compile/src/build/mod.rs:436-472`). The visionGraph workflow drawn in the consumer diagram is stamped at `556a015`, where it pins release v0.7.0 and passes `--certificate-out "$RUNNER_TEMP/certificate"`, outside `--out site-data` (`publish.yml:49`, `publish.yml:180`). That pairing runs only because the pinned release predates the guard. visionGraph is changing this workflow in a lane today, so the visionGraph key is left for phase 2.

**Open (ADR-OM-005 reconciliation, 2026-11-02):** the record now says its "sole producer" wording is about publication ownership, not a claim that every consumer is current, and that Agentbox's grouped-proposal adapter and VisionClaw's retained `vault propose` are distinct consumers: a signed multi-page proposal is not evidence that the single-page `apply` path can execute it (`../ontomark/docs/adr/ADR-OM-005-estate-alignment-one-producer-explicit-projections.md:25`). Its 2026-11-03 repair adds a resource-identity guard to single-page apply and leaves grouped signing, admission and digest checks to the consumers (`{ADR}:21`); see OM-00.
