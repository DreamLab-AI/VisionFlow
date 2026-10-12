---
id: OM-01
title: Input layer — vault, pages, frontmatter gates, secrets, slugs, vocabulary
area: ontomark
governing:
  - docs/engineering/sovereign-corpus-contracts.md
adrs: [ADR-OM-001, ADR-OM-005]
sources:
  - ../ontomark/crates/ontomark-compile/src/input/mod.rs
  - ../ontomark/crates/ontomark-compile/src/input/page.rs
  - ../ontomark/crates/ontomark-compile/src/input/frontmatter.rs
  - ../ontomark/crates/ontomark-compile/src/input/okf.rs
  - ../ontomark/crates/ontomark-compile/src/input/secrets.rs
  - ../ontomark/crates/ontomark-compile/src/input/slug.rs
  - ../ontomark/crates/ontomark-compile/src/input/consistency.rs
  - ../ontomark/crates/ontomark-compile/src/input/vocabulary.rs
  - ../ontomark/crates/ontomark-compile/src/input/promotion.rs
  - ../ontomark/crates/ontomark-compile/src/input/definition.rs
  - ../ontomark/crates/ontomark-compile/src/input/code.rs
  - ../ontomark/crates/ontomark-compile/src/ir/projection.rs
  - ../ontomark/crates/ontomark-compile/src/ir/model.rs
  - ../ontomark/crates/ontomark-compile/src/check/validate.rs
  - ../ontomark/crates/ontomark-compile/src/build/publish.rs
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../ontomark/crates/ontomark-compile/src/build/redact.rs
  - ../ontomark/crates/ontomark-compile/src/migrate/plan.rs
  - ../ontomark/crates/ontomark-model/src/mint.rs
  - ../ontomark/crates/ontomark-model/src/profile.rs
  - ../ontomark/crates/ontomark-wasm/src/profile.rs
  - ../ontomark/crates/ontomark-parse/src/frontmatter.rs
  - ../ontomark/crates/ontomark-cli/src/main.rs
  - ../ontomark/crates/ontomark-compile/src/check/repair.rs
  - ../ontomark/crates/ontomark-compile/src/check/bodies.rs
  - ../ontomark/crates/ontomark-compile/src/check/conflicts.rs
  - ../ontomark/crates/ontomark-compile/src/ir/closure.rs
  - ../ontomark/crates/ontomark-compile/src/build/indexes.rs
  - ../ontomark/crates/ontomark-compile/src/build/turtle.rs
  - ../ontomark/crates/ontomark-compile/tests/probe_consistency.rs
verified_commit: ee9e04a25715e70d3de211c17a012e878d14f339
---

## OM-01.1 Vault root to Vault — pages, journals, skips

```mermaid
flowchart TD
    CLI["ontomark-cli load per VaultKind dir<br/>ontomark-cli/src/main.rs:306-312"]
    BUILD["build::run_with — walk_pages once, inspect_walk census, then Vault::load_with_walk<br/>build/mod.rs:512-518"]
    LOADW["Vault::load_with_walk — loads the files of a PageWalk the caller already made<br/>input/page.rs:436"]
    LV["load_vault — kind = Working iff dir name is working, else Knowledge<br/>input/page.rs:609-623"]
    LOAD["Vault::load root, kind<br/>input/page.rs:413"]
    NOPAGES["no pages dir → VaultError::NotAVault<br/>input/page.rs:415-421"]
    WALK["walk_pages pages_dir — see OM-01.2<br/>input/page.rs:422"]
    ID["id = path under pages without .md, backslash to slash<br/>input/page.rs:444-447"]
    READ["read_to_string, io error carries the path<br/>input/page.rs:452"]
    PARSE["Page::parse path, rel_path, id, text — see OM-01.3<br/>input/page.rs:453"]
    JDIR{"root/journals is a dir?<br/>input/page.rs:457-460"}
    JWALK["walk_pages journals_dir — same walker<br/>input/page.rs:461"]
    JID["journal id = relative stem, not a date check<br/>input/page.rs:466-469"]
    JPARSE["Page::parse into journals Vec<br/>input/page.rs:475"]
    JSKIP["journal skips re-rooted under journals/<br/>input/page.rs:477-480"]
    VAULT["Vault root, kind, pages, journals, skipped<br/>input/page.rs:385-401"]
    JDATE["journal_date only checked later, by validate → JOURNAL_NOT_DATED<br/>check/validate.rs:637-643"]
    CLI --> LV
    BUILD --> LOADW
    LV --> LOAD --> NOPAGES
    LOAD --> WALK --> LOADW --> ID --> READ --> PARSE --> JDIR
    JDIR -- yes --> JWALK --> JID --> JPARSE --> JSKIP --> VAULT
    JDIR -- no --> VAULT
    VAULT -.-> JDATE
    note1["INVARIANT: journals are kept out of Vault.pages so projections keyed on pages cannot silently grow,<br/>input/page.rs:392-396"]
    note2["INVARIANT: the first malformed frontmatter or unreadable file aborts the whole load — Page::parse errors propagate,<br/>input/page.rs:453"]
    note3["INVARIANT: one listing feeds the census, the load and the held-page identities — the PageWalk made at build/mod.rs:512 is passed on,<br/>ir/projection.rs:95, ir/projection.rs:227 — see OM-01.2"]
    VAULT --- note1
    PARSE --- note2
    LV --- note3
```
- **EXTERNAL:** the whole `input/` tree is a temporary stand-in relocated from VisionClaw `vault-core`, to be deleted once the input adapter reads `ontomark-model` (`input/mod.rs:3-8`); ADR-OM-001 owns that replacement.
- **DIVERGENCE:** `ontomark-parse` carries its own frontmatter reader (`ontomark-parse/src/frontmatter.rs:1-9`) beside the one used here (`input/frontmatter.rs:128`); the two are not shared in production.

## OM-01.2 walk_pages — what is declined, reported, and the one listing every consumer shares

```mermaid
flowchart TD
    W["walk_pages pages_dir — WalkDir, follow_links false<br/>input/page.rs:532-536"]
    ISMD{"extension is md?<br/>input/page.rs:542-544"}
    DROPNONMD["non-md → continue, not reported<br/>input/page.rs:542-544"]
    ISFILE{"entry is a regular file?<br/>input/page.rs:548"}
    DROPSYM["symlink or device → Skip NotARegularFile, reported; a directory named .md is not a page and is not reported<br/>input/page.rs:549-558, input/page.rs:63-72"]
    HIDDEN{"any path component starts with a dot?<br/>input/page.rs:560"}
    SKIP["Skip HiddenDirectory<br/>input/page.rs:561-565"]
    KEEP["rels.push parts<br/>input/page.rs:567"]
    SORT["component-wise sort of rel paths, skips sorted by path<br/>input/page.rs:569-570"]
    OUT["PageWalk files, skipped<br/>input/page.rs:98-103"]
    W --> ISMD
    ISMD -- no --> DROPNONMD
    ISMD -- yes --> ISFILE
    ISFILE -- no --> DROPSYM
    ISFILE -- yes --> HIDDEN
    HIDDEN -- yes --> SKIP
    HIDDEN -- no --> KEEP
    KEEP --> SORT --> OUT
    UNPUB["_misc is NOT declined here — publication rule, kept as a page<br/>input/page.rs:38, input/page.rs:525-528"]
    CALLERS["callers: Vault::load, build::run_with, migrate read_considered, repair, bodies<br/>input/page.rs:422, build/mod.rs:512, migrate/plan.rs:314, check/repair.rs:375, check/bodies.rs:811"]
    CENSUS["inspect_walk — the census over the SAME PageWalk: a skip counts excluded_directory or excluded_not_regular_file<br/>ir/projection.rs:95, ir/projection.rs:104-108"]
    CENSUSF["per file: _misc excluded_directory, unreadable or unterminated fence invalid_input, public not a YAML bool invalid_publication_flag<br/>ir/projection.rs:113-115, ir/projection.rs:122-125, ir/projection.rs:139-141"]
    HELD["excluded_identities_of reads the held pages out of the loaded Vault — no second walk or parse<br/>ir/projection.rs:227-235"]
    RUN["build::run_with walks once, then census, load and held identities all read that listing<br/>build/mod.rs:512-514, build/mod.rs:599"]
    OUT --- CALLERS
    OUT --- UNPUB
    RUN --> OUT
    RUN --> CENSUS --> CENSUSF
    RUN --> HELD
    note1["INVARIANT: every skip is reported — hidden directories and non-regular files reach Vault.skipped and the census counts, input/page.rs:85-87, input/page.rs:397-400"]
    note2["INVARIANT: the doc names both skip classes, input/page.rs:525-527; the hidden test is on every component including the file name, input/page.rs:560"]
    OUT --- note1
    HIDDEN --- note2
```
- **INVARIANT:** `UNPUBLISHED_DIRS = ["_misc"]` (`input/page.rs:38`) is not the only held-back rule: `build/publish.rs` adds `misc` at any depth (`build/publish.rs:130-142`), so the publish scope still has two lists.
- **INVARIANT:** hidden-directory and non-regular-file skips are reported on `Vault.skipped` (`input/page.rs:397-400`) and travel into the validation report (`check/validate.rs:378`).

## OM-01.3 Frontmatter parse and the public, type and resource gates

```mermaid
flowchart TD
    TXT["page text"]
    SPLIT["frontmatter::split — strip BOM, opening --- must be line 1<br/>input/frontmatter.rs:128-133"]
    NOFM["no opening fence → yaml None, body is whole file, body_line 1, unterminated false<br/>input/frontmatter.rs:134-139"]
    CLOSE["closing --- found → yaml and body, body_line computed<br/>input/frontmatter.rs:141-152"]
    UNCLOSED["opening fence but no closing fence → yaml None, Split.unterminated true<br/>input/frontmatter.rs:164-169"]
    FMPARSE["Frontmatter::parse — empty or null → default, non-mapping → error<br/>input/frontmatter.rs:178-190"]
    PERR["YAML error → VaultError::Frontmatter with path<br/>input/page.rs:172-175"]
    PAGE["Page with Frontmatter map<br/>input/page.rs:178-187"]
    UFLAG["Page.unterminated_frontmatter carries the split flag to the census and validate<br/>input/page.rs:185"]
    PUBLIC["Page::is_public — only Value::Bool(true) publishes; absent, false, a string or a number do not<br/>input/page.rs:231-236"]
    BOOL["Frontmatter::boolean takes a YAML bool only — a quoted true is None, never coerced<br/>input/frontmatter.rs:247-252"]
    OKF["Page::okf → OkfBlock::from_frontmatter, never fails<br/>input/page.rs:240-242, input/okf.rs:264"]
    OKFF["type_name, resource, status parse, generated, verified, sources<br/>input/okf.rs:274-288"]
    HV["is_human_verified — any verified stamp by a human<br/>input/okf.rs:294-296"]
    GTYPE["MISSING_TYPE if knowledge page has no type<br/>check/validate.rs:775-781"]
    GINV["INVALID_TYPE if type not in vocab types or working_types<br/>check/validate.rs:782-795"]
    GREQ["MISSING_REQUIRED for each required key; resource needs iri and has_ontology<br/>check/validate.rs:796-811"]
    GONT["has_ontology = has resource OR has type<br/>ir/model.rs:429"]
    GIRI["MISSING_CLASS_IRI when has_ontology but iri empty<br/>check/validate.rs:507-513"]
    GRES["resource = frontmatter resource else namespace + slug<br/>input/page.rs:217-222"]
    TXT --> SPLIT
    SPLIT --> NOFM
    SPLIT --> CLOSE
    SPLIT --> UNCLOSED
    CLOSE --> FMPARSE --> PERR
    FMPARSE --> PAGE
    NOFM --> PAGE
    UNCLOSED --> PAGE
    PAGE --> UFLAG
    PAGE --> PUBLIC --> BOOL
    PAGE --> OKF --> OKFF --> HV
    OKFF --> GTYPE --> GINV --> GREQ
    PAGE --> GRES --> GONT --> GIRI
    note1["INVARIANT: an unfenced or absent frontmatter means private — Frontmatter::default has no public key,<br/>input/page.rs:176, input/page.rs:231-236"]
    note2["INVARIANT: an unclosed opening fence is no longer silently private prose — the census refuses it as invalid_input and validate reports UNCLOSED_FRONTMATTER (S3.3),<br/>ir/projection.rs:122-125, check/validate.rs:342-348"]
    PUBLIC --- note1
    UNCLOSED --- note2
```
- **INVARIANT:** every in-tree "is public" decision now reads the same rule: `Page::is_public` (`input/page.rs:231-236`), `Frontmatter::boolean` (`input/frontmatter.rs:247-252`) and the build census (`ir/projection.rs:138-142`) accept only a real YAML bool; `validate` reports anything else as `INVALID_PUBLIC_FLAG` (`check/validate.rs:354-367`).
- **DIVERGENCE:** `ontomark-model` is still lenient when the profile sets `lenient_booleans` (`ontomark-model/src/profile.rs:604-611`) and the wasm profile has its own strict copy (`ontomark-wasm/src/profile.rs:316-321`); both sit outside the build path.

## OM-01.4 Secrets scanner — body and rendered frontmatter, kind and line never value

```mermaid
sequenceDiagram
    autonumber
    participant V as check/validate.rs<br/>check/validate.rs:305
    participant P as build/publish.rs<br/>build/publish.rs:224
    participant S as input/secrets.rs<br/>input/secrets.rs:166
    participant R as OnceLock regex set<br/>input/secrets.rs:114
    V->>S: scan page.body for EVERY page and journal (:305)
    V->>S: scan the rendered frontmatter block too — S1.2 (:323)
    P->>S: scan page.body for PUBLIC pages only (build/publish.rs:220-224)
    P->>S: scan the rendered frontmatter, the bytes render re-emits (build/publish.rs:236)
    S->>R: patterns compiled once, 11 CredentialKind variants (input/secrets.rs:115-157)
    R-->>S: each regex find_iter, byte offset to line by binary search (input/secrets.rs:170-180)
    S-->>V: Finding line and kind, sorted and deduplicated (input/secrets.rs:181-183)
    V->>V: body file line = body_line + finding.line - 1 (check/validate.rs:307) - frontmatter line = finding.line, the block opens on file line 1 (check/validate.rs:332)
    V->>V: Severity::Error SECRET_DETECTED, value never reported (check/validate.rs:308-318)
    S-->>P: Finding list (input/secrets.rs:98-104)
    P->>P: same line arithmetic, SecretsFound blocks the staging (build/publish.rs:228, build/publish.rs:258-260)
    Note over S: INVARIANT: Finding has no value field, so a report cannot echo a secret — input/secrets.rs:94-104
    Note over R: INVARIANT: no credential character class contains a dollar sign, so env references are not matches — input/secrets.rs:106-113
    Note over V: INVARIANT: S1.2 closed — a credential in a sources or title value is seen by validate and by the staging gate alike — check/validate.rs:323, build/publish.rs:236
```
- **INVARIANT:** the build runs the credential gate first, before validation, so a key on a public page gets its own exit code rather than being swallowed by the generic validation failure (`build/mod.rs:522-535`, `build/mod.rs:574`).
- **INVARIANT:** `journals/` pages are secret-scanned by validate (`check/validate.rs:633-634`) but are never staged for publication (`build/publish.rs:207-209`).

## OM-01.5 Slug derivation and the consistency vocabulary

```mermaid
flowchart TD
    TITLE["Page::title — frontmatter title, else file stem<br/>input/page.rs:192-203"]
    SLUGP["Page::slug — frontmatter slug, else slugify of title<br/>input/page.rs:207-212"]
    SLUGIFY["slugify — lowercase, non a-z0-9 runs to dash, trim dashes<br/>input/slug.rs:28-32"]
    RE["Regex non-slug compiled once<br/>input/slug.rs:14-17"]
    REFSLUG["ref_slug — text after the last colon, no re-slugify<br/>input/slug.rs:46-48"]
    TARGET["target_slug — kebab target verbatim, else slugify<br/>input/slug.rs:65-75"]
    THING["is_thing — thing, owl-thing, owl:thing<br/>input/slug.rs:80-85"]
    U1["closure.rs uses ref_slug and is_thing<br/>ir/closure.rs:19, ir/closure.rs:123-124"]
    U2["indexes.rs uses ref_slug<br/>build/indexes.rs:13, build/indexes.rs:58"]
    U3["Corpus links use target_slug<br/>ir/model.rs:424"]
    U4["Corpus resolve fallback uses plain slugify, not target_slug<br/>ir/model.rs:342"]
    CKEY["class_key — tail after last slash hash or colon, lowercased<br/>input/consistency.rs:62-69"]
    TAX["is_taxonomic → Taxonomy::is_taxonomy_node<br/>input/consistency.rs:87-89"]
    CDP["check_disjoint_pair — refuse taxonomic, self, or no shared parent<br/>input/consistency.rs:100-131"]
    CALL["check_disjointness calls it per disjoint-with target<br/>check/validate.rs:727-743"]
    TTL["turtle uses is_taxonomic and OWL_DISJOINT_WITH<br/>build/turtle.rs:393-411"]
    WHELK["whelk_inconsistent blocker shape<br/>input/consistency.rs:144-149"]
    TITLE --> SLUGP --> SLUGIFY --> RE
    SLUGIFY --> TARGET
    REFSLUG --> U1
    REFSLUG --> U2
    TARGET --> U3
    SLUGIFY --> U4
    CKEY --> TAX --> CDP --> CALL
    TAX --> TTL
    CDP -.-> WHELK
    note1["DIVERGENCE: a second slugify exists, hand-rolled without regex, in ontomark-model — ontomark-model/src/mint.rs:73-88"]
    note2["DIVERGENCE: whelk_inconsistent has no production caller; only tests/probe_consistency.rs:119 builds it,<br/>despite input/consistency.rs:3-6 saying both write paths read it"]
    note3["DIVERGENCE: link refs use target_slug at ir/model.rs:424 but the resolve closure uses slugify at ir/model.rs:342; they differ on kebab input with leading, trailing or doubled dashes"]
    SLUGIFY --- note1
    WHELK --- note2
    U4 --- note3
```
- **DIVERGENCE:** `iri_for` had no caller and was removed in 8d654d1; the class IRI is built inline from `Page::resource` (`input/page.rs:217-222`) and the resolve fallback (`ir/model.rs:342`).
- **EXTERNAL:** the module doc says the sibling-only rule is shared with VisionClaw's elevation actor (`input/consistency.rs:3-6`); that consumer is not in this repo.

## OM-01.6 Vocabulary loading, and the input modules with no production caller

```mermaid
flowchart TD
    DISC["Vocabulary::discover — walk up for ontology/vocabulary.yaml<br/>input/vocabulary.rs:470-485"]
    CLIUSE["only caller of discover: ontomark-cli resolve_repo<br/>ontomark-cli/src/main.rs:275"]
    LOADV["Vocabulary::load — read file, wrap error with path<br/>input/vocabulary.rs:457-464"]
    LOADERS["other load callers: migrate plan<br/>migrate/plan.rs:413"]
    FROMYAML["from_yaml_str — serde_yaml then check<br/>input/vocabulary.rs:495-499"]
    CHECK["check — EL profile, inverse must be a relation, relation vs scalar clash, logical IRI cannot carry signature, min over max<br/>input/vocabulary.rs:501-529"]
    NONEL["error codes NON_EL_VOCABULARY and SIGNATURE_ON_LOGICAL_RELATION<br/>input/vocabulary.rs:73-79"]
    RESERVED["reserved keys always legal: type title resource public ...<br/>input/vocabulary.rs:427-445"]
    KNOWN["is_known_key = relation OR scalar OR reserved<br/>input/vocabulary.rs:661-665"]
    STRICT["UNKNOWN_KEY only when VaultKind::Knowledge<br/>input/page.rs:125-129, check/validate.rs:285-295"]
    DEFN["definition::parse — defines-as conjunction of names and existentials<br/>input/definition.rs:167"]
    CODE["CodeMap::scan — only paired fences are code<br/>input/code.rs:42-49"]
    CODEUSE["used by validate residue check and repair<br/>check/validate.rs:229, check/repair.rs:270"]
    PROMO["promotion::apply — state machine, blockers veto, only a human approves<br/>input/promotion.rs:219-226"]
    PROMOUSE["only Blocker is used outside the module<br/>input/consistency.rs:35, check/conflicts.rs:20, check/validate.rs:751"]
    DISC --> LOADV --> FROMYAML --> CHECK --> NONEL
    DISC --- CLIUSE
    LOADV --- LOADERS
    CHECK --> RESERVED --> KNOWN --> STRICT
    FROMYAML -.-> DEFN
    CODE --> CODEUSE
    PROMO --> PROMOUSE
    note1["DIVERGENCE: PromotionState, Transition and apply have no caller outside input/promotion.rs — a state machine with no driver in this workspace, input/promotion.rs:30-93"]
    note2["DIVERGENCE: dead or test-only public API — parse_page input/page.rs:598, Vault::get and get_mut input/page.rs:493-503; all_pages is now read by the build digest loop, ontomark-compile/src/build/mod.rs:866"]
    PROMO --- note1
    PROMOUSE --- note2
```
- **DOC-DRIFT:** `parse_page` is documented as the entry point `VisionClaw`'s `CorpusSource::LocalDirectory` calls (`input/page.rs:592-596`), but no caller exists in this workspace.
- **INVARIANT:** an unparseable or inconsistent vocabulary is a load error naming the file, never a partial vocabulary (`input/vocabulary.rs:460-463`, `input/vocabulary.rs:495-499`).
