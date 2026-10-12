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
verified_commit: ffe06e45bfebd388a1d85849b24a8810e4d0d610
---

## OM-01.1 Vault root to Vault — pages, journals, skips

```mermaid
flowchart TD
    CLI["ontomark-cli load per VaultKind dir<br/>ontomark-cli/src/main.rs:272-278"]
    BUILD["build::run load_vault on vault_root<br/>build/mod.rs:311"]
    LV["load_vault — kind = Working iff dir name is working, else Knowledge<br/>input/page.rs:568-576"]
    LOAD["Vault::load root, kind<br/>input/page.rs:390"]
    NOPAGES["no pages dir → VaultError::NotAVault<br/>input/page.rs:392-398"]
    WALK["walk_pages pages_dir — see OM-01.2<br/>input/page.rs:399"]
    ID["id = path under pages without .md, backslash to slash<br/>input/page.rs:405-408"]
    READ["read_to_string, io error carries the path<br/>input/page.rs:413"]
    PARSE["Page::parse path, rel_path, id, text — see OM-01.3<br/>input/page.rs:414"]
    JDIR{"root/journals is a dir?<br/>input/page.rs:418-421"}
    JWALK["walk_pages journals_dir — same walker<br/>input/page.rs:422"]
    JID["journal id = relative stem, not a date check<br/>input/page.rs:427-430"]
    JPARSE["Page::parse into journals Vec<br/>input/page.rs:436"]
    JSKIP["journal skips re-rooted under journals/<br/>input/page.rs:438-441"]
    VAULT["Vault root, kind, pages, journals, skipped<br/>input/page.rs:362-378"]
    JDATE["journal_date only checked later, by validate → JOURNAL_NOT_DATED<br/>check/validate.rs:584-590"]
    CLI --> LV
    BUILD --> LV
    LV --> LOAD --> NOPAGES
    LOAD --> WALK --> ID --> READ --> PARSE --> JDIR
    JDIR -- yes --> JWALK --> JID --> JPARSE --> JSKIP --> VAULT
    JDIR -- no --> VAULT
    VAULT -.-> JDATE
    note1["INVARIANT: journals are kept out of Vault.pages so projections keyed on pages cannot silently grow,<br/>input/page.rs:369-373"]
    note2["INVARIANT: the first malformed frontmatter or unreadable file aborts the whole load — Page::parse errors propagate,<br/>input/page.rs:414"]
    note3["DOC-DRIFT: module doc says one reader is shared by VisionClaw ingest and the vault CLI, input/page.rs:1-2, but<br/>ir/projection.rs:77 and ir/projection.rs:172 walk and split independently — see OM-01.2"]
    VAULT --- note1
    PARSE --- note2
    LV --- note3
```
- **EXTERNAL:** the whole `input/` tree is a temporary stand-in relocated from VisionClaw `vault-core`, to be deleted once the input adapter reads `ontomark-model` (`input/mod.rs:3-8`); ADR-OM-001 owns that replacement.
- **DIVERGENCE:** `ontomark-parse` carries its own frontmatter reader (`ontomark-parse/src/frontmatter.rs:1-9`) beside the one used here (`input/frontmatter.rs:119`); the two are not shared in production.

## OM-01.2 walk_pages — what is declined, and the three other walkers

```mermaid
flowchart TD
    W["walk_pages pages_dir — WalkDir, follow_links false<br/>input/page.rs:492-496"]
    ISFILE{"entry is a regular file?<br/>input/page.rs:501-503"}
    DROPSYM["not a file → continue, NO Skip recorded — a symlinked .md vanishes silently<br/>input/page.rs:501-503"]
    ISMD{"extension is md?<br/>input/page.rs:505-507"}
    DROPNONMD["non-md → continue, not reported<br/>input/page.rs:505-507"]
    HIDDEN{"any path component starts with a dot?<br/>input/page.rs:512"}
    SKIP["Skip HiddenDirectory — the only SkipReason variant<br/>input/page.rs:513-517, input/page.rs:63-67"]
    KEEP["rels.push parts<br/>input/page.rs:519"]
    SORT["component-wise sort of rel paths, skips sorted by path<br/>input/page.rs:521-522"]
    OUT["PageWalk files, skipped<br/>input/page.rs:92-97"]
    W --> ISFILE
    ISFILE -- no --> DROPSYM
    ISFILE -- yes --> ISMD
    ISMD -- no --> DROPNONMD
    ISMD -- yes --> HIDDEN
    HIDDEN -- yes --> SKIP
    HIDDEN -- no --> KEEP
    KEEP --> SORT --> OUT
    UNPUB["_misc is NOT declined here — publication rule, kept as a page<br/>input/page.rs:38, input/page.rs:486-488"]
    CALLERS["callers: Vault::load, migrate read_considered, repair, bodies<br/>input/page.rs:399, migrate/plan.rs:314, check/repair.rs:375, check/bodies.rs:811"]
    P2["SECOND WALKER inspect_inputs — raw WalkDir, errors swallowed, plain sort<br/>ir/projection.rs:77-84"]
    P2H["its own hidden and unpublished test, counted not reported as Skip<br/>ir/projection.rs:93-96"]
    P3["THIRD WALKER excluded_identities — only unpublished pages, parses each via Page::parse<br/>ir/projection.rs:172-189"]
    RUN["build::run runs the census BEFORE load_vault<br/>build/mod.rs:309-311"]
    OUT --- CALLERS
    OUT --- UNPUB
    RUN --> P2 --> P2H
    P2 -.- P3
    note1["DIVERGENCE: three walkers decide what a page is. Only walk_pages follows the Skip contract —<br/>input/page.rs:79-81 promises every skip is reported, but symlinks are dropped unreported, input/page.rs:501-503"]
    note2["DOC-DRIFT: walk_pages doc says only dot-directories are skipped, input/page.rs:486, but the test is on every component<br/>including the file name, input/page.rs:512"]
    OUT --- note1
    HIDDEN --- note2
```
- **DIVERGENCE:** `UNPUBLISHED_DIRS = ["_misc"]` (`input/page.rs:38`) is not the only held-back rule: `build/publish.rs` adds `misc` at any depth (`build/publish.rs:130-142`), so the publish scope has two lists.
- **INVARIANT:** hidden-directory pages are reported on `Vault.skipped` (`input/page.rs:374-377`) and surface in the validation report (`check/validate.rs:329`).

## OM-01.3 Frontmatter parse and the public, type and resource gates

```mermaid
flowchart TD
    TXT["page text"]
    SPLIT["frontmatter::split — strip BOM, opening --- must be line 1<br/>input/frontmatter.rs:119-123"]
    NOFM["no opening fence → yaml None, body is whole file, body_line 1<br/>input/frontmatter.rs:125-129"]
    CLOSE["closing --- found → yaml and body, body_line computed<br/>input/frontmatter.rs:131-142"]
    UNCLOSED["opening fence but no closing fence → falls through to yaml None, no error<br/>input/frontmatter.rs:153-157"]
    FMPARSE["Frontmatter::parse — empty or null → default, non-mapping → error<br/>input/frontmatter.rs:166-178"]
    PERR["YAML error → VaultError::Frontmatter with path<br/>input/page.rs:159-162"]
    PAGE["Page with Frontmatter map<br/>input/page.rs:165-172"]
    PUBLIC["Page::is_public — boolean public, absent = private<br/>input/page.rs:211-213"]
    BOOL["boolean accepts real bool OR the strings true false, any case<br/>input/frontmatter.rs:230-240"]
    OKF["Page::okf → OkfBlock::from_frontmatter, never fails<br/>input/page.rs:217-219, input/okf.rs:264"]
    OKFF["type_name, resource, status parse, generated, verified, sources<br/>input/okf.rs:274-288"]
    HV["is_human_verified — any verified stamp by a human<br/>input/okf.rs:294-296"]
    GTYPE["MISSING_TYPE if knowledge page has no type<br/>check/validate.rs:722-728"]
    GINV["INVALID_TYPE if type not in vocab types or working_types<br/>check/validate.rs:729-742"]
    GREQ["MISSING_REQUIRED for each required key; resource needs iri and has_ontology<br/>check/validate.rs:743-758"]
    GONT["has_ontology = has resource OR has type<br/>ir/model.rs:429"]
    GIRI["MISSING_CLASS_IRI when has_ontology but iri empty<br/>check/validate.rs:454-460"]
    GRES["resource = frontmatter resource else namespace + slug<br/>input/page.rs:202-207"]
    TXT --> SPLIT
    SPLIT --> NOFM
    SPLIT --> CLOSE
    SPLIT --> UNCLOSED
    CLOSE --> FMPARSE --> PERR
    FMPARSE --> PAGE
    NOFM --> PAGE
    UNCLOSED --> PAGE
    PAGE --> PUBLIC --> BOOL
    PAGE --> OKF --> OKFF --> HV
    OKFF --> GTYPE --> GINV --> GREQ
    PAGE --> GRES --> GONT --> GIRI
    note1["INVARIANT: an unfenced or absent frontmatter means private — Frontmatter::default has no public key,<br/>input/page.rs:163, input/page.rs:212"]
    note2["DIVERGENCE: unclosed opening fence is silently read as no-frontmatter, so the page is private and its YAML becomes body text,<br/>input/frontmatter.rs:153-157"]
    PUBLIC --- note1
    UNCLOSED --- note2
```
- **DIVERGENCE:** there are four "is public" decisions: `Page::is_public` lenient to the strings `true`/`false` (`input/frontmatter.rs:233-236`), the build census strict to a real bool and refusing anything else (`ir/projection.rs:112-116`), `ontomark-model` lenient only if the profile says so (`ontomark-model/src/profile.rs:604-611`), and the wasm profile strict (`ontomark-wasm/src/profile.rs:316-321`).
- **DOC-DRIFT:** the census promises `"true"` the string never becomes publication permission (`ir/projection.rs:8-9`), yet validate, publish and redact all call the lenient `Page::is_public` (`check/validate.rs:384`, `build/publish.rs:220`, `build/redact.rs:166`); the census alone guards the string case, and only on the build path (`build/mod.rs:309`).

## OM-01.4 Secrets scanner — body only, kind and line never value

```mermaid
sequenceDiagram
    autonumber
    participant V as check/validate.rs<br/>check/validate.rs:305
    participant P as build/publish.rs<br/>build/publish.rs:224
    participant S as input/secrets.rs<br/>input/secrets.rs:166
    participant R as OnceLock regex set<br/>input/secrets.rs:114
    V->>S: scan page.body for EVERY page and journal (:305)
    P->>S: scan page.body for PUBLIC pages only (build/publish.rs:220-224)
    S->>R: patterns compiled once, 11 CredentialKind variants (input/secrets.rs:115-157)
    R-->>S: each regex find_iter, byte offset to line by binary search (input/secrets.rs:170-180)
    S-->>V: Finding line and kind, sorted and deduplicated (input/secrets.rs:181-183)
    V->>V: file line = body_line + finding.line - 1 (check/validate.rs:307)
    V->>V: Severity::Error SECRET_DETECTED, value never reported (check/validate.rs:308-318)
    S-->>P: Finding list (input/secrets.rs:98-104)
    P->>P: same line arithmetic, SecretsFound blocks the staging (build/publish.rs:228, build/publish.rs:246-248)
    Note over S: INVARIANT: Finding has no value field, so a report cannot echo a secret — input/secrets.rs:94-104
    Note over R: INVARIANT: no credential character class contains a dollar sign, so env references are not matches — input/secrets.rs:106-113
    Note over V: DIVERGENCE: only page.body is scanned. A credential in frontmatter, for example a sources or title value, is never seen — check/validate.rs:305, build/publish.rs:224
    Note over P: DIVERGENCE: has_secret is public API with no production caller — input/secrets.rs:191-193
```
- **INVARIANT:** the build runs the credential gate first, before validation, so a key on a public page gets its own exit code rather than being swallowed by the generic validation failure (`build/mod.rs:315-324`).
- **INVARIANT:** `journals/` pages are secret-scanned by validate (`check/validate.rs:580-581`) but are never staged for publication (`build/publish.rs:207-209`).

## OM-01.5 Slug derivation and the consistency vocabulary

```mermaid
flowchart TD
    TITLE["Page::title — frontmatter title, else file stem<br/>input/page.rs:177-188"]
    SLUGP["Page::slug — frontmatter slug, else slugify of title<br/>input/page.rs:192-197"]
    SLUGIFY["slugify — lowercase, non a-z0-9 runs to dash, trim dashes<br/>input/slug.rs:29-33"]
    RE["Regex non-slug compiled once<br/>input/slug.rs:15-18"]
    REFSLUG["ref_slug — text after the last colon, no re-slugify<br/>input/slug.rs:47-49"]
    TARGET["target_slug — kebab target verbatim, else slugify<br/>input/slug.rs:66-76"]
    IRIFOR["iri_for namespace + slugify title<br/>input/slug.rs:88-90"]
    THING["is_thing — thing, owl-thing, owl:thing<br/>input/slug.rs:95-100"]
    U1["closure.rs uses ref_slug and is_thing<br/>ir/closure.rs:19, ir/closure.rs:123-124"]
    U2["indexes.rs uses ref_slug<br/>build/indexes.rs:13, build/indexes.rs:58"]
    U3["Corpus links use target_slug<br/>ir/model.rs:424"]
    U4["Corpus resolve fallback uses plain slugify, not target_slug<br/>ir/model.rs:342"]
    CKEY["class_key — tail after last slash hash or colon, lowercased<br/>input/consistency.rs:62-69"]
    TAX["is_taxonomic → Taxonomy::is_taxonomy_node<br/>input/consistency.rs:87-89"]
    CDP["check_disjoint_pair — refuse taxonomic, self, or no shared parent<br/>input/consistency.rs:100-131"]
    CALL["check_disjointness calls it per disjoint-with target<br/>check/validate.rs:674-690"]
    TTL["turtle uses is_taxonomic and OWL_DISJOINT_WITH<br/>build/turtle.rs:393-411"]
    WHELK["whelk_inconsistent blocker shape<br/>input/consistency.rs:144-149"]
    TITLE --> SLUGP --> SLUGIFY --> RE
    SLUGIFY --> IRIFOR
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
- **DIVERGENCE:** `iri_for` is exported (`input/slug.rs:88`) but nothing in the workspace calls it; Corpus builds the IRI inline instead (`ir/model.rs:342`, `input/page.rs:206`).
- **EXTERNAL:** the module doc says the sibling-only rule is shared with VisionClaw's elevation actor (`input/consistency.rs:3-6`); that consumer is not in this repo.

## OM-01.6 Vocabulary loading, and the input modules with no production caller

```mermaid
flowchart TD
    DISC["Vocabulary::discover — walk up for ontology/vocabulary.yaml<br/>input/vocabulary.rs:470-485"]
    CLIUSE["only caller of discover: ontomark-cli resolve_repo<br/>ontomark-cli/src/main.rs:241"]
    LOADV["Vocabulary::load — read file, wrap error with path<br/>input/vocabulary.rs:457-464"]
    LOADERS["other load callers: migrate plan<br/>migrate/plan.rs:413"]
    FROMYAML["from_yaml_str — serde_yaml then check<br/>input/vocabulary.rs:495-499"]
    CHECK["check — EL profile, inverse must be a relation, relation vs scalar clash, logical IRI cannot carry signature, min over max<br/>input/vocabulary.rs:501-529"]
    NONEL["error codes NON_EL_VOCABULARY and SIGNATURE_ON_LOGICAL_RELATION<br/>input/vocabulary.rs:73-79"]
    RESERVED["reserved keys always legal: type title resource public ...<br/>input/vocabulary.rs:427-445"]
    KNOWN["is_known_key = relation OR scalar OR reserved<br/>input/vocabulary.rs:661-665"]
    STRICT["UNKNOWN_KEY only when VaultKind::Knowledge<br/>input/page.rs:119-123, check/validate.rs:285-295"]
    DEFN["definition::parse — defines-as conjunction of names and existentials<br/>input/definition.rs:167"]
    CODE["CodeMap::scan — only paired fences are code<br/>input/code.rs:42-49"]
    CODEUSE["used by validate residue check and repair<br/>check/validate.rs:229, check/repair.rs:270"]
    PROMO["promotion::apply — state machine, blockers veto, only a human approves<br/>input/promotion.rs:219-226"]
    PROMOUSE["only Blocker is used outside the module<br/>input/consistency.rs:35, check/conflicts.rs:20, check/validate.rs:698"]
    DISC --> LOADV --> FROMYAML --> CHECK --> NONEL
    DISC --- CLIUSE
    LOADV --- LOADERS
    CHECK --> RESERVED --> KNOWN --> STRICT
    FROMYAML -.-> DEFN
    CODE --> CODEUSE
    PROMO --> PROMOUSE
    note1["DIVERGENCE: PromotionState, Transition and apply have no caller outside input/promotion.rs — a state machine with no driver in this workspace, input/promotion.rs:30-93"]
    note2["DIVERGENCE: dead or test-only public API — page_files input/page.rs:536, parse_page input/page.rs:557, Vault::all_pages input/page.rs:468, Vault::get, get_mut, public input/page.rs:454-475"]
    PROMO --- note1
    PROMOUSE --- note2
```
- **DOC-DRIFT:** `parse_page` is documented as the entry point `VisionClaw`'s `CorpusSource::LocalDirectory` calls (`input/page.rs:550-553`), but no caller exists in this workspace.
- **INVARIANT:** an unparseable or inconsistent vocabulary is a load error naming the file, never a partial vocabulary (`input/vocabulary.rs:460-463`, `input/vocabulary.rs:495-499`).
