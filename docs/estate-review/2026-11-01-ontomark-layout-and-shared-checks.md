---
title: Ontomark layout contracts and shared crate consumption
date: 2026-11-01
status: assessed
type: explanation
---

# Ontomark layout contracts and shared crate consumption

The audit reproduced disagreement between validation and publication for custom vault layouts. It also found aliases, nested roles and escaping roots that could classify or ingest files incorrectly. The operator authorised fixes in the shared Ontomark owner and consumption through VisionClaw crate features. The implementation keeps the existing producer migration gates: sharing filesystem checks does not establish bundle parity or retire VisionClaw's producer.

## Scope and evidence

This is a whole-document assessment of **ADR-OM-008**. ADR-OM-007 is context only: this batch does not reverify every ported read/write verb in that record. Selected historical panels **OM-01.1, OM-01.2 and OM-05.4** are reconciled; the new **OM-12** panels describe the implemented owner/consumer boundary. No other panels receive review credit from rendering their topics.

The baseline is Ontomark `0caae5d4b695cb720942b9b9212017c00a80ad41` and VisionClaw `ecd78b6cb25d6e702e6c2459c209ccdd2c5c6b4b`. The [evidence directory](evidence/2026-11-01/om-layout/README.md) retains pinned source exports, hashes and actual CLI probes. All fixtures are synthetic. No real private corpus was published, no production incident was inferred and no service was redeployed.

## Findings and disposition

| ID | Reproduced baseline behaviour | Shared-check disposition and remaining boundary |
| --- | --- | --- |
| OM-L1 | With knowledge `nested/drafts`, working `drafts` and required status, `validate` refuses the missing field; `build` infers working by suffix and publishes under working. | The build resolves the complete declared role with `Layout::kind_of_at`. The permanent regression requires validation refusal and no promoted output. The legacy lexical helper remains documented for source compatibility. |
| OM-L2 | Knowledge `.` plus working `drafts` loads a draft into both roles; its class enters the ontology in public and machine scopes. | Canonical roots may not contain each other. Flat knowledge requires no working role. Fixtures test refusal before output creation in both scopes. |
| OM-L3 | `notes` plus `./notes` or `notes/../notes` is accepted as two roles. | Equality after canonical resolution is refused. Parent components are invalid even when normalisation would remain inside the repository. |
| OM-L4 | Implicit absolute cwd finds a malformed ancestor vocabulary; explicit `--repo .` misses it and succeeds with a generic profile. | Owner discovery canonicalises the start before walking ancestors. VisionClaw consumes `discover_path` and retains its own vocabulary parser. Real CLI tests exercise the relative/implicit equivalence. |
| OM-L5 | Passing no exclusions loads a template as a page and loads the same journal as both a page and journal. | Built-in furniture exclusions always apply to `load_excluding`; profile entries add to them. Low-level `walk_pages` still takes an explicit list for separate journal traversal. |
| OM-L6 | Knowledge `../outside` reads a sibling directory beyond the repository. | The owner rejects absolute paths, parent components and escaping root symlinks; it checks nearest existing ancestors for optional missing roots. This is configuration validation, not a concurrent-filesystem sandbox. |
| OM-L7 | A custom `required_keys: [owner]` accepts a page without `owner`. | **Open.** The list also feeds generated schema; runtime enforcement is deliberately limited to knowledge `type`/`status`. A first enforcement attempt broke an existing publication test because the estate profile also lists `title` and `generated`. That expansion was removed. Full enforcement needs a separate policy/corpus migration. |
| OM-L8 | A working-root symlink to knowledge loads the same directory as two roles despite `follow_links(false)` on child entries. | The owner resolves root links before checking overlap. VisionClaw consumes the root check and skips child links under its default feature, including cycles. Feature-off local ingestion retains legacy behaviour. |
| OM-L9 | `migrate plan --vault working` on a flat layout with no working role returns an empty ready plan. | CLI and library planning now validate selected roots explicitly; the CLI returns 2 for the absent role. Missing optional ingest halves remain skippable in VisionClaw, which is a different operation. |
| OM-L10 | Security review flagged rollback's asymmetric plain-name guard. A `..` migration name reaches `.ontomark/migrations/../abc/journal.jsonl`. | A red CLI test reproduced journal access; the guard now rejects `..` in migration names as it already did in digests. Separators already limited this to one level under `.ontomark`; this is not evidence of arbitrary filesystem escape. |

The fresh post-release review exposed three further boundaries:

| ID | Follow-up evidence | Disposition |
| --- | --- | --- |
| OM-L11 | `journals/` is a separate walk root. A symlink bypasses the child-link setting; loader and planner read it, and body/fence repair can operate on the linked scope. | Six red probes included loader, planner and body-repair failures. The follow-up rejects linked journal roots and linked repair scopes before writing; a separate fence-repair regression covers the sibling entry point. |
| OM-L12 | A planned journal directory is moved outside the repository and replaced by a link before apply or rollback. Matching content hashes still permit external writes. A linked `.ontomark` redirects migration bookkeeping. | Apply/rollback reuse the owner's resolved-path check on target parents and journal/log/originals paths. Conflicting targets remain unchanged; an additional test covers an existing linked log file. This handles stable redirected paths, not concurrent replacement during a filesystem operation. |
| OM-L13 | Source review finds `FrontmatterGrammar.public_gate` declared but unused by legacy publication, whose `Page::is_public` reads `public` directly. | **Open policy finding.** No custom-key runtime guarantee is established. This filesystem repair does not thread a new visibility policy through projection, redaction and migration; doing that needs a separate publication contract and privacy regression suite. |

The nine baseline probes assert actual loader or CLI output. The eight layout repairs have permanent CLI regressions; the required-key probe remains audit evidence for an open finding. The rollback case is an additional regression prompted by the security review.

## ADR-OM-008 section assessment

| Section or obligation | Assessment |
| --- | --- |
| Metadata, implementation list and related decisions | The record's partial/live metadata is a coarse description of a feature introduced in 0.4.0. Its verification paragraph explicitly records a documentation commit, with no armed paths; it is not fresh implementation evidence. New owner/consumer decisions separately record this repair and staged activation. |
| Context: plain vault access and hard-coded estate layout | Plain-vault mode and profile-based folders exist. The context's concern about role inference remained in the suffix helper and build caller, reproduced by OM-L1. |
| Context: estate contract must not move | The fixes retain role output names and generation naming. Existing bundle/CLI fixtures are evidence for those cases. The historical full-corpus parity claim is not freshly rerun here. |
| Decision 1: absent vocabulary means a plain vault | Implemented fallback is reachable; generic vocabulary and a built-in flat layout are exercised by the existing five plain-vault CLI cases. Discovery was sensitive to the lexical spelling of the start (OM-L4); canonical discovery repairs that difference. |
| Decision 1: malformed vocabulary remains an error | Correct when the file is discovered. The baseline counterexample bypassed discovery rather than accepting malformed YAML. The fixed relative and implicit starts find the same file. |
| Decision 1: built-in versus supplied profiles | Built-ins flatten only a plain vault. Supplied profile YAML preserves its declared layout and must now pass shared root checks. This keeps data-driven layouts without permitting aliases or escapes. |
| Decision 1: estate behaviour is byte-identical | Retain the original `conformance/parity.sh`/estate-small receipt as historical. Fresh unit and integration tests do not establish a current real-corpus producer parity window. |
| Decision 2: folders are layout data | `dir_of`, `path_of` and `root_of` place roles. The suffix-based inverse was not a true inverse for custom paths. `kind_of_at` now compares complete resolved roots; the legacy helper is no longer used by production build classification. |
| Decision 2: `.` and working constraints | Textual inequality did not prevent canonical aliases or nested roots (OM-L2/L3/L8). Validation now checks the raw declared roles together, including existing symlinks and missing leaves. |
| Decision 2: output role names remain stable | Role keys remain `knowledge` and `working`; output staging remains `publish/pages` and `publish/working`. No generation or identity remint is introduced. |
| Decision 2: absent roles refuse by name | Normal CLI selection already uses `MissingVault`. Migration planning bypassed the check (OM-L9); both entry points now check it. This review does not claim every lower-level API refuses every absent path identically. |
| Decision 3: profile supplies generation prefix | The implementation uses the profile override or canonical repository basename. Existing profile tests and six bundle-contract tests passed at baseline. `BundleManifest` continues validating the prefix carried in the ID itself. The omitted-profile digest finding in the preceding report remains open. |
| Decision 4: unknown-key policy | Role `unknown_keys` selects fail versus permissive validation. This change does not alter that policy. |
| Decision 4: required keys | The named runtime rules are `MISSING_TYPE` and `MISSING_STATUS` on knowledge ontology pages. The broader field name and schema use can imply more than is enforced (OM-L7). Working remains tolerant. The repair clarifies this boundary rather than expanding policy implicitly. |
| Alternative: generic layout should always be flat | The original rejection still explains why the serde/default estate layout is not globally flattened. The CLI continues applying flat layout only to a plain vault with a built-in profile. |
| Alternative: consumers validate prefixes from profiles | The original choice remains: bundle readers validate the ID's embedded prefix without loading the producer profile. This is structural validation, not complete source/configuration provenance. |
| Alternative: retain `dir_name` alias | Historical breaking rename to role-oriented terminology is preserved. No old directory alias is reintroduced. The new exact-root API addresses a different inverse-mapping problem. |
| Consequence: 0.4.0 API/version transition | Historical release statement, not the version of this repair. Current compiler and CLI release versions are recorded in the implementation receipt. Unchanged workspace crates do not need unrelated version bumps. |
| Consequence: installable CLI and estate profile dependency | The CLI remains the `ontomark` package with a separately versioned estate-profile dependency. Consumer checks need only `ontomark-compile` with default features disabled. |
| Consequence: validate/build/query/migrate any vault | Positive plain-vault fixtures support the named cases; custom-layout counterexamples show why this was not an unconditional guarantee. The shared validation now refuses invalid layouts before work. This batch does not test every verb/profile/filesystem combination. |

## Implementation boundary

Ontomark owns `profile::resolve_roots`, `Layout::validate`, `Layout::kind_of_at` and canonical vocabulary discovery. VisionClaw's default `vault/shared-layout` and server `shared-corpus-layout` features consume the published compiler crate without its default reasoner feature. The local source checks all configured roots before listing, bounds explicit prefixes and fetches, and skips child symlinks. The CLI keeps its existing parser and producer.

The security gates reported no high-severity finding. The first owner medium finding led to OM-L10; the fresh review led to OM-L11–L13. Its consumer findings describe the retained feature-off behaviour, including symlink recursion; they are recorded compatibility limits, not evidence that the feature-enabled path is affected. The review inspected an ephemeral CPU harness as well as production source, so its file labels are not a substitute for source verification.

The focused local-reader harness compiles the actual production module and the actual port definitions, exercising the same unit tests without the server's GPU dependency graph. It establishes those functions' behaviour; it is not server linking, boot or deployment evidence. The receipt separately records the full server build check and all package/feature gates.

## Diagram reconciliation

Historical OM-01.1 now states the caller-dependent journal separation at its retained pin. OM-01.2 distinguishes child-link traversal from root-link resolution. OM-05.4 names lexical discovery and the migration exception. Their original pins remain because only selected panels were assessed. The retained old-pin diff includes `input/page.rs` and `check/bodies.rs`: CommonMark changes and shifted lines mean it would be false to call every old runtime file byte-identical.

OM-12 shows current owner resolution, validation/build role agreement and the feature-gated consumer boundary. Rendering sibling panels confers no source-review credit.

## Comparisons and user utility

[Obsidian's storage model](https://obsidian.md/help/Files%2Band%2Bfolders/How%2BObsidian%2Bstores%2Bdata) treats a vault as a folder with subfolders and warns about nesting vaults because links may not update reliably. That supports presenting distinct source roots clearly, but is not an external security guarantee for this compiler's role checks.

[Quartz's private-page controls](https://quartz.jzhao.xyz/features/private-pages) distinguish publication filtering from paths excluded from parsing. Ontomark's built-in furniture exclusion is a parsing boundary; public flags and publication projection are separate. Neither folder placement nor a website filter should be presented as a complete confidentiality policy.

The practical improvement is consistent feedback before work: the same invalid layout is refused by validation, publication and consuming applications, with the conflicting paths named. Keep a configuration preview showing resolved roots, role assignment, exclusions and optional missing sources. Pair it with the separate audience preview and provenance receipt proposed in the machine-bundle review; layout validity alone cannot establish publication privacy or consumer parity.

## Evidence preservation incident

The first owner worktree was removed before an asynchronous scratch-preservation
operation finished. Moving its build cache across filesystems filled the small
`/tmp` mount; the original owner security scratch was lost and one consumer
registry-test log was incomplete. Committed/pushed source, published crates and
prior completed test logs survived. The partial copied cache was removed, the
consumer registry tests were rerun successfully, and a fresh security receipt at
`a55033c` was preserved outside the worktree before further work. That fresh review
also exposed the journal-root follow-up. The evidence package records the incident;
it does not claim the original receipt or cache was preserved byte-for-byte.

## Release and verification receipt

The owner releases are **ontomark-compile 0.6.3** and **ontomark 0.7.2**, published
from `4b9d15df48056971321a62b8c4e905f4536c74db` (journal implementation
`395c18b3c8f037bd9fe883121b515506f4430d8d`). The preceding layout release was
compiler 0.6.2 / CLI 0.7.1 at `a55033cfc0ce906d9230ffc62dfe6ea8de1e0219`.
VisionClaw consumes the final registry patch at
`efbd7f6f080df06ecb580254bddabcee59d4fbf8`; its lock records SHA-256
`72cf2bc78aebf9430941cc5525900383b325b79be7eb67ca0a4283185631ffbc`.
The actual consumer branch parent was `5e1bd90a434818b066f9c0d769940e5f61640804`:
a pre-existing metadata-only update after the recorded `ecd78b6` baseline.
Its product source was unchanged; the affected ADR-2166 review has a scoped annex.

| Final gate | Result and boundary |
| --- | --- |
| Owner workspace tests | 1,113 passed, 8 ignored |
| Owner compiler all features | 693 passed, 7 ignored |
| Owner compiler without default features | 589 passed, 2 ignored |
| Owner format, strict workspace Clippy and rustdoc | Passed |
| Owner WASM release builds | `ontomark-wasm` and `ontomark-ingest` passed |
| Owner package and registry publication | Compiler package verified; both releases published |
| VisionClaw registry vault/core tests | 546 passed, 5 ignored |
| VisionClaw vault without default features | 388 passed, 5 ignored |
| Actual local-reader module | 13 feature-enabled tests passed against 0.6.3; 9 feature-off tests passed in the preceding unchanged-module run |
| Consumer strict Clippy | Vault and actual-module harness passed against 0.6.3 |
| Consumer default server library | `cargo check` passed; existing unused-import and quick-xml future-compatibility warnings remain |
| Consumer formatting | Changed Rust files passed; whole-workspace formatting has pre-existing unrelated differences |
| Consumer ADR index | Scoped changed-path re-verification; no fresh historical runtime claims |

Six journal regressions were observed failing before the patch, then passing.
Two further permanent cases cover fence repair and a linked journal log; the
final suite contains 29 migration tests. Tests ran with four test threads and
eight build jobs after an earlier unbounded doctest-linking run was interrupted.
Interrupted and failed preliminary logs remain labelled evidence, not final gates.

All four OM-12 renders and the corrected OM-05.4 annotation were visually
inspected, along with OM-01.1–2. The new panels have no clipped labels.
Existing sibling panels rendered during topic checks receive no additional
review credit. Full-corpus citation and completion receipts accompany the
[evidence manifest](evidence/2026-11-01/om-layout/README.md).

Concurrent consumer integration: incoming main `434afb6b7` was merged at
`138532019`, preserving both ADR-2005 evidence records; the dependency-direction
checks and server library compile check passed. Incoming `d6c091f0f` then changed
GPU settle behaviour. Merge `fcb924be9` passed the server compile check again;
the shared-layout paths stayed unchanged. The final scoped receipt is retained
separately. Four incoming ADR staleness failures (2035, 2140, 2143, 2169) remain
outside this compiler change; the earlier passing index is not represented as
a clean final global gate. The estate ledger also exposes three changed prior
assessments (2143, 2166, 2167) rather than rebinding their source hashes without
review. This batch completes the layout repair, not the wider estate audit.
