# WebVOWL WASM breakout — scoping note

Read-only survey, 2026-09-06. Scopes a proposed standalone crate/repo for the
estate's Rust/WASM `webvowl-wasm` code, currently duplicated three times.

## 0. What actually exists (corrects the brief: three copies, not two)

| # | Path | Repo (git remote) | Crate ver | npm name | Role |
|---|---|---|---|---|---|
| A | `WasmVOWL/rust-wasm` | `DreamLab-AI/WasmVOWL` (standalone) | 1.0.0 | `webvowl-wasm` | Smaller, cleaner rewrite (2,783 LOC). No live deployment found. |
| B | `knowledgeGraph/explorer/rust-wasm` | `DreamLab-AI/knowledgeGraph` (standalone) | 0.3.4 | `@dreamlab-ai/webvowl-wasm` | Larger fork (10,335 LOC): pinning, statistics, Barnes-Hut quadtree + SIMD layout, OWL2 validator, NGG1 binary reader, markdown ontology parser. Consumed via a `file:` path dep + dynamic import in `explorer/modern`. No live deploy found in this repo (its own CI is explicitly "build and verify, no deploy, by design"). |
| C | `visionGraph/publishing-tools/WasmVOWL/rust-wasm` | inside `visionGraph` repo, plain committed files (no `.gitmodules`, no nested `.git`) | 0.3.4, byte-identical to B per the repo's own estate-review docs | — | **The only currently-live, production-facing consumer.** `visionGraph/.github/workflows/publish.yml` runs `wasm-pack build` here and `npm install ../rust-wasm/pkg` into `modern/`, then deploys — this is the narrativegoldmine.com corpus-explorer pipeline (`VisionFlow/docs/terminology.md:97`). A **built artefact, `pkg.tar.gz`, is git-committed** in this tree. |

A fourth copy was noticed in passing at `project4/publishing-tools/WasmVOWL` — out of the requested scope, not investigated further, flagged only so it isn't rediscovered as a surprise later.

This changes the shape of the breakout: it is not "pick a winner between two dry-run-clean crates", it is "pick a winner between A and B, then **cut the one live production pipeline (C) over to the published package** and delete the two vendored trees (B's rust-wasm and all of C)."

---

## 1. Licence provenance

### Upstream — VisualDataWeb/WebVOWL

- GitHub API confirms: `VisualDataWeb/WebVOWL`, default branch `master`, `license.key: mit`, JavaScript, org-owned (TIB), 983 stars, still active (pushed 2026-06-04).
- Fetched `license.txt` from `master` directly:

  > The MIT License (MIT)
  > Copyright (c) 2014-2019 Vincent Link, Steffen Lohmann, Eduard Marbach, Stefan Negru, Vitalis Wiens

  Plain MIT, no additional NOTICE file, no dual-licensing, no CLA artefacts visible at repo root. The owner's belief that it "may be Python" does not hold — it is JavaScript (Gruntfile/Webpack/D3), confirmed by both the API `language` field and the root file listing (`Gruntfile.js`, `webpack.config.js`, `src/`). No conflation risk there.

### WasmVOWL/rust-wasm (copy A, v1.0.0)

- `Cargo.toml`: `license = "MIT"`, `authors = ["DreamLab-AI"]`, `repository = "https://github.com/DreamLab-AI/WasmVOWL"` — correctly self-attributed.
- `rust-wasm/package.json`, however, still carries `"author": "WebVOWL Contributors"` and `"repository": "https://github.com/VisualDataWeb/WebVOWL.git"` — **misattributed to upstream**, inconsistent with its own `Cargo.toml`. Minor hygiene defect, not a legal problem, but wrong for an npm publish (a consumer resolving the repo field lands on the wrong project).
- `license.txt` at the repo root is byte-identical to upstream's, copyright notice intact (Link/Lohmann/Marbach/Negru/Wiens 2014–2019) — correct MIT practice for a derivative work.
- Top-level `README.md` states outright: *"DreamLab-AI's WASM-native rebuild of [VisualDataWeb/WebVOWL]"* — explicit, correct upstream attribution in prose.
- No vendored upstream `.js` files, no upstream data/test fixtures (`.owl`/`.ttl`/`.rdf`) found anywhere under `rust-wasm/src` or `tests/`.
- No per-file "ported from" / "adapted from" header comments anywhere in `src/` (grep across all `.rs` files: zero hits).
- Git history for the path is a single squash: `git log --diff-filter=A -- rust-wasm` shows one add-commit, "fully completed first tranche" (2025-11-10) — this is agent-authored greenfield Rust, not an incremental line-by-line port with visible provenance in the commit log.

### knowledgeGraph/explorer/rust-wasm (copy B, v0.3.4)

- `Cargo.toml`: `license = "MIT"`, `authors = ["WebVOWL Contributors"]`, `repository = "https://github.com/VisualDataWeb/WebVOWL"` — **wrongly points at the upstream project**, not this fork (matches the crate-survey's finding). This is the more serious of the two metadata defects: publishing to crates.io with this field unmodified would make the crate's own metadata claim to *be* the upstream JS project's repository.
- `explorer/rust-wasm/README.md:16` claims the package "is published to npm as `@dreamlab-ai/webvowl-wasm` version 0.3.3". **Checked directly against the live registry (`registry.npmjs.org/@dreamlab-ai/webvowl-wasm` → 200)**: the package is real and published, but at `dist-tags.latest: 0.3.2` — one patch *behind* even the README's stale claim, and two behind the local Cargo.toml's 0.3.4. So the publish pipeline that produced 0.3.2 has not kept pace with the source tree for at least two patch releases. Its own package metadata reads *"Rust/WASM powered semantic graph visualization for Logseq knowledge graphs"* — Logseq-specific framing not present in the local README. The unscoped `webvowl-wasm` name (no `@dreamlab-ai/` scope) is confirmed unclaimed on the npm registry too (404), consistent with the crates.io check.
- `explorer/README.md` (top-level, human-facing): *"WasmVOWL is a modern web application for visualizing ontologies, based on the original WebVOWL project... a complete rewrite... Built on the solid foundation of the original WebVOWL"* — explicit, correct attribution in prose, same as copy A.
- `LICENSE-EXPLORER` and `explorer/license.txt` both carry the exact upstream MIT text with the original copyright holders — correct.
- No vendored upstream JS, no upstream data fixtures, no per-file attribution comments (grep: zero hits) — same clean-room shape as copy A.
- Git history: single squash-add commit, "feat: build out main as a self-contained ontology + pipeline release" (2026-07-25) — same pattern as copy A, no incremental port trail.

### Verdict

**Neither copy is a derivative work carrying vendored upstream source or data.** Both are clean-room Rust reimplementations of WebVOWL's *behaviour* (force-directed OWL/RDF graph layout, SVG rendering, the WebVOWL JSON/statistics shape), not the JS code itself — no copied files, no copied fixtures, no line-level provenance trail. That said, both explicitly market themselves in prose as a "rewrite of" / "rebuild of" WebVOWL, both preserve the exact upstream MIT licence text with the original copyright notice, and both are themselves MIT — which is the *correct* posture regardless of whether a court would call this legally a "derivative work" under MIT's terms: MIT's only real obligation (retain copyright notice + permission notice "in all copies or substantial portions") is already satisfied by the checked-in `license.txt`/`LICENSE-EXPLORER` files.

**What a published crate must carry to comply:**
1. Keep the upstream MIT `LICENSE` text with the original 2014–2019 copyright notice intact in the new repo root (both copies already have a correct copy to draw from).
2. Add a `NOTICE`/README attribution line naming VisualDataWeb/WebVOWL as the design/behavioural origin (both READMEs already have serviceable prose — reuse it, don't invent new language).
3. Fix both misattributed metadata fields before publishing anything: copy A's `package.json` `author`/`repository` (currently claims to be upstream), copy B's `Cargo.toml` `repository` (currently points at `VisualDataWeb/WebVOWL` instead of the fork). The published crate's `Cargo.toml` must point at the *new* breakout repo.
4. No upstream NOTICE-style attribution file exists to carry forward (upstream ships only `license.txt`), so nothing else is owed.

No trademark clearance is needed for using MIT-licensed upstream *code* — but see §3 on the "WebVOWL" *name* itself, which is a separate question from the licence.

---

## 2. Consumer inventory

Two independent read-only scans covered: WasmVOWL, knowledgeGraph, VisionFlow, visionGraph, VisionClaw (`project`, incl. `client/`), `project/agentbox`, `dreamlab-ai-website`, and every `.github/workflows/*.yml` in all of them.

### VisionClaw core + client, agentbox, dreamlab-ai-website — **zero code consumers**

Explicit negative, checked thoroughly (the brief specifically flagged the owner's suspicion that agentbox holds/consumes this code):

- `project/agentbox` has 9 files mentioning "webvowl" — all documentation or fixture-label text, zero code dependency:
  - `skills/ontology-core/SKILL.md`, `skills/ontology-core/references/ttl-authoring.md`, `skills/ontology-enrich/SKILL.md`, `skills/SKILL-DIRECTORY.md` — describe agentbox's own OWL2/TTL export as "WebVOWL-compatible" (a Turtle `@prefix`-ordering convention note), not a dependency on the crate.
  - `ontology/code-harness.ttl`, `ontology/decision-layer.ttl` — same TTL-convention comment.
  - `docs/estate-closeout/2026-09-05/*.json`, `scripts/recall-fixtures/recall-fixture.v1.json` — an unrelated RuVector memory-namespace label string `"projects/WasmVOWL"` used in recall-benchmark fixtures.
  - `flake.nix:905` — `wasm-pack`/`wasm-bindgen-cli`/`binaryen` are generic Nix WASM toolchain packages shared by other in-repo Rust→WASM crates (e.g. xr-client); not wired to WasmVOWL specifically.
- `project/client` has a *separate, unrelated* Three.js/JSS-based ontology visualizer (`features/ontology/OntologyBrowser.tsx`, `sparqlService.ts`, `JssOntologyService.ts`) with zero relation to WebVOWL/WASM.
- `dreamlab-ai-website` and every workflow file across all three repos: zero hits.

**Conclusion: publishing has zero downstream blast radius in VisionClaw, its client, agentbox, or dreamlab-ai-website.**

### WasmVOWL (copy A) — 4 points, no live deployment found

| File:line | Mode | Change on publish |
|---|---|---|
| `rust-wasm/Cargo.toml` + `package.json` | crate/npm definition | Rename target for the name collision |
| `modern/package.json:10` `"build:wasm": "cd ../rust-wasm && npm run build"` | in-repo relative build step | Replace with `npm install <published-pkg>` |
| `modern/vite.config.ts` `optimizeDeps.exclude: ['webvowl-wasm']` | bare-specifier config; no matching `import` found in `App.tsx`/hooks (dead or lazily resolved elsewhere) | Update to new package name if renamed |
| `Dockerfile` | downloads upstream WebVOWL 1.1.7 `.war` — a legacy/unrelated path | No change needed |

### knowledgeGraph/explorer (copy B) — 6 points, no live deployment found (repo's own CI is explicitly "build and verify, no deploy, by design")

| File:line | Mode | Change on publish |
|---|---|---|
| `explorer/rust-wasm/Cargo.toml` + `package.json` | crate `webvowl-wasm` / npm `@dreamlab-ai/webvowl-wasm` 0.3.4 | Collision partner; misattributed `repository` field must be fixed regardless |
| `explorer/modern/package.json:42` `"webvowl-wasm": "file:../rust-wasm/pkg"` | **path dependency on built pkg/** | Replace with a real semver pin |
| `explorer/modern/src/workers/physics.worker.ts:99,101` `await import('webvowl-wasm')` | dynamic import by bare specifier (imports as `webvowl-wasm`, relying on the `file:` alias — not the scoped npm name) | Import specifier must match whatever name is finally published |
| `explorer/modern/vite.config.ts` `optimizeDeps.exclude: [...]` | build config | Keep in sync |
| `explorer/.github/workflows/wasm-publish.yml` | references `publishing-tools/WasmVOWL/**` paths that **don't exist in this repo's layout** — GitHub only runs root `.github/workflows`, and this one lives at `explorer/.github/workflows/`, so it is not live CI at all | Dead file — delete regardless of the publish decision |
| `explorer/rust-wasm/README.md:16` | stale/unverified npm-publish claim (0.3.3) | Verify against the real registry before any rename |

### visionGraph — 4 points, **the one live production consumer**

| File:line | Mode | Change on publish |
|---|---|---|
| `publishing-tools/WasmVOWL/` (whole tree) | full vendored copy of copy B, plain committed files, no submodule | Deletion target once cutover lands |
| `.github/workflows/publish.yml:57-189` | **active CI**: installs wasm-pack, runs `wasm-pack build --target web` in `publishing-tools/WasmVOWL/rust-wasm`, then `npm install ../rust-wasm/pkg` in `modern/`, builds and deploys the live narrativegoldmine.com corpus-explorer site | Operative deployment pipeline — swap for `npm install webvowl-wasm@<ver>`, drop the wasm-pack build step entirely |
| `publishing-tools/WasmVOWL/rust-wasm/pkg.tar.gz` | **git-committed built artefact** | Remove once the real dependency lands |
| `publishing-tools/WasmVOWL/.github/workflows/wasm-publish.yml` | inert nested duplicate, same broken-path pattern as knowledgeGraph's copy | Delete |

### VisionFlow — 0 code consumers, ecosystem-map gap confirmed

- `docs/architecture/repository-map.md` lists exactly six repos (VisionFlow, VisionClaw/`project`, agentbox, solid-pod-rs, nostr-rust-forum, dreamlab-ai-website) and its Mermaid dependency-direction diagram. **WasmVOWL, knowledgeGraph, and visionGraph appear nowhere in it** — a real gap, not an oversight to route around: this map is the estate's canonical "what exists" document and currently doesn't know about any of the three copies.
- `docs/architecture/licensing.md` frames the whole named ecosystem as AGPL-3.0 with an MPL-2.0 relicense proposed. WebVOWL/MIT sits entirely outside that boundary today — correctly so; keeping the breakout as an independent MIT repo avoids the AGPL-linking questions that document raises for everything else.
- `docs/estate-closeout/2026-09-06/crate-survey.md` and `closeout-table.md` (row 22, status **OPEN**) already record this exact problem and prescribe a target: *"merge into the knowledgeGraph tree per the survey plan, feature-gate the NGG1 binary path, fix metadata, then publish."* This scoping note refines that plan (§3) rather than contradicting it — the main refinement is that visionGraph's live pipeline, not knowledgeGraph, is the one that must actually be cut over.
- `docs/terminology.md:97` confirms narrativegoldmine's live front page is served from the vendored `publishing-tools/WasmVOWL/modern` tree in visionGraph.

**Consumer counts**: VisionClaw/client 0, agentbox 0 (code), dreamlab-ai-website 0, WasmVOWL 4 (no live deploy), knowledgeGraph 6 (no live deploy), visionGraph 4 (1 live production pipeline + 1 committed build artefact), VisionFlow 0 code / ~10 doc references + 1 confirmed map gap.

---

## 3. Breakout plan

### Repository and ecosystem placement

- New standalone repo, MIT-licensed, sitting outside the AGPL boundary the same way `prose-sanitiser` and `diagram-ir` already do (both already-published, MIT/Apache-dual, standalone in this estate).
- Add a row to `VisionFlow/docs/architecture/repository-map.md` following its existing table+Mermaid convention: role "Ontology visualization WASM engine (WebVOWL-derived), consumed by visionGraph's corpus explorer", primary docs entry `README.md`. Also add a row/line to `docs/architecture/licensing.md` noting it's MIT and outside the AGPL substrate set.
- Suggested repo name: **`vowl-wasm`** (short, drops the "WebVOWL" trademark-adjacent string from the *repo* name while staying instantly recognisable to anyone who knows the domain; matches the estate's existing terse naming, e.g. `diagram-ir`, `solid-pod-rs`).

### Crate name

`webvowl-wasm` is unclaimed on crates.io (confirmed directly: `crates.io/api/v1/crates/webvowl-wasm` → 404 with a proper User-Agent header, matching the crate-survey's finding), so there is no *name-squatting* blocker. The concern is different: "WebVOWL" is the upstream project's own brand (TIB-hosted, `vowl.visualdataweb.org`), and shipping a crate under that exact name reads as if it were the official WASM build. Recommend **not** claiming the bare name even though it's technically available.

Proposed alternatives (all confirmed unclaimed via the same 404 check): `vowl-wasm`, `vowl-render`, `vowl-layout`, `ontograph-wasm`. **`vowl-wasm`** is the pick — keeps the recognisable "VOWL" abbreviation (which is WebVOWL's own name for its visual notation, not a trademarked product name on its own), drops "Web" (the part that reads as the specific upstream product), and matches the proposed repo name.

npm is a different, live situation: `@dreamlab-ai/webvowl-wasm` is **already published and real** — checked directly against `registry.npmjs.org`, `dist-tags.latest: 0.3.2`, described as "Rust/WASM powered semantic graph visualization for Logseq knowledge graphs." This means the npm side isn't a clean rename the way crates.io is: publishing under a new scoped name (`@dreamlab-ai/vowl-wasm`) is still right for consistency with the new crate name, but the existing `@dreamlab-ai/webvowl-wasm` package needs an explicit `npm deprecate` pointing at the replacement rather than silent abandonment, since it may have external installs (its Logseq-specific description suggests it was published for exactly that kind of independent consumer, not only for the two in-estate copies this survey traced).

### What the clean-room merge keeps

Base the merge on **copy B's superset** (knowledgeGraph/explorer/rust-wasm, 10,335 LOC) as the feature-complete parent, cross-checking copy A (WasmVOWL, 2,783 LOC) for anything cleaner in the shared modules (`ontology/parser.rs`, `graph/`, `layout/force.rs`, `render/`, `bindings/`) since A is the smaller and likely easier-to-read implementation of the same core behaviour. Concretely:

- **Kept unconditionally** (present in both, B is the superset): `ontology::{parser, model}`, `graph::{node, edge, builder}`, `layout::{force, simulation}`, `render`, `bindings`, `error`.
- **Kept from B only, made default-off feature gates** (estate-specific, not general-purpose WebVOWL behaviour):
  - `ngg1` — the NGG1 binary format is knowledgeGraph's own pipeline artefact (`ADR-NG-001`, `FORMAT-NGG1.md`), not something a general WASM ontology-visualization consumer needs. Feature: `ngg1`.
  - `ontology::markdown_parser` — parses knowledgeGraph's markdown-fronted ontology pages, a project-specific input format. Feature: `markdown-ontology`.
  - `ontology::owl2_validator` — genuinely general-purpose (OWL2-functional-syntax validation), candidate to ship **on by default** rather than gated; revisit once API surface is drafted.
  - `debug`, `interaction` — knowledgeGraph explorer-specific UI plumbing (click/selection state). Feature: `interaction`.
- **Kept from B as default-on, general-purpose performance work**: `graph::{pinning, statistics}`, `layout::{quadtree (Barnes-Hut), simd, csr_sim}`, the `parallel` (rayon) and `simd` Cargo features already defined in B's `Cargo.toml`, and the existing `debug-serde` legacy-export flag (keep off by default, as B already has it).
- **Dropped**: `wee_alloc` optional dependency (unmaintained upstream, marginal WASM binary-size win, not worth the maintenance liability in a fresh crate) and B's `regex` dependency if it turns out to be markdown-parser-only (verify at merge time; gate behind `markdown-ontology` if so).
- **Version**: start the new crate at `0.1.0` — neither 1.0.0 (A) nor 0.3.4 (B) accurately describes a freshly-merged, freshly-scoped public API, and starting over avoids any implied compatibility promise to either existing consumer.

### Public API surface

The two READMEs already document a de facto stable surface both copies share — this is what consumers depend on and what must not silently change shape in the merge:

- `WebVowl::new()`, `.loadOntology(json: string)`, `.initSimulation()`, `.runSimulation(n)`, `.tick()`, `.isFinished()`, `.getAlpha()`, `.setCenter/.setLinkDistance/.setChargeStrength`, `.getGraphData()`, `.getNodeCount()/.getEdgeCount()`, `.getStatistics()`.
- B additionally exposes an `NggExplorer` zero-copy positions path (per its README's ADR-NG-001 note) and a `.getMetadata()` call for OWL/RDF header extraction — both become part of the stable surface if their owning features are enabled, clearly documented as feature-gated rather than universal.
- Freeze this surface in the new crate's `lib.rs` with `#[wasm_bindgen]` doc comments before the first 0.1.0 publish; this is the contract `visionGraph`'s cutover and any future consumer build against.

### CI shape

Neither existing copy has *working* CI (knowledgeGraph's and visionGraph's `wasm-publish.yml` files both reference non-existent paths and never actually run; visionGraph's real, live workflow is `publish.yml`, which builds-in-place rather than testing a standalone crate). Build fresh, following two precedents already live in this estate:

- **SHA-pinned actions**: follow `agentbox-of-empires/.github/workflows/ci.yml`'s exact pattern — `actions/checkout@<40-char-sha> # vX.Y.Z`, `dtolnay/rust-toolchain@<sha> # stable`, `Swatinem/rust-cache@<sha>`.
- **fmt/clippy/test/doc gates**: follow `prose-sanitiser/.github/workflows/ci.yml`'s shape — `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --all-features --locked`, `cargo test --workspace --all-features --locked`, `cargo doc --workspace --no-deps --all-features --locked`, plus a `cargo deny check licenses advisories bans sources` job (same repo's `release.yml` pattern).
- **wasm32 build**: follow `RuView`'s precedent (`.github/workflows/rust.yml`) — `dtolnay/rust-toolchain` with `targets: wasm32-unknown-unknown`, then `cargo build --target wasm32-unknown-unknown --release`. Add a `wasm-pack build --target web` step and `wasm-pack test --headless --chrome` (both copies already script this in `package.json`).
- **wasm-pack pack**: `wasm-pack build --target web --release` then verify `pkg/` contents (adapt WasmVOWL's own `build.yml` size/structure checks — those are the one genuinely useful piece of prior art here, just needs its paths fixed and its parent workflow trigger corrected) — but do **not** upload the `pkg/` tarball as a committed artefact; upload it as a CI artifact (`actions/upload-artifact`) and publish to npm from CI instead, ending the pattern of committing built output.
- **MSRV**: pin `rust-version` in `Cargo.toml` following `diagram-ir`'s precedent (`rust-version = "1.85"`); verify against whatever `wasm-bindgen`/`petgraph`/`nalgebra` versions land in the merged `Cargo.toml` and adjust if needed.
- **Feature matrix**: a matrix job building `--no-default-features`, `--features ngg1,markdown-ontology,interaction`, and `--all-features`, since this crate now genuinely has optional surfaces that must each compile standalone.

### Publish and cutover sequence

1. Create the new repo, seed from copy B's tree (superset), replace `ontology::markdown_parser`/`ngg1`/`interaction`/`debug` with feature gates as above, fix the licence/attribution metadata from §1, freeze the public API, add the CI in §3.
2. `cargo publish --dry-run` (as the crate-survey already validated dry-runs pass cleanly for both existing copies — the merged crate should be no harder), then publish `vowl-wasm` 0.1.0 to crates.io.
3. `wasm-pack build` + `npm publish` for `@dreamlab-ai/vowl-wasm` 0.1.0, then `npm deprecate @dreamlab-ai/webvowl-wasm "renamed to @dreamlab-ai/vowl-wasm"` against the live 0.3.2 package (do not unpublish — npm's unpublish window has long since passed and unpublishing a package with unknown external installs is the wrong move regardless).
4. Cut visionGraph's `publish.yml` over first, since it is the only live consumer: replace the `wasm-pack build` + `npm install ../rust-wasm/pkg` steps (lines 167–189) with `npm install @dreamlab-ai/vowl-wasm@0.1.0`; remove the `publishing-tools/WasmVOWL/rust-wasm/pkg.tar.gz` commit and the vendored `rust-wasm/` source tree from that repo; delete the inert nested `wasm-publish.yml`. Ship this as its own PR, verify narrativegoldmine.com still renders correctly (browser check), before touching anything else.
5. Update knowledgeGraph/explorer: replace the `file:../rust-wasm/pkg` path dependency in `explorer/modern/package.json` with the published npm package, update the one dynamic `import('webvowl-wasm')` call and the `vite.config.ts` exclude entry to the new name, delete `explorer/rust-wasm/` entirely, delete the dead `wasm-publish.yml`.
6. Update WasmVOWL: same treatment for `modern/package.json`'s build script and `vite.config.ts`; delete `rust-wasm/` entirely (keep the repo itself only if it still has standalone value as a demo/frontend shell — otherwise consider archiving it, since its only reason to exist was housing this crate).
7. Add the `repository-map.md` and `licensing.md` rows in VisionFlow (§3, first bullet).
8. Update `closeout-table.md` row 22 from OPEN to closed, citing this document and the completed cutover.

### What to delete locally afterwards

`WasmVOWL/rust-wasm/`, `knowledgeGraph/explorer/rust-wasm/`, `visionGraph/publishing-tools/WasmVOWL/` (the whole vendored tree, including the committed `pkg.tar.gz` and the dead nested workflow), and both dead `wasm-publish.yml` files. Keep `WasmVOWL/modern/` and `knowledgeGraph/explorer/modern/` only if they still serve as useful frontend shells against the new npm package; otherwise fold whichever is more current into `visionGraph`'s deployed frontend and archive the rest.

### Effort estimate

| Task | Hours |
|---|---|
| Merge B into new repo, feature-gate NGG1/markdown/interaction, fix licence/attribution metadata, freeze API surface | 6–8 |
| Write CI (fmt/clippy/test/doc/deny + wasm32 build + wasm-pack pack + feature matrix, SHA-pinned) | 3–4 |
| First crates.io + npm publish, including dry-run fixes if any surface | 1–2 |
| visionGraph cutover (the live pipeline) + browser verification of narrativegoldmine.com | 3–4 |
| knowledgeGraph cutover + WasmVOWL cutover (both dead-consumer repos, lower risk) | 2–3 |
| repository-map.md / licensing.md / closeout-table.md doc updates | 1 |
| **Total** | **16–22 agent-hours** |

---

## Summary for the team lead

**Licence verdict**: Clean-room MIT reimplementation in both copies — no vendored upstream JS, no copied fixtures, no per-file attribution, but both correctly ship the exact upstream MIT licence text with the original copyright notice and both READMEs correctly credit WebVOWL in prose. Not a legal blocker. Two metadata defects need fixing before publish: copy A's `package.json` misattributes author/repository to upstream; copy B's `Cargo.toml` `repository` field does the same. Upstream is confirmed JavaScript (not Python, contrary to the owner's belief), MIT-licensed, still active.

**Consumer counts**: VisionClaw core + client — 0. agentbox — 0 code (9 doc/fixture mentions only; explicit no on the owner's suspicion it holds this code). dreamlab-ai-website — 0. WasmVOWL — 4 (no live deploy). knowledgeGraph — 6 (no live deploy; its own CI is deploy-free by design). **visionGraph — 4, including the one and only live production consumer** (`publish.yml` deploys narrativegoldmine.com from a vendored, git-committed copy with a committed `pkg.tar.gz` build artefact). VisionFlow — 0 code, but its canonical `repository-map.md` currently has **no row at all** for any of the three repos — a real ecosystem-map gap this breakout should close.

**Top risks**:
1. **The brief undercounted the copies** — there are three, not two, and the third (visionGraph) is the only one actually deployed to production. Any plan that treats knowledgeGraph as the cutover target (as the existing closeout-table row 22 currently implies) will miss the real consumer.
2. **A committed build artefact** (`pkg.tar.gz` in visionGraph) and **two dead, non-functional CI files** (both named `wasm-publish.yml`, both referencing paths that don't exist in their own repos) suggest this area hasn't had a working release pipeline for some time — the cutover is also the first time this code gets real CI.
3. **`@dreamlab-ai/webvowl-wasm` is confirmed live on npm at 0.3.2** (checked directly against the registry) — two patches behind the local source and described for "Logseq knowledge graphs" specifically, meaning it may already have external consumers this survey couldn't trace. The rename must ship an `npm deprecate` pointer, not a silent abandonment; do not unpublish it.
