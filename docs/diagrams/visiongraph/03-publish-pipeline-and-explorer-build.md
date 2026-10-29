---
id: VG-03
title: publish.yml — ontomark build, Quartz notes, and the React/vowl-wasm explorer build
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-004, ADR-VG-008]
sources:
  - ../visionGraph/.github/workflows/publish.yml
  - ../visionGraph/publishing-tools/WasmVOWL/modern/scripts/check-space-domains.mjs
  - ../visionGraph/publishing-tools/WasmVOWL/modern/scripts/check-supply-chain-domain.mjs
  - ../visionGraph/publishing-tools/WasmVOWL/modern/package.json
  - ../visionGraph/docs/adr/ADR-VG-004-space-and-earth-domains.md
  - ../visionGraph/docs/adr/ADR-VG-008-supply-chains-and-logistics-domain.md
verified_commit: fd5269a55e52b8b1d1e9d7d59cb5f0c7835a5a2e
---

## VG-03.1 publish.yml — checkout to deploy, one job, three producers into one site

```mermaid
flowchart TD
    CO["Checkout fetch-depth:0, Node 22, Rust 1.99.0<br/>publish.yml:64-79"]
    CONTRACT["Consumer contract tests, fail if the legacy producer returns<br/>publish.yml:88-89"]
    BIN["Fetch the released ontomark CLI, pinned by tag and sha256 — see VG-03.5<br/>publish.yml:98-117"]
    VALIDATE["ontomark validate --vault all — source gate<br/>publish.yml:127-135"]
    BUILD["ontomark build — machine artefacts, content staging, certificate<br/>see VG-03.5<br/>publish.yml:137-198"]
    CERT["oo-cert re-checks the derivation certificate, digest-pinned<br/>see VG-03.5<br/>publish.yml:210-227"]
    GATE["Hard gate: artefacts present AND substantial<br/>see VG-03.5<br/>publish.yml:253-307"]
    DOMAINS["Space and Earth, then Supply Chains domain contracts over site-data<br/>see VG-03.5<br/>publish.yml:309-313"]
    QUARTZ["Quartz build -d content -o www/notes — see VG-03.6<br/>publish.yml:318-328"]
    PLACE["Place site-data at the site root, hash-verified, gzip the large artefacts<br/>publish.yml:330-362"]
    FIN["Finalize — CNAME, .nojekyll, 1GB size gate<br/>publish.yml:364-388"]
    SPA["Honesty gate, then explorer SPA build — see VG-03.3<br/>publish.yml:395-451"]
    ROUTES["Stage static routes and third-party notices, hash-compared<br/>publish.yml:453-479"]
    SMOKE["Explorer smoke — Playwright/CDP against www/<br/>publish.yml:481-485"]
    SEC["Refuse credential material — see VG-03.6<br/>publish.yml:493-504"]
    URLCHK["Site layout check — see VG-03.6<br/>publish.yml:506-510"]
    DEPLOY["Deploy — peaceiris/actions-gh-pages, SHA-pinned<br/>external_repository: DreamLab-AI/knowledgeGraph<br/>publish.yml:515-523"]
    DISPATCH["corpus-sync repository_dispatch to DreamLab-AI/VisionClaw<br/>payload source_repo + source_sha — see VG-04.1<br/>publish.yml:525-531"]
    CO --> CONTRACT --> BIN --> VALIDATE --> BUILD --> CERT --> GATE --> DOMAINS --> QUARTZ --> PLACE --> FIN --> SPA --> ROUTES --> SMOKE --> SEC --> URLCHK --> DEPLOY --> DISPATCH
    note1["INVARIANT: concurrency group deploy-ontology serialises<br/>deploys to the shared gh-pages branch — name kept for<br/>continuity with a pre-Quartz workflow, publish.yml:33-39"]
```
- **Invariant:** `ONTOMARK_RELEASE_TAG` and `ONTOMARK_SHA256` are both pinned (v0.6.0 and a digest), the step fails if either is empty, and there is no build-from-source fallback: "a run that cannot obtain the pinned release fails, because nothing else produces the site's artefacts" (`publish.yml:41-51`, `publish.yml:100-105`).
- **Drift:** the earlier topic described a `VAULT_RELEASE_TAG` left empty so every run built `vault` from VisionClaw at `VAULT_GIT_REV` (`90a7a85c`). That path no longer exists in the workflow: the `vault` fetch, VisionClaw checkout, cache and `cargo build` steps were replaced by the digest-checked ontomark download (`publish.yml:91-117`), and the Python `scripts/tests/test_publish_cli.py` consumer contract fails the build if the legacy producer is reintroduced (`publish.yml:85-89`). The source-gate comment still names VisionClaw's `docs/VAULT-corpus-format.md` and ADR-2112 as the contract (`publish.yml:119-120`).
- **Invariant:** the deploy is followed by a `corpus-sync` `repository_dispatch` carrying `source_repo` and `source_sha` (`publish.yml:525-531`), so VisionClaw's ontology release is rebuilt from the exact revision this run validated rather than from whatever `main` is when it runs; it runs only after the deploy step succeeds, so a failed deploy never refreshes the VisionClaw release.
- **Open:** the workflow header says the owner authorised estate deployment "during the ontomark resumption on 2026-10-11" (`publish.yml:17-18`) where the previous text said agents never push; the authorisation is recorded in a comment, not in a governing document this topic cites.

## VG-03.3 Explorer SPA build — vowl-wasm consumed as a pinned package, reading ontomark-built artefacts

```mermaid
sequenceDiagram
    autonumber
    participant JOB as publish.yml explorer SPA step<br/>publish.yml:410
    participant WWW as www/ — ontomark artefacts already placed<br/>see VG-03.1 PLACE
    participant STAGE as public/data + public/api<br/>staged for local vite preview
    participant NPM as npm ci --no-audit --no-fund<br/>publish.yml:419
    participant VW as at-dreamlab-ai slash vowl-wasm<br/>EXTERNAL — see VW-*
    participant TSC as npx tsc -b, runtime-notices cargo checks, npm run test<br/>publish.yml:420-425
    participant VITE as npm run build<br/>publish.yml:428

    JOB->>WWW: read ontology.json.gz, ontology.ttl, graph slash star,<br/>search-index.json — publish.yml:414-417
    JOB->>STAGE: gunzip and copy into public/data, public/api for vite preview
    JOB->>NPM: install dependencies incl. vowl-wasm package
    NPM->>VW: release tarball URL pinned in package.json,<br/>integrity-locked in package-lock.json — package.json:25
    JOB->>TSC: type-check, fmt/test/clippy/doc of tools/runtime-notices, unit tests
    JOB->>VITE: NODE_OPTIONS max-old-space-size 4096
    VITE-->>JOB: dist slash — index.html, 404.html, assets, coi-serviceworker.js
    Note over JOB: copies the SPA shell only, not dist slash api or<br/>dist slash data — ontomark-built artefacts in www stay<br/>authoritative — publish.yml:430-442
    JOB->>WWW: also writes an at slash explorer redirect stub<br/>for the address that briefly served the app — publish.yml:444-451
```
- The explorer now consumes vowl-wasm from the 0.2.0 GitHub release tarball rather than from an npm version range (`publishing-tools/WasmVOWL/modern/package.json:25`), and the job stages five explorer routes as copies of the SPA shell, compared by hash, plus the vowl-wasm, ontomark-graph and JavaScript runtime notices (`publish.yml:453-479`).

## VG-03.5 ontomark build — the sole producer of machine artefacts and Quartz content staging

```mermaid
flowchart TD
    BIN["ontomark CLI — release download checked with sha256sum --strict<br/>publish.yml:98-117"]
    VALIDATE["ontomark validate --vault all — hard-fails on errors,<br/>warnings printed non-fatal<br/>publish.yml:127-135"]
    BUILDCMD["ontomark build --vault all --out site-data --publish-out quartz/content<br/>--with-markdown-mirror --certificate-out --stats<br/>publish.yml:165-171"]
    SPLIT["Contract C3 staging tree, normally site-data/publish,<br/>moved to quartz/content when --publish-out is given<br/>publish.yml:176-179"]
    ASSETS["Symlink the repo-level assets/ into quartz/content/assets<br/>publish.yml:180-187"]
    BOTH["Guard: quartz/content/pages AND /working must both exist<br/>publish.yml:189-198"]
    CERT["oo-cert v2.0.1, sha256-pinned, re-derives every certificate line<br/>from its premises, then copies the small reports to site-data/certificate<br/>publish.yml:210-227"]
    KEEP["Full certificate kept as an artefact for 7 days<br/>publish.yml:229-235"]
    CTX["Stage tracked JSON-LD context docs from static slash<br/>into site-data<br/>publish.yml:250-251"]
    GATE["Hard gate: every required artefact path present<br/>publish.yml:260-285"]
    SUBST["Substance check: search-index entries, api/pages count,<br/>ontology.ttl line count each at least 1000<br/>publish.yml:293-302"]
    BIN --> VALIDATE --> BUILDCMD --> SPLIT --> ASSETS --> BOTH --> CERT --> KEEP --> CTX --> GATE --> SUBST --> SPACE
    SPACE["Space and Earth domain contract — domains 6 and 7 present with<br/>stable ids and minimum member counts, published roots and tiers<br/>check-space-domains.mjs:14-27"]
    SPACECNT["Invariants not totals: no duplicate title in the catalogue or the researched list,<br/>every researched title published with article text, every accounted identity public draft<br/>check-space-domains.mjs:39-73"]
    SUPPLY["Supply Chains and Logistics domain 8 — id, root, tier, 15 category hubs,<br/>catalogue identity checks skipped when the catalogue is absent<br/>check-supply-chain-domain.mjs:13-54"]
    SPACE --> SPACECNT --> SUPPLY
    note1["INVARIANT: --vault all is required — omitting it silently<br/>falls back to knowledge-only and every later gate still<br/>passes green, publish.yml:152-157"]
    note2["INVARIANT: the build is refused with exit 2 when the certificate does not<br/>cover the classifier's closure, so no generation ships without its warrant<br/>publish.yml:158-163"]
```
- **Invariant:** presence of an artefact path proves nothing — a producer can write every expected file empty, as the retired Python pipeline did against the migrated corpus; `SUBST` exists specifically to catch that (`publish.yml:287-302`).
- **Drift:** the earlier topic recorded a **Debt** that the space and Earth gate pinned exact totals of 934 expansion identities and 660 researched drafts, forcing every research batch to edit the gate. At this stamp the gate asserts invariants and prints the totals instead (`check-space-domains.mjs:39-74`), so that debt is cleared in the script; the ADR still describes the two roots as staged rather than live (`docs/adr/ADR-VG-004-space-and-earth-domains.md:35-36`).
- **Tension:** ADR-VG-008 establishes the Supply Chains and Logistics domain as domain 8 and is accepted and complete but its activation is `staged` (`docs/adr/ADR-VG-008-supply-chains-and-logistics-domain.md:5-7`), while `publish.yml` already runs its gate on every deploy (`publish.yml:312-313`), so the deploy enforces a domain whose activation has not been declared.
- **Open:** the certificate step re-checks a build that `ontomark build` itself refuses to finish without a certificate; the checker is a different program (a Lean 4 binary) pinned by its own tag and digest (`publish.yml:212-214`), separate from the `ontomark` pin (`publish.yml:49-51`), but both are fetched with the same `ACCESS_TOKEN` and neither pin has a recorded rotation rule in this workflow (`publish.yml:117`, `publish.yml:215`).

## VG-03.6 Quartz notes build, credential and layout gates, deploy

```mermaid
flowchart TD
    QBUILD["Quartz build -d content -o www/notes --concurrency 4<br/>NODE_OPTIONS max-old-space-size 8192<br/>publish.yml:318-328"]
    PLACE["cp -a site-data/. www/ — hash-verify every<br/>staged artefact survived unaltered<br/>publish.yml:330-345"]
    ZIP["gzip api/pages, api/markdown and data/ontology.json at rest,<br/>the explorer inflates them in the browser<br/>publish.yml:347-362"]
    SIZE["Finalize: CNAME, .nojekyll, apparent-size gate<br/>hard-fails over 1000MB, warns over 900MB<br/>publish.yml:364-388"]
    SECSELF["check-secrets.sh --self-test<br/>publish.yml:503"]
    SEC["check-secrets.sh www — refuse credential material<br/>publish.yml:504"]
    URLCHK["check-urls.sh www quartz/content — explorer owns root,<br/>notes complete under /notes/<br/>publish.yml:506-510"]
    DEPLOY["Deploy — peaceiris/actions-gh-pages<br/>external_repository DreamLab-AI/knowledgeGraph<br/>publish.yml:515-523"]
    QBUILD --> PLACE --> ZIP --> SIZE --> SECSELF --> SEC --> URLCHK --> DEPLOY
    note1["INVARIANT: the self-test runs before the real scan — a<br/>credential rule that stops matching must fail loudly<br/>rather than pass silently, publish.yml:501-504"]
```
- **Drift:** the corpus once had a live credential leak reach a build artefact (`publish.yml:495-499`, referencing `docs/migration-2026-09-22/credential-redaction.md`); `check-secrets.sh` guards the artefact precisely because `ontomark validate` only guards the source, where the earlier text named `vault validate`.
