---
id: VG-03
title: publish.yml — vault build, Quartz notes, and the React/vowl-wasm explorer build
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: []
sources:
  - ../visionGraph/.github/workflows/publish.yml
verified_commit: ac6274f9f5e12375f92086ccb9adba50c965ecf9
---

## VG-03.1 publish.yml — checkout to deploy, one job, three producers into one site

```mermaid
flowchart TD
    CO["Checkout fetch-depth:0 + setup Node 22<br/>publish.yml:61-70"]
    VAULTBIN["Obtain the vault binary — release or built from<br/>VAULT_GIT_REV — see VG-03.5<br/>publish.yml:81-145"]
    VALIDATE["vault validate --vault all — source gate<br/>publish.yml:155-163"]
    BUILD["vault build — machine artefacts + content staging<br/>see VG-03.5<br/>publish.yml:165-217"]
    GATE["Hard gate: artefacts present AND substantial<br/>see VG-03.5<br/>publish.yml:238-292"]
    QUARTZ["Quartz build -d content -o www/notes — see VG-03.6<br/>publish.yml:297-307"]
    PLACE["Place site-data at the site root, hash-verified<br/>publish.yml:309-324"]
    FIN["Finalize — CNAME, .nojekyll, 1GB size gate<br/>publish.yml:326-349"]
    SPA["Explorer SPA build — see VG-03.3<br/>publish.yml:356-408"]
    SMOKE["Explorer smoke — Playwright/CDP against www/<br/>publish.yml:410-414"]
    SEC["Refuse credential material — see VG-03.6<br/>publish.yml:422-433"]
    URLCHK["Site layout check — see VG-03.6<br/>publish.yml:435-439"]
    DEPLOY["Deploy — peaceiris/actions-gh-pages, SHA-pinned<br/>external_repository: DreamLab-AI/knowledgeGraph<br/>publish.yml:444-452"]
    CO --> VAULTBIN --> VALIDATE --> BUILD --> GATE --> QUARTZ --> PLACE --> FIN --> SPA --> SMOKE --> SEC --> URLCHK --> DEPLOY
    note1["INVARIANT: concurrency group deploy-ontology serialises<br/>deploys to the shared gh-pages branch — name kept for<br/>continuity with a pre-Quartz workflow, publish.yml:30-36"]
```
- **DEBT:** `VAULT_RELEASE_TAG` is empty, so every run builds `vault` from source at `VAULT_GIT_REV`; the owner must set the tag once a VisionClaw release ships the binary (`publish.yml:41-48`).

## VG-03.3 Explorer SPA build — vowl-wasm consumed as a pinned package, reading vault-built artefacts

```mermaid
sequenceDiagram
    autonumber
    participant JOB as publish.yml explorer SPA step<br/>publish.yml:371
    participant WWW as www/ — vault artefacts already placed<br/>see VG-03.1 PLACE
    participant STAGE as public/data + public/api<br/>staged for local vite preview
    participant NPM as npm ci --no-audit --no-fund<br/>publish.yml:380
    participant VW as at-dreamlab-ai slash vowl-wasm<br/>EXTERNAL — see VW-*
    participant TSC as npx tsc -b + npm run test<br/>publish.yml:381-382
    participant VITE as npm run build<br/>publish.yml:385

    JOB->>WWW: read ontology.json, ontology.ttl, graph slash star,<br/>search-index.json — publish.yml:375-378
    JOB->>STAGE: copy into public/data, public/api for vite preview
    JOB->>NPM: install dependencies incl. vowl-wasm package
    NPM->>VW: version pinned in package.json,<br/>integrity-locked in package-lock.json
    JOB->>TSC: type-check plus unit tests
    JOB->>VITE: NODE_OPTIONS max-old-space-size 4096
    VITE-->>JOB: dist slash — index.html, 404.html, assets, coi-serviceworker.js
    Note over JOB: copies the SPA shell only, not dist slash api or<br/>dist slash data — vault-built artefacts in www stay<br/>authoritative — publish.yml:387-399
    JOB->>WWW: also writes an at slash explorer redirect stub<br/>for the address that briefly served the app — publish.yml:401-408
```

## VG-03.5 vault build — the sole producer of machine artefacts and Quartz content staging

```mermaid
flowchart TD
    BIN["vault binary — release download or built from<br/>VAULT_GIT_REV<br/>publish.yml:81-145"]
    VALIDATE["vault validate --vault all — hard-fails on errors,<br/>warnings printed non-fatal<br/>publish.yml:155-163"]
    BUILDCMD["vault build --vault all --out site-data<br/>--publish-out quartz/content --with-markdown-mirror --stats<br/>publish.yml:186-191"]
    SPLIT["Contract C3 staging tree, normally site-data/publish,<br/>moved to quartz/content when --publish-out is given<br/>publish.yml:193-199"]
    ASSETS["Symlink knowledge/assets into quartz/content/assets<br/>publish.yml:200-204"]
    BOTH["Guard: quartz/content/pages AND /working must both exist<br/>publish.yml:205-214"]
    CTX["Stage tracked JSON-LD context docs from static slash<br/>into site-data<br/>publish.yml:235-236"]
    GATE["Hard gate: every required artefact path present<br/>publish.yml:245-270"]
    SUBST["Substance check: search-index entries, api/pages count,<br/>ontology.ttl line count each at least 1000<br/>publish.yml:278-287"]
    BIN --> VALIDATE --> BUILDCMD --> SPLIT --> ASSETS --> BOTH --> CTX --> GATE --> SUBST
    note1["INVARIANT: --vault all is required — omitting it silently<br/>falls back to knowledge-only and every later gate still<br/>passes green, publish.yml:180-185"]
```
- **INVARIANT:** presence of an artefact path proves nothing — a producer can write every expected file empty, as the retired Python pipeline did against the migrated corpus; `SUBST` exists specifically to catch that (`publish.yml:272-287`).

## VG-03.6 Quartz notes build, credential and layout gates, deploy

```mermaid
flowchart TD
    QBUILD["Quartz build -d content -o www/notes --concurrency 4<br/>NODE_OPTIONS max-old-space-size 8192<br/>publish.yml:297-307"]
    PLACE["cp -a site-data/. www/ — hash-verify every<br/>staged artefact survived unaltered<br/>publish.yml:309-324"]
    SIZE["Finalize: CNAME, .nojekyll, apparent-size gate<br/>hard-fails over 1000MB, warns over 900MB<br/>publish.yml:326-349"]
    SECSELF["check-secrets.sh --self-test<br/>publish.yml:432"]
    SEC["check-secrets.sh www — refuse credential material<br/>publish.yml:433"]
    URLCHK["check-urls.sh www quartz/content — explorer owns root,<br/>notes complete under /notes/<br/>publish.yml:435-439"]
    DEPLOY["Deploy — peaceiris/actions-gh-pages<br/>external_repository DreamLab-AI/knowledgeGraph<br/>publish.yml:444-452"]
    QBUILD --> PLACE --> SIZE --> SECSELF --> SEC --> URLCHK --> DEPLOY
    note1["INVARIANT: the self-test runs before the real scan — a<br/>credential rule that stops matching must fail loudly<br/>rather than pass silently, publish.yml:430-432"]
```
- **DOC-DRIFT:** the corpus once had a live credential leak reach a build artefact (`publish.yml:424-427`, referencing `docs/migration-2026-09-22/credential-redaction.md`); `check-secrets.sh` guards the artefact precisely because `vault validate` only guards the source.
