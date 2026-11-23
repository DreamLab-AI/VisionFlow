---
id: DW-01
title: Repo composition and deploy topology
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/BASELINE-architecture.md
adrs: [ADR-2001, ADR-2002, ADR-2003, ADR-2004]
sources:
  - ../dreamlab-ai-website/CLAUDE.md
  - ../dreamlab-ai-website/README.md
  - ../dreamlab-ai-website/CNAME
  - ../dreamlab-ai-website/.github/workflows/deploy.yml
  - ../dreamlab-ai-website/.github/workflows/workers-deploy.yml
  - ../dreamlab-ai-website/.github/workflows/rust-ci.yml
  - ../dreamlab-ai-website/docs/BASELINE-architecture.md
  - ../dreamlab-ai-website/docs/architecture/kit-compatibility-record.md
  - ../dreamlab-ai-website/forum-config/Cargo.toml
  - ../dreamlab-ai-website/forum-config/Cargo.lock
  - ../dreamlab-ai-website/docs/adr/ADR-2002-split-hosting-pages-workers.md
  - ../dreamlab-ai-website/forum-config/dreamlab.toml
  - ../dreamlab-ai-website/forum-config/src/workers.rs
  - ../dreamlab-ai-website/src/App.tsx
verified_commit: e255b2b113ee47d58a0c872c6b4702afc62a46e5
---

## DW-01.1 Repo composition — thin operator overlay, not a protocol owner
```mermaid
flowchart TB
    REPO["dreamlab-ai-website"] --> REACT["React 18 marketing SPA<br/>src/, this repo<br/>BASELINE-architecture.md:50"]
    REPO --> FCFG["forum-config/ overlay<br/>branding, zones, CF resource ids, kit pin<br/>CLAUDE.md:5-11"]
    REPO --> DOCS["docs/ — BASELINE, IDENTITY-zones,<br/>adr, api, security, deployment"]
    FCFG -. "cloned at KIT_REF" .-> KIT["nostr-rust-forum kit<br/>forum client, BBS client, 5 Workers<br/>see NF-*"]
    REPO -. "published crates" .-> CRATES["Published kit crates<br/>core, config, mesh, rate-limit<br/>=1.0.0-beta.20<br/>forum-config/Cargo.toml:50-53"]
    REPO -. "poker route" .-> POKER["/community/table<br/>practice + two asset tables<br/>DREAM and BLAKES7 house seats<br/>dreamlab.toml:320"]
```
- "What this repo is" (BASELINE-architecture.md:35-41): a thin operator overlay — the forum source, Nostr crates, and five Workers all live upstream; this repo carries only the React site, `forum-config/`, and docs.
- INVARIANT: `forum-config/Cargo.toml` license is `AGPL-3.0-only` (Cargo.toml:9) because it statically links the AGPL kit crates — the package comment (Cargo.toml:6-8) explicitly corrects an earlier "Proprietary" framing as "legally incoherent".

## DW-01.2 Three frontends, one origin
```mermaid
flowchart LR
    ORIGIN["dreamlab-ai.com<br/>CNAME:1"] --> ROOT["/  React 18 marketing SPA<br/>src/App.tsx, Vite+React Router<br/>BASELINE-architecture.md:50"]
    ORIGIN --> COMM["/community/  Leptos 0.7 CSR-WASM forum<br/>kit crate nostr-bbs-forum-client, Trunk-built<br/>deploy.yml:245 'Build Leptos forum with Trunk'"]
    ORIGIN --> BBS["/community/bbs/  retro ASCII/BBS terminal<br/>kit crate nostr-bbs-bbs-client, Trunk-built<br/>deploy.yml:336 'Build retro ASCII/BBS client with Trunk'"]
    ORIGIN --> TBL["/community/table  poker route<br/>practice + DREAM and BLAKES7 tables<br/>dreamlab.toml:320"]
    LEGACY["/bbs"] -. "legacy client redirect" .-> BBS
```
- **Drift:** `BASELINE-architecture.md:146-148` records that the README says "Two SPAs, one origin" while the deploy ships a third client at `/community/bbs/` (`deploy.yml:336`); the README has since been rewritten and now states "three independently built frontends" (`README.md:54`), so that divergence entry describes a README that no longer exists, and `CLAUDE.md:14` still reads "static dual-SPA".
- The forum client also carries the poker table at `/community/table` (ADR-2009, kit ADR-2020): a practice table plus DREAM and BLAKES7 tables with separate configured house seats (`dreamlab.toml:320`). The baseline paragraph names only DREAM (`BASELINE-architecture.md:54-58`); DW-10 covers current credit, recovery and receipt boundaries.
- All three are static assets after build; React gets Vite build variables, forum/BBS get a `window.__ENV__` block injected by `sed` at deploy time (BASELINE-architecture.md:68-72).

## DW-01.3 Deploy topology — GitHub Pages primary, Cloudflare Pages gated off
```mermaid
flowchart TB
    BUILD["build-and-deploy job<br/>deploy.yml"] --> GHPAGES["peaceiris/actions-gh-pages<br/>publish_branch gh-pages, cname dreamlab-ai.com<br/>deploy.yml:406-413"]
    BUILD --> MIRROR{"DREAMLAB_UK_TOKEN present?<br/>deploy.yml step 'Check mirror token'"}
    MIRROR -->|yes| MIRRORDEPLOY["mirror to TheDreamLabUK/website<br/>cname thedreamlab.uk<br/>deploy.yml:429-440"]
    MIRROR -->|no| SKIP["skip, ::notice::"]
    BUILD --> CFGATE{"vars.CLOUDFLARE_PAGES_ENABLED == 'true'?<br/>deploy.yml:443"}
    CFGATE -->|yes| CFPAGES["cloudflare/wrangler-action<br/>pages deploy dist/ --project-name=dreamlab-ai"]
    CFGATE -->|no, default| CFOFF["Cloudflare Pages step does not run"]
```
- **Drift:** the baseline's first divergence says `README.md` frames the site as a "dual-SPA Cloudflare-edge deployment" (`BASELINE-architecture.md:141-145`); the README now says "GitHub Pages: one origin" (`README.md:43`) and the only surviving "dual-SPA" wording is `CLAUDE.md:14`. The origin is GitHub Pages, Cloudflare only hosts the backend Workers (five: auth, pod, relay, search, preview — `forum-config/src/workers.rs:155-172`), and Cloudflare Pages is opt-in behind an explicit repo variable (`BASELINE-architecture.md:82`).
- INVARIANT: GitHub Pages is the origin of record for `dreamlab-ai.com` until DNS is re-cut; the Cloudflare Pages step must stay gated behind that repo variable (BASELINE-architecture.md:185-187, Invariant 4).
- The branded custom domains for those Workers (`relay./api./pods./search./preview.dreamlab-ai.com`) are the documented end-state but are **not provisioned in DNS** — the client instead talks to raw `*.workers.dev` hosts baked into the build env (`deploy.yml:49-61`; verified 2026-06-09 `ERR_NAME_NOT_RESOLVED`, not re-checked as of 2026-08-31) — shipping the branded domains baked in previously severed every client API call (BASELINE-architecture.md:91-97).

## DW-01.5 The pin rule — one kit SHA at four sites, one crate version beside it
```mermaid
flowchart TB
    RULE["dual-pin rule, four sites move together<br/>CLAUDE.md:246-257"]
    subgraph SHAT["SHA track: the kit source the clients and workers build from"]
        A["1. KIT_REF<br/>.github/workflows/deploy.yml:131"] --- SHA["fd31193090a29471972bd408e07a7e5b8654f087<br/>kit main, tag v1.0.0-beta.21-rc.3<br/>kit-compatibility-record.md:26"]
        B["2. KIT_REF<br/>.github/workflows/workers-deploy.yml:45"] --- SHA
        C["3. KIT_REF<br/>.github/workflows/rust-ci.yml:20"] --- SHA
        D["4. CANONICAL_KIT_SHA<br/>kit-compatibility-record.md:30"] --- SHA
    end
    subgraph VERT["Library track: the crates forum-config links"]
        E["four exact = pins, no git rev<br/>forum-config/Cargo.toml:50-53"] --- VER["1.0.0-beta.20 from crates.io<br/>Cargo.lock:959-962 registry source + checksum"]
        F["CANONICAL_KIT_VERSION<br/>kit-compatibility-record.md:31"] --- VER
        G["RESOLVED lines x4<br/>kit-compatibility-record.md:86-89"] --- VER
    end
    RULE --> SHAT
    RULE -.->|"crates move only when the kit publishes<br/>CLAUDE.md:259-263"| VERT
    SHA -.->|"runs ahead of the published crates by design<br/>BASELINE-architecture.md:117-125"| VER
```
- `workers-deploy.yml` fires on `forum-config/Cargo.lock` and its own file, which holds `KIT_REF`, so a kit re-pin never ships a new client against old workers — the client/worker skew that "wiped the forum on 2026-06-15" (BASELINE-architecture.md:127-130, `workers-deploy.yml:13-20`).
- **Drift (closed):** the governing doc now records `KIT_REF = fd31193…` and `=1.0.0-beta.20` (`BASELINE-architecture.md:110,118`), equal to every pin site; it had lagged at `ea7d95e` and beta.16. `CLAUDE.md:246-263` replaced the old fourth site (the `rev` pins in `Cargo.toml`) with `CANONICAL_KIT_SHA`; the attribution of that check is in DW-05.8.
- **Drift (closed):** the git-rev wording no longer outlives the rule. The pin-site comments now say `Cargo.lock` holds no git rev and name the three-workflow plus `CANONICAL_KIT_SHA` equality (`deploy.yml:122-126`, `workers-deploy.yml:13-15,36-40`); the record compares the SHA only against the `KIT_REF`s (`kit-compatibility-record.md:34-38`) and its bump procedure moves the crate pins only on a library release (`kit-compatibility-record.md:368-373`); Invariant 1 now states the same split (`BASELINE-architecture.md:172-178`). The lockfile has no git source: all four crates resolve from the registry (`Cargo.lock:945-948,959-962`).
- **Drift (closed):** the README no longer names a kit SHA; it points at `KIT_REF` and `CANONICAL_KIT_SHA` (`README.md:65`). The record's History table now marks `931898a` (tag `v1.0.0-beta.10`) "Superseded" (`kit-compatibility-record.md:382`). pin-parity compares only the SHA field, so neither had failed the gate.
- INVARIANT: the machine-readable pin lives in one place the gate reads, `CANONICAL_KIT_SHA` / `CANONICAL_KIT_VERSION` (`kit-compatibility-record.md:30-31`). The pin-site comments defer provenance to it rather than restating versions (`rust-ci.yml:18-19`, `deploy.yml:127-130`, `workers-deploy.yml:41-44`), which is why surrounding prose can rot without the gate noticing.

## DW-01.6 Deploy job sequence — clone kit, build three frontends, merge, inject, deploy
```mermaid
sequenceDiagram
    autonumber
    participant GH as gate job<br/>test-and-lint.yml, deploy.yml:135-150
    participant CO as checkout + clone kit at KIT_REF<br/>deploy.yml:157-166
    participant RB as Build React main site<br/>deploy.yml:178-179 npm run build
    participant LB as Build Leptos forum with Trunk<br/>deploy.yml:245-247 --public-url /community/
    participant MG as Merge React + Forum into dist/<br/>deploy.yml step 'Merge...'
    participant ENV as Inject window.__ENV__<br/>deploy.yml step 'Inject runtime env config'
    participant BB as Build retro BBS with Trunk<br/>deploy.yml:336
    participant PG as Deploy to gh-pages<br/>deploy.yml:406-413
    GH->>CO: needs.gate.outputs.passed == 'true'
    CO->>RB: repo source, npm run build
    CO->>LB: kit source at KIT_REF
    RB->>MG: dist/index.html + assets
    LB->>MG: kit/dist, copied by cp -r kit/dist/* dist/community/<br/>deploy.yml:271-274
    MG->>ENV: sed inject <script>window.__ENV__=...SIDESTR_WALLET,ENCRYPTION_ENABLED,ZONE_CONFIG...
    ENV->>BB: build BBS after the forum merge so `cp kit/dist/*` does not grab the bbs subtree
    BB->>PG: dist/community/bbs/ merged, rebrand step, 404 shims
```
- Since `d3ecd05` (2026-10-02) the deploy also fires on a lockfile-only change: `package-lock.json` joined the push path filter (`deploy.yml:13-28`), so a dependency bump that touches nothing else still republishes the site. The same filter now carries `scripts/**` (`deploy.yml:27`, comment `deploy.yml:21-26`) since the postbuild prerender and discovery check (DW-02.3) and the content generators run in the build, so a generator-only commit rebuilds and ships; the checkout also takes full history (`fetch-depth: 0`, `deploy.yml:157-161`) because the sitemap `<lastmod>` is each page's last commit.
- Supply-chain hardening: every tool the deploy job downloads (Trunk, binaryen/`wasm-opt`, Tailwind CLI) is pinned to an exact version **and** SHA256-verified before use, because this job carries the Cloudflare API token (BASELINE-architecture.md:134-137, `deploy.yml:41-48`).
- The injected payload has grown since ADR-2015/ADR-2016 shipped: `SIDESTR_WALLET:'on'` (deploy.yml:90) gates the testnet4 wallet nav/tip UI, `SIDESTR_CHAINS_JSON` (deploy.yml:95) lists the two sidestr chains (kit ADR-2021), `ENCRYPTION_ENABLED:'true'` (deploy.yml:106) is the zone E2EE master switch, `ZONE_CONFIG_JSON` (deploy.yml:107) marks zones 2-4 `encrypted:true` with zone4 carrying `agent_keys:true`, and `POKER:'on'` + `POKER_CONFIG_JSON` (deploy.yml:101-102) gate and parameterise the poker table — see DW-03/DW-04 for the zone model and encryption detail this step only injects.

- The React build gained one more public input, `VITE_MONITOR_PUBKEY` (`deploy.yml:80`, passed to the React build at `deploy.yml:188`): the nightly inbox-watch co-recipient to whom every contact, signup and chat message is also wrapped, because the relay serves a kind-1059 wrap only to its addressee (`deploy.yml:75-79`); ADR-2011 names that key the `website-info` identity (DW-04). It is another hand-synced mirror (DW-03). `workers-deploy.yml` now retries its post-deploy `/health` probe three times because a worker's Durable Object is cold straight after a deploy (`workers-deploy.yml:368-371`).
## DW-01.7 SPA deep-link 404 shim — load-bearing, not incidental
```mermaid
stateDiagram-v2
    [*] --> HardLoad: GET /workshops/foo (no server-side routing on GH Pages)
    HardLoad --> Pages404: GitHub Pages serves dist/404.html
    Pages404 --> Redirect: script rewrites to /?__p=%2Fworkshops%2Ffoo
    Redirect --> ReactBoot: React app boots, reads __p via public/spa-redirect.js
    HardLoad --> Community404: path starts with /community
    Community404 --> CommunityRedirect: /community/?__p=%2Flogin
    CommunityRedirect --> ForumPickup: inline script in dist/community/index.html<br/>restores /community/login BEFORE WASM reads window.location
    HardLoad --> BbsRedirect: path is /bbs or /bbs/*
    BbsRedirect --> BbsServed: replaced with /community/bbs/ directly (single-screen terminal)
```
- "SPA deep links depend on a 404-redirect shim" is called out as load-bearing in the Known-divergences section, not incidental (BASELINE-architecture.md:165-168, `deploy.yml:368-398`, `291-301`).
- The React pickup script is deliberately external (`public/spa-redirect.js`), not an inline `<script>`, because `index.html`'s CSP (`script-src 'self'`, no `'unsafe-inline'`) would block an injected inline script — exactly the bug class that broke `/workshops` deep links (`deploy.yml:400-403`, the comment above the "Deploy" step).

## DW-01.8 EXTERNAL — estate authority map beyond this repo
```mermaid
flowchart LR
    DW["dreamlab-ai-website<br/>this area"] -->|"clones at KIT_REF"| NF["nostr-rust-forum kit<br/>EXTERNAL, see NF-*"]
    DW -->|"crates.io consumption"| NF
    DW -->|"CF Tunnel, native pod card<br/>DIVERGENCE: advertised, host has no DNS record<br/>ADR-2002-split-hosting-pages-workers.md:130"| AB["agentbox native solid-pod-rs<br/>EXTERNAL, see AB-*"]
    DW -->|"BrokerActor governance publisher,<br/>visionclaw-server admin key<br/>forum-config/dreamlab.toml:275"| VC["VisionClaw server<br/>EXTERNAL, see VC-*"]
    DW -->|"junkiejarvis website chat bridge"| AB
```
- `visionclaw-server` is a governance publisher pubkey shared with the forum's primary admin pubkey — see DW-03/DW-04 for the identity implications and IDENTITY-zones.md. The `[governance].agent_pubkeys` list this repo authors is at `forum-config/dreamlab.toml:264,275-277`; the ecosystem statement itself is at `CLAUDE.md:17`.
- DIVERGENCE: the native pod tier is a third plane beside the two ADR-2002 decides. `[native_pod] enabled = true` points at `pods-native.dreamlab-ai.com` (`dreamlab.toml:253-258`), which had no DNS record when ADR-2002's open item was written on 2026-11-22 (NXDOMAIN, not re-checked here), so the Pod Browser's mesh card cannot reach it; the item is parked by the owner (`ADR-2002-split-hosting-pages-workers.md:130-146`, `BASELINE-architecture.md:156-161`). Both now record that the card is gated by the `NATIVE_POD_URL` build secret, not by `enabled` (`ADR-2002-split-hosting-pages-workers.md:156-163`, `BASELINE-architecture.md:159-161`); the overlay side is DW-03.12.
