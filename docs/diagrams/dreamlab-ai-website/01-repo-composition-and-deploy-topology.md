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
  - ../dreamlab-ai-website/forum-config/dreamlab.toml
  - ../dreamlab-ai-website/forum-config/src/workers.rs
  - ../dreamlab-ai-website/src/App.tsx
verified_commit: 81ec18c4d56240dcf8e9dd8b07a2dca8239adaea
---

## DW-01.1 Repo composition — thin operator overlay, not a protocol owner
```mermaid
flowchart TB
    REPO["dreamlab-ai-website"] --> REACT["React 18 marketing SPA<br/>src/, this repo<br/>BASELINE-architecture.md:50"]
    REPO --> FCFG["forum-config/ overlay<br/>branding, zones, CF resource ids, kit pin<br/>CLAUDE.md:5-11"]
    REPO --> DOCS["docs/ — BASELINE, IDENTITY-zones,<br/>adr, api, security, deployment"]
    FCFG -. "EXTERNAL, cloned at KIT_REF" .-> KIT["nostr-rust-forum kit<br/>forum client, BBS client, 5 Workers<br/>see NF-*"]
    REPO -. "config crates, crates.io" .-> CRATES["nostr-bbs-core/config/mesh/rate-limit<br/>=1.0.0-beta.11<br/>forum-config/Cargo.toml:49-52"]
```
- "What this repo is" (BASELINE-architecture.md:35-41): a thin operator overlay — the forum source, Nostr crates, and five Workers all live upstream; this repo carries only the React site, `forum-config/`, and docs.
- INVARIANT: `forum-config/Cargo.toml` license is `AGPL-3.0-only` (Cargo.toml:9) because it statically links the AGPL kit crates — the package comment (Cargo.toml:6-8) explicitly corrects an earlier "Proprietary" framing as "legally incoherent".

## DW-01.2 Three frontends, one origin
```mermaid
flowchart LR
    ORIGIN["dreamlab-ai.com<br/>CNAME:1"] --> ROOT["/  React 18 marketing SPA<br/>src/App.tsx, Vite+React Router<br/>BASELINE-architecture.md:50"]
    ORIGIN --> COMM["/community/  Leptos 0.7 CSR-WASM forum<br/>kit crate nostr-bbs-forum-client, Trunk-built<br/>deploy.yml:218 'Build Leptos forum with Trunk'"]
    ORIGIN --> BBS["/community/bbs/  retro ASCII/BBS terminal<br/>kit crate nostr-bbs-bbs-client, Trunk-built<br/>deploy.yml:309 'Build retro ASCII/BBS client'"]
    LEGACY["/bbs"] -. "301-style client redirect" .-> BBS
```
- DOC-DRIFT: `README.md` frames the site as "Two SPAs, one origin"; the deploy ships a third client at `/community/bbs/` (BASELINE-architecture.md:122-124, `deploy.yml:309`).
- All three are static assets after build; React gets Vite build variables, forum/BBS get a `window.__ENV__` block injected by `sed` at deploy time (BASELINE-architecture.md:54-58).

## DW-01.3 Deploy topology — GitHub Pages primary, Cloudflare Pages gated off
```mermaid
flowchart TB
    BUILD["build-and-deploy job<br/>deploy.yml"] --> GHPAGES["peaceiris/actions-gh-pages<br/>publish_branch gh-pages, cname dreamlab-ai.com<br/>deploy.yml:379-385"]
    BUILD --> MIRROR{"DREAMLAB_UK_TOKEN present?<br/>deploy.yml step 'Check mirror token'"}
    MIRROR -->|yes| MIRRORDEPLOY["mirror to TheDreamLabUK/website<br/>cname thedreamlab.uk<br/>deploy.yml:402-413"]
    MIRROR -->|no| SKIP["skip, ::notice::"]
    BUILD --> CFGATE{"vars.CLOUDFLARE_PAGES_ENABLED == 'true'?<br/>deploy.yml:416"}
    CFGATE -->|yes| CFPAGES["cloudflare/wrangler-action<br/>pages deploy dist/ --project-name=dreamlab-ai"]
    CFGATE -->|no, default| CFOFF["Cloudflare Pages step does not run"]
```
- DOC-DRIFT: `README.md` frames the site as a "dual-SPA Cloudflare-edge deployment"; the origin is GitHub Pages, Cloudflare only hosts the backend Workers (five: auth, pod, relay, search, preview — `forum-config/src/workers.rs:155-172`), and Cloudflare Pages is opt-in behind an explicit repo variable (BASELINE-architecture.md:117-121).
- INVARIANT: GitHub Pages is the origin of record for `dreamlab-ai.com` until DNS is re-cut; the Cloudflare Pages step must stay gated behind that repo variable (BASELINE-architecture.md:155-157, Invariant 4).
- The branded custom domains for those Workers (`relay./api./pods./search./preview.dreamlab-ai.com`) are the documented end-state but are **not provisioned in DNS** — the client instead talks to raw `*.workers.dev` hosts baked into the build env (`deploy.yml:50-54`; verified 2026-06-09 `ERR_NAME_NOT_RESOLVED`, not re-checked as of 2026-08-31) — shipping the branded domains baked in previously severed every client API call (BASELINE-architecture.md:77-82).

## DW-01.5 The dual-pin rule — four locations that must move together
```mermaid
flowchart TB
    A["1. KIT_REF<br/>.github/workflows/deploy.yml:106"] --- SHA["13cbe6cbad7ee7ff3b609233a8bee3dd8eae1f3e"]
    B["2. KIT_REF<br/>.github/workflows/workers-deploy.yml:45"] --- SHA
    C["3. KIT_REF<br/>.github/workflows/rust-ci.yml:21"] --- SHA
    D["4. rev pin, resolved version<br/>forum-config/Cargo.toml:49-52"] --- VER["1.0.0-beta.11"]
    SHA -.->|"CANONICAL_KIT_SHA"| REC["kit-compatibility-record.md:30"]
    VER -.->|"CANONICAL_KIT_VERSION"| REC2["kit-compatibility-record.md:31"]
```
- `workers-deploy.yml` fires on `forum-config/Cargo.lock` and `KIT_REF` changes precisely so a kit re-pin never ships a new client against old workers — the client/worker skew that "wiped the forum on 2026-06-15" (BASELINE-architecture.md:103-106).
- DOC-DRIFT: `BASELINE-architecture.md:99,101` still cites `KIT_REF = a7544687b4d1c09807862d749b27f8c8da307a12` and crate version `"1.0.0-beta.9"` (line 96) as current; the live pins (`deploy.yml:106`, `workers-deploy.yml:45`, `rust-ci.yml:21`, `forum-config/Cargo.toml:49-52`, `kit-compatibility-record.md:30-31`) are `13cbe6cbad7ee7ff3b609233a8bee3dd8eae1f3e` / `1.0.0-beta.11` (re-pinned 2026-10-02 by `81ec18c`: cohort merge-not-replace, author-scoped tier lookup, member wallet off by default) — the kit has been re-pinned repeatedly since this governing doc's `verified_commit: d852f61` without a doc update. All four pin sites and the compatibility record agree with each other; only the governing doc has drifted.
- DOC-DRIFT (Wave 2, the repo's most-read file): `README.md:298` still states "Live pin `2d693ed2…` (beta.6, re-pinned 2026-07-21)" — a THIRD, even-older value distinct from both the governing doc's stale beta.9 citation above and the live beta.11 pin, and not covered by any `verified_commit` mechanism at all.
- The pin-site comments no longer restate release notes (`325734e`, 2026-10-02): `rust-ci.yml:19`, `deploy.yml:101-105` and `workers-deploy.yml:40-44` now say only that the kit crates resolve at `v1.0.0-beta.11` and defer provenance to the compatibility record, which pin-parity checks against the exact SHA. The earlier drift (a beta.9 annotation beside a beta.11 pin) is closed at the source.
- **Drift:** the compatibility record still contradicts itself, and the `81ec18c` re-pin widened the gap. The deployment row's SHA column and the machine-readable field now carry `13cbe6c` (`kit-compatibility-record.md:26`, `kit-compatibility-record.md:30`), but the same row's "Kit branch/tag at pin" column still reads "`main` at 341c5d2" and its notes describe nothing after 341c5d2, so the cohort merge-not-replace, author-scoped tier lookup and wallet change set that 13cbe6c brings is recorded only in the commit message. The History table still marks `931898a` (tag `v1.0.0-beta.10`) as "Current (canonical — matches `CANONICAL_KIT_SHA` above and the `KIT_REF` pins)" (`kit-compatibility-record.md:347`). pin-parity compares only the SHA, so none of this fails the gate.
- INVARIANT: the machine-readable pin lives in exactly one place the gate reads — `CANONICAL_KIT_SHA` / `CANONICAL_KIT_VERSION` in `kit-compatibility-record.md:30-31` — and every other site is compared against it, which is why the surrounding comments can rot without the gate noticing.

## DW-01.6 Deploy job sequence — clone kit, build three frontends, merge, inject, deploy
```mermaid
sequenceDiagram
    autonumber
    participant GH as gate job<br/>test-and-lint.yml, deploy.yml:109-114
    participant CO as checkout + clone kit at KIT_REF<br/>deploy.yml:132-138
    participant RB as Build React main site<br/>deploy.yml:151 npm run build
    participant LB as Build Leptos forum with Trunk<br/>deploy.yml:218-220 --public-url /community/
    participant MG as Merge React + Forum into dist/<br/>deploy.yml step 'Merge...'
    participant ENV as Inject window.__ENV__<br/>deploy.yml step 'Inject runtime env config'
    participant BB as Build retro BBS with Trunk<br/>deploy.yml:309
    participant PG as Deploy to gh-pages<br/>deploy.yml:379-385
    GH->>CO: needs.gate.outputs.passed == 'true'
    CO->>RB: dist/index.html + assets
    CO->>LB: kit/dist (forum WASM)
    RB->>MG: cp -r kit/dist/* dist/community/
    LB->>MG: (forum output)
    MG->>ENV: sed inject <script>window.__ENV__=...SIDESTR_WALLET,ENCRYPTION_ENABLED,ZONE_CONFIG...
    ENV->>BB: build BBS after the forum merge so `cp kit/dist/*` does not grab the bbs subtree
    BB->>PG: dist/community/bbs/ merged, rebrand step, 404 shims
```
- Since `d3ecd05` (2026-10-02) the deploy also fires on a lockfile-only change: `package-lock.json` joined the push path filter (`deploy.yml:13-21`), so a dependency bump that touches nothing else still republishes the site.
- Supply-chain hardening: every tool the deploy job downloads (Trunk, binaryen/`wasm-opt`, Tailwind CLI) is pinned to an exact version **and** SHA256-verified before use, because this job carries the Cloudflare API token (BASELINE-architecture.md:110-113, `deploy.yml:34-41`).
- The injected payload has grown since ADR-2015/ADR-2016 shipped: `SIDESTR_WALLET:'on'` (deploy.yml:77) gates the testnet4 wallet nav/tip UI, `ENCRYPTION_ENABLED:'true'` (deploy.yml:81) is the zone E2EE master switch, and `ZONE_CONFIG_JSON` (deploy.yml:82) now marks zones 2-4 `encrypted:true` with zone4 carrying `agent_keys:true` — see DW-03/DW-04 for the zone model and encryption detail this step only injects.

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
- "SPA deep links depend on a 404-redirect shim" is called out as load-bearing in the Known-divergences section, not incidental (BASELINE-architecture.md:138-141, `deploy.yml:341-370`, `266-275`).
- The React pickup script is deliberately external (`public/spa-redirect.js`), not an inline `<script>`, because `index.html`'s CSP (`script-src 'self'`, no `'unsafe-inline'`) would block an injected inline script — exactly the bug class that broke `/workshops` deep links (`deploy.yml` comment above the "Deploy" step).

## DW-01.8 EXTERNAL — estate authority map beyond this repo
```mermaid
flowchart LR
    DW["dreamlab-ai-website<br/>this area"] -->|"clones at KIT_REF"| NF["nostr-rust-forum kit<br/>EXTERNAL, see NF-*"]
    DW -->|"crates.io consumption"| NF
    DW -->|"CF Tunnel, native pod card"| AB["agentbox native solid-pod-rs<br/>EXTERNAL, see AB-*"]
    DW -->|"BrokerActor governance publisher,<br/>visionclaw-server admin key<br/>forum-config/dreamlab.toml:275"| VC["VisionClaw server<br/>EXTERNAL, see VC-*"]
    DW -->|"junkiejarvis website chat bridge"| AB
```
- `visionclaw-server` is a governance publisher pubkey shared with the forum's primary admin pubkey — see DW-03/DW-04 for the identity implications and IDENTITY-zones.md. The `[governance].agent_pubkeys` list this repo authors is at `forum-config/dreamlab.toml:263,274-275`; the ecosystem statement itself is at `CLAUDE.md:17`.
