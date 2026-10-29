---
id: DW-06
title: Content pipeline, ADR/security-doc inventory and the invariants register
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/BASELINE-architecture.md
  - ../dreamlab-ai-website/docs/IDENTITY-zones.md
adrs: [ADR-2001, ADR-2002, ADR-2003, ADR-2004, ADR-2005, ADR-2006, ADR-2007, ADR-2008, ADR-2011]
sources:
  - ../dreamlab-ai-website/CLAUDE.md
  - ../dreamlab-ai-website/README.md
  - ../dreamlab-ai-website/docs/BASELINE-architecture.md
  - ../dreamlab-ai-website/docs/IDENTITY-zones.md
  - ../dreamlab-ai-website/forum-config/dreamlab.toml
  - ../dreamlab-ai-website/.github/workflows/rust-ci.yml
  - ../dreamlab-ai-website/package.json
  - ../dreamlab-ai-website/scripts/check-claims.mjs
  - ../dreamlab-ai-website/scripts/lib/claims.mjs
  - ../dreamlab-ai-website/docs/claims-register.json
  - ../dreamlab-ai-website/.github/workflows/workers-deploy.yml
  - ../dreamlab-ai-website/scripts/generate-workshop-list.mjs
  - ../dreamlab-ai-website/scripts/generate-testimonials.mjs
  - ../dreamlab-ai-website/scripts/lib/content-generators.mjs
  - ../dreamlab-ai-website/docs/security/SECURITY_OVERVIEW.md
  - ../dreamlab-ai-website/docs/api/AUTH_API.md
  - ../dreamlab-ai-website/docs/api/POD_API.md
  - ../dreamlab-ai-website/docs/api/NOSTR_RELAY.md
  - ../dreamlab-ai-website/docs/api/SEARCH_API.md
  - ../dreamlab-ai-website/docs/api/MODERATION_API.md
verified_commit: 07cf8642bddb44df2388a5d1e9d2074923ab6ac0
---

## DW-06.1 Pre-build content pipeline — two generators, run before every dev/build
```mermaid
sequenceDiagram
    autonumber
    participant NPM as npm run dev / npm run build
    participant WS as generate-workshop-list.mjs
    participant WD as public/data/workshops/
    participant TS as generate-testimonials.mjs
    participant YAML as content/site-content.yaml
    participant OUT as src/data/*.json
    NPM->>WS: predev/prebuild step
    WS->>WD: readdir(workshopsBaseDir)<br/>generate-workshop-list.mjs:5,33
    WD-->>WS: per-workshop dirs, .md files
    WS->>WS: extractTitleFromMd: first H1, fallback to<br/>Title Cased filename<br/>generate-workshop-list.mjs:8-25
    WS->>WD: queue the manifest, then write manifest.json per workshop dir<br/>{title, pages: [{slug, title, pageTitle?}]}<br/>generate-workshop-list.mjs:86-90, 112
    WS->>OUT: write src/data/workshop-list.json<br/>generate-workshop-list.mjs:117-118
    NPM->>TS: predev/prebuild step
    TS->>YAML: parseYaml(site-content.yaml)<br/>generate-testimonials.mjs:11-12
    TS->>TS: data.pages.testimonials.items<br/>generate-testimonials.mjs:14
    TS->>OUT: write src/data/testimonials.json<br/>generate-testimonials.mjs:18
```
- Both generators fail soft: a missing `public/data/workshops/` writes an empty `workshop-list.json` instead of failing the build (`generate-workshop-list.mjs:121-124`), and a YAML parse error writes an empty `testimonials.json` (`generate-testimonials.mjs:22-25`) — CLAUDE.md's Data directory note explicitly marks these two JSON files as GENERATED, do-not-hand-edit.

## DW-06.2 Workshop manifest generation — page ordering rules
```mermaid
flowchart TB
    FILES["workshop dir .md files"] --> SORT["pageItems.sort<br/>generate-workshop-list.mjs:73-84"]
    SORT --> R1{"slug === readme.md?"}
    R1 -->|yes| FIRST["sorts first, always"]
    R1 -->|no| R2{"both slugs match ^(digits)[_.]?"}
    R2 -->|yes| NUMERIC["numeric prefix order<br/>e.g. 00_intro.md before 01_setup.md"]
    R2 -->|no| ALPHA["localeCompare fallback"]
    FIRST --> MANIFEST["workshopManifest {title, pages}<br/>written to <workshop>/manifest.json"]
    NUMERIC --> MANIFEST
    ALPHA --> MANIFEST
```
- The workshop id itself is reformatted for display: `workshop-01-foo` becomes `01 - Foo` via a regex that splits the leading number from the rest and title-cases each word (`generate-workshop-list.mjs:40-46`).
- Manifests are now written in a second pass: every workshop's manifest is queued first (`generate-workshop-list.mjs:86-90`), then lessons whose title duplicates another lesson's, or repeats their own workshop's title, gain a `pageTitle` (`generate-workshop-list.mjs:98-110`) before the write (`generate-workshop-list.mjs:112`). The ordering rules above are unchanged. Content correctness is also checked by `scripts/lib/content-generators.mjs`, which re-derives counts from the generators' inputs (`scripts/lib/content-generators.mjs:7-11`), because both JSON artefacts are committed yet rebuilt by `prebuild`; it also now fails when a workshop's `path` is not `/workshops/<id>` (`scripts/lib/content-generators.mjs:88-99`), since prerender and the discovery check both re-read that field.

## DW-06.5 API and security documentation inventory
```mermaid
flowchart TB
    DOCS["docs/"] --> API["api/"]
    DOCS --> SEC["security/"]
    API --> A1["AUTH_API.md — auth-worker: WebAuthn, NIP-98, D1 schema"]
    API --> A2["POD_API.md — pod-worker: WAC access control, R2 layout"]
    API --> A3["NOSTR_RELAY.md — relay-worker: WebSocket protocol, NIP-11, Durable Objects"]
    API --> A4["SEARCH_API.md — search-worker: WASM RVF search flow"]
    API --> A5["MODERATION_API.md — auth-worker: moderation/WoT/invites, kinds 30910-30914"]
    SEC --> S1["SECURITY_OVERVIEW.md — threat model, trust boundaries, CORS, SSRF"]
    SEC --> S2["AUTHENTICATION.md — auth flows"]
    SEC --> S3["QE_AUDIT_FORUM_CLIENT.md"]
    SEC --> S4["QE_AUDIT_FORUM_MESH_2026-06-07.md"]
    SEC --> S5["QE_AUDIT_NOSTR_CORE.md"]
    SEC --> S6["QE_AUDIT_WORKERS.md"]
    SEC --> S7["QE_COVERAGE_REPORT.md"]
```
- `MODERATION_API.md`'s four endpoint families are backed by Nostr parameterised-replaceable event kinds **30910-30914** (`docs/api/MODERATION_API.md:15`), with D1 projections into `moderation_actions`/`reports` for fast querying; every mutating endpoint requires a NIP-98 header binding method, URL and body-SHA256 to prevent replay (`docs/api/MODERATION_API.md:9`).

## DW-06.6 Invariants register — consolidated from both governing docs
```mermaid
flowchart TB
    I1["1. KIT_REF + Cargo pin lockstep<br/>BASELINE-architecture.md:163-166 — see DW-01.5"]
    I2["2. Privileged deploy downloads nothing unverified<br/>BASELINE-architecture.md:167-169 — see DW-01.6"]
    I3["3. window.__ENV__ vs Vite build vars, validate every endpoint<br/>BASELINE-architecture.md:170-172"]
    I4["4. GitHub Pages is origin of record until DNS re-cut<br/>BASELINE-architecture.md:173-175 — see DW-01.3"]
    I5["5. Identity stays raw 64-hex pubkey, no silent DID move<br/>IDENTITY-zones.md:175-177 — see DW-04.1"]
    I6["6. Zone required_cohorts stays dual-accept<br/>IDENTITY-zones.md:178-179 — see DW-03.1"]
    I7["7. BREACHED at HEAD: doc says only zone3 is encrypted<br/>IDENTITY-zones.md:180-181, but dreamlab.toml has zone2/3/4<br/>encrypted=true (dreamlab.toml:123,137,148); the toml's own<br/>[encryption] gate + zone4.agent_keys cite 'ADR-2016'<br/>(dreamlab.toml:34,149) but no ADR-2016-*.md exists in<br/>docs/adr/ at HEAD — the record was never written<br/>see DW-03.1"]
    I8["8. Talk-to-AI reply relays subset of of agent publish fan-out<br/>IDENTITY-zones.md:182-183 — see DW-04.3"]
    I9["9. Admin/Jarvis pubkeys + ZONE_CONFIG hand-mirrored, rotation touches every mirror<br/>IDENTITY-zones.md:184-186 — see DW-03.3/DW-03.5"]
    I10["10. website-info holds no cohort, never agent, no admin or zone key;<br/>monitor key and [[ingress]] entries equal across mirrors<br/>IDENTITY-zones.md:187-191 — see DW-04.9"]
```
- These ten invariants are the compliance surface for this area: any change touching them requires a governing-doc update plus a thin ADR under `docs/adr/`, per each doc's own Change-process section — the pack itself is a **thin ledger** amending `BASELINE-architecture.md` / `IDENTITY-zones.md` (the living docs are normative); the pre-2026-08-31 legacy corpus (numbered 013+) is frozen under `docs/archive/adr/` as evidence, not authority (`CLAUDE.md:21-23`). Its ADR-2xxx numbering overlaps VisionFlow/agentbox's own ADR-2xxx space by design — each area's ids are area-scoped, resolved by directory, not globally unique across repos.

## DW-06.7 Known divergences / DOC-DRIFT register — this area's open items
```mermaid
flowchart TB
    D1["BASELINE divergence: README said two SPAs, now says three<br/>README.md:54, CLAUDE.md:14 still dual-SPA, see DW-01.2"]
    D2["BASELINE divergence: Cloudflare-edge vs GitHub Pages origin<br/>README now says one GitHub Pages origin, README.md:43, see DW-01.3"]
    D3["forum-config/README.md pin note stale: git rev vs crates.io version<br/>BASELINE-architecture.md:150-152"]
    D4["Branded worker custom domains undeployed, workers.dev is shipped reality<br/>see DW-01.3"]
    D5["dreamlab.toml README: 'Friends' zone does not exist<br/>see DW-03.1"]
    D6["DID/Multikey convergence is documentation-only<br/>see DW-04.1"]
    D7["Admin/governance key unsplit, staged not applied<br/>see DW-03.4"]
    D8["Roster authorised_by authored but not rendered by the kit<br/>see DW-03.5"]
```

## DW-06.9 Known divergences / DOC-DRIFT register — continued (D9-D15)
```mermaid
flowchart TB
    D9["BASELINE-architecture.md cites a stale KIT_REF (BASELINE-architecture.md:117) and a beta.9 crate version (BASELINE-architecture.md:152)<br/>see DW-01.5"]
    D10["branding.rs dead code contradicts live dreamlab.toml branding values<br/>see DW-03.8/DW-03.1"]
    D11["BASELINE-architecture.md estate-closeout note predates the 2026-09-05 gate fix<br/>see DW-05.3"]
    D12["RESOLVED 07cf8642: README.md names kit source 8a5cf81, agreeing with the pins<br/>dreamlab-ai-website/README.md:65 and rust-ci.yml:20,<br/>workers-deploy.yml:44 — see DW-06.8"]
    D13["RESOLVED 2026-10-07: root-level seed scripts with a hardcoded key<br/>deleted by 3baf2fa4, historical — see DW-07.2"]
    D14["embeddings/ tooling targets RuVector Postgres + all-MiniLM-L6-v2,<br/>unrelated to the search worker; the README claim was removed — see DW-07.4"]
    D15["IDENTITY-zones.md I7 breached: zone2/zone4 now encrypted=true too<br/>(dreamlab.toml:123,148), master gate [encryption].enabled added<br/>(dreamlab.toml:34) citing an uncommitted ADR-2016 — DW-03.1/DW-04.1<br/>still state 'only zone3 encrypted', not yet re-verified against this<br/>see DW-06.6 I7"]
```
- D9 and D11 are findings from this diagram-authoring pass, not pre-existing entries in either governing doc's own Known-divergences section — both are consequences of the governing docs' `verified_commit: d852f61` being older than this repo's HEAD at this topic's stamp (`3afa374f`) and the estate-closeout note's own explicit 2026-09-04 dating.
- D12/D13/D14 are Wave 2 findings, closing the material gaps the estate audit (`reports/audit-web.md`) identified for this area; D15 is a Wave 3 finding — see DW-06.6/DW-06.6 for the full evidence.

## DW-06.8 README feature claims — checked against a claims register, not a feature list
```mermaid
flowchart TB
    CLI["npm run check:claims, package.json:31<br/>advisory, always exits 0, scripts/check-claims.mjs:2-3,33"] --> EXTRACT["extractClaims: list items of five or more words<br/>outside code fences, scripts/lib/claims.mjs:28-45"]
    README["README.md list items, default input<br/>scripts/check-claims.mjs:19"] --> EXTRACT
    EXTRACT --> CLASS["classifyDeterministic: first register match wins,<br/>else unclassified, scripts/lib/claims.mjs:48-56"]
    REG["docs/claims-register.json<br/>3 feature notes, 2 claims, docs/claims-register.json:3-40"] --> CLASS
    CLASS --> REPORT["formatReport lists only stale and unsupported<br/>scripts/lib/claims.mjs:135-142"]
    CLASS -.->|"DREAMLAB_JEV=1 and feature notes present"| JEV["advisory notes, never a verdict<br/>scripts/lib/claims.mjs:89-131"]
```
- The earlier nine-bullet README feature list this diagram once classified is gone with the README rewrite; the register is seeded only with README claims that cite a concrete in-repo `file:line` (web push in the operator overlay, declared recipients and an inbox-watch path), so uncited README claims report `unclassified` by design (`docs/claims-register.json:2`).
- **Tension (closed):** the README no longer says "no register file"; it now says the source-grounded register ships in this repository, that its evidence paths are checked in the test suite, and that the checker runs at the end of `postbuild` (`README.md:125`). `docs/claims-register.json` carries two claims and three feature notes, each now with an `evidence` path (`docs/claims-register.json:3-40`), and `checkClaims` still skips Jev only when no feature notes exist (`scripts/lib/claims.mjs:98-101`).
- **Drift (closed):** `README.md:65` used to state kit source `631c09f…`; it now states `8a5cf81…`, matching the pin sites that carry `8a5cf817e81b20e2b761899eab753768c0cb4988` (`.github/workflows/rust-ci.yml:20`, `.github/workflows/workers-deploy.yml:44`).
