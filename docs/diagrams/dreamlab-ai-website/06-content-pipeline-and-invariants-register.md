---
id: DW-06
title: Content pipeline, ADR/security-doc inventory and the invariants register
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/BASELINE-architecture.md
  - ../dreamlab-ai-website/docs/IDENTITY-zones.md
adrs: [ADR-2001, ADR-2002, ADR-2003, ADR-2004, ADR-2005, ADR-2006, ADR-2007, ADR-2008]
sources:
  - ../dreamlab-ai-website/scripts/generate-workshop-list.mjs
  - ../dreamlab-ai-website/scripts/generate-testimonials.mjs
  - ../dreamlab-ai-website/public/data/team/manifest.json
  - ../dreamlab-ai-website/package.json
  - ../dreamlab-ai-website/docs/adr/README.md
  - ../dreamlab-ai-website/docs/security/SECURITY_OVERVIEW.md
  - ../dreamlab-ai-website/docs/api/AUTH_API.md
  - ../dreamlab-ai-website/docs/api/POD_API.md
  - ../dreamlab-ai-website/docs/api/NOSTR_RELAY.md
  - ../dreamlab-ai-website/docs/api/SEARCH_API.md
  - ../dreamlab-ai-website/docs/api/MODERATION_API.md
verified_commit: 9a3dd8830
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
    WS->>WD: readdir(workshopsBaseDir)<br/>generate-workshop-list.mjs:5,32
    WD-->>WS: per-workshop dirs, .md files
    WS->>WS: extractTitleFromMd: first H1, fallback to<br/>Title Cased filename<br/>generate-workshop-list.mjs:8-25
    WS->>WD: write manifest.json per workshop dir<br/>{title, pages: [{slug, title}]}<br/>generate-workshop-list.mjs:85-88
    WS->>OUT: write src/data/workshop-list.json<br/>generate-workshop-list.mjs:97-98
    NPM->>TS: predev/prebuild step
    TS->>YAML: parseYaml(site-content.yaml)<br/>generate-testimonials.mjs:11-12
    TS->>TS: data.pages.testimonials.items<br/>generate-testimonials.mjs:14
    TS->>OUT: write src/data/testimonials.json<br/>generate-testimonials.mjs:18
```
- Both generators fail soft: a missing `public/data/workshops/` writes an empty `workshop-list.json` instead of failing the build (`generate-workshop-list.mjs:97-99`), and a YAML parse error writes an empty `testimonials.json` (`generate-testimonials.mjs:22-25`) — CLAUDE.md's Data directory note explicitly marks these two JSON files as GENERATED, do-not-hand-edit.

## DW-06.2 Workshop manifest generation — page ordering rules
```mermaid
flowchart TB
    FILES["workshop dir .md files"] --> SORT["pageItems.sort<br/>generate-workshop-list.mjs:72-82"]
    SORT --> R1{"slug === readme.md?"}
    R1 -->|yes| FIRST["sorts first, always"]
    R1 -->|no| R2{"both slugs match ^(digits)[_.]?"}
    R2 -->|yes| NUMERIC["numeric prefix order<br/>e.g. 00_intro.md before 01_setup.md"]
    R2 -->|no| ALPHA["localeCompare fallback"]
    FIRST --> MANIFEST["workshopManifest {title, pages}<br/>written to <workshop>/manifest.json"]
    NUMERIC --> MANIFEST
    ALPHA --> MANIFEST
```
- The workshop id itself is reformatted for display: `workshop-01-foo` becomes `01 - Foo` via a regex that splits the leading number from the rest and title-cases each word (`generate-workshop-list.mjs:39-45`).

## DW-06.3 Team roster — manifest + per-member markdown, fetched at runtime
```mermaid
flowchart LR
    MANIFEST["public/data/team/manifest.json<br/>members: array of zero-padded ids 01..44"] --> FETCH["client fetches<br/>/data/team/<id>.md at runtime"]
    FETCH --> DEVGUARD["dev server: path-traversal guard<br/>see DW-02.4"]
    FETCH --> PROD["production: static files, no runtime guard"]
```
- Team portraits live under `public/images/team/` (`01..44.webp`), loaded via the same manifest (CLAUDE.md Project Structure, `public/images/` section).

## DW-06.4 ADR pack — living-doc ledger, ADR-2001 through ADR-2008
```mermaid
flowchart TB
    PACK["docs/adr/"] --> T["TEMPLATE.md, PREAMBLE.md, README.md"]
    PACK --> A1["ADR-2001 corpus-consolidation"]
    PACK --> A2["ADR-2002 split-hosting-pages-workers"]
    PACK --> A3["ADR-2003 three-frontends-one-origin"]
    PACK --> A4["ADR-2004 kit-pin-version-and-sha-lockstep"]
    PACK --> A5["ADR-2005 config-hand-synced-mirrors"]
    PACK --> A6["ADR-2006 raw-schnorr-nip42-identity"]
    PACK --> A7["ADR-2007 four-zone-dual-accept-cohorts"]
    PACK --> A8["ADR-2008 talk-to-ai-nostr-dm-routing"]
```
- This is a **thin ledger** amending `BASELINE-architecture.md` / `IDENTITY-zones.md` (the living docs are normative); the pre-2026-08-31 legacy corpus (numbered 013+) is frozen under `docs/archive/adr/` as evidence, not authority (CLAUDE.md Architecture ground truth section).
- Numbering overlaps VisionFlow/agentbox's own ADR-2xxx space by design — the brief's ledger-overlap note applies: each area's ADR ids are area-scoped, resolved by directory, not globally unique across repos.

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
- `MODERATION_API.md`'s four endpoint families are backed by Nostr parameterised-replaceable event kinds **30910-30914**, with D1 projections into `moderation_actions`/`reports` for fast querying; every mutating endpoint requires a NIP-98 header binding method, URL and body-SHA256 to prevent replay (`docs/api/MODERATION_API.md:1-10`).

## DW-06.6 Invariants register — consolidated from both governing docs
```mermaid
flowchart TB
    I1["1. KIT_REF + Cargo pin lockstep<br/>BASELINE-architecture.md:145-148 — see DW-01.5"]
    I2["2. Privileged deploy downloads nothing unverified<br/>BASELINE-architecture.md:149-151 — see DW-01.6"]
    I3["3. window.__ENV__ vs Vite build vars, validate every endpoint<br/>BASELINE-architecture.md:152-154"]
    I4["4. GitHub Pages is origin of record until DNS re-cut<br/>BASELINE-architecture.md:155-157 — see DW-01.3"]
    I5["5. Identity stays raw 64-hex pubkey, no silent DID move<br/>IDENTITY-zones.md:141-143 — see DW-04.2"]
    I6["6. Zone required_cohorts stays dual-accept<br/>IDENTITY-zones.md:144-145 — see DW-03.1"]
    I7["7. Only zone3 is encrypted; changing it is a recorded event<br/>IDENTITY-zones.md:146-147 — see DW-03.1"]
    I8["8. Talk-to-AI reply relays subset of of agent publish fan-out<br/>IDENTITY-zones.md:148-149 — see DW-04.3"]
    I9["9. Admin/Jarvis pubkeys + ZONE_CONFIG hand-mirrored, rotation touches every mirror<br/>IDENTITY-zones.md:150-152 — see DW-03.3/DW-03.5"]
```
- These nine invariants are the compliance surface for this area: any change touching them requires a governing-doc update plus a thin ADR under `docs/adr/`, per each doc's own Change-process section.

## DW-06.7 Known divergences / DOC-DRIFT register — this area's open items
```mermaid
flowchart TB
    D1["README: two SPAs vs three shipped clients<br/>see DW-01.2"]
    D2["README: Cloudflare-edge vs GitHub Pages origin<br/>see DW-01.3"]
    D3["forum-config/README.md pin note stale: git rev vs crates.io version<br/>BASELINE-architecture.md:132-134"]
    D4["Branded worker custom domains undeployed, workers.dev is shipped reality<br/>see DW-01.4"]
    D5["dreamlab.toml README: 'Friends' zone does not exist<br/>see DW-03.1"]
    D6["DID/Multikey convergence is documentation-only<br/>see DW-04.2"]
    D7["Admin/governance key unsplit, staged not applied<br/>see DW-03.4"]
    D8["Roster authorised_by authored but not rendered by the kit<br/>see DW-03.5"]
    D9["BASELINE-architecture.md itself cites a stale KIT_REF and crate version<br/>see DW-01.5"]
    D10["branding.rs dead code contradicts live dreamlab.toml branding values<br/>see DW-03.8/DW-03.2"]
    D11["BASELINE-architecture.md estate-closeout note predates the 2026-09-05 gate fix<br/>see DW-05.3"]
```
- D9 and D11 are findings from this diagram-authoring pass, not pre-existing entries in either governing doc's own Known-divergences section — both are consequences of the governing docs' `verified_commit: d852f61` being older than this area's current HEAD (`9a3dd8830`) and the estate-closeout note's own explicit 2026-09-04 dating.
