---
id: DW-02
title: Site build — React SPA, routing and the Vite pipeline
area: dreamlab-ai-website
governing:
  - ../dreamlab-ai-website/docs/BASELINE-architecture.md
adrs: []
sources:
  - ../dreamlab-ai-website/src/App.tsx
  - ../dreamlab-ai-website/vite.config.ts
  - ../dreamlab-ai-website/package.json
  - ../dreamlab-ai-website/CLAUDE.md
  - ../dreamlab-ai-website/index.html
  - ../dreamlab-ai-website/src/lib/og-meta.ts
  - ../dreamlab-ai-website/src/pages/Contact.tsx
  - ../dreamlab-ai-website/src/lib/nostr.ts
  - ../dreamlab-ai-website/src/prerender.tsx
  - ../dreamlab-ai-website/scripts/prerender.mjs
  - ../dreamlab-ai-website/scripts/generate-llms-full.mjs
  - ../dreamlab-ai-website/scripts/check-discovery.mjs
  - ../dreamlab-ai-website/scripts/check-claims.mjs
  - ../dreamlab-ai-website/docs/archive/adr/043-ai-search-visibility-aeo.md
verified_commit: 07cf8642bddb44df2388a5d1e9d2074923ab6ac0
---

## DW-02.1 Route table — lazy-loaded, code-split
```mermaid
flowchart TB
    APP["App.tsx:31 <App>"] --> BR["BrowserRouter<br/>v7_startTransition, v7_relativeSplatPath<br/>src/App.tsx:41-43"]
    BR --> IDX["/ -> Index<br/>src/App.tsx:49"]
    BR --> PRG["/programmes -> Programmes<br/>src/App.tsx:50"]
    BR --> CC["/co-create -> CoCreate<br/>src/App.tsx:51"]
    BR --> RES["/research -> Research<br/>src/App.tsx:52"]
    BR --> ECO["/ecosystem -> Ecosystem<br/>src/App.tsx:53"]
    BR --> TEAM["/team -> Team<br/>src/App.tsx:54"]
    BR --> WSI["/workshops -> WorkshopIndex<br/>src/App.tsx:57"]
    BR --> WSP["/workshops/:workshopId(/:pageSlug) -> WorkshopPage<br/>src/App.tsx:58-59"]
    BR --> TST["/testimonials -> Testimonials<br/>src/App.tsx:62"]
    BR --> VEN["/ventures -> Ventures, unlinked direct-URL only<br/>src/App.tsx:64-65"]
    BR --> CHN["/chain -> ChainMirror<br/>src/App.tsx:66"]
    BR --> CNT["/contact -> Contact<br/>src/App.tsx:70"]
    BR --> PRIV["/privacy -> Privacy<br/>src/App.tsx:71"]
    BR --> CLK["/clickthrough -> ClickThrough, hidden: noindex, absent from the sitemap<br/>src/App.tsx:67-68, src/prerender.tsx:34"]
    BR --> NF["* -> NotFound<br/>src/App.tsx:81"]
```
- All route components are `lazy()`-loaded (src/App.tsx:10-24); `RouteErrorBoundary` (src/App.tsx:46) contains a failed lazy chunk so it does not white-screen the whole SPA (Sprint v9 D3, src/App.tsx:44-45 comment).
- `v7_relativeSplatPath` is audited as a no-op here (the sole splat route's links are all absolute); `v7_startTransition` only changes how a navigation to a not-yet-loaded chunk holds the outgoing page (src/App.tsx:34-40).
- Route-level lazy loading is supplemented by Rollup manual chunks (`vendor`, `nostr`, `ui`) grouping React/router, `nostr-tools` and Radix primitives into separate bundles (vite.config.ts:61-70).

## DW-02.3 Build & test commands
```mermaid
flowchart TB
    DEV["npm run dev<br/>package.json:22"] --> PRE1["pre-step: generate-workshop-list.mjs<br/>+ generate-testimonials.mjs<br/>package.json:21 predev"]
    BUILD["npm run build<br/>package.json:24"] --> PRE1B["package.json:23 prebuild, same pre-step"]
    PRE1B --> VITE["Vite 5.4, SWC plugin, production build"]
    VITE --> POST["package.json:32 postbuild, four scripts in series"]
    POST --> PRER["prerender.mjs: render each route to dist/route/index.html,<br/>redirect stubs, sitemap.xml, workshops/llms.txt<br/>scripts/prerender.mjs:143-158"]
    PRER --> LLMS["generate-llms-full.mjs: dist/llms-full.txt<br/>from the prerendered HTML<br/>scripts/generate-llms-full.mjs:2-4"]
    LLMS --> DISC["check-discovery.mjs: asserts canonical, schema,<br/>sitemap and no __BUILD_DATE__ token<br/>scripts/check-discovery.mjs:32"]
    DISC --> CLM["check-claims.mjs: advisory README claim check, always exits 0,<br/>appended to postbuild by 07cf8642<br/>scripts/check-claims.mjs:2-3"]
    LINT["npm run lint<br/>package.json:26"] --> ESLINT["ESLint 9 flat config<br/>eslint.config.js, ignores dist/"]
    TEST["npm run test<br/>package.json:28"] --> VITEST["Vitest + Testing Library, jsdom<br/>src/**/__tests__"]
    RUSTT["cd forum-config && cargo test"] --> RUSTOVERLAY["operator-overlay tests:<br/>config parsing, branding, deploy manifests"]
```
- The postbuild step is new since the freshness plugin left vite.config.ts: `prerender.mjs` reads the built `dist/index.html` as its template (scripts/prerender.mjs:51) and renders every route through Vite's SSR loader (scripts/prerender.mjs:62), so the deploy ships static HTML per route; `deploy.yml` now triggers on all of `scripts/**` (`.github/workflows/deploy.yml:27`) because of it.
- CLAUDE.md's Behavioral Rules require `npm run build` and `npm run lint` before committing.
- TypeScript strict mode is **partial**: `noImplicitAny: false`, `strictNullChecks: true` (CLAUDE.md:199).

## DW-02.4 Dev-server hardening — `/data/team` path-traversal guard
```mermaid
sequenceDiagram
    autonumber
    participant B as browser dev request
    participant MW as configureServer middleware<br/>vite.config.ts:20-21 server.middlewares.use('/data/team', ...)
    participant FS as public/data/team/
    B->>MW: GET /data/team/<path>
    MW->>MW: reject if req.url includes '..' or '%2e'<br/>vite.config.ts:23-27 400 Invalid request
    MW->>FS: readdirSync(teamDir) when req.url === '/'
    FS-->>MW: file list
    MW->>MW: filter to *.md, drop dotfiles<br/>vite.config.ts:34 safeFiles filter
    MW-->>B: text/plain file list
```
- This middleware exists only in the Vite dev server (`configureServer`); the production static build serves `public/data/team/` as plain files with no equivalent runtime guard.

## DW-02.5 `index.html` — CSP, and where build-date handling went
```mermaid
flowchart LR
    HTML["index.html"] --> CSP["script-src 'self', no 'unsafe-inline'<br/>index.html:64-67, 80"]
    HTML --> NODATE["no __BUILD_DATE__ placeholder remains<br/>index.html keeps only a static Organization graph<br/>index.html:18; per-route schema is src/lib/og-meta.ts:251-260"]
    VITE["vite.config.ts plugins: react, configure-server only<br/>vite.config.ts:16-19"] -. "inject-build-date removed" .-> NODATE
    PRER["prerender.mjs per-route meta, canonical, JSON-LD<br/>scripts/prerender.mjs:94-127<br/>hidden routes: noindex, nofollow, scripts/prerender.mjs:110-115"] --> LASTMOD["sitemap lastmod from git log of page source, hidden routes left out, scripts/prerender.mjs:157<br/>scripts/prerender.mjs:44-47"]
    DISC["check-discovery.mjs:32 fails the build on any token"] -. guards .-> NODATE
```
- The CSP is why the React `__p` deep-link pickup lives in the external `public/spa-redirect.js` rather than an inline script injected at deploy time (see DW-01.7; `index.html:85-88`); an inline script would be blocked.
- **Drift:** the earlier reading of this topic (and the archived ADR-043, `docs/archive/adr/043-ai-search-visibility-aeo.md:5-6`) that a Vite `inject-build-date` plugin replaces `__BUILD_DATE__` in `index.html` is false at HEAD: the plugin and every placeholder are gone, `dateModified` is no longer stamped with the build date, and freshness is the per-page git commit date in the sitemap `<lastmod>` (`scripts/prerender.mjs:44-47`), omitted rather than faked on a shallow clone (`scripts/prerender.mjs:43`), which is why `deploy.yml` checks out with `fetch-depth: 0` (`.github/workflows/deploy.yml:159-160`).
- OG/social-card image URLs (src/lib/og-meta.ts:42-51) must point at files that already exist under `public/`; there is no OG-image generation pipeline, so meta and imagery are hand-kept in sync.

## DW-02.7 Contact form — RHF + Zod boundary, no database write
```mermaid
flowchart TB
    SCHEMA["formSchema = z.object({...})<br/>src/pages/Contact.tsx:47-53<br/>name >=2 chars, email format,<br/>projectType required, message >=10 chars"]
    HOOK["useForm(FormValues)<br/>resolver: zodResolver(formSchema)<br/>src/pages/Contact.tsx:456-457"]
    SCHEMA --> HOOK
    HOOK --> VIEW["ContactMobile / ContactDesktop<br/>share one form + onSubmit via ContactViewProps<br/>src/pages/Contact.tsx:57-62"]
    VIEW --> GUARD["onSubmit: if RELAY_URL/ADMIN_PUBKEY unset<br/>toast.error(unavailable), return<br/>src/pages/Contact.tsx:482-485"]
    GUARD --> SEND["generateEphemeralIdentity -> buildEnquiryRumor<br/>-> wrapDm -> publishGiftWrap<br/>src/pages/Contact.tsx:495-509, see DW-04.3/DW-04.4"]
    SEND --> OK["result.ok? best-effort publishMonitorCopy (void), toast.success, form.reset<br/>: toast.error<br/>src/pages/Contact.tsx:511-517, src/lib/nostr.ts:352"]
```
- No REST endpoint and no database row: validated form data is packaged straight into the NIP-17 gift-wrap DM ingress that DW-04 diagrams; success is gated strictly on the relay's OK-true (src/pages/Contact.tsx:489-491 comment, ADR-041 D4).
