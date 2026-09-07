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
verified_commit: 9a3dd8830
---

## DW-02.1 Route table — lazy-loaded, code-split
```mermaid
flowchart TB
    APP["App.tsx:29 <App>"] --> BR["BrowserRouter<br/>v7_startTransition, v7_relativeSplatPath<br/>src/App.tsx:39-40"]
    BR --> IDX["/ -> Index<br/>src/App.tsx:47"]
    BR --> PRG["/programmes -> Programmes<br/>src/App.tsx:48"]
    BR --> CC["/co-create -> CoCreate<br/>src/App.tsx:49"]
    BR --> RES["/research -> Research<br/>src/App.tsx:50"]
    BR --> ECO["/ecosystem -> Ecosystem<br/>src/App.tsx:51"]
    BR --> TEAM["/team -> Team<br/>src/App.tsx:52"]
    BR --> WSI["/workshops -> WorkshopIndex<br/>src/App.tsx:55"]
    BR --> WSP["/workshops/:workshopId(/:pageSlug) -> WorkshopPage<br/>src/App.tsx:56-57"]
    BR --> TST["/testimonials -> Testimonials<br/>src/App.tsx:60"]
    BR --> VEN["/ventures -> Ventures, unlinked direct-URL only<br/>src/App.tsx:62-63"]
    BR --> CNT["/contact -> Contact<br/>src/App.tsx:65"]
    BR --> PRIV["/privacy -> Privacy<br/>src/App.tsx:66"]
    BR --> NF["* -> NotFound<br/>src/App.tsx:76"]
```
- All route components are `lazy()`-loaded (src/App.tsx:10-22); `RouteErrorBoundary` (src/App.tsx:44) contains a failed lazy chunk so it does not white-screen the whole SPA (Sprint v9 D3, src/App.tsx:42-43 comment).
- `v7_relativeSplatPath` is audited as a no-op here (the sole splat route's links are all absolute); `v7_startTransition` only changes how a navigation to a not-yet-loaded chunk holds the outgoing page (src/App.tsx:32-38).

## DW-02.2 Legacy redirects
```mermaid
flowchart LR
    R1["/residential-training"] -->|Navigate replace| PRG["/programmes<br/>src/App.tsx:69"]
    R2["/masterclass"] -->|Navigate replace| PRG2["/programmes<br/>src/App.tsx:70"]
    R3["/system-design"] -->|Navigate replace| RES["/research<br/>src/App.tsx:71"]
    R4["/research-paper"] -->|Navigate replace| RES2["/research<br/>src/App.tsx:72"]
    R5["/work"] -->|Navigate replace| RES3["/research<br/>src/App.tsx:73"]
```
- `public/sitemap.xml` must be kept in sync with the route table by hand when routes change (CLAUDE.md route-table section).

## DW-02.3 Build & test commands
```mermaid
flowchart TB
    DEV["npm run dev"] --> PRE1["pre-step: generate-workshop-list.mjs<br/>+ generate-testimonials.mjs"]
    BUILD["npm run build"] --> PRE1
    PRE1 --> VITE["Vite 5.4, SWC plugin, production build"]
    LINT["npm run lint"] --> ESLINT["ESLint 9 flat config<br/>eslint.config.js, ignores dist/"]
    TEST["npm run test"] --> VITEST["Vitest + Testing Library, jsdom<br/>src/**/__tests__"]
    RUSTT["cd forum-config && cargo test"] --> RUSTOVERLAY["operator-overlay tests:<br/>config parsing, branding, deploy manifests"]
```
- CLAUDE.md's Behavioral Rules require `npm run build` and `npm run lint` before committing.

## DW-02.4 Dev-server hardening — `/data/team` path-traversal guard
```mermaid
sequenceDiagram
    autonumber
    participant B as browser dev request
    participant MW as configureServer middleware<br/>vite.config.ts:29-30 server.middlewares.use('/data/team', ...)
    participant FS as public/data/team/
    B->>MW: GET /data/team/<path>
    MW->>MW: reject if req.url includes '..' or '%2e'<br/>vite.config.ts:31-36 400 Invalid request
    MW->>FS: readdirSync(teamDir) when req.url === '/'
    FS-->>MW: file list
    MW->>MW: filter to *.md, drop dotfiles<br/>vite.config.ts safeFiles filter
    MW-->>B: text/plain file list
```
- This middleware exists only in the Vite dev server (`configureServer`); the production static build serves `public/data/team/` as plain files with no equivalent runtime guard.

## DW-02.5 `index.html` — CSP and build-date injection
```mermaid
flowchart LR
    HTML["index.html"] --> CSP["script-src 'self', no 'unsafe-inline'<br/>referenced by deploy.yml comment on the pickup script"]
    HTML --> DATE["__BUILD_DATE__ placeholder"]
    PLUGIN["inject-build-date plugin<br/>vite.config.ts transformIndexHtml"] --> DATE
    DATE --> REPLACED["replaced with new Date().toISOString().slice(0,10)<br/>vite.config.ts"]
```
- The CSP is why the React `__p` deep-link pickup lives in the external `public/spa-redirect.js` rather than an inline script injected at deploy time (see DW-01.7); an inline script would be blocked.

## DW-02.6 Key patterns
```mermaid
flowchart TB
    COMP["Components<br/>shadcn/ui: Radix + Tailwind + CVA"]
    FORMS["Forms<br/>React Hook Form + Zod schemas at form boundaries"]
    DATA["Data fetching<br/>TanStack React Query"]
    CONTENT["Content<br/>team bios / workshops: markdown under public/data/, fetched at runtime<br/>testimonials: content/site-content.yaml via pre-build script"]
    OG["OG/social meta<br/>src/lib/og-meta.ts — image URLs must point at files<br/>that actually exist under public/, no generation pipeline"]
    SPLIT["Code splitting<br/>Vite manual chunks (vendor, ui) + route-level lazy loading"]
```
- TypeScript: strict mode is **partial** — `noImplicitAny: false`, `strictNullChecks: true`; target ES2020, module ESNext, JSX react-jsx (CLAUDE.md TypeScript Configuration section).
