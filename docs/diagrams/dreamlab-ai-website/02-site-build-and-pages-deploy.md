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
verified_commit: 9b8ea495da80aaa5b45795af916bda4470467481
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
- Route-level lazy loading is supplemented by Rollup manual chunks (`vendor`, `nostr`, `ui`) grouping React/router, `nostr-tools` and Radix primitives into separate bundles (vite.config.ts:70-79).

## DW-02.3 Build & test commands
```mermaid
flowchart TB
    DEV["npm run dev<br/>package.json:22"] --> PRE1["pre-step: generate-workshop-list.mjs<br/>+ generate-testimonials.mjs<br/>package.json:21 predev"]
    BUILD["npm run build<br/>package.json:24"] --> PRE1B["package.json:23 prebuild, same pre-step"]
    PRE1B --> VITE["Vite 5.4, SWC plugin, production build"]
    LINT["npm run lint<br/>package.json:26"] --> ESLINT["ESLint 9 flat config<br/>eslint.config.js, ignores dist/"]
    TEST["npm run test<br/>package.json:28"] --> VITEST["Vitest + Testing Library, jsdom<br/>src/**/__tests__"]
    RUSTT["cd forum-config && cargo test"] --> RUSTOVERLAY["operator-overlay tests:<br/>config parsing, branding, deploy manifests"]
```
- CLAUDE.md's Behavioral Rules require `npm run build` and `npm run lint` before committing.
- TypeScript strict mode is **partial**: `noImplicitAny: false`, `strictNullChecks: true` (CLAUDE.md:190).

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
    HTML --> DATE["__BUILD_DATE__ placeholder<br/>index.html:176,187,269"]
    PLUGIN["inject-build-date plugin, ADR-043 W4<br/>vite.config.ts:22-23 transformIndexHtml"] --> DATE
    DATE --> REPLACED["replaced with new Date().toISOString().slice(0,10)<br/>vite.config.ts:24"]
```
- The CSP is why the React `__p` deep-link pickup lives in the external `public/spa-redirect.js` rather than an inline script injected at deploy time (see DW-01.7); an inline script would be blocked.
- OG/social-card image URLs (src/lib/og-meta.ts:40-41) must point at files that already exist under `public/`; there is no OG-image generation pipeline, so meta and imagery are hand-kept in sync.

## DW-02.7 Contact form — RHF + Zod boundary, no database write
```mermaid
flowchart TB
    SCHEMA["formSchema = z.object({...})<br/>src/pages/Contact.tsx:43-48<br/>name >=2 chars, email format,<br/>projectType required, message >=10 chars"]
    HOOK["useForm(FormValues)<br/>resolver: zodResolver(formSchema)<br/>src/pages/Contact.tsx:364-365"]
    SCHEMA --> HOOK
    HOOK --> VIEW["ContactMobile / ContactDesktop<br/>share one form + onSubmit via ContactViewProps<br/>src/pages/Contact.tsx:53-58"]
    VIEW --> GUARD["onSubmit: if RELAY_URL/ADMIN_PUBKEY unset<br/>toast.error(unavailable), return<br/>src/pages/Contact.tsx:380-385"]
    GUARD --> SEND["generateEphemeralIdentity -> buildEnquiryRumor<br/>-> wrapDm -> publishGiftWrap<br/>src/pages/Contact.tsx:395-409, see DW-04.3/DW-04.4"]
    SEND --> OK["result.ok? toast.success + form.reset<br/>: toast.error<br/>src/pages/Contact.tsx:411-417"]
```
- No REST endpoint and no database row: validated form data is packaged straight into the NIP-17 gift-wrap DM ingress that DW-04 diagrams; success is gated strictly on the relay's OK-true (src/pages/Contact.tsx:389-391 comment, ADR-041 D4).
