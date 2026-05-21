# Site Verification Status

**Status:** Current-state note
**Date:** 2026-05-21

The website PRD defines the target acceptance criteria for `visionflow.info`. The current repository does not yet implement every verification gate.

| PRD item | Current status | Evidence / next action |
|---|---|---|
| Static build | Passing | `website/build.sh` builds both WASM crates and writes `website/dist/` |
| Lighthouse CI | Superseded | Browser verification now uses the external Chrome DevTools sidecar; PRD perf budgets are checked through Playwright/CDP resource metrics |
| axe-core accessibility scan | Passing | `tests/site.spec.js` runs axe through Playwright connected to the sidecar |
| Contact/demo form | Not present in current static site | Add form endpoint or revise PRD scope |
| Content-hashed assets | Not implemented | Build copies stable names such as `css/styles.css` and `js/main.js`; either hash assets or mark deferred |
| Tailwind Play CDN | Superseded by local CSS in implementation | ADR-001 amended on 2026-05-20 |
| Navigation covers all PRD sections | Improved | Static nav now links problem, evolution, substrates, governance, economics, cases, landscape, scaling, and repos |
| WASM build | Passing | `wasm-pack` release builds complete for mesh hero and particle field |

## Last Verification

```sh
npm run build
npm run check:sidecar
npm run test:site
```

Result on 2026-05-21: build completed successfully; the sidecar reported Chrome/149.0.7827.14 over CDP; Playwright ran 10 browser tests through `browsercontainer` with 10 passing.

## Verification Commands

```sh
npm ci
npm run build
npm run verify
```

Browser automation contract:

- From inside the Docker network, CDP is `http://browsercontainer:9223`.
- From the host, CDP is `http://localhost:9222`.
- `SITE_BASE_URL` must resolve from the sidecar browser. By default, Playwright serves `website/dist` on `0.0.0.0:4173` and advertises the current container's Docker-network IPv4 address to the sidecar.
- The sidecar rejects non-localhost hostnames in the CDP `Host` header, so the test harness resolves `browsercontainer` to its Docker-network IP before connecting.

`npm run check:sidecar` verifies `/json/version` before running browser tests. CI/CD uses the same external sidecar and does not install or launch a local Chromium.

## Current Deferred Items

| Item | Disposition |
|---|---|
| Contact/demo form | Deferred until a static-form provider is selected |
| Content-hashed static assets | Deferred; stable names remain for now |
| WebP-only image budget | Deferred; repo currently includes PNG/JPG/SVG assets copied into `dist/img` |
| Lighthouse score gate | Deferred; current gate records CDP resource budgets and axe results without launching Lighthouse's own Chrome |
