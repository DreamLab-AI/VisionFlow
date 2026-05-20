# Site Verification Status

**Status:** Current-state note
**Date:** 2026-05-20

The website PRD defines the target acceptance criteria for `visionflow.info`. The current repository does not yet implement every verification gate.

| PRD item | Current status | Evidence / next action |
|---|---|---|
| Static build | Passing | `website/build.sh` builds both WASM crates and writes `website/dist/` |
| Lighthouse CI | Not present | Add Lighthouse CI workflow/config before claiming AC-02 |
| axe-core accessibility scan | Not present | Add automated axe or Playwright accessibility check before claiming AC-08 |
| Contact/demo form | Not present in current static site | Add form endpoint or revise PRD scope |
| Content-hashed assets | Not implemented | Build copies stable names such as `css/styles.css` and `js/main.js`; either hash assets or mark deferred |
| Tailwind Play CDN | Superseded by local CSS in implementation | ADR-001 should be amended or superseded |
| Navigation covers all PRD sections | Partial | Site nav omits some PRD sections even where content exists |
| WASM build | Passing | `wasm-pack` release builds complete for mesh hero and particle field |

## Last Local Verification

```sh
cd website
./build.sh
```

Result on 2026-05-20: build completed successfully. No Lighthouse, axe, or browser rendering validation was run in this pass.

