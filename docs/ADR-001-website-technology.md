# ADR-001: visionflow.info Website Technology Stack

**Date:** 2026-05-20
**Status:** Accepted
**Deciders:** DreamLab AI

---

## Context

VisionFlow requires a public marketing website at visionflow.info. The site must showcase Rust/WASM technical competence, deploy without infrastructure overhead, and align with the existing DreamLab aesthetic. Key constraints: no server-side runtime, gh-pages hosting, and visual fidelity to the dreamlab-ai-website design system.

---

## Decisions

### 1. Static Site with Rust WASM Modules

**Decision:** Vanilla HTML/CSS/JS with Rust WASM modules built via wasm-pack. No framework (Next.js, Leptos, SvelteKit).

**Rationale:**
- gh-pages serves static assets; no SSR or edge runtime is available or needed.
- WASM modules provide a live demonstration of Rust competence to the technical audience.
- Mirrors the established voronoi-hero pattern from dreamlab-ai-website, reusing proven integration patterns.
- Eliminates framework lock-in and hydration complexity for a site with no dynamic data requirements.

**Consequences:**
- Manual DOM manipulation where a framework would generate boilerplate; acceptable at this site's scale.
- WASM binary delivery adds initial load; mitigated by separating crates (see decision 3).

---

### 2. Tailwind CSS via Play CDN

**2026-05-20 amendment:** The current implementation uses local static CSS rather than Tailwind Play CDN. Treat this decision as superseded in implementation until a follow-up ADR either adopts local CSS as the permanent path or reinstates a Tailwind build/CDN strategy.

**Decision:** Style via Tailwind CSS Play CDN (`<script src="https://cdn.tailwindcss.com">`). No PostCSS build step.

**Rationale:**
- Eliminates a Node.js build pipeline for CSS, keeping the toolchain to Rust/wasm-pack only.
- The Play CDN is the same utility system as a local install; class semantics are identical.
- Rapid iteration: class changes are visible immediately without a watch process.
- Consistent with DreamLab's existing design vocabulary.

**Consequences:**
- Play CDN is not recommended for production at high scale (runtime JIT compilation, no purging). Acceptable for a marketing site with modest traffic; can be swapped for a CLI build if traffic or performance requirements change.
- Network dependency at load time; mitigated by CDN caching and the site's non-critical-path nature.

---

### 3. Cargo Workspace with Separate WASM Crates

**Decision:** Cargo workspace containing discrete crates: `mesh-hero` (WebGL mesh animation) and `particle-field` (background particle system).

**Rationale:**
- Independent compilation: changing one crate does not invalidate the other's build cache.
- Smaller per-module `.wasm` binaries improve initial load and allow lazy loading.
- Clean separation of concerns maps directly to independent visual components.
- Each crate can be versioned and tested independently.

**Consequences:**
- Slightly more Cargo workspace boilerplate.
- wasm-pack must be invoked once per crate in CI; GitHub Actions matrix handles this without significant complexity.

---

### 4. GitHub Actions Deployment to gh-pages

**2026-05-20 amendment:** The current workflow uses GitHub Pages artifact upload/deploy actions rather than manually pushing a built `dist/` directory to the `gh-pages` branch. The deployment target remains GitHub Pages.

**Decision:** CI/CD via GitHub Actions: build WASM with wasm-pack, assemble static assets, push to `gh-pages` branch. Domain mapped via CNAME record pointing visionflow.info at the Pages endpoint.

**Rationale:**
- Zero hosting cost; fully managed TLS via GitHub Pages.
- wasm-pack is a single-binary install in CI, keeping the workflow self-contained.
- Push-to-main triggers automatic deployment; no manual release step.
- CNAME record is the standard gh-pages custom domain mechanism.

**Consequences:**
- Deployment pipeline is coupled to GitHub Actions availability.
- Cold CI runs require downloading the Rust toolchain and wasm-pack; caching `~/.cargo` and `target/` mitigates build time.

---

### 4A. Browser Verification Through External Chrome DevTools Sidecar

**2026-05-21 amendment:** Browser automation uses the external `browsercontainer` Chrome DevTools sidecar from agentbox. From inside the Docker network the CDP endpoint is `http://browsercontainer:9223`; from the host it is `http://localhost:9222`.

**Decision:** Playwright tests connect over CDP to the sidecar instead of installing or launching Chromium in the VisionFlow runtime. CI/CD runs on a self-hosted Linux runner that can reach the sidecar, serves `website/dist` on `0.0.0.0:4173`, and exposes the site to the sidecar through `SITE_BASE_URL` or the runner container's Docker-network IPv4 address.

**Rationale:**
- Agentbox has standardized browser automation on the GPU-backed sidecar; VisionFlow should not reintroduce a local browser dependency.
- The sidecar is the environment used for Chrome DevTools, screenshots, DOM inspection, and accessibility checks across the ecosystem.
- Removing `playwright install --with-deps chromium` avoids drift between CI and the operator runtime.

**Consequences:**
- Browser tests fail fast if the sidecar is not reachable via `/json/version`.
- GitHub-hosted runners are not sufficient for the full browser gate unless they are connected to an equivalent sidecar network.
- Lighthouse CI is deferred because its default launcher model conflicts with the no-local-Chrome decision; payload and accessibility gates remain active through CDP.

---

### 5. DreamLab Design System

**Decision:** Apply the DreamLab anthracite-and-bronze palette and glassmorphic component language:

| Token | Value |
|---|---|
| Background | `#0e0e11` (dark anthracite) |
| Accent bronze | `#CD7F32` |
| Accent gold | `#D4A574` |
| Accent bright-gold | `#FFD700` |
| Typeface | Inter (Google Fonts) |
| Card style | Glassmorphic (`backdrop-filter: blur`, semi-transparent borders) |

**Rationale:**
- Visual coherence across DreamLab properties reduces cognitive load for returning visitors.
- Established palette; no design decisions required.

**Consequences:**
- Site is visually identifiable as a DreamLab product, which is the intent.
- Any future rebrand of the DreamLab system propagates here.

---

### 6. No JavaScript Framework

**Decision:** Vanilla JS with the Intersection Observer API for scroll-triggered animations. No React, Vue, or Svelte.

**Rationale:**
- WASM handles all computationally intensive visual work; the JS layer is thin orchestration.
- Minimal bundle size maximises First Contentful Paint on cold loads.
- Intersection Observer is well-supported (>97 % global) and requires no polyfill.
- Removes the npm dependency graph and associated supply-chain surface area.

**Consequences:**
- Animation sequencing is written imperatively; manageable given the limited number of scroll triggers.
- No component model; HTML structure must be maintained by hand.

---

## Summary

The current stack (static HTML/CSS + Rust WASM + GitHub Pages artifact deploy + external Chrome DevTools sidecar verification) prioritises deployment simplicity and technical demonstration over application-framework ergonomics. It is deliberately minimal: WASM replaces a framework's complexity budget with Rust's type system. The DreamLab design system ensures aesthetic consistency without bespoke design work.
