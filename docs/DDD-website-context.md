# Domain-Driven Design: visionflow.info Website

**Bounded contexts for the public-facing explanation layer of the VisionFlow coordination engineering ecosystem.**

## 1. Presentation Context

**Purpose:** Owns the website as a navigable artefact — structure, layout, scroll state, and responsive behaviour.

**Aggregates**

- `Page` — root aggregate. Holds an ordered collection of `Section[]`. Governs scroll-position tracking and active-section resolution.
- `Section` — a discrete content block. Owns its content payload, animation hook, and optional diagram reference. Identity: `sectionId`.

**Value objects**

- `Viewport { width, height, devicePixelRatio }` — immutable snapshot of the browser viewport.
- `ScrollPosition { y, normalised }` — vertical offset expressed both in pixels and as a 0–1 fraction of total page height.
- `IntersectionState { sectionId, ratio, isVisible }` — derived from IntersectionObserver; drives active-nav highlighting.

**Domain rules**

- A `Section` is *active* when `IntersectionState.ratio ≥ 0.4`.
- Navigation renders the `Page` section list; it never owns content directly.
- Layout breakpoints are properties of `Viewport`, not of individual `Section` instances.


## 2. Animation Context

**Purpose:** Owns all WASM-powered visual elements. Isolated from the DOM content model; communicates only via domain events.

**Aggregates**

- `MeshHero` — the node-graph hero animation. Owns `nodes[]`, `edges[]`, and a `PhysicsState`. Accepts `Tick` and `Resize` events to advance simulation.
- `ParticleField` — the background particle system. Owns `particles[]` and `forces[]`. Driven by `Tick` and `ScrollUpdate`.

**Value objects**

- `Seed(u64)` — deterministic PRNG seed; identical seeds produce identical layouts.
- `Edge { sourceId, targetId, weight }` — immutable connection between two nodes.
- `Color { r, g, b, a }` — normalised float colour; computed, never stored as hex.
- `NoiseField { scale, octaves, persistence }` — Perlin/simplex noise configuration for organic motion.

**Domain events** (published by WASM, consumed by the Presentation Context)

| Event | Payload | Consumer |
|---|---|---|
| `Tick` | `{ deltaMs }` | `MeshHero`, `ParticleField` |
| `Resize` | `Viewport` | `MeshHero` |
| `ScrollUpdate` | `ScrollPosition` | `ParticleField` |

**Domain rule:** The Animation Context has no knowledge of section IDs or page semantics.


## 3. Content Context

**Purpose:** Owns the ecosystem knowledge that the website renders. Authoritative source for all copy, comparisons, and economic claims.

**Aggregates**

- `EcosystemNarrative` — the primary explanatory arc. Owns structured content for: `problem`, `evolution`, `substrates[]`, `broker`, and `scaling`.
- `CompetitiveAnalysis` — owns `platforms[]` and the cross-cutting `capabilities[]` matrix used to populate the comparison table.
- `EconomicCase` — owns `scenarios[]` and `costModels[]` used in the ROI and cost-avoidance sections.

**Value objects**

- `Substrate { id, name, description, protocolRef }` — an individual execution environment (e.g. CLAUDE.md, A2A, MCP).
- `CaseStudy { title, problem, outcome, metricsRef }` — a self-contained evidence unit; immutable once published.
- `ComparisonRow { platform, capabilities: Map<capabilityId, SupportLevel> }` — a single row in the competitive table. `SupportLevel` is an enum: `Full | Partial | None`.

**Domain rules**

- `EcosystemNarrative` is the only aggregate permitted to reference `Substrate` instances directly.
- `CompetitiveAnalysis` must not embed prose; it references `CaseStudy` by ID only.
- All monetary figures in `EconomicCase` are expressed in USD and carry a `confidenceLevel` enum (`Indicative | Validated`).


## 4. Deployment Context

**Purpose:** Owns the build pipeline from source to live hosting. No runtime behaviour; purely a build-time concern.

**Aggregate**

- `BuildPipeline` — orchestrates `wasm-pack` compilation, static asset assembly, verification, and GitHub Pages artifact deployment. Identity: `pipelineRunId`.

**Value objects**

- `WasmArtifact { hash, sizeByes, targetPath }` — the compiled `.wasm` + JS glue output from `wasm-pack`.
- `StaticAsset { path, contentType, hash }` — any non-WASM file (HTML, CSS, fonts, images) included in the `dist/` bundle.
- `CNAME { hostname }` — the custom domain record written to `dist/CNAME` on every build.
- `BrowserSidecar { cdpEndpoint, siteBaseUrl, networkScope }` — the external Chrome DevTools runtime used for browser verification.

**Domain rules**

- Browser verification must connect to `BrowserSidecar`; the VisionFlow pipeline does not launch local Chrome.
- A `BuildPipeline` run is deployable only after sidecar reachability, render smoke, navigation, accessibility, and payload-budget checks pass.
- A deployable artifact is only valid if `WasmArtifact.hash` differs from the previously deployed hash or any `StaticAsset.hash` has changed. No-op pushes are prohibited.


## Context Map

```
Presentation ──events──► Animation
Presentation ──renders──► Content
Deployment ──artefacts──► Presentation  (build time only)
Deployment ──verifies──► BrowserSidecar (external dependency)
```

All runtime cross-boundary communication uses domain events or read-only value objects. No aggregate leaks across a boundary.


## Ubiquitous Language Glossary

| Term | Definition |
|---|---|
| **Broker** | The VisionFlow coordination broker — the runtime that routes tasks across substrates. |
| **Substrate** | An execution environment understood by the broker (e.g. CLAUDE.md conventions, A2A, MCP). |
| **MeshHero** | The animated node-graph canvas rendered at the top of the landing page via WASM. |
| **ParticleField** | The scroll-reactive background particle animation. |
| **Tick** | A single animation-frame event carrying elapsed milliseconds; the heartbeat of the Animation context. |
| **Section** | A discrete, named vertical block of page content (e.g. "Problem", "Architecture", "Economics"). |
| **IntersectionState** | The visibility ratio of a Section within the viewport, as reported by IntersectionObserver. |
| **Seed** | A deterministic integer that fully determines the initial layout of a WASM visual element. |
| **ComparisonRow** | One platform's row in the competitive feature matrix. |
| **WasmArtifact** | The compiled binary and its JS glue produced by `wasm-pack build`. |
| **CNAME** | The `visionflow.info` hostname record written into the static bundle for GitHub Pages routing. |
| **BrowserSidecar** | The external `browsercontainer` Chrome DevTools runtime used for all browser automation. |
| **Coordination engineering** | The discipline of designing systems that reliably orchestrate autonomous agents across heterogeneous substrates. |
