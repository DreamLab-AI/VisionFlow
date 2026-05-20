# PRD: visionflow.info Website

**Owner:** Dr John O'Hare, DreamLab AI
**Status:** Draft
**Date:** 2026-05-20
**Version:** 1.0

---

## 1. Purpose

Deliver a public-facing marketing and technical reference site for the VisionFlow coordination engineering ecosystem. The site must convert technically sophisticated visitors (AI researchers, enterprise architects, open-source contributors) into engaged users, contributors, and commercial prospects.

---

## 2. Success Criteria

| Metric | Target |
|---|---|
| First Contentful Paint | < 3 s on 4G mobile |
| Lighthouse Performance | >= 90 |
| Lighthouse Accessibility | >= 90 (WCAG 2.1 AA) |
| Core Web Vitals | All green |
| GitHub repo link click-through | >= 15% of sessions |
| Contact / demo form submission | >= 2% of sessions |
| Bounce rate | < 55% |

---

## 3. User Stories

**US-01** As an AI researcher, I land on the hero section and within 10 seconds understand that VisionFlow is a federated coordination layer for human-AI intelligence, so I continue scrolling.

**US-02** As an enterprise architect, I read the Economic Case section and understand why paying high token costs for coordination harnesses is cheaper than failed AI deployments, so I request a demo.

**US-03** As an open-source contributor, I navigate the Repository Map and reach the correct GitHub repo for the substrate I want to contribute to, within two clicks.

**US-04** As a mobile visitor, I experience the full site without horizontal scroll or illegible text.

**US-05** As a screen-reader user, all interactive elements have ARIA labels and focus order is logical.

**US-06** As a returning visitor, the WASM hero animation loads without blocking the main thread.

---

## 4. Sections (in order)

| # | Section | Primary Goal |
|---|---|---|
| 1 | Hero | Communicate VisionFlow's identity; animate mesh topology |
| 2 | Problem | Frame the coordination gap in federated AI systems |
| 3 | Evolution Line | Timeline from monolithic AI to coordination engineering |
| 4 | Five Substrates | Detail VisionClaw, Agentbox, solid-pod-rs, nostr-rust-forum, dreamlab-ai-website |
| 5 | Judgment Broker | Explain the cross-substrate coordination primitive |
| 6 | Scaling Model | Show how token spend scales with problem complexity |
| 7 | Competitive Landscape | Position against alternatives with a comparison matrix |
| 8 | Case Studies | Three concrete problem-solving scenarios |
| 9 | Economic Case | ROI argument for coordination harnesses vs raw LLM spend |
| 10 | Repository Map | Linked cards for all public repos in DreamLab-AI org |

---

## 5. Technical Constraints

### 5.1 Deployment
- Static site only; no server-side runtime at visionflow.info
- Built by GitHub Actions; deployed to `gh-pages` branch of the canonical repo
- All assets must be cache-busted via content-hash filenames

### 5.2 Frontend Stack
- Rust compiled to WASM via `wasm-pack` for all interactive animations
- Static HTML/CSS shell for content; WASM loads asynchronously post-FCP
- CSS custom properties for the design token set; no runtime CSS-in-JS
- No client-side framework dependency for the non-WASM content layer

### 5.3 Design Tokens (mandatory)
```
--color-bg:        #0e0e11
--color-surface:   #18181d
--color-bronze:    #CD7F32
--color-gold-mid:  #D4A574
--color-gold-hi:   #FFD700
--color-text:      #e8e8f0
--color-text-muted:#8888a8
--font-display:    'Inter', system-ui, sans-serif
--font-mono:       'JetBrains Mono', monospace
```

### 5.4 WASM Modules (two required)
1. **mesh-hero**: WebGL2 canvas; animated agent-node mesh with spring physics; 60 fps target; graceful CPU fallback via `prefers-reduced-motion`
2. **particle-field**: Lightweight particle system used in the Scaling Model and Economic Case sections; driven by scroll position

### 5.5 Performance Budget
| Asset type | Budget |
|---|---|
| Total JS (including WASM glue) | < 120 KB gzip |
| WASM binary (mesh-hero) | < 300 KB gzip |
| Total CSS | < 20 KB gzip |
| Images | WebP only; max 200 KB each |
| Total page weight (initial load) | < 800 KB |

### 5.6 Accessibility
- WCAG 2.1 Level AA
- All colour pairs must pass 4.5:1 contrast (text) or 3:1 (UI components)
- WASM canvas elements expose `role="img"` with descriptive `aria-label`
- Full keyboard navigation; no focus traps outside modal dialogs

### 5.7 External Links
- DreamLab-AI GitHub org: `https://github.com/DreamLab-AI`
- DreamLab corporate site: `https://dreamlab-ai.com`
- All external links open in new tab with `rel="noopener noreferrer"`

---

## 6. Content Requirements

### 6.1 Five Substrates — required facts per substrate

| Substrate | Key facts to surface |
|---|---|
| VisionClaw | Knowledge engineering; OWL 2 EL reasoning; 92 CUDA kernels |
| Agentbox | Hardened agent runtime; 90+ skills; 180+ tools |
| solid-pod-rs | Rust Solid Protocol implementation; DID:Nostr identity; WAC; Web Ledgers |
| nostr-rust-forum | Governance UI; passkey authentication; Agent Control Surface Protocol |
| dreamlab-ai-website | React SPA; Leptos WASM forum |

### 6.2 Economic Case — required arguments
- Token cost as a fraction of failed-project cost at enterprise scale
- Coordination harness reduces hallucination surface per decision
- Federated substrate removes single-vendor lock-in, changing negotiating position
- Quantitative example: cost model for a 10-agent problem vs a 100-agent coordinated swarm

### 6.3 Competitive Landscape — required comparisons
- LangChain / LangGraph
- AutoGen
- CrewAI
- OpenAI Assistants API
- Semantic Kernel
- Dimensions: federation, knowledge reasoning, identity layer, governance, substrate composability

---

## 7. Acceptance Criteria

**AC-01** GitHub Actions workflow builds the site and deploys to gh-pages on every push to `main`; a failing build blocks the deploy.

**AC-02** Lighthouse CI runs in the Actions pipeline; scores below the targets in §2 fail the build.

**AC-03** `mesh-hero` WASM module renders at >= 55 fps on a mid-range Android device (Moto G Power equivalent) in Chrome; verified via automated Playwright test.

**AC-04** All ten sections are present in the HTML source with matching `id` attributes for anchor navigation.

**AC-05** The navigation bar links to each section and remains accessible at all viewport widths >= 320 px.

**AC-06** All five substrate cards link to their respective GitHub repositories.

**AC-07** The contact / demo form submits to a Formspree (or equivalent static-form) endpoint; submission confirmation is displayed without page reload.

**AC-08** `axe-core` automated accessibility scan reports zero violations at Level AA.

**AC-09** Total page weight on initial load (network tab, no cache) is <= 800 KB on a desktop Chrome session.

**AC-10** `prefers-reduced-motion: reduce` disables all WASM animations and substitutes a static SVG fallback.

---

## 8. Out of Scope

- Server-side personalisation or authentication
- Blog or CMS integration
- Internationalisation (English only for v1)
- E-commerce or payment flows
- Documentation site (separate concern, handled per-substrate)

---

## 9. Milestones

| Milestone | Deliverable |
|---|---|
| M1 | Repo scaffold, CI pipeline, design tokens, HTML shell |
| M2 | WASM mesh-hero and particle-field modules rendering |
| M3 | All ten sections copy-complete and styled |
| M4 | Lighthouse CI green; axe-core clean |
| M5 | Production deploy to visionflow.info |
