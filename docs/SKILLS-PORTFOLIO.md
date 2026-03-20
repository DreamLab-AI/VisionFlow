# Skills Portfolio

68 skills organised across 11 categories. 5 deprecated redirects. All skills have YAML frontmatter, description, and "When Not To Use" sections.

## Portfolio Map

```mermaid
mindmap
  root((68 Skills))
    Browser & UI
      browser
      browser-automation
      chrome-cdp
      playwright
      host-webserver-debug
      daisyui
      ui-ux-pro-max-skill
      wasm-js
    Development & Quality
      build-with-quality
      pair-programming
      verification-quality
      sparc-methodology
      skill-builder
      rust-development
      cuda
      pytorch-ml
    Content & Docs
      report-builder
      mermaid-diagrams
      paperbanana
      latex-documents
      docs-alignment
      prd2build
      fossflow
      web-summary
    Media & Creative
      ffmpeg-processing
      imagemagick
      comfyui
      blender
    Research & AI
      perplexity-research
      gemini-url-context
      deepseek-reasoning
      openai-codex
    AgentDB & Memory
      agentdb-advanced
      agentdb-learning
      agentdb-memory-patterns
      agentdb-vector-search
    Swarm & Coordination
      swarm-advanced
      hive-mind-advanced
      hooks-automation
      performance-analysis
      stream-chain
    GIS & Specialised
      qgis
      ontology-core
      ontology-enrich
      agentic-jujutsu
      jupyter-notebooks
    GitHub Integration
      github-code-review
      github-multi-repo
      github-project-management
      github-release-management
      github-workflow-automation
    V3 Implementation
      v3-cli-modernization
      v3-core-implementation
      v3-ddd-architecture
      v3-integration-deep
      v3-mcp-optimization
      v3-memory-unification
      v3-performance-optimization
      v3-security-overhaul
      v3-swarm-coordination
    Flow Nexus
      flow-nexus-neural
      flow-nexus-platform
      flow-nexus-swarm
```

## Browser Automation Decision Tree

```mermaid
flowchart TD
    START[Browser Task?] --> Q1{Need to interact<br/>with page?}
    Q1 -->|No| FETCH[WebFetch / curl<br/>No browser needed]
    Q1 -->|Yes| Q2{Which scenario?}

    Q2 -->|Forms, scraping, clicks| Q3{Need visual<br/>rendering?}
    Q3 -->|No| AB[agent-browser<br/>Fastest, smallest context]
    Q3 -->|Yes| PW[Playwright<br/>Display :1 + VNC]

    Q2 -->|Screenshots, visual testing| PW

    Q2 -->|Debug live page| Q4{Where is the page?}
    Q4 -->|In container Chromium| CDP[Chrome CDP<br/>Live session, raw CDP]
    Q4 -->|On Docker host| HWD[host-webserver-debug<br/>HTTPS bridge]

    Q2 -->|Network inspection, tracing| PW
    Q2 -->|Logged-in session access| CDP
    Q2 -->|Parallel multi-page| MULTI[agent-browser<br/>--session isolation]
    Q2 -->|Multiple needs| COMBINE[Combine tools<br/>See Combination Patterns]

    style AB fill:#4ade80,color:#000
    style PW fill:#60a5fa,color:#000
    style CDP fill:#f59e0b,color:#000
    style HWD fill:#c084fc,color:#000
    style FETCH fill:#94a3b8,color:#000
    style MULTI fill:#4ade80,color:#000
    style COMBINE fill:#f472b6,color:#000
```

## Skill Categories

| Category | Count | Key Skills |
|----------|-------|------------|
| Browser & UI | 8 | browser-automation (meta), playwright, chrome-cdp, daisyui |
| Development & Quality | 8 | build-with-quality, sparc-methodology, cuda, pytorch-ml |
| Content & Docs | 8 | report-builder, mermaid-diagrams, latex-documents, paperbanana |
| Media & Creative | 4 | ffmpeg-processing, comfyui, blender, imagemagick |
| Research & AI | 4 | perplexity-research, gemini-url-context, deepseek-reasoning |
| AgentDB & Memory | 4 | agentdb-vector-search, agentdb-learning, agentdb-memory-patterns |
| Swarm & Coordination | 5 | swarm-advanced, hive-mind-advanced, hooks-automation |
| GIS & Specialised | 5 | qgis, ontology-core, jupyter-notebooks |
| GitHub Integration | 5 | github-code-review, github-release-management |
| V3 Implementation | 9 | v3-core-implementation, v3-security-overhaul |
| Flow Nexus | 3 | flow-nexus-neural, flow-nexus-swarm |
| Deprecated (redirects) | 5 | agentdb-optimization, perplexity, swarm-orchestration |

## Conformance

All 68 skills pass:
- SKILL.md present with YAML frontmatter
- `name:` and `description:` fields
- "When Not To Use" section (non-deprecated)
- Live ↔ Mirror sync (multi-agent-docker/skills/)

## Invocation

| Method | Example |
|--------|---------|
| Slash command | `/browser-automation` |
| Natural language | "I need to scrape a website" (auto-triggers browser skill) |
| Direct tool | `agent-browser open https://example.com` |
