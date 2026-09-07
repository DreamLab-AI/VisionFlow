# Diagrams as code

Machine-readable coverage of the whole VisionFlow estate — nine repositories, one identity spine — verified
against the code and each repository's ADR pack and governing documents. Sequence diagrams first; no narrative —
every fact lives inside a diagram as a participant `path:line`, a message, or a `Note`. This tree lives in the
estate canon (VisionFlow) so that cross-repo edges are first-class; it was moved here from VisionClaw
`docs/diagrams` on 2026-09-07 (history there up to 4d1a698e7).

All `sources:` / `governing:` paths are relative to the VisionFlow repo root and reach sibling checkouts through
the local layout in [`docs/architecture/repository-map.md`](../architecture/repository-map.md) (`../project`,
`../project/agentbox`, `../solid-pod-rs`, …). Citations inside diagrams stay short (basename or trailing
fragment) and resolve by suffix against the file's own `sources:`.

| Path | Area id | Repository (local path) |
|------|---------|-------------------------|
| `visionflow/NN-*.md` | `VF-NN` | VisionFlow — canon, website, estate-health, dream-cycle, governance gates (`.`) |
| `visionclaw/NN-*.md` | `VC-NN` | VisionClaw — Rust server, GPU/wire, knowledge/data, React + Godot clients (`../project`) |
| `agentbox/NN-*.md` | `AB-NN` | agentbox — container runtime, ingress/identity/governance, memory/learning (`../project/agentbox`) |
| `solid-pod-rs/NN-*.md` | `SP-NN` | solid-pod-rs — Solid/JSS library and server, did:nostr, git pods (`../solid-pod-rs`) |
| `nostr-rust-forum/NN-*.md` | `NF-NN` | nostr-rust-forum — forum kit, Workers, passkey auth, relay, ACS kinds (`../nostr-rust-forum`) |
| `dreamlab-ai-website/NN-*.md` | `DW-NN` | dreamlab-ai-website — DreamLab deployment and operator overlay (`../dreamlab-ai-website`) |
| `vowl-wasm/NN-*.md` | `VW-NN` | vowl-wasm — VOWL notation engine, WASM (`../vowl-wasm`) |
| `knowledgegraph/NN-*.md` | `KG-NN` | knowledgeGraph — published corpus and build pipeline (`../knowledgeGraph`) |
| `visiongraph/NN-*.md` | `VG-NN` | visionGraph — authoring vault and publishing pipeline (`../visionGraph`) |
| `estate/NN-*.md` | `ES-NN` | cross-repo and infrastructure interfaces; `verified_commit` is a `{repo: sha}` map |
| `COVERAGE.md` | | generated inverted indexes: diagram → file, ADR → files, governing doc → files, source path → files |
| `hero/` | | marketing hero images and their Mermaid → Nano Banana pipeline; not part of the coverage tree |
| `rendered/` | | mmdc SVG output, gitignored, regenerable |

The pre-2026-09-05 technical diagrams and narrative docs are history only, kept in VisionClaw at
[`docs/archive/diagrams/2026-09-pre-overhaul/`](https://github.com/DreamLab-AI/VisionClaw/tree/main/docs/archive/diagrams/2026-09-pre-overhaul).

## File contract

````
---
id: VC-03                      # <AREA>-<NN>, unique
title: REST request lifecycle
area: visionclaw               # = the directory name (see the table above)
governing: [../project/docs/IDENTITY-authority-chain.md]
adrs: [ADR-2009, ADR-2011]
sources: [../project/src/main.rs, ../project/src/middleware/rbac_gate.rs]   # relative to VisionFlow root, must exist
verified_commit: b00c28a0d     # one sha, or {visionclaw: …, agentbox: …} for a topic whose sources span repos
---
## VC-03.1 GET /api/graph/data — read path
```mermaid
sequenceDiagram
    autonumber
    participant RG as RbacGate<br/>src/middleware/rbac_gate.rs:122
    ...
```
````

Every mermaid block sits under an `## <file-id>.<n> <title>` heading; ids are unique tree-wide; at most
three prose lines per diagram. Notes use the prefixes `INVARIANT:`, `DIVERGENCE:` (governing-doc open
item), `DOC-DRIFT:` (doc says X, code does Y), `EXTERNAL:` (asserted by this repo about another repo),
`see XX-NN.n` (cross-reference).

## Tooling

```bash
node scripts/diagram-index-gen.cjs docs/diagrams --check              # frontmatter, ids, paths, prose limit
node scripts/diagram-index-gen.cjs docs/diagrams --check --render     # + parse every block with mmdc → rendered/
node scripts/diagram-index-gen.cjs docs/diagrams --check --cite-check # + resolve every path:line citation
node scripts/diagram-index-gen.cjs docs/diagrams                      # regenerate the index below + COVERAGE.md
```

`--cite-check` resolves each `path:line` inside a diagram against the file's own `sources:` list, asserts the
file is long enough, warns when the anchor line is blank or a lone closing brace, and — for a participant
labelled with a function name — warns when the cited line falls outside that function's body. It **warns,
never fails**.

That last check exists because relocating a citation by diff, however carefully, preserves whatever the
citation meant: if it was already pointing at the wrong line, a re-anchoring pass moves the error and stamps a
fresh `verified_commit` on it. Re-derive a citation from the SYMBOL (`grep -n` the name, then read the body),
never from a computed offset.

Its blind spots: a bare `:NNN` continuation (`Dockerfile.unified:255<br/>… ENTRYPOINT:340`), an extensionless
path (`Makefile:12`, `Dockerfile:40` — the matcher requires a `.ext`), a basename two `sources:` entries share
(`ci.yml`), a `governing:` doc that is not also a source, and — unfixably — a citation that resolves to a real,
non-blank line describing behaviour that has since been deleted. Those still need a human reading the code at
each cited line.

One rule follows from all of it: **verify a landing line by reading it, never by adding a shift to the old
one.** A diff shift tells you where a line moved, not whether the citation was pointing at the right line
before it moved — and a wrong citation plus a correct shift is still a wrong citation, now wearing a fresh
`verified_commit`.

`--check --cite-check` passing is not sufficient: `--render` catches grammar defects the checker cannot see, so
re-render after EVERY edit, including a one-line note. Traps measured on 2026-09-07 across ~1,300 diagrams (each
cost a lane a render cycle):

| Breaks | Where | Do instead |
|---|---|---|
| bare `;` (and HTML entities such as `&lt;`, which contain one) | sequenceDiagram message text and `Note over X:` | plain prose, commas, `<br/>`; quoted flowchart labels and `note … end note` blocks are safe |
| `::` or a quote (`Foo::bar`) | stateDiagram-v2 transition labels | `Foo.bar` |
| `{ }` in a member line (`Replayed { ttl }`) | classDiagram | drop the braces |
| escaped `\"` | flowchart node labels | reword |
| `&&`, `|`, `>` | sequence messages | prose |
| `call` as a classDef name | flowchart | another name |
| `mindmap`, `timeline` | anywhere | rejected by the generator as low-density kinds; use flowchart |
| a wide `erDiagram`, an `LR` subgraph of many unconnected nodes, an unwrapped `note for` | render > 4500 px | `flowchart TB` with subgraphs, wrap notes at ~55 chars with `<br/>`, flip `TD`↔`LR` |

`mmdc` is the Nix-installed Mermaid CLI (11.16). For a visual check, copy an SVG from `rendered/` to
`/home/devuser/gui-tools/` and open `file:///home/devuser/exchange/<name>.svg` in the browsercontainer
sidecar (chrome-devtools MCP `browser-gpu`); `agentbox/scripts/mmdc-sidecar.sh` renders through the same
sidecar. Hero images regenerate with `hero/src/batch-generate.sh` (Nano Banana Pro, `GOOGLE_API_KEY`).

## Diagram index

<!-- BEGIN GENERATED DIAGRAM INDEX -->
_123 topic files, 1323 diagrams. Regenerate with_ `node scripts/diagram-index-gen.js docs/diagrams`.

### visionflow

| ID | Topic | Diagrams | Kinds | Governing | ADRs |
|----|-------|----------|-------|-----------|------|
| VF-01 | [Repo composition and the canon — document taxonomy, ADR lifecycle, index gate](visionflow/01-repo-composition-and-canon.md) | 12 | flowchart, stateDiagram-v2, erDiagram, sequenceDiagram | [BASELINE-visionflow.md](../../docs/BASELINE-visionflow.md), [README.md](../../docs/README.md), [repository-map.md](../../docs/architecture/repository-map.md) | ADR-2001, ADR-2002, ADR-2005, ADR-2006, ADR-2007 |
| VF-02 | [Website build and deploy — copy-only build, asset inventory, gated Pages publication](visionflow/02-website-build-and-deploy.md) | 12 | sequenceDiagram, flowchart, erDiagram, stateDiagram-v2 | [BASELINE-visionflow.md](../../docs/BASELINE-visionflow.md), [site-verification.md](../../docs/site-verification.md), [PRD-website.md](../../docs/PRD-website.md) | ADR-2002, ADR-2003, ADR-2004, ADR-2005 |
| VF-03 | [Estate health — nightly CI collection, offline check, and the snapshot the site renders](visionflow/03-estate-health.md) | 10 | flowchart, sequenceDiagram, stateDiagram-v2, erDiagram | [BASELINE-visionflow.md](../../docs/BASELINE-visionflow.md), [repository-map.md](../../docs/architecture/repository-map.md) | ADR-2006, ADR-2008 |
| VF-04 | [Dream-cycle integration — rotation slots, evaluator contracts, ledger and the human gate](visionflow/04-dream-cycle-integration.md) | 10 | flowchart, stateDiagram-v2, sequenceDiagram, erDiagram | [BASELINE-visionflow.md](../../docs/BASELINE-visionflow.md), [README.md](../../docs/adr/README.md) | ADR-2008, ADR-2009 |
| VF-05 | [Governance gates — every CI gate, what it reads, its pass/fail rule and whether it blocks](visionflow/05-governance-gates.md) | 12 | flowchart, sequenceDiagram, stateDiagram-v2 | [BASELINE-visionflow.md](../../docs/BASELINE-visionflow.md), [compatibility-matrix.md](../../docs/architecture/compatibility-matrix.md) | ADR-2005, ADR-2006 |
| VF-06 | [Diagram tooling — the report render gate and this tree's own generator](visionflow/06-diagram-tooling.md) | 13 | flowchart, sequenceDiagram, stateDiagram-v2 | [BASELINE-visionflow.md](../../docs/BASELINE-visionflow.md) | ADR-2004 |
| VF-07 | [Pitch, presentation and report pipelines — release manifests, Wardley exports, LaTeX and the content that has no pipeline](visionflow/07-pitch-presentation-reports.md) | 11 | flowchart, sequenceDiagram, classDiagram | [BASELINE-visionflow.md](../../docs/BASELINE-visionflow.md), [compatibility-matrix.md](../../docs/architecture/compatibility-matrix.md) | ADR-2006 |
| VF-08 | [Compatibility matrix and dependency direction across the estate](visionflow/08-compatibility-and-dependency-direction.md) | 10 | flowchart, classDiagram, sequenceDiagram, stateDiagram-v2 | [compatibility-matrix.md](../../docs/architecture/compatibility-matrix.md), [repository-map.md](../../docs/architecture/repository-map.md), [BASELINE-visionflow.md](../../docs/BASELINE-visionflow.md) | ADR-2006, ADR-2007 |

### visionclaw

| ID | Topic | Diagrams | Kinds | Governing | ADRs |
|----|-------|----------|-------|-----------|------|
| VC-01 | [Server boot, AppState construction and the full route table](visionclaw/01-boot-and-app-state.md) | 15 | sequenceDiagram, flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2004, ADR-2005, ADR-2007, ADR-2008, ADR-2026, ADR-2037, ADR-2038, ADR-2045, ADR-2053 |
| VC-02 | [Actor supervision tree, GraphServiceSupervisor routing and peer actor surfaces](visionclaw/02-actor-supervision.md) | 20 | flowchart, sequenceDiagram, stateDiagram-v2, classDiagram | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2005, ADR-2007, ADR-2045 |
| VC-03 | [Request lifecycle, identity and the RBAC lattice](visionclaw/03-request-lifecycle-and-rbac.md) | 16 | sequenceDiagram, flowchart | [IDENTITY-authority-chain.md](../../../project/docs/IDENTITY-authority-chain.md), [SECURITY-profiles.md](../../../project/docs/SECURITY-profiles.md), [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2002, ADR-2003, ADR-2009, ADR-2010, ADR-2011, ADR-2012, ADR-2013, ADR-2026, ADR-2039, ADR-2043, ADR-2044 |
| VC-04 | [Handler internals — graph, state, and domain route families](visionclaw/04-handlers-graph-and-state.md) | 29 | sequenceDiagram, flowchart, classDiagram | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2005, ADR-2007, ADR-2011 |
| VC-05 | [Governance and identity handler families](visionclaw/05-handlers-governance-and-identity.md) | 18 | sequenceDiagram, flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [IDENTITY-authority-chain.md](../../../project/docs/IDENTITY-authority-chain.md) | ADR-2006, ADR-2010, ADR-2011, ADR-2013, ADR-2016 |
| VC-06 | [Settings round trip — REST, actors, SQLite adapter and generated client types](visionclaw/06-settings-round-trip.md) | 10 | flowchart, sequenceDiagram, classDiagram | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2005, ADR-2011, ADR-2041, ADR-2046, ADR-2047, ADR-2080 |
| VC-07 | [Hexagonal ports, adapters, the CQRS application layer and the crate split](visionclaw/07-hexagonal-ports-and-crates.md) | 11 | flowchart, sequenceDiagram, classDiagram | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2004, ADR-2005, ADR-2016 |
| VC-08 | [Observability, liveness canaries, health composition and the dev/production build loop](visionclaw/08-observability-health-and-dev-loop.md) | 12 | sequenceDiagram, flowchart, classDiagram | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [SECURITY-profiles.md](../../../project/docs/SECURITY-profiles.md) | ADR-2008, ADR-2026, ADR-2037, ADR-2038, ADR-2049 |
| VC-09 | [Configuration loading, boot-time profile assertion and the environment-flag register](visionclaw/09-config-and-env-flags.md) | 15 | sequenceDiagram, flowchart, classDiagram | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [SECURITY-profiles.md](../../../project/docs/SECURITY-profiles.md) | ADR-2012, ADR-2026, ADR-2037, ADR-2038, ADR-2039, ADR-2041, ADR-2043, ADR-2046, ADR-2094 |
| VC-10 | [GPU supervision and context bus](visionclaw/10-gpu-supervision-and-context-bus.md) | 10 | flowchart, sequenceDiagram, stateDiagram-v2, classDiagram | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [GPU-wire-abi.md](../../../project/docs/GPU-wire-abi.md) | ADR-2007, ADR-2053 |
| VC-11 | [Physics step and force channels](visionclaw/11-physics-step-and-force-channels.md) | 9 | sequenceDiagram, flowchart, classDiagram, stateDiagram-v2 | [GPU-wire-abi.md](../../../project/docs/GPU-wire-abi.md), [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2007, ADR-2028, ADR-2029, ADR-2055, ADR-2060 |
| VC-12 | [SimParams ABI, GPU buffers and PTX](visionclaw/12-simparams-abi-and-ptx.md) | 7 | classDiagram, sequenceDiagram, flowchart | [GPU-wire-abi.md](../../../project/docs/GPU-wire-abi.md) | ADR-2028, ADR-2030, ADR-2054, ADR-2055, ADR-2056 |
| VC-13 | [Position broadcast pipeline and WebSocket](visionclaw/13-broadcast-pipeline-and-websocket.md) | 8 | sequenceDiagram, stateDiagram-v2 | [PROTOCOL-registry.md](../../../project/docs/PROTOCOL-registry.md), [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2003, ADR-2018, ADR-2009, ADR-2002 |
| VC-14 | [Wire frames and tag registry](visionclaw/14-wire-frames-and-tag-registry.md) | 9 | classDiagram, flowchart, sequenceDiagram | [PROTOCOL-registry.md](../../../project/docs/PROTOCOL-registry.md), [GPU-wire-abi.md](../../../project/docs/GPU-wire-abi.md) | ADR-2018, ADR-2019, ADR-2020, ADR-2024, ADR-2057, ADR-2060 |
| VC-15 | [GPU analytics kernels and pathfinding](visionclaw/15-gpu-analytics.md) | 13 | sequenceDiagram, classDiagram, flowchart | [GPU-wire-abi.md](../../../project/docs/GPU-wire-abi.md) | ADR-2007, ADR-2053, ADR-2054, ADR-2061 |
| VC-16 | [Interaction — drag, pin, layout, constraints and agent beams](visionclaw/16-interaction-drag-pin-layout.md) | 7 | sequenceDiagram, flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [PROTOCOL-registry.md](../../../project/docs/PROTOCOL-registry.md) | ADR-2020, ADR-2029, ADR-2055 |
| VC-17 | [XR presence crate and co-presence](visionclaw/17-xr-presence-crate.md) | 6 | sequenceDiagram, stateDiagram-v2, classDiagram | [PROTOCOL-registry.md](../../../project/docs/PROTOCOL-registry.md), [XR-client.md](../../../project/docs/XR-client.md) | ADR-2019, ADR-2020 |
| VC-18 | [Analytics support handlers and the analytics WebSocket](visionclaw/18-analytics-support-handlers.md) | 9 | flowchart, sequenceDiagram, classDiagram | [PROTOCOL-registry.md](../../../project/docs/PROTOCOL-registry.md), [GPU-wire-abi.md](../../../project/docs/GPU-wire-abi.md) | ADR-2007, ADR-2009, ADR-2059 |
| VC-20 | [Ontology pipeline - OWL extraction, Oxigraph, Whelk reasoning, governed mutation](visionclaw/20-ontology-pipeline-oxigraph-whelk.md) | 12 | sequenceDiagram, erDiagram, classDiagram, flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2004, ADR-2071, ADR-2064, ADR-2066, ADR-2068 |
| VC-21 | [Corpus ingest (GitHub/local vault) and the vault-migrate converter](visionclaw/21-corpus-ingest-and-vault.md) | 12 | sequenceDiagram, flowchart, stateDiagram-v2, classDiagram | [VAULT-corpus-format.md](../../../project/docs/VAULT-corpus-format.md), [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2014, ADR-2040, ADR-2041, ADR-2042, ADR-2070 |
| VC-22 | [Data authority, provenance and erasure](visionclaw/22-data-authority-provenance-erasure.md) | 11 | flowchart, erDiagram, sequenceDiagram | [DATA-authority-erasure.md](../../../project/docs/DATA-authority-erasure.md), [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2004, ADR-2015, ADR-2016, ADR-2017, ADR-2069, ADR-2070 |
| VC-23 | [Identifier taxonomy — typed URN, did:nostr, sha256-12, federation crossing, wire node-id](visionclaw/23-identifiers-urn-did-sha12.md) | 10 | classDiagram, sequenceDiagram, flowchart | [IDENTIFIER-taxonomy.md](../../../project/docs/IDENTIFIER-taxonomy.md) | ADR-2021, ADR-2022, ADR-2023, ADR-2024, ADR-2025, ADR-2070, ADR-2072 |
| VC-24 | [ACSP — governed decision/elevation pipeline](visionclaw/24-acsp-decision-elevation.md) | 11 | stateDiagram-v2, sequenceDiagram, flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2006, ADR-2101 |
| VC-25 | [Insight loop, KPI, briefing, NLQ and semantic classification](visionclaw/25-insight-kpi-nlq-semantics.md) | 13 | sequenceDiagram, erDiagram, flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2014, ADR-2040, ADR-2004, ADR-2063, ADR-2065 |
| VC-26 | [Solid Pod integration — embedded pod, proxy, client stack](visionclaw/26-solid-pod-and-jss.md) | 13 | flowchart, sequenceDiagram | [DATA-authority-erasure.md](../../../project/docs/DATA-authority-erasure.md), [IDENTITY-authority-chain.md](../../../project/docs/IDENTITY-authority-chain.md) | ADR-2067, ADR-2068, ADR-2070, ADR-2106 |
| VC-27 | [Agent estate integration — MCP relay, discovery, monitoring, ingest](visionclaw/27-agent-integration-mcp-relay.md) | 13 | sequenceDiagram, classDiagram | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [IDENTIFIER-taxonomy.md](../../../project/docs/IDENTIFIER-taxonomy.md) | ADR-2025 |
| VC-28 | [External services — outbound integrations](visionclaw/28-external-services.md) | 9 | sequenceDiagram, flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2066 |
| VC-30 | [React client boot sequence and state layer](visionclaw/30-client-boot-and-state.md) | 12 | sequenceDiagram, classDiagram, flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2074, ADR-2077 |
| VC-31 | [R3F/Three.js graph render pipeline and WASM scene effects](visionclaw/31-client-graph-render-pipeline.md) | 10 | flowchart, sequenceDiagram, classDiagram | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) |  |
| VC-32 | [Client WebSocket transport and binary position protocol](visionclaw/32-client-websocket-and-binary.md) | 16 | sequenceDiagram, classDiagram, flowchart | [PROTOCOL-registry.md](../../../project/docs/PROTOCOL-registry.md), [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2002, ADR-2019, ADR-2020, ADR-2047, ADR-2057, ADR-2078, ADR-2080 |
| VC-33 | [Browser-client identity — NIP-07, NIP-98, passkeys, RBAC gating](visionclaw/33-client-auth-and-identity.md) | 9 | sequenceDiagram, classDiagram | [IDENTITY-authority-chain.md](../../../project/docs/IDENTITY-authority-chain.md), [SECURITY-profiles.md](../../../project/docs/SECURITY-profiles.md) | ADR-2002, ADR-2009, ADR-2011, ADR-2012, ADR-2074, ADR-2075 |
| VC-34 | [Client feature directories — API and WebSocket surface](visionclaw/34-client-features.md) | 21 | sequenceDiagram, flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2041, ADR-2006, ADR-2074, ADR-2077 |
| VC-35 | [Voice end to end — PTT, STT, intent, TTS](visionclaw/35-voice-end-to-end.md) | 12 | stateDiagram-v2, sequenceDiagram, classDiagram, flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [IDENTITY-authority-chain.md](../../../project/docs/IDENTITY-authority-chain.md) | ADR-2002, ADR-2039, ADR-2075 |
| VC-36 | [Godot + gdext OpenXR immersive client](visionclaw/36-godot-xr-client.md) | 19 | sequenceDiagram, flowchart, stateDiagram-v2, classDiagram | [XR-client.md](../../../project/docs/XR-client.md), [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2032, ADR-2033, ADR-2034, ADR-2035, ADR-2036, ADR-2039, ADR-2076, ADR-2079 |
| VC-37 | [Browser XR surface and desktop spatial input](visionclaw/37-browser-xr-and-desktop-input.md) | 8 | sequenceDiagram, flowchart, stateDiagram-v2 | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [XR-client.md](../../../project/docs/XR-client.md) | ADR-2032, ADR-2081 |

### agentbox

| ID | Topic | Diagrams | Kinds | Governing | ADRs |
|----|-------|----------|-------|-----------|------|
| AB-01 | [Nix flake composition and apply-class gates](agentbox/01-nix-flake-composition.md) | 11 | flowchart, stateDiagram-v2, sequenceDiagram, classDiagram | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2003, ADR-2006, ADR-2029, ADR-2039, ADR-2080 |
| AB-02 | [Boot sequence, supervision tree and readiness](agentbox/02-boot-sequence-and-readiness.md) | 20 | sequenceDiagram, flowchart, stateDiagram-v2 | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2003, ADR-2007, ADR-2028, ADR-2029, ADR-2034, ADR-2063, ADR-2080 |
| AB-03 | [Management API request lifecycle and route table](agentbox/03-management-api-request-lifecycle.md) | 17 | sequenceDiagram, flowchart | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md), [INGRESS-identity.md](../../../project/agentbox/docs/INGRESS-identity.md) | ADR-2005, ADR-2013, ADR-2003 |
| AB-04 | [Five-slot adapter spine, dispatch middleware and connect lifecycle](agentbox/04-adapter-spine.md) | 16 | flowchart, classDiagram, sequenceDiagram, stateDiagram-v2 | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2004, ADR-2005, ADR-2035, ADR-2036, ADR-2037, ADR-2064 |
| AB-05 | [Manifest gate catalogue, vault path authority and the agentbox.sh CLI](agentbox/05-manifest-gates-and-cli.md) | 11 | sequenceDiagram, flowchart, stateDiagram-v2 | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2003, ADR-2028, ADR-2029, ADR-2036, ADR-2037, ADR-2038, ADR-2039, ADR-2080 |
| AB-06 | [Compose overlays, sidecar topology and the loopback-publish invariant](agentbox/06-sidecars-and-compose-overlays.md) | 8 | flowchart, sequenceDiagram | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2013, ADR-2003, ADR-2040 |
| AB-07 | [Daemon classes, argv-boundary reaping, cron and backups](agentbox/07-daemons-reapers-cron-backups.md) | 9 | flowchart, stateDiagram-v2, sequenceDiagram | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2032, ADR-2003, ADR-2039, ADR-2040 |
| AB-08 | [Claude Code hook pipeline and its handlers](agentbox/08-hooks-pipeline.md) | 14 | flowchart, sequenceDiagram | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2015, ADR-2026, ADR-2007 |
| AB-09 | [MCP registry, boot projector and the server catalogue](agentbox/09-mcp-servers-catalogue.md) | 7 | flowchart, sequenceDiagram | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2008, ADR-2003, ADR-2039 |
| AB-10 | [Ingress — nip98-proxy and the AoE door](agentbox/10-ingress-nip98-proxy-and-aoe.md) | 13 | flowchart, sequenceDiagram, stateDiagram-v2 | [INGRESS-identity.md](../../../project/agentbox/docs/INGRESS-identity.md), [SECURITY-profiles.md](../../../project/agentbox/docs/SECURITY-profiles.md) | ADR-2002, ADR-2009, ADR-2010, ADR-2011, ADR-2013, ADR-2047, ADR-2080 |
| AB-11 | [Identity — DID, URN, mandate, authority](agentbox/11-identity-did-mandate-authority.md) | 16 | classDiagram, sequenceDiagram, flowchart, stateDiagram-v2 | [INGRESS-identity.md](../../../project/agentbox/docs/INGRESS-identity.md), [PROTOCOL-registry.md](../../../project/agentbox/docs/PROTOCOL-registry.md) | ADR-2011, ADR-2025, ADR-2027, ADR-2064 |
| AB-12 | [tab0-bridge and the interaction plane](agentbox/12-tab0-bridge-and-interaction-plane.md) | 14 | flowchart, sequenceDiagram, classDiagram | [INGRESS-identity.md](../../../project/agentbox/docs/INGRESS-identity.md), [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md) | ADR-2009, ADR-2010, ADR-2011, ADR-2047 |
| AB-13 | [Nostr — relay, gateway, pod bridge, session mirror](agentbox/13-nostr-relay-gateway-bridge-mirror.md) | 17 | flowchart, sequenceDiagram, classDiagram, stateDiagram-v2 | [INGRESS-identity.md](../../../project/agentbox/docs/INGRESS-identity.md), [SECURITY-profiles.md](../../../project/agentbox/docs/SECURITY-profiles.md), [PROTOCOL-registry.md](../../../project/agentbox/docs/PROTOCOL-registry.md) | ADR-2012, ADR-2025, ADR-2026, ADR-2061 |
| AB-14 | [Governance — journal, action pipeline, approvals](agentbox/14-governance-journal-actions-approvals.md) | 14 | flowchart, stateDiagram-v2, sequenceDiagram, classDiagram | [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md), [SECURITY-profiles.md](../../../project/agentbox/docs/SECURITY-profiles.md) | ADR-2022, ADR-2027, ADR-2041 |
| AB-15 | [Capability gating, spend caps and consultants](agentbox/15-capability-gating-spend-consultants.md) | 14 | flowchart, sequenceDiagram, stateDiagram-v2 | [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md), [SECURITY-profiles.md](../../../project/agentbox/docs/SECURITY-profiles.md) | ADR-2020, ADR-2031, ADR-2033, ADR-2080 |
| AB-16 | [Secrets custody, seccomp and runtime profiles](agentbox/16-secrets-custody-seccomp-profiles.md) | 11 | flowchart, classDiagram, sequenceDiagram, stateDiagram-v2 | [SECURITY-profiles.md](../../../project/agentbox/docs/SECURITY-profiles.md), [INGRESS-identity.md](../../../project/agentbox/docs/INGRESS-identity.md) | ADR-2007, ADR-2026, ADR-2027, ADR-2033 |
| AB-17 | [Agent events and the BC20 provenance bridge](agentbox/17-agent-events-and-provenance-bridge.md) | 10 | classDiagram, sequenceDiagram | [PROTOCOL-registry.md](../../../project/agentbox/docs/PROTOCOL-registry.md), [INGRESS-identity.md](../../../project/agentbox/docs/INGRESS-identity.md) | ADR-2011, ADR-2022, ADR-2025, ADR-2061 |
| AB-20 | [RuVector memory path — every MCP memory tool end to end](agentbox/20-ruvector-memory-path.md) | 12 | sequenceDiagram, flowchart, stateDiagram-v2, erDiagram | [LEARNING-memory.md](../../../project/agentbox/docs/LEARNING-memory.md) | ADR-2014, ADR-2018, ADR-2019, ADR-2051 |
| AB-21 | [Learning loop — capture, judge, distil, consume](agentbox/21-learning-loop.md) | 10 | sequenceDiagram, classDiagram, stateDiagram-v2, flowchart | [LEARNING-memory.md](../../../project/agentbox/docs/LEARNING-memory.md) | ADR-2015, ADR-2016, ADR-2017, ADR-2018, ADR-2051, ADR-2052 |
| AB-22 | [Skills estate — discovery, lint gate, routing, harness/precedent MCP bridges](agentbox/22-skills-and-routing.md) | 13 | sequenceDiagram, stateDiagram-v2, flowchart | [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md) | ADR-2020, ADR-2021, ADR-2028, ADR-2056, ADR-2057 |
| AB-23 | [Dream machine — nightly cycle, gates and acceptance path](agentbox/23-dream-engine.md) | 14 | stateDiagram-v2, sequenceDiagram, classDiagram, flowchart | [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md) | ADR-2024, ADR-2053, ADR-2081 |
| AB-24 | [Ontology Loom facade and the model-swap seam](agentbox/24-loom-facade.md) | 9 | flowchart, sequenceDiagram, classDiagram, stateDiagram-v2 | [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md) | ADR-2023, ADR-2053, ADR-2055 |
| AB-25 | [Ontology tools and governed writes](agentbox/25-ontology-tools-and-governed-writes.md) | 10 | flowchart, sequenceDiagram, classDiagram, stateDiagram-v2 | [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md) | ADR-2022, ADR-2023, ADR-2028, ADR-2054 |
| AB-26 | [Headroom compression, the beads work-DAG, typed spawn and RuvNet grounding](agentbox/26-headroom-beads-spawn-brain.md) | 9 | sequenceDiagram, classDiagram, stateDiagram-v2, flowchart | [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md), [LEARNING-memory.md](../../../project/agentbox/docs/LEARNING-memory.md) | ADR-2004, ADR-2005, ADR-2020 |
| AB-27 | [Media and GPU capability services — manifest-gated dispatch](agentbox/27-media-gpu-capability-services.md) | 12 | sequenceDiagram, flowchart, classDiagram | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md), [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md) | ADR-2006, ADR-2020, ADR-2040, ADR-2057 |
| AB-28 | [Agentbox service crates and the manifest binary](agentbox/28-agentbox-services-and-manifest-binary.md) | 10 | flowchart, sequenceDiagram, classDiagram | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2030, ADR-2031, ADR-2032 |
| AB-29 | [Metaharness router console — AoE dispatch plane phase 0](agentbox/29-metaharness-router-console.md) | 8 | flowchart, sequenceDiagram, stateDiagram-v2 | [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md) | ADR-2079, ADR-2080 |

### solid-pod-rs

| ID | Topic | Diagrams | Kinds | Governing | ADRs |
|----|-------|----------|-------|-----------|------|
| SP-01 | [Workspace composition, crate dependency graph and the feature-flag matrix](solid-pod-rs/01-workspace-composition.md) | 11 | flowchart, classDiagram | [README.md](../../../solid-pod-rs/README.md), [BASELINE-solid-pod-rs.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md), [ecosystem-integration.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md) | ADR-2001, ADR-2004, ADR-2005 |
| SP-02 | [Server boot, configuration layering and the full route table](solid-pod-rs/02-server-boot-and-routes.md) | 13 | sequenceDiagram, classDiagram, flowchart, stateDiagram-v2 | [README.md](../../../solid-pod-rs/README.md), [BASELINE-solid-pod-rs.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md) | ADR-2004, ADR-2007 |
| SP-03 | [LDP request lifecycle — verbs, containers, content negotiation and PATCH](solid-pod-rs/03-ldp-request-lifecycle.md) | 13 | sequenceDiagram, flowchart, classDiagram, stateDiagram-v2 | [README.md](../../../solid-pod-rs/README.md), [BASELINE-solid-pod-rs.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md) | ADR-2002, ADR-2004, ADR-2005 |
| SP-04 | [Web Access Control — policy resolution, evaluation, conditions and the sidecar rule](solid-pod-rs/04-wac-authorisation.md) | 14 | sequenceDiagram, stateDiagram-v2, flowchart, classDiagram | [README.md](../../../solid-pod-rs/README.md), [BASELINE-solid-pod-rs.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md) | ADR-2002, ADR-2005 |
| SP-05 | [Identity — NIP-98, replay, did:nostr, did:key, WebID, Solid-OIDC and the IdP](solid-pod-rs/05-identity-nip98-did-oidc.md) | 20 | flowchart, sequenceDiagram, stateDiagram-v2, classDiagram | [README.md](../../../solid-pod-rs/README.md), [BASELINE-solid-pod-rs.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md), [ecosystem-integration.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md) | ADR-2003, ADR-2006 |
| SP-06 | [Storage backends, quota, multitenancy, provisioning and git-versioned pods](solid-pod-rs/06-storage-quota-and-git-pods.md) | 14 | classDiagram, flowchart, sequenceDiagram, stateDiagram-v2 | [README.md](../../../solid-pod-rs/README.md), [BASELINE-solid-pod-rs.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md) | ADR-2004, ADR-2005 |
| SP-07 | [Provenance — git-marks, block-trails, Bitcoin anchoring — and the web ledger](solid-pod-rs/07-provenance-and-payments.md) | 17 | flowchart, stateDiagram-v2, sequenceDiagram, classDiagram | [README.md](../../../solid-pod-rs/README.md), [BASELINE-solid-pod-rs.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md) | ADR-2004, ADR-2007 |
| SP-08 | [Federation and outward surfaces — ActivityPub, NIP-01 relay, Solid Notifications, the forge, MCP](solid-pod-rs/08-federation-notifications-forge-mcp.md) | 16 | flowchart, sequenceDiagram, stateDiagram-v2, classDiagram | [README.md](../../../solid-pod-rs/README.md), [ecosystem-integration.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md) | ADR-2005, ADR-2006 |
| SP-09 | [CI gates, the release pipeline, versioning and the consumer pin matrix](solid-pod-rs/09-ci-release-and-consumer-pins.md) | 10 | flowchart, sequenceDiagram | [README.md](../../../solid-pod-rs/README.md), [ecosystem-integration.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/explanation/ecosystem-integration.md), [BASELINE-solid-pod-rs.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md) | ADR-2001 |
| SP-10 | [Failure modes, fail-closed boundaries and the invariant register](solid-pod-rs/10-failure-modes-and-invariants.md) | 11 | flowchart, sequenceDiagram | [README.md](../../../solid-pod-rs/README.md), [BASELINE-solid-pod-rs.md](../../../solid-pod-rs/crates/solid-pod-rs/docs/BASELINE-solid-pod-rs.md) | ADR-2002, ADR-2003, ADR-2004, ADR-2005, ADR-2006, ADR-2007 |

### nostr-rust-forum

| ID | Topic | Diagrams | Kinds | Governing | ADRs |
|----|-------|----------|-------|-----------|------|
| NF-01 | [Cargo workspace, the five Workers, entry points and the upstream-absorption canary](nostr-rust-forum/01-workspace-and-worker-topology.md) | 6 | flowchart, classDiagram, stateDiagram-v2 | [BASELINE-architecture.md](../../../nostr-rust-forum/docs/BASELINE-architecture.md) | ADR-2002, ADR-2007 |
| NF-02 | [auth-worker — passkey ceremonies, NIP-98 gating, membership and the REST surface](nostr-rust-forum/02-auth-worker-passkey-nip98-and-membership.md) | 9 | sequenceDiagram, flowchart, stateDiagram-v2 | [IDENTITY-keys-and-trust.md](../../../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md), [BASELINE-architecture.md](../../../nostr-rust-forum/docs/BASELINE-architecture.md) | ADR-2003, ADR-2004 |
| NF-03 | [relay-worker — NIP-42 AUTH, the EVENT admission pipeline, trust ladder and federation](nostr-rust-forum/03-relay-worker-admission-auth-and-trust.md) | 13 | flowchart, sequenceDiagram, stateDiagram-v2, classDiagram | [BASELINE-architecture.md](../../../nostr-rust-forum/docs/BASELINE-architecture.md), [IDENTITY-keys-and-trust.md](../../../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md) | ADR-2004, ADR-2005, ADR-2006, ADR-2010 |
| NF-04 | [pod-worker — Solid LDP surface, WAC evaluation, delegation, quota and payments](nostr-rust-forum/04-pod-worker-ldp-wac-and-delegation.md) | 10 | flowchart, sequenceDiagram, classDiagram | [BASELINE-architecture.md](../../../nostr-rust-forum/docs/BASELINE-architecture.md), [IDENTITY-keys-and-trust.md](../../../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md) | ADR-2009, ADR-2007 |
| NF-05 | [The two Leptos clients — boot, identity, transport, onboarding, messaging and the retro BBS](nostr-rust-forum/05-leptos-clients-onboarding-messaging-governance.md) | 12 | sequenceDiagram, flowchart, classDiagram, stateDiagram-v2 | [BASELINE-architecture.md](../../../nostr-rust-forum/docs/BASELINE-architecture.md), [IDENTITY-keys-and-trust.md](../../../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md) | ADR-2008 |
| NF-06 | [Agent Control Surface Protocol — kinds 31400-31405, the broker aggregate and its consumers](nostr-rust-forum/06-agent-control-surface-31400-31405.md) | 10 | classDiagram, flowchart, sequenceDiagram, stateDiagram-v2, erDiagram | [BASELINE-architecture.md](../../../nostr-rust-forum/docs/BASELINE-architecture.md) | ADR-2010 |
| NF-07 | [search-worker, preview-worker and the shared rate-limit / ASCII utilities](nostr-rust-forum/07-search-preview-and-shared-worker-utilities.md) | 8 | flowchart, sequenceDiagram, classDiagram | [BASELINE-architecture.md](../../../nostr-rust-forum/docs/BASELINE-architecture.md) |  |
| NF-08 | [Config projection, zones, the KV/D1/R2 store map and admin authority](nostr-rust-forum/08-config-zones-stores-and-admin-authority.md) | 8 | flowchart, sequenceDiagram | [BASELINE-architecture.md](../../../nostr-rust-forum/docs/BASELINE-architecture.md), [IDENTITY-keys-and-trust.md](../../../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md) | ADR-2004, ADR-2006, ADR-2007 |
| NF-09 | [CI gates, wrangler deploy, fixtures, benchmarks, e2e and the setup-skill scaffold](nostr-rust-forum/09-ci-deploy-canary-and-validation.md) | 9 | flowchart, sequenceDiagram, classDiagram | [BASELINE-architecture.md](../../../nostr-rust-forum/docs/BASELINE-architecture.md) | ADR-2002, ADR-2003, ADR-2007 |
| NF-10 | [Invariants, the re-verified anomaly register and the doc-drift ledger](nostr-rust-forum/10-invariants-and-anomaly-register.md) | 9 | flowchart | [BASELINE-architecture.md](../../../nostr-rust-forum/docs/BASELINE-architecture.md), [IDENTITY-keys-and-trust.md](../../../nostr-rust-forum/docs/IDENTITY-keys-and-trust.md) | ADR-2002, ADR-2003, ADR-2004, ADR-2005, ADR-2006, ADR-2007, ADR-2008, ADR-2009, ADR-2010 |

### dreamlab-ai-website

| ID | Topic | Diagrams | Kinds | Governing | ADRs |
|----|-------|----------|-------|-----------|------|
| DW-01 | [Repo composition and deploy topology](dreamlab-ai-website/01-repo-composition-and-deploy-topology.md) | 8 | flowchart, sequenceDiagram, stateDiagram-v2 | [BASELINE-architecture.md](../../../dreamlab-ai-website/docs/BASELINE-architecture.md) | ADR-2001, ADR-2002, ADR-2003, ADR-2004 |
| DW-02 | [Site build — React SPA, routing and the Vite pipeline](dreamlab-ai-website/02-site-build-and-pages-deploy.md) | 6 | flowchart, sequenceDiagram | [BASELINE-architecture.md](../../../dreamlab-ai-website/docs/BASELINE-architecture.md) |  |
| DW-03 | [forum-config overlay — zones, admin, KV and the hand-synced mirror set](dreamlab-ai-website/03-forum-config-overlay-zones-admin-kv-sync.md) | 8 | flowchart, stateDiagram-v2, sequenceDiagram | [BASELINE-architecture.md](../../../dreamlab-ai-website/docs/BASELINE-architecture.md) | ADR-2005, ADR-2007 |
| DW-04 | [Identity, zones and security posture](dreamlab-ai-website/04-identity-zones-and-security-posture.md) | 8 | flowchart, sequenceDiagram | [IDENTITY-zones.md](../../../dreamlab-ai-website/docs/IDENTITY-zones.md) | ADR-2006, ADR-2007, ADR-2008 |
| DW-05 | [CI gates — rust-ci, KIT_REF pin guard, playwright, config mirrors](dreamlab-ai-website/05-ci-gates-rust-ci-kit-pin-playwright.md) | 6 | flowchart, sequenceDiagram | [BASELINE-architecture.md](../../../dreamlab-ai-website/docs/BASELINE-architecture.md) | ADR-2004, ADR-2005 |
| DW-06 | [Content pipeline, ADR/security-doc inventory and the invariants register](dreamlab-ai-website/06-content-pipeline-and-invariants-register.md) | 7 | sequenceDiagram, flowchart | [BASELINE-architecture.md](../../../dreamlab-ai-website/docs/BASELINE-architecture.md), [IDENTITY-zones.md](../../../dreamlab-ai-website/docs/IDENTITY-zones.md) | ADR-2001, ADR-2002, ADR-2003, ADR-2004, ADR-2005, ADR-2006, ADR-2007, ADR-2008 |

### vowl-wasm

| ID | Topic | Diagrams | Kinds | Governing | ADRs |
|----|-------|----------|-------|-----------|------|
| VW-01 | [Crate composition, modules and the WASM boundary](vowl-wasm/01-crate-composition-and-wasm-boundary.md) | 9 | flowchart, sequenceDiagram, classDiagram | [README.md](../../../vowl-wasm/README.md) |  |
| VW-02 | [OWL parse → graph model](vowl-wasm/02-owl-parse-to-graph-model.md) | 8 | sequenceDiagram, flowchart, classDiagram, stateDiagram-v2 | [README.md](../../../vowl-wasm/README.md) |  |
| VW-03 | [Force layout — Barnes-Hut, SIMD and CSR simulation lifecycle](vowl-wasm/03-force-layout-lifecycle.md) | 8 | sequenceDiagram, flowchart, stateDiagram-v2, classDiagram | [README.md](../../../vowl-wasm/README.md) |  |
| VW-04 | [JS API surface, examples and downstream consumers](vowl-wasm/04-js-api-examples-and-consumers.md) | 7 | classDiagram, flowchart, sequenceDiagram | [README.md](../../../vowl-wasm/README.md) |  |
| VW-05 | [Build, test, bench and the npm publish pipeline](vowl-wasm/05-build-test-bench-and-publish.md) | 6 | flowchart, sequenceDiagram, classDiagram | [README.md](../../../vowl-wasm/README.md) |  |

### knowledgegraph

| ID | Topic | Diagrams | Kinds | Governing | ADRs |
|----|-------|----------|-------|-----------|------|
| KG-01 | [Repo composition and the three-way data/pipeline/explorer licensing boundary](knowledgegraph/01-repo-composition-and-licensing.md) | 4 | flowchart, sequenceDiagram | [BASELINE-narrativegoldmine.md](../../../knowledgeGraph/docs/BASELINE-narrativegoldmine.md) | ADR-2001, ADR-2002 |
| KG-02 | [Corpus → ontology build pipeline — 8 stages, census to graph tiers](knowledgegraph/02-corpus-to-ontology-pipeline.md) | 6 | flowchart, sequenceDiagram, classDiagram | [BASELINE-narrativegoldmine.md](../../../knowledgeGraph/docs/BASELINE-narrativegoldmine.md) | ADR-2001, ADR-2002, ADR-2003, ADR-2004 |
| KG-03 | [Ontology architecture — page anatomy, Turtle mapping, taxonomy resolution, bridging](knowledgegraph/03-ontology-architecture.md) | 5 | flowchart, classDiagram, sequenceDiagram | [BASELINE-narrativegoldmine.md](../../../knowledgeGraph/docs/BASELINE-narrativegoldmine.md) | ADR-2002, ADR-2003 |
| KG-04 | [Explorer — NGG1 tier reader, worker transport, React SPA, and the externalised WASM crate](knowledgegraph/04-explorer-and-site-build.md) | 5 | flowchart, sequenceDiagram | [BASELINE-narrativegoldmine.md](../../../knowledgeGraph/docs/BASELINE-narrativegoldmine.md) | ADR-2001 |
| KG-05 | [CI/CD — the six build.yml gates, and the deploy that happens in a different repo](knowledgegraph/05-ci-cd-and-publish.md) | 4 | flowchart, sequenceDiagram | [build-and-gates.md](../../../knowledgeGraph/docs/ci-cd/build-and-gates.md) | ADR-2003 |
| KG-06 | [Consumers — VisionClaw ingest, agentbox Loom grounding, vowl-wasm, OntoCast staging](knowledgegraph/06-consumers-and-integration.md) | 4 | flowchart, sequenceDiagram | [ecosystem.md](../../../knowledgeGraph/docs/ecosystem.md) |  |
| KG-07 | [Invariants register — the 11 baseline invariants and the open items they qualify](knowledgegraph/07-invariants-register.md) | 3 | flowchart, stateDiagram-v2 | [BASELINE-narrativegoldmine.md](../../../knowledgeGraph/docs/BASELINE-narrativegoldmine.md) | ADR-2001, ADR-2002, ADR-2003, ADR-2004 |

### visiongraph

| ID | Topic | Diagrams | Kinds | Governing | ADRs |
|----|-------|----------|-------|-----------|------|
| VG-01 | [Vault structure and the publication contract — frontmatter gate, inclusion divergence](visiongraph/01-vault-structure-and-contract.md) | 4 | flowchart | [PUBLICATION-contract.md](../../../visionGraph/docs/PUBLICATION-contract.md) | ADR-VG-001, ADR-VG-002 |
| VG-02 | [Authoring → pipeline → published corpus — the 9-stage build and its swarm-authoring gates](visiongraph/02-authoring-to-pipeline-flow.md) | 6 | flowchart, sequenceDiagram, classDiagram | [PUBLICATION-contract.md](../../../visionGraph/docs/PUBLICATION-contract.md) | ADR-VG-002 |
| VG-03 | [publish.yml — the actual publisher, quality gates, and the React/vowl-wasm explorer build](visiongraph/03-publish-pipeline-and-explorer-build.md) | 4 | flowchart, sequenceDiagram | [PUBLICATION-contract.md](../../../visionGraph/docs/PUBLICATION-contract.md) |  |
| VG-04 | [Downstream contracts — VisionClaw's pull model, and knowledgeGraph as deploy target](visiongraph/04-downstream-contracts.md) | 3 | sequenceDiagram, flowchart | [PUBLICATION-contract.md](../../../visionGraph/docs/PUBLICATION-contract.md) | ADR-VG-002 |
| VG-05 | [Invariants register — vault contract facts vs the proposed, inactive closeout ADRs](visiongraph/05-invariants-register.md) | 3 | flowchart | [PUBLICATION-contract.md](../../../visionGraph/docs/PUBLICATION-contract.md) | ADR-VG-001, ADR-VG-002 |

### estate

| ID | Topic | Diagrams | Kinds | Governing | ADRs |
|----|-------|----------|-------|-----------|------|
| ES-01 | [Estate topology — substrates, network fabric, service ports, compose networks](estate/01-estate-topology.md) | 6 | flowchart | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2023, ADR-2013, ADR-2027, ADR-2025, ADR-2009, ADR-2012, ADR-2062 |
| ES-02 | [Agent-event path — agentbox action to rendered beam, plus legacy paths](estate/02-agent-events-agentbox-to-visionclaw.md) | 11 | sequenceDiagram, classDiagram | [PROTOCOL-registry.md](../../../project/docs/PROTOCOL-registry.md), [GPU-wire-abi.md](../../../project/docs/GPU-wire-abi.md), [PROTOCOL-registry.md](../../../project/agentbox/docs/PROTOCOL-registry.md) | ADR-2020, ADR-2015, ADR-2083, ADR-2084, ADR-2085, ADR-2088, ADR-2089, ADR-2090, ADR-2091 |
| ES-03 | [Cross-repo federation contract (agentbox <-> VisionClaw)](estate/03-cross-repo-federation-contract.md) | 10 | classDiagram, flowchart, sequenceDiagram | [IDENTIFIER-taxonomy.md](../../../project/docs/IDENTIFIER-taxonomy.md), [PROTOCOL-registry.md](../../../project/docs/PROTOCOL-registry.md), [PROTOCOL-registry.md](../../../project/agentbox/docs/PROTOCOL-registry.md), [DATA-authority-erasure.md](../../../project/docs/DATA-authority-erasure.md), [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2023, ADR-2025, ADR-2061 |
| ES-04 | [did:nostr identity mesh — signing, verification, custody](estate/04-identity-mesh-did-nostr.md) | 6 | flowchart, classDiagram, sequenceDiagram | [IDENTITY-authority-chain.md](../../../project/docs/IDENTITY-authority-chain.md), [INGRESS-identity.md](../../../project/agentbox/docs/INGRESS-identity.md), [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md), [SECURITY-profiles.md](../../../project/docs/SECURITY-profiles.md) | ADR-2002, ADR-2009, ADR-2010, ADR-2011, ADR-2013, ADR-2026 |
| ES-05 | [Human-approval governance loop across the estate](estate/05-governance-loop-across-estate.md) | 10 | flowchart, classDiagram, sequenceDiagram, stateDiagram-v2 | [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md), [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md) | ADR-2006 |
| ES-06 | [Ontology Loom and the email privacy path](estate/06-loom-and-email-privacy-path.md) | 10 | flowchart, sequenceDiagram, stateDiagram-v2 | [GOVERNANCE-capabilities.md](../../../project/agentbox/docs/GOVERNANCE-capabilities.md), [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2023, ADR-2079, ADR-2080 |
| ES-07 | [RuVector memory and embedding estate](estate/07-memory-and-embedding-estate.md) | 9 | flowchart, sequenceDiagram, stateDiagram-v2 | [LEARNING-memory.md](../../../project/agentbox/docs/LEARNING-memory.md), [DATA-authority-erasure.md](../../../project/docs/DATA-authority-erasure.md) | ADR-2014, ADR-2015, ADR-2016 |
| ES-08 | [Solid-pod estate — four deployments, write identity, access control](estate/08-solid-pod-estate.md) | 11 | flowchart, sequenceDiagram, classDiagram, stateDiagram-v2 | [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md), [DATA-authority-erasure.md](../../../project/docs/DATA-authority-erasure.md), [INGRESS-identity.md](../../../project/agentbox/docs/INGRESS-identity.md) | ADR-2015, ADR-2016, ADR-2017, ADR-2064, ADR-2068, ADR-2106 |
| ES-09 | [Build, deploy and CI estate — source to running container, every gate](estate/09-build-deploy-and-ci-estate.md) | 21 | flowchart, sequenceDiagram, stateDiagram-v2 | [BASELINE-architecture.md](../../../project/docs/BASELINE-architecture.md), [BASELINE-container.md](../../../project/agentbox/docs/BASELINE-container.md) | ADR-2008, ADR-2037, ADR-2013, ADR-2028 |
| ES-10 | [Deployment and security profiles across the estate](estate/10-deployment-and-security-profiles.md) | 10 | flowchart, sequenceDiagram, stateDiagram-v2 | [SECURITY-profiles.md](../../../project/docs/SECURITY-profiles.md), [SECURITY-profiles.md](../../../project/agentbox/docs/SECURITY-profiles.md), [INGRESS-identity.md](../../../project/agentbox/docs/INGRESS-identity.md), [IDENTITY-authority-chain.md](../../../project/docs/IDENTITY-authority-chain.md) | ADR-2003, ADR-2010, ADR-2012, ADR-2013, ADR-2026, ADR-2027, ADR-2037, ADR-2038, ADR-2039, ADR-2062, ADR-2086, ADR-2087 |
<!-- END GENERATED DIAGRAM INDEX -->
