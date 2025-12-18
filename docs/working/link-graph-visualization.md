---
title: Link Graph Visualization
description: Visual representation of documentation link structure and connectivity
category: reference
tags:
  - visualization
  - graph
  - links
  - analysis
  - documentation
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Documentation Link Graph Visualization

## Network Overview

```
Total Nodes (Files):     281
Total Edges (Links):   1,469
Graph Density:         1.86%
Average Links/File:     5.23
```

## Hub Analysis

### Central Hub Files (Highest Inbound Links)

```
guides/index.md                    ████████████████ 15 links
guides/readme.md                   ████████████████ 15 links
README.md                          ████████████ 12 links
OVERVIEW.md                        ██████████ 10 links
INDEX.md                           █████████ 9 links
```

### Source Hub Files (Highest Outbound Links)

```
ARCHITECTURE_COMPLETE.md           ████████████████████ 47 links
archive/INDEX-QUICK-START-old.md   ███████████████████ 45 links
guides/navigation-guide.md         ██████████████ 32 links
guides/developer/readme.md         ████████████ 28 links
QUICK_NAVIGATION.md                ███████████ 25 links
```

## Link Type Distribution

```
Internal Links (51.3%)     ██████████████████████████
Anchor Links (22.0%)       ███████████
External URLs (8.1%)       ████
Broken Links (17.2%)       █████████  ⚠
Wiki Links (1.5%)          █
```

## Health Metrics by Category

### Architecture Documentation
```
Total Files: 45
Orphaned:    12 (26.7%)   ⚠
Isolated:    18 (40.0%)   ⚠
Broken:      38 links     ✗
Health:      72/100       ~
```

### Guides
```
Total Files: 68
Orphaned:    22 (32.4%)   ⚠
Isolated:    35 (51.5%)   ⚠
Broken:      67 links     ✗
Health:      65/100       ⚠
```

### Reference Documentation
```
Total Files: 34
Orphaned:    8 (23.5%)    ~
Isolated:    12 (35.3%)   ⚠
Broken:      18 links     ⚠
Health:      78/100       ~
```

### Tutorials
```
Total Files: 12
Orphaned:    3 (25.0%)    ~
Isolated:    4 (33.3%)    ~
Broken:      5 links      ~
Health:      82/100       ✓
```

### Archived Content
```
Total Files: 89
Orphaned:    35 (39.3%)   ✗
Isolated:    67 (75.3%)   ✗
Broken:      98 links     ✗
Health:      45/100       ✗
```

## Connectivity Patterns

### Strongly Connected Components

```
Main Documentation Core
├─ README.md
├─ OVERVIEW.md
├─ INDEX.md
└─ guides/
    ├─ index.md (15 inbound)
    ├─ readme.md (15 inbound)
    └─ navigation-guide.md (32 outbound)
```

### Isolated Clusters

```
Feature Documentation (Weak Links)
├─ guides/features/ (23 files)
│   ├─ Connected: 8
│   └─ Isolated: 15  ⚠
└─ Link density: 1.2%

Architecture Details (Moderate Links)
├─ explanations/architecture/ (45 files)
│   ├─ Connected: 27
│   └─ Isolated: 18  ~
└─ Link density: 2.1%

API Reference (Good Links)
├─ reference/api/ (18 files)
│   ├─ Connected: 12
│   └─ Isolated: 6   ✓
└─ Link density: 3.4%
```

## Link Flow Diagram

```
Entry Points          Core Docs              Deep Content
─────────────         ─────────             ─────────────

README.md ──────────> guides/index.md ───> Feature Guides
    │                      │                     │
    │                      ├───────────────> API Reference
    │                      │                     │
OVERVIEW.md ─────────>     │              Tutorials
    │                      │                     │
    │                      └───────────────> Architecture
    │                                            │
INDEX.md ────────────> navigation-guide ──> Explanations
    │                      │
    │                      └───────────────> Operations
    │
QUICK_NAV.md ──────────────────────────────> Advanced Topics


Legend:
───> Strong links (3+ connections)
──>  Moderate links (1-2 connections)
⚠   Weak/broken links
```

## Orphan Island (86 files)

Files with **no inbound links** from documentation:

```
High-Value Orphans (Should be linked):
├─ explanations/architecture/
│   ├─ adapter-patterns.md
│   ├─ analytics-visualization.md
│   ├─ cqrs-directive-template.md
│   ├─ github-sync-service-design.md
│   ├─ integration-patterns.md
│   ├─ ontology-analysis.md
│   ├─ pipeline-integration.md
│   └─ semantic-physics-system.md
│
├─ guides/features/
│   ├─ auth-user-settings.md
│   ├─ filtering-nodes.md
│   ├─ intelligent-pathfinding.md
│   ├─ local-file-sync-strategy.md
│   ├─ natural-language-queries.md
│   ├─ nostr-auth.md
│   ├─ ontology-sync-enhancement.md
│   └─ settings-authentication.md
│
└─ guides/infrastructure/
    ├─ goalie-integration.md
    ├─ port-configuration.md
    └─ tools.md

Low-Priority Orphans (Archive/Deprecated):
└─ archive/ (35 orphaned files)
```

## Isolated Peninsula (150 files)

Files with **no outbound links** to other docs:

```
Should Add Links:
├─ Implementation guides → Link to API reference
├─ Feature docs → Link to configuration
├─ Tutorials → Link to next steps
└─ Architecture → Link to implementation
```

## External Link Map

```
github.com (45 links)
├─ Repository: /ruvnet/...
├─ Issues & PRs
└─ Example code

neo4j.com (28 links)
├─ Database docs
├─ Cypher reference
└─ Driver API

threejs.org (15 links)
├─ Core API
├─ Examples
└─ Documentation

vircadia.com (12 links)
├─ Platform docs
└─ Integration guides

Other domains (19 links)
└─ Various references
```

## Broken Link Hotspots

```
ARCHITECTURE_COMPLETE.md          ████████████ 13 broken
archive/INDEX-QUICK-START-old.md  ███████████████████████████ 58 broken
OVERVIEW.md                       ███ 3 broken
README.md                         ██ 2 broken
guides/navigation-guide.md        █████ 8 broken
```

## Link Quality Heatmap

```
                  Healthy    Needs Work    Critical
                  (90-100)   (70-89)       (<70)
─────────────────────────────────────────────────
Tutorials           ████         ░░           ░
API Reference       ███          █░           ░
Architecture        ██           ██           ░
Guides              █░           ██           ░░
Features            ░            █            ██
Archive             ░            ░            ████
```

## Network Metrics

### Centrality Analysis

**Betweenness Centrality** (bridge documents):
1. `guides/index.md` - Critical hub
2. `guides/navigation-guide.md` - Navigation bridge
3. `README.md` - Entry point
4. `OVERVIEW.md` - Overview hub
5. `guides/developer/readme.md` - Developer entry

**PageRank** (importance):
1. `guides/index.md` - High importance
2. `README.md` - High importance
3. `OVERVIEW.md` - Medium-high
4. Core architecture docs - Medium
5. Tutorial entry points - Medium

### Link Reciprocity

```
Bidirectional Links: 45 pairs
One-way Links:      663 edges
Reciprocity Rate:   12.7%  (Low - suggests navigation issues)
```

**Top Bidirectional Pairs:**
```
guides/index.md ↔ guides/readme.md
architecture/ ↔ implementation guides
features/ ↔ configuration docs
tutorials/ ↔ API reference
```

### Clustering Coefficient

```
Global Clustering:  0.089  (Low - documentation is fragmented)
Local Clustering:
  - High clusters: Main guides (0.34)
  - Medium: Architecture (0.18)
  - Low: Features (0.06)
```

## Recommendations

### Improve Hub Connectivity
```
Add links to orphaned files from:
├─ guides/index.md (+20 links)
├─ explanations/architecture/README.md (+12 links)
└─ reference/api/README.md (+8 links)
```

### Build Topic Bridges
```
Create navigation pages for:
├─ Feature catalog (features/index.md)
├─ Architecture overview (architecture/index.md)
└─ Operations guide (operations/index.md)
```

### Increase Reciprocity
```
Add "See Also" sections in:
├─ Isolated files (150 files)
├─ Feature documentation
└─ Implementation guides
```

### Strengthen Weak Clusters
```
guides/features/
├─ Current density: 1.2%
├─ Target density: 3.5%
└─ Action: Add 30+ cross-links

archive/
├─ Current: Mostly broken
├─ Action: Update or deprecate
```

## Graph Statistics Summary

```
Metric                    Current    Target    Gap
────────────────────────────────────────────────────
Graph Density             1.86%      3.50%    -47%
Avg Links/File            5.23       8.00     -35%
Broken Link Rate          17.2%      2.0%     760%
Orphan Rate               30.6%      7.0%     337%
Isolation Rate            53.4%      18.0%    197%
Reciprocity               12.7%      25.0%    -49%
Clustering Coeff          0.089      0.200    -55%
────────────────────────────────────────────────────
Overall Health Score      65/100     90/100   -28%
```

## Visualization Legend

```
████ = Strong/Healthy
███░ = Moderate
██░░ = Weak
█░░░ = Critical
░░░░ = None/Broken

✓ = Good (>80%)
~ = Acceptable (60-80%)
⚠ = Needs Work (40-60%)
✗ = Critical (<40%)
```

---

**Generated**: 2025-12-18T21:13:09Z
**Source**: complete-link-graph.json
**Tool**: Link analysis visualization generator
