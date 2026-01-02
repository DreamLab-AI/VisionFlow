---
layout: default
title: Research
nav_order: 91
nav_exclude: true
has_children: true
description: Technical research and state-of-the-art analysis documents
---

# Research

Technical research and state-of-the-art analysis documents for VisionFlow development.

These documents are for internal development use and are excluded from public navigation.

## Protocol Research

| Document | Description |
|----------|-------------|
| [QUIC HTTP/3 Analysis](QUIC_HTTP3_ANALYSIS.md) | Research on QUIC and HTTP/3 for real-time graph visualization |

## Visualization Research

| Document | Description |
|----------|-------------|
| [Graph Visualization SOTA Analysis](graph-visualization-sota-analysis.md) | State-of-the-art high-performance graph visualization research |
| [Three.js vs Babylon.js Comparison](threejs-vs-babylonjs-graph-visualization.md) | Technical comparison for graph visualization requirements |

## Key Findings

### QUIC/HTTP3

- **Recommendation**: Implement QUIC/WebTransport using `quinn` + `web-transport-quinn`
- Benefits: 0-RTT connection, no head-of-line blocking, built-in encryption
- 50-98% latency improvements in real-world conditions

### Graph Visualization

- GPU-accelerated force layout: 40-123x speedup via compute shaders
- WebGL rendering optimisations: point sprites, instanced rendering, texture atlases
- Systems achieving 1M+ nodes all implement GPU force simulation

### Three.js vs Babylon.js

- **Three.js**: Smaller bundle (168KB), mature force-directed graph libraries
- **Babylon.js**: Better WebXR support, built-in optimisations
- **Recommendation for 1000+ nodes**: Three.js for graph visualisation

## Usage

These research documents inform technology choices and architectural direction. They should be updated as new research becomes available.
