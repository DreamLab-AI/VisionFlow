---
layout: default
title: GPU Architecture
parent: Architecture
grand_parent: Explanations
nav_order: 4
has_children: true
permalink: /explanations/architecture/gpu/
---

# GPU Architecture

GPU-accelerated computation and WASM-based physics simulation.

## Overview

VisionFlow uses WebGPU and WASM SIMD for high-performance graph layout and physics simulation:

- Force-directed graph layouts
- Semantic force calculations
- Real-time physics simulation
- GPU-accelerated rendering

## Contents

| Document | Description |
|----------|-------------|
| [README](./README.md) | GPU architecture overview |
| [Communication Flow](./communication-flow.md) | GPU-CPU communication patterns |
| [Optimizations](./optimizations.md) | Performance optimization techniques |
