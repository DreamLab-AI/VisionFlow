---
title: Semantic Forces User Guide
description: Semantic forces enable GPU-accelerated physics where forces convey semantic meaning about relationships, hierarchies, and node types.
category: guide
tags:
  - tutorial
  - api
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Semantic Forces User Guide

## Overview

Semantic forces enable GPU-accelerated physics where forces convey semantic meaning about relationships, hierarchies, and node types.

## Features

### 1. DAG Layout

Hierarchical layout for directed acyclic graphs:

```typescript
// Enable DAG layout
const dagConfig = {
  enabled: true,
  verticalSpacing: 100,
  horizontalSpacing: 50,
  levelAttraction: 0.5,
  siblingRepulsion: 0.3
};
```

Automatically arranges parent-child hierarchies vertically or radially.

### 2. Type Clustering

Groups nodes by semantic type:

```typescript
const typeClusterConfig = {
  enabled: true,
  clusterAttraction: 0.4,
  clusterRadius: 150,
  interClusterRepulsion: 0.2
};
```

Person nodes cluster together, organizations cluster separately.

### 3. Collision Detection

Prevents node overlap with semantic awareness:

```typescript
const collisionConfig = {
  enabled: true,
  minDistance: 5,
  collisionStrength: 1.0,
  nodeRadius: 10
};
```

### 4. Attribute-Weighted Springs

Edge forces based on semantic attributes:

```typescript
const springConfig = {
  enabled: true,
  baseSpringK: 0.01,
  weightMultiplier: 0.5,
  restLengthMin: 30,
  restLengthMax: 200
};
```

Stronger relationships = shorter spring rest lengths.

## API Endpoints

### Get Semantic Configuration

```bash
GET /api/semantic-forces/config
```

### Update Configuration

```bash
POST /api/semantic-forces/config
Content-Type: application/json

{
  "dag": { "enabled": true, ... },
  "typeCluster": { "enabled": true, ... }
}
```

## Best Practices

1. **Enable DAG for hierarchies**: Use when you have clear parent-child relationships
2. **Type clustering for ontologies**: Groups OWL classes naturally
3. **Combine forces**: DAG + Type clustering works well together
4. **Tune strengths**: Start with defaults, adjust based on graph density

---

---

## Related Documentation

- [Natural Language Queries Tutorial](natural-language-queries.md)
- [Intelligent Pathfinding Guide](intelligent-pathfinding.md)
- [Contributing Guidelines](../developer/06-contributing.md)
- [Goalie Integration - Goal-Oriented AI Research](../infrastructure/goalie-integration.md)
- [VisionFlow Guides](../index.md)

## Examples

See `/api/schema/examples` for common query patterns.
