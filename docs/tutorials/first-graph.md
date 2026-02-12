---
title: "Tutorial: Creating Your First Knowledge Graph"
description: Launch VisionFlow, create nodes and relationships through the UI, and explore your data in an interactive 3D space backed by Neo4j.
category: tutorial
tags:
  - getting-started
  - tutorial
  - neo4j
  - knowledge-graph
  - 3d-visualization
updated-date: 2026-02-12
difficulty-level: beginner
---

# Tutorial: Creating Your First Knowledge Graph

This tutorial walks you through starting the VisionFlow server, creating nodes and
relationships with the browser-based UI, and viewing the result as a live 3D
force-directed graph. All data is persisted in Neo4j, so everything you build
here survives restarts.

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Docker + Compose | v20.10+ | v24+ |
| RAM | 8 GB | 16 GB |
| Browser | Chrome 90+, Firefox 88+, Safari 14+ | Chrome latest with WebGL 2 |
| GPU (optional) | -- | NVIDIA with CUDA 11.8+ |

Make sure you have completed the [Installation Guide](installation.md) and that
`docker compose ps` shows all services in the **Up** state.

## Step 1 -- Start the Server

From the repository root, bring the stack up in development mode:

```bash
cd VisionFlow
docker compose --profile dev up -d
```

Wait roughly 30 seconds for Neo4j and the Rust backend to initialise. Confirm
readiness by hitting the health endpoint:

```bash
curl http://localhost:3030/api/health
# Expected: {"status":"ok"}
```

## Step 2 -- Open the Browser UI

Navigate to **http://localhost:3030** in your browser. You will see:

- A dark 3D viewport in the centre of the screen.
- A left sidebar with graph controls and the multi-agent panel.
- A right sidebar with visual and physics settings.
- A green **Connected** indicator in the bottom status bar.

If the viewport stays blank for more than 60 seconds, check the container logs
with `docker compose logs -f visionflow-container`.

## Step 3 -- Create Your First Nodes

1. Click the **Add Node** button in the left control panel.
2. Enter a label -- for example, `Machine Learning`.
3. Click **Create**. A glowing sphere appears in the viewport.
4. Repeat to add a few more nodes: `Neural Networks`, `Training Data`,
   `Backpropagation`, `Loss Function`.

Each node is written to Neo4j as a labelled vertex. You can verify this with a
Cypher query in the Neo4j browser at **http://localhost:7474**:

```cypher
MATCH (n) RETURN n LIMIT 25
```

## Step 4 -- Add Relationships

1. Click the first node (`Machine Learning`) so it highlights.
2. Hold **Shift** and click a second node (`Neural Networks`).
3. Click **Add Edge** in the toolbar that appears.
4. Choose a relationship type -- for example, `USES` -- and confirm.
5. A line connects the two nodes and the physics engine adjusts the layout.

Add a few more edges to build a small network:

| Source | Relationship | Target |
|--------|-------------|--------|
| Neural Networks | REQUIRES | Training Data |
| Neural Networks | APPLIES | Backpropagation |
| Backpropagation | MINIMISES | Loss Function |
| Machine Learning | EVALUATES | Loss Function |

## Step 5 -- Explore in 3D

With your graph created, experiment with the interactive controls:

- **Left-click + drag** -- Rotate the camera around the graph.
- **Scroll wheel** -- Zoom in and out.
- **Double-click a node** -- Centre the camera on that node.
- **Spacebar** -- Pause or resume the physics simulation.
- **R** -- Reset the camera to the default viewpoint.

Open the **Physics** panel on the right to tune spring strength, repulsion
force, and damping. Higher repulsion spreads the graph out; stronger springs
pull connected nodes closer together.

## Step 6 -- Verify Persistence in Neo4j

Because VisionFlow uses Neo4j as its primary backing store, your graph survives
container restarts. Stop the stack and bring it back up:

```bash
docker compose down && docker compose --profile dev up -d
```

Refresh the browser. Your nodes and relationships reappear exactly as you left
them.

## What Happens Under the Hood

1. The React + Three.js frontend sends node and edge mutations over the binary
   WebSocket protocol (34-byte frames).
2. The Rust backend (Actix Web, hexagonal architecture) validates the mutation
   and writes it to Neo4j via the Bolt driver.
3. The GPU-accelerated physics engine recalculates forces and streams updated
   positions back to all connected clients at 60 FPS.

## Next Steps

- [Neo4j Basics](neo4j-basics.md) -- Learn Cypher queries and dual-persistence
  patterns used by VisionFlow.
- [Building a Digital Twin](digital-twin.md) -- Connect real-time sensor data
  to an entity graph.
- [Architecture Overview](../explanation/architecture/README.md) -- Understand
  the full system design.

---

**Document Version**: 1.0
**Last Updated**: 2026-02-12
**Maintained By**: VisionFlow Documentation Team
