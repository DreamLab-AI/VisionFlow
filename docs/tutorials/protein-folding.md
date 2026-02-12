---
title: "Tutorial: Visualizing Protein Folding Networks"
description: Import PDB protein structure data into VisionFlow's knowledge graph and use GPU-accelerated physics to generate molecular dynamics-style layouts.
category: tutorial
tags:
  - protein-folding
  - molecular-dynamics
  - pdb
  - gpu-physics
  - tutorial
  - scientific-computing
updated-date: 2026-02-12
difficulty-level: intermediate
---

# Tutorial: Visualizing Protein Folding Networks

This tutorial shows you how to import Protein Data Bank (PDB) files into
VisionFlow, represent amino acid residues and bonds as a knowledge graph, and
use the GPU-accelerated physics engine to produce an interactive 3D layout that
reflects molecular structure.

## Prerequisites

- VisionFlow stack running (`docker compose --profile dev up -d`).
- Familiarity with the [First Graph tutorial](first-graph.md).
- A PDB file. This tutorial uses `1CRN` (crambin, 46 residues) as a
  compact example. Download it from the RCSB:

```bash
curl -o /tmp/1crn.pdb https://files.rcsb.org/download/1CRN.pdb
```

## What You Will Build

1. A residue-level graph where each amino acid is a node and covalent bonds
   are edges, stored in Neo4j.
2. A GPU physics layout that approximates the protein's spatial fold using
   VisionFlow's semantic force engine.
3. An interactive 3D viewer where you can rotate, zoom, and query structural
   properties.

## Step 1 -- Parse the PDB File into Graph Format

VisionFlow's import pipeline converts PDB ATOM records into graph primitives.
Run the import command:

```bash
curl -s -X POST http://localhost:3030/api/import/pdb \
  -H "Content-Type: application/octet-stream" \
  --data-binary @/tmp/1crn.pdb
```

The importer performs the following transformations:

| PDB Record | Graph Element |
|-----------|--------------|
| ATOM (CA) | Node per residue (alpha-carbon position) |
| CONECT | Edge with type `COVALENT_BOND` |
| HELIX / SHEET | Node metadata (`secondaryStructure: helix` or `sheet`) |
| SSBOND | Edge with type `DISULFIDE_BRIDGE` |

After import, verify the graph in Neo4j:

```cypher
MATCH (r:Residue) RETURN count(r)
// Expected: 46
```

## Step 2 -- Configure Molecular Physics

VisionFlow's semantic constraint engine maps molecular interactions to physics
forces. Open the **Physics** panel and apply the **Molecular Dynamics** preset,
or configure manually:

| Parameter | Value | Molecular Analogue |
|-----------|-------|-------------------|
| Spring strength | 0.8 | Covalent bond stiffness |
| Repulsion force | 200 | Van der Waals exclusion radius |
| Type clustering | 0.4 | Hydrophobic interaction strength |
| Damping | 0.7 | Solvent viscosity approximation |
| Central force | 0.05 | Prevents the chain from drifting off-screen |

With these settings the GPU physics engine arranges residues so that:

- Covalently bonded neighbours sit close together (backbone continuity).
- Hydrophobic residues (Ala, Val, Leu, Ile, Phe) cluster toward the core.
- Charged residues (Arg, Lys, Glu, Asp) migrate toward the surface.

The layout converges in 2--5 seconds on an RTX 4090. Press **Space** to pause
the simulation once you are satisfied with the fold.

## Step 3 -- Colour by Secondary Structure

Use the **Visual Settings** panel to map the `secondaryStructure` metadata to
node colour:

- **Helix** residues -- red.
- **Sheet** residues -- blue.
- **Coil** residues -- grey.

This immediately reveals the distribution of secondary structure elements
across the 3D fold, matching the known crambin topology (two helices and a
small beta-sheet).

## Step 4 -- Query Structural Neighbourhoods

Click any residue node to select it. VisionFlow highlights all nodes within a
configurable graph distance (default: 2 hops). This is useful for identifying
residues in spatial contact that are distant in sequence -- a hallmark of
tertiary folding.

For programmatic queries, use Cypher:

```cypher
MATCH (a:Residue {name: "CYS_3"})-[:DISULFIDE_BRIDGE]-(b:Residue)
RETURN a.name, b.name
```

## Step 5 -- Scale to Larger Proteins

For proteins larger than a few hundred residues, enable GPU acceleration
explicitly:

```bash
# In .env
GPU_ACCELERATION=true
CUDA_VISIBLE_DEVICES=0
```

Performance reference:

| Protein Size | Nodes | Edges | FPS (RTX 4090) |
|-------------|-------|-------|----------------|
| Crambin (1CRN) | 46 | 90 | 60 |
| Lysozyme (1LYZ) | 129 | 260 | 60 |
| Haemoglobin (1HBB) | 574 | 1,200 | 60 |
| Ribosome subunit | 4,500 | 12,000 | 55 |
| Full ribosome | 15,000+ | 40,000+ | 40 (multi-GPU recommended) |

## Step 6 -- Export and Share

VisionFlow supports several export paths:

- **Screenshot** -- Press `P` to capture the current viewport as a PNG.
- **Graph JSON** -- `GET /api/export/json` returns the full node/edge list
  with coordinates for use in external tools.
- **Collaborative session** -- Share the URL with colleagues; all connected
  browsers see the same live layout via the binary WebSocket sync.

## How It Works Under the Hood

1. The PDB importer (Rust) parses ATOM/CONECT records and writes residue nodes
   and bond edges to Neo4j via the Bolt driver.
2. The semantic constraint generator reads node types (amino acid identity) and
   generates per-type force parameters from a lookup table aligned with the
   Kyte-Doolittle hydrophobicity scale.
3. The CUDA physics kernel evaluates all pairwise forces in parallel, producing
   updated XYZ positions each frame.
4. Positions are streamed to the React + Three.js frontend over the 34-byte
   binary WebSocket protocol at up to 60 FPS.

## Next Steps

- [GPU Acceleration Concepts](../explanation/concepts/gpu-acceleration.md) --
  Deep dive into the CUDA kernel architecture.
- [Semantic Forces System](../explanation/architecture/semantic-forces-system.md)
  -- How ontological relationships become physics forces.
- [Industry Applications -- Scientific Computing](../use-cases/industry-applications.md#2-scientific-computing)
  -- Broader context on research use cases.

---

**Document Version**: 1.0
**Last Updated**: 2026-02-12
**Maintained By**: VisionFlow Documentation Team
