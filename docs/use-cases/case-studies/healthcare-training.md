---
title: "Case Study: Medical Ontology Visualization for Surgical Training"
description: Using VisionFlow to visualize medical ontologies and anatomical hierarchies for surgical training simulations with OWL reasoning and XR immersion.
category: case-study
tags:
  - healthcare
  - ontology
  - owl
  - surgical-training
  - xr
  - visualization
updated-date: 2026-02-12
difficulty-level: advanced
---

# Case Study: Medical Ontology Visualization for Surgical Training

## Overview

Surgical training programmes require residents to develop a deep, spatial
understanding of anatomical relationships before they enter the operating
theatre. This case study explores how a university hospital network deployed
VisionFlow to transform OWL-encoded medical ontologies into interactive 3D
knowledge graphs, enabling residents to navigate anatomical hierarchies in
extended reality (XR) and receive real-time feedback from ontology-driven
reasoning.

## Problem Statement

The hospital's surgical education department faced three interconnected
challenges:

1. **Flat learning materials.** Textbooks and 2D atlases cannot convey the
   three-dimensional spatial relationships between organs, vessels, and nerves
   that surgeons must internalise.
2. **Disconnected knowledge.** Anatomical facts, procedural steps, and
   contraindication rules lived in separate systems with no formal linkage.
3. **High simulator cost.** Physical manikins and proprietary VR trainers cost
   upwards of $500,000 per unit, limiting access to a handful of time slots per
   week.

## Solution Architecture

### OWL Ontology as the Knowledge Backbone

The department encoded its curriculum using OWL 2 EL ontologies aligned to the
Foundational Model of Anatomy (FMA). Key axiom patterns include:

- `Heart SubClassOf hasPart some LeftVentricle`
- `CoronaryArtery SubClassOf supplies some Myocardium`
- `LeftVentricle DisjointWith RightVentricle`

VisionFlow's integrated Whelk reasoner (10--100x faster than Java-based
reasoners) classifies the ontology at import time, automatically inferring
transitive part-of chains and detecting logical contradictions in newly added
axioms.

### Anatomical Hierarchy as a 3D Graph

Each OWL class becomes a node in VisionFlow's Neo4j-backed knowledge graph.
Object properties become typed edges. The GPU-accelerated semantic physics
engine translates ontological relationships into spatial forces:

- **SubClassOf** links pull child nodes beneath their parents, producing a
  natural top-down hierarchy.
- **hasPart** edges act as strong springs, clustering organs with their
  structural components.
- **supplies / drainedBy** edges create lateral connections between vascular
  and organ subsystems.

The result is a 3D anatomical map where spatial proximity reflects semantic
relatedness -- a property that 2D tree views cannot provide.

### XR Immersion for Residents

Using Meta Quest 3 headsets connected through Vircadia, residents enter the
graph in room-scale VR. They can:

- Walk through the thoracic cavity cluster, seeing heart, lungs, and great
  vessels arranged according to their ontological relationships.
- Pinch a node to expand its subclass hierarchy on demand (level-of-detail
  controlled by the client-side hierarchical LOD system).
- Trigger a reasoning query by voice -- for example, asking "What does the left
  anterior descending artery supply?" -- and see the answer highlighted as a
  glowing subgraph.

Mentors join the same Vircadia domain from a desktop browser, observing the
resident's gaze direction and annotations in real time.

## Key Results

| Metric | Traditional Approach | VisionFlow Approach |
|--------|---------------------|---------------------|
| Equipment cost per station | $500,000 (manikin) | $15,000 (GPU workstation + Quest 3) |
| Concurrent trainees | 1 per manikin | 10+ per VisionFlow instance |
| Ontology update cycle | Months (vendor release) | Minutes (edit OWL, re-import) |
| Reasoning feedback | None (static content) | Real-time inference on interaction |
| Knowledge retention (30-day) | 48% (2D materials) | 71% (3D + XR, internal study) |

## Compliance and Privacy

All patient-derived data was excluded from the ontology. The system runs
entirely on-premises behind the hospital firewall, satisfying HIPAA
administrative safeguard requirements (45 CFR 164.308). Audit trails are
maintained through Neo4j transaction logs and Git-versioned ontology files.

## Lessons Learned

- Aligning the custom curriculum ontology to FMA took the most calendar time
  but paid dividends in reusability and reasoning quality.
- Residents initially found the 3D graph overwhelming; enabling hierarchical
  LOD collapse (showing only top-level systems by default) resolved the issue.
- Voice-driven reasoning queries via Whisper STT proved more natural in VR than
  controller-based text input.

## Related Documentation

- [Ontology Reasoning Pipeline](../../explanation/architecture/ontology-reasoning-pipeline.md)
- [XR Immersive System](../../explanation/architecture/xr-immersive-system.md)
- [Client-Side Hierarchical LOD](../../explanation/architecture/ontology/client-side-hierarchical-lod.md)
- [Industry Applications -- Healthcare](../industry-applications.md#4-healthcare--biotech)

---

**Document Version**: 1.0
**Last Updated**: 2026-02-12
**Maintained By**: VisionFlow Documentation Team
