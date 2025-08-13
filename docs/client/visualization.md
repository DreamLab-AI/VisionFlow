# Client-Side Visualization Concepts

This document outlines the higher-level concepts behind the visualisation of the knowledge graph in the LogseqXR client. It explains *what* is being visualized and *how* different visual elements represent data, distinguishing itself from `rendering.md` which details the technical "how-to" of drawing these elements.

## Core Visualization Metaphor

The primary goal is to transform an abstract knowledge graph, typically represented by Markdown files and their links, into a tangible and interactive 3D spatial environment.

-   **Nodes as Entities:** Each primary piece of information (e.g., a Logseq page, a specific block, or a concept) is represented as a **node** in the 3D space.
    -   Typically, nodes correspond to individual Markdown files.
    -   The visual appearance of a node (size, colour, shape) can be mapped to its attributes (e.g., file size, type, metadata tags).
-   **Edges as Relationships:** Links and connections between these entities (e.g., hyperlinks, block references, tags) are represented as **edges** connecting the corresponding nodes.
    -   The visual properties of edges (thickness, colour, style) can signify the type or strength of the relationship.

## Mapping Data to Visual Elements

**Multi-Graph Architecture**: The visualisation now supports multiple independent graphs (logseq, visionflow) with separate visual themes through `settings.visualisation.graphs.*`.

The effectiveness of the visualisation hinges on how data attributes are mapped to visual properties.

### Node Visuals

-   **Size:**
    -   **Multi-graph**: `settings.visualisation.graphs.logseq.nodes.nodeSize` vs `settings.visualisation.graphs.visionflow.nodes.nodeSize`
    -   **Legacy**: `visualisation.nodes.nodeSize` (automatically migrated)
    -   Modulated by data attributes (file size, connection count)
-   **Color:**
    -   **Multi-graph**: `settings.visualisation.graphs.[graphType].nodes.baseColor`
    -   **Default themes**: Logseq (blue `#4B5EFF`), VisionFlow (green `#10B981`)
    -   Dynamic coloring based on metadata (file type, tags)
-   **Shape / Form:**
    -   **Primary**: Spheres (performance optimized)
    -   **Metadata-driven**: `enableMetadataShape` allows geometry variation
    -   **Quality levels**: Low/medium/high affecting geometry detail
-   **Holograms:**
    -   Enabled by `visualisation.nodes.enableHologram` and configured via `visualisation.hologram` (which is `HologramSettings`) in [`settings.ts`](../../client/src/features/settings/config/settings.ts).
    -   Rendered by [`HologramManager.tsx`](../../client/src/features/visualisation/renderers/HologramManager.tsx) using both React components and class-based implementations.
    -   **Current Implementation** (✅ **VERIFIED**):
      - `HologramRing`: Individual animated ring components with configurable rotation
      - `HologramSphere`: Wireframe icosahedron spheres with rotation animation
      - `HologramManager`: Main React component managing multiple hologram elements
      - `HologramManagerClass`: Class-based implementation for non-React usage
    -   **Material System**: Uses basic Three.js materials with wireframe, transparency, and bloom layer support
    -   **Animation**: Frame-based rotation with configurable speeds per element type

### Edge Visuals

-   **Thickness/Width:**
    -   Controlled by settings like `visualisation.edges.baseWidth` in [`settings.ts`](../../client/src/features/settings/config/settings.ts).
    -   Can represent link strength.
-   **Color:**
    -   Default from `visualisation.edges.colour` in [`settings.ts`](../../client/src/features/settings/config/settings.ts).
    -   Can indicate link type or use gradients (e.g., `visualisation.edges.useGradient`, `visualisation.edges.gradientColors`).
-   **Style:**
    -   Arrows for directionality (e.g., `visualisation.edges.enableArrows`).
    -   Flow effects for activity (e.g., `visualisation.edges.enableFlowEffect`).
    -   All relevant settings are within `visualisation.edges` in [`settings.ts`](../../client/src/features/settings/config/settings.ts).

### Text Labels

-   Appearance controlled by `visualisation.labels` (which is `LabelSettings`) in [`settings.ts`](../../client/src/features/settings/config/settings.ts).
-   Node labels typically display titles or filenames.
-   Edge labels can show relationship types.

## Interactive Visualization

The visualisation supports rich real-time interaction:

-   **Navigation:** 
    - Mouse/touch controls for panning, zooming, rotating
    - SpacePilot 3D mouse support (`spacePilot` settings)
    - XR/VR hand tracking and controller input
-   **Selection & Highlighting:** 
    - Click/tap nodes and edges for detailed information
    - Selection effects with bloom and animation
    - Metadata visualisation overlay
-   **Multi-Graph Interaction:**
    - Switch between graph views (logseq ↔ visionflow)
    - Parallel visualisation of multiple graphs
    - Independent camera controls per graph
-   **Real-time Updates:**
    - WebSocket binary protocol for position updates
    - Server-side physics simulation pushes to client
    - ~60fps smooth animation with motion damping
-   **Spatial Arrangement:** 
    - Server-side physics (`GraphService` in Rust)
    - Spring-based layout algorithms
    - Collision detection and boundary constraints
    - Connected nodes attracted, unconnected repelled

## Metadata Visualization

### `MetadataVisualizer.tsx` ([`client/src/features/visualisation/components/MetadataVisualizer.tsx`](../../client/src/features/visualisation/components/MetadataVisualizer.tsx))

This component is responsible for displaying additional information or visual cues based on the metadata associated with nodes.

**Possible Implementations:**
-   **Icons/Glyphs:** Displaying small icons on or near nodes to represent file types, tags, or status.
-   **Auras/Halos:** Using subtle visual effects around nodes to indicate certain metadata properties (e.g., a glowing aura for unread items).
-   **Dynamic Text Panels:** Showing detailed metadata in a 2D overlay when a node is selected or hovered.

## Distinction from `rendering.md`

-   **`visualisation.md` (this document):** Focuses on the *conceptual* aspects.
    -   What do nodes and edges *represent*?
    -   How is data (size, type, connections) *encoded* into visual properties (size, colour, shape)?
    -   What insights is the user intended to gain from these visual mappings?
    -   The *meaning* behind the visual design choices.
-   **[`rendering.md`](./rendering.md):** Focuses on the *technical implementation*.
    -   How are spheres, lines, and text *drawn* using React Three Fiber and Three.js?
    -   What specific components (`GraphCanvas`, `GraphManager`, `TextRenderer`) are involved?
    -   What techniques (instancing, shaders like `HologramMaterial.tsx`) are used for performance and visual effects?
    -   The *mechanics* of putting pixels on the screen.

In essence, `visualisation.md` is about the "language" of the visual representation, while `rendering.md` is about the "grammar and tools" used to speak that language.