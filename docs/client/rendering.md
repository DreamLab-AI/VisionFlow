# Client Rendering System

This document provides a detailed overview of the 3D rendering system in the LogseqXR client, which is built using React Three Fiber (R3F) and Three.js. It covers the setup, key components, and how various visual elements are rendered.

## Overview

The rendering system is responsible for transforming the graph data (nodes, edges, metadata) into an interactive 3D visualisation. It leverages the declarative nature of React with the power of Three.js through R3F, allowing for efficient management of the 3D scene graph and performance optimisations like instancing. This document serves as the primary documentation for the technical implementation of the client-side visualisation.

## React Three Fiber (R3F) Setup

The core of the 3D scene is established by `@react-three/fiber`.

### `GraphCanvas.tsx` (`client/src/features/graph/components/GraphCanvas.tsx`)

This is the main entry point for the 3D scene. It wraps the Three.js `Canvas` component from R3F and sets up the fundamental rendering environment.

**Responsibilities:**
-   initialises the WebGL renderer.
-   Configures the camera (e.g., `PerspectiveCamera`).
-   Sets up basic scene elements like ambient lighting.
-   Hosts other 3D components that make up the graph visualisation.
-   Manages the `dpr` (device pixel ratio) for rendering quality.

**Key Properties:**
-   `camera`: Defines the camera properties (e.g., `fov`, `near`, `far`, `position`).
-   `gl`: WebGL renderer settings (e.g., `antialias`, `alpha`).

## Core Rendering Components

### `GraphManager.tsx` (`client/src/features/graph/components/GraphManager.tsx`)

This component is central to rendering the graph's nodes and edges. It receives processed graph data from the `GraphDataManager` and efficiently renders it in the 3D scene using React Three Fiber.

**Responsibilities:**
-   Manages the creation and updating of 3D objects for nodes and edges
-   Utilizes **instanced rendering** for nodes and edges to draw many similar objects with a single draw call, significantly improving performance for large graphs
-   Applies visual properties (colour, size, material) based on settings and node/edge attributes
-   Orchestrates updates to node positions and other dynamic properties based on physics simulation data received from the server via binary protocol
-   Supports dual-graph rendering for both Logseq and VisionFlow graph types simultaneously
-   Integrates with post-processing effects and visual enhancements

### `GraphViewport.tsx` (`client/src/features/graph/components/GraphViewport.tsx`)

This component manages the camera controls and applies post-processing effects to the rendered scene.

**Responsibilities:**
-   Integrates camera controls (e.g., `OrbitControls` or custom controls) to allow user navigation.
-   Applies visual enhancements like bloom, depth of field, or other effects using R3F's `Postprocessing` components.

## Node and Edge Rendering

### Nodes

Nodes are typically rendered as instanced meshes (e.g., spheres or custom geometries).

-   **Geometry:** A basic geometry (e.g., `SphereGeometry`) is created once.
-   **Material:** A material (e.g., `MeshStandardMaterial`, `MeshBasicMaterial`) is applied, often with custom shaders for unique visual effects.
-   **Instancing:** `InstancedMesh` is used to render thousands of nodes efficiently. Each instance has its own position, rotation, and scale, which are updated dynamically.

### Edges

Edges are rendered as lines or thin cylinders connecting nodes.

-   **Geometry:** Line geometries are dynamically generated based on connected node positions. For curved edges, `CatmullRomCurve3` or similar can be used.
-   **Material:** Line materials (`LineBasicMaterial`, `LineDashedMaterial`) or custom shader materials are used.
-   **Instancing:** For very large graphs, instancing can also be applied to edges, though it's more complex than for nodes due to varying lengths and orientations.

## Text Rendering

### `TextRenderer.tsx` (`client/src/features/visualisation/renderers/TextRenderer.tsx`)

This component is responsible for rendering text labels (e.g., node names, metadata) in the 3D scene.

**Key Features:**
-   Uses **Signed Distance Field (SDF) fonts** for crisp, scalable text that looks good at any distance and angle.
-   Handles text positioning, alignment, and scaling relative to the 3D objects they label.
-   Optimized for performance, often by batching text geometries or using instancing for common labels.

## Custom Shaders

### `HologramMaterial.tsx` (`client/src/features/visualisation/renderers/materials/HologramMaterial.tsx`)

This module defines a custom Three.js `ShaderMaterial` used to create a distinctive holographic visual effect for certain nodes or elements.

**Key Features:**
-   Utilizes GLSL shaders (vertex and fragment shaders) to achieve effects like:
    -   Animated scan lines or noise patterns.
    -   colour tinting and transparency.
    -   Edge glow or outline effects.
-   Integrates with R3F by being exposed as a custom material component.

## Graph Data Management

### `GraphDataManager` (`client/src/features/graph/managers/graphDataManager.ts`)

The GraphDataManager serves as the central hub for managing graph data flow between the WebSocket service and the 3D rendering components.

**Key Features:**
- **Binary Protocol Integration**: Processes incoming binary position updates via the 28-byte format
- **Dual Graph Support**: Manages separate data streams for Logseq and VisionFlow graphs
- **Real-time Updates**: Handles 60fps position synchronisation from server-side physics
- **Event Broadcasting**: Notifies rendering components of data changes
- **Memory Management**: Efficiently manages large graph datasets with cleanup utilities

### `GraphWorkerProxy` (`client/src/features/graph/managers/graphWorkerProxy.ts`)

Handles heavy graph processing operations in Web Workers to maintain 60fps UI performance.

**Responsibilities:**
- **Background Processing**: Offloads complex calculations from main thread
- **Data Transformation**: Converts server data to render-ready format
- **Performance optimisation**: Prevents UI blocking during large graph updates

## Distinction from `visualisation.md`

While `rendering.md` focuses on the "how" of drawing elements in 3D space (technical implementation, R3F components, performance techniques), `visualisation.md` focuses on the "what" and "why" – the higher-level concepts of how data is mapped to visual properties, the meaning of different visual elements (e.g., node colour representing file type), and the overall user experience of interacting with the visualized knowledge graph.