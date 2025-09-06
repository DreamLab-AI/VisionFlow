# VisionFlow Client Documentation

*[Client](../index.md)*

Welcome to the comprehensive documentation for the VisionFlow client. This document serves as the main index and table of contents for all client-side documentation, providing a clear overview of the architecture, features, and systems that power the VisionFlow application.

## Table of Contents

### 1. Core Architecture & Systems
- **[Architecture](architecture.md)**: A high-level overview of the client's architecture, including the technology stack, component structure, and data flow.
- **[State Management](state-management.md)**: A deep dive into the Zustand-based state management system, covering `settingsStore` and the management of transient graph state.
- **[Graph System](graph-system.md)**: Detailed documentation on the React Three Fibre-based rendering pipeline, including the web worker for physics, `InstancedMesh` for performance, custom shaders, and metadata-driven visualisation.
- **[UI Components](ui-components.md)**: A comprehensive reference for the design system and reusable UI components, including props and usage examples.
- **[Types](types.md)**: An overview of the core TypeScript types used throughout the application, including the binary protocol and settings structures.

### 2. Features & Integrations
- **[GPU Analytics](features/gpu-analytics.md)**: Documentation on the client-side implementation of GPU-accelerated shortest path analytics, including the UI, state management, and API interaction.
- **[XR Integration](xr-integration.md)**: A guide to the WebXR integration, covering the provider-based architecture, Quest 3 AR implementation, and hand interaction system.
- **[WebSocket Communication](websocket.md)**: A detailed explanation of the WebSocket service, including the singleton pattern, readiness protocol, and adapter pattern for real-time communication.
- **[Settings Panel](settings-panel.md)**: A description of the settings panel, its unified state architecture, and how it interacts with the `settingsStore`.

### 3. Other Key Documentation
- **[Command Palette](command-palette.md)**: Documentation for the command palette feature.
- **[Core](core.md)**: Core client concepts.
- **[Help System](help-system.md)**: Documentation for the help system.
- **[Onboarding](onboarding.md)**: Documentation for the user onboarding flow.
- **[Parallel Graphs](parallel-graphs.md)**: Documentation on the parallel graph rendering system.
- **[Rendering](rendering.md)**: General rendering documentation.
- **[User Controls Summary](user-controls-summary.md)**: A summary of user controls.
- **[Visualisation](visualisation.md)**: General visualisation documentation.

## Technology Stack

- **React 18**: Modern React with hooks and concurrent features.
- **TypeScript**: Type-safe development with strict mode.
- **React Three Fibre**: 3D rendering with declarative Three.js.
- **Zustand**: Lightweight, performant state management.
- **WebXR**: Virtual and augmented reality support via `@react-three/xr`.
- **Tailwind CSS**: Utility-first styling framework.
- **Vite**: Fast build tool and development server.
## Documents

- [Client Architecture](./architecture.md)
- [Command Palette](./command-palette.md)
- [Client Core Utilities and Hooks](./core.md)
- [GPU-Accelerated Analytics](./gpu-analytics.md)
- [Graph System](./graph-system.md)
- [Help System](./help-system.md)
- [Onboarding System](./onboarding.md)
- [Parallel Graphs Feature](./parallel-graphs.md)
- [Client Rendering System](./rendering.md)
- [Settings Panel](./settings-panel.md)
- [State Management](./state-management.md)
- [Client TypeScript Types](./types.md)
- [UI Component Library](./ui-components.md)
- [User Controls Summary - Settings Panel](./user-controls-summary.md)
- [Client-Side visualisation Concepts](./visualization.md)
- [WebSocket Communication](./websocket.md)
- [WebXR Integration](./xr-integration.md)


## Related Topics

- [Client Architecture](../client/architecture.md)
- [Client Core Utilities and Hooks](../client/core.md)
- [Client Rendering System](../client/rendering.md)
- [Client TypeScript Types](../client/types.md)
- [Client side DCO](../archive/legacy/old_markdown/Client side DCO.md)
- [Client-Side visualisation Concepts](../client/visualization.md)
- [Command Palette](../client/command-palette.md)
- [GPU-Accelerated Analytics](../client/features/gpu-analytics.md)
- [Graph System](../client/graph-system.md)
- [Help System](../client/help-system.md)
- [Onboarding System](../client/onboarding.md)
- [Parallel Graphs Feature](../client/parallel-graphs.md)
- [RGB and Client Side Validation](../archive/legacy/old_markdown/RGB and Client Side Validation.md)
- [Settings Panel](../client/settings-panel.md)
- [State Management](../client/state-management.md)
- [UI Component Library](../client/ui-components.md)
- [User Controls Summary - Settings Panel](../client/user-controls-summary.md)
- [WebSocket Communication](../client/websocket.md)
- [WebXR Integration](../client/xr-integration.md)
