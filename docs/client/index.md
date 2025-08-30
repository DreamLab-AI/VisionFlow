# VisionFlow Client Documentation

Welcome to the comprehensive documentation for the VisionFlow client. This document serves as the main index and table of contents for all client-side documentation, providing a clear overview of the architecture, features, and systems that power the VisionFlow application.

## Table of Contents

### 1. Core Architecture & Systems
- **[Architecture](architecture.md)**: A high-level overview of the client's architecture, including the technology stack, component structure, and data flow.
- **[State Management](state-management.md)**: A deep dive into the Zustand-based state management system, covering `settingsStore` and the management of transient graph state.
- **[Graph System](graph-system.md)**: Detailed documentation on the React Three Fiber-based rendering pipeline, including the web worker for physics, `InstancedMesh` for performance, custom shaders, and metadata-driven visualisation.
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
- **[Visualization](visualization.md)**: General visualization documentation.

## Technology Stack

- **React 18**: Modern React with hooks and concurrent features.
- **TypeScript**: Type-safe development with strict mode.
- **React Three Fiber**: 3D rendering with declarative Three.js.
- **Zustand**: Lightweight, performant state management.
- **WebXR**: Virtual and augmented reality support via `@react-three/xr`.
- **Tailwind CSS**: Utility-first styling framework.
- **Vite**: Fast build tool and development server.