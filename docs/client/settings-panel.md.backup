# Settings Panel

*[Client](../index.md)*

This document describes the redesigned, fully functional settings panel and its underlying architecture.

## Overview

The settings panel provides a user-friendly interface for configuring all aspects of the application's visualisation and behaviour. It is organised, responsive, and provides real-time feedback.

## Design Principles

### 1. **Tabbed Organisation**
Settings are organised into logical tabs for clarity:
- **Appearance**: Visual customisation (nodes, edges, labels, effects)
- **Performance**: Quality and optimisation settings
- **XR/VR**: Virtual and augmented reality configuration
- **Advanced**: Power user features that may require authentication

### 2. **Collapsible Sections**
Within each tab, settings are grouped into collapsible cards to reduce clutter and improve focus.

### 3. **Smart Controls**
Each setting uses the most appropriate UI control for an intuitive user experience, including sliders, colour pickers, switches, and select dropdowns.

### 4. **Visual Feedback**
The panel provides immediate visual feedback, such as hover effects and save confirmations. It also clearly indicates which features are gated for authenticated users.

## Implementation Details

### Component Structure
```
SettingsPanel
├── Header (title + description)
├── Tabs Component
│   ├── Tab List (Appearance, Performance, XR/VR, Advanced)
│   └── Tab Content
│       └── Collapsible Setting Groups
│           └── Individual Setting Controls
└── Status Bar (auto-save info + user status)
```

### Key Features

- **Real-time Updates**: Changes to settings are reflected instantly in the visualisation without requiring a manual save.
- **Power User Gating**: Advanced settings are visible and enabled only for authenticated users.
- **Responsive Layout**: The panel is designed with a fixed height and scrollable content area to work well across different screen sizes.

### File Structure
- **Main Component**: [`client/src/features/settings/components/panels/SettingsPanelRedesign.tsx`](../../client/src/features/settings/components/panels/SettingsPanelRedesign.tsx)
- **Host Panel**: [`client/src/app/components/RightPaneControlPanel.tsx`](../../client/src/app/components/RightPaneControlPanel.tsx)

## Settings Architecture: Single Source of Truth

The entire settings system is powered by a single, authoritative settings store. This eliminates conflicts and ensures data consistency.

- **Authoritative Store**: [`client/src/store/settingsStore.ts`](../../client/src/store/settingsStore.ts)

This Zustand-based store is the single source of truth for all application settings. It features:
- **Persistence**: Settings are automatically saved to `localStorage` and synchronized with the server.
- **Immutability**: Uses Immer for safe and predictable state updates.
- **Reactivity**: Components subscribe to the store and reactively update when settings change.
- **Multi-Graph Support**: Manages separate configuration trees for different graph visualisations (e.g., `visualisation.graphs.logseq` and `visualisation.graphs.visionflow`).

All UI components, including the settings panel, interact exclusively with this unified store, ensuring that the application state is always consistent and reliable.



## See Also

- [Configuration Guide](../getting-started/configuration.md)
- [Getting Started with VisionFlow](../getting-started/index.md)
- [Guides](../guides/README.md)
- [Installation Guide](../getting-started/installation.md)
- [Quick Start Guide](../getting-started/quickstart.md)
- [VisionFlow Quick Start Guide](../guides/quick-start.md)
- [VisionFlow Settings System Guide](../guides/settings-guide.md)

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
- [State Management](../client/state-management.md)
- [UI Component Library](../client/ui-components.md)
- [User Controls Summary - Settings Panel](../client/user-controls-summary.md)
- [VisionFlow Client Documentation](../client/index.md)
- [WebSocket Communication](../client/websocket.md)
- [WebXR Integration](../client/xr-integration.md)
