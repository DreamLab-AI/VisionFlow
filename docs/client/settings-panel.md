# Settings Panel

This document describes the redesigned, fully functional settings panel and its underlying architecture.

## Overview

The settings panel provides a user-friendly interface for configuring all aspects of the application's visualisation and behavior. It is organized, responsive, and provides real-time feedback.

## Design Principles

### 1. **Tabbed Organization**
Settings are organized into logical tabs for clarity:
- **Appearance**: Visual customisation (nodes, edges, labels, effects)
- **Performance**: Quality and optimisation settings
- **XR/VR**: Virtual and augmented reality configuration
- **Advanced**: Power user features that may require authentication

### 2. **Collapsible Sections**
Within each tab, settings are grouped into collapsible cards to reduce clutter and improve focus.

### 3. **Smart Controls**
Each setting uses the most appropriate UI control for an intuitive user experience, including sliders, color pickers, switches, and select dropdowns.

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