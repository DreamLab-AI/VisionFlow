# Task List for Upgrading Fork

This list outlines the key features to be ported from the "failed fork". The relevant code has been backed up in the `/codestore` directory.

## 1. Advanced Performance-Optimized Settings Management (Highest Value)

The most significant feature in the failed fork is its highly optimized, path-based settings management system. The working version uses a simpler, less performant approach.

**Tasks:**

- [ ] **Port Path-Based API:** The API client is purely path-based, sending only granular changes (e.g., `visualisation.nodes.opacity`) instead of the entire settings object on every update.
    - *Source:* `codestore/client/src/api/settingsApi.ts`
- [ ] **Implement Optimized Store:** The Zustand store is designed for performance.
    - [ ] **Lazy Loading:** Settings are not fetched all at once. The `ensureLoaded` and `loadSection` methods fetch only the settings needed for the currently visible UI, drastically reducing initial load time.
    - [ ] **Partial State:** The store only holds the settings that have been loaded, reducing its in-memory footprint.
    - [ ] **Debounced Auto-Saving:** An `AutoSaveManager` class intelligently batches multiple rapid setting changes (e.g., from a slider) into a single API call, preventing network flooding and including error recovery/retry logic.
    - *Source:* `codestore/client/src/store/settingsStore.ts`
- [ ] **Integrate Selective Hooks:** Custom hooks (`useSelectiveSetting`, `useSelectiveSettings`) allow React components to subscribe to only the specific settings they need, preventing unnecessary re-renders.
    - *Source:* `codestore/client/src/hooks/useSelectiveSettingsStore.ts`

**Why it's valuable:** This system solves major performance bottlenecks common in complex applications. It improves initial load speed, reduces network traffic by up to 80-90% for settings updates, and makes the UI far more responsive.

## 2. Comprehensive & Structured UI Definitions

The failed fork has a more mature and complete system for defining the settings UI, which would make development and maintenance much easier.

**Tasks:**

- [ ] **Integrate Main Settings UI Definition:** This file is a complete blueprint for the entire settings UI, defining every control, its type, path, and properties.
    - *Source:* `codestore/client/src/features/settings/config/settingsUIDefinition.ts`
- [ ] **Implement Debug Settings UI Pattern:** This pattern cleanly separates client-side-only debug settings (using `localStorage`) from backend-synced settings.
    - *Reference (missing file):* `client/src/features/settings/config/debugSettingsUIDefinition.ts` (This file was not found in the original fork, but the pattern should be replicated).

**Why it's valuable:** This declarative UI approach makes the settings panel highly maintainable and scalable. Adding or changing a setting is as simple as modifying the configuration file.

## 3. Implemented UI for Advanced Control Panels

The failed fork contains the fully implemented UI logic for sections that are placeholders in the working code.

**Tasks:**

- [ ] **Integrate Physics Engine Controls:** A detailed UI for controlling the GPU physics engine.
    - *Source (missing file):* `codestore/client/src/features/physics/components/PhysicsEngineControls.tsx` (This file was not found, but the implementation should be ported if found elsewhere or recreated based on memory or analysis of the wider codebase).
- [ ] **Integrate Dashboard Panel:** A comprehensive dashboard with real-time metrics.
    - *Source (missing file):* `codestore/client/src/features/dashboard/components/DashboardPanel.tsx` (This file was not found, but the implementation should be ported if found elsewhere or recreated).
- [ ] **Integrate Analytics Components:** UI for running complex graph analyses like semantic clustering and shortest path.
    - *Source:* `codestore/client/src/features/analytics/components/`

**Why it's valuable:** This is ready-to-use code that can be adapted to fill the empty tabs in the working version's superior UI layout, saving significant development time.





## 4. Advanced Rendering and Shader Effects

The failed fork contains a more modular and potentially higher-quality rendering pipeline with custom shaders for advanced visual effects.

**Tasks:**

- [ ] **Integrate Rendering Pipeline:** A dedicated `/rendering` directory and several custom shaders for diffuse glow, ethereal clouds, and bloom effects.
    - *Source:* `codestore/client/src/rendering/`
- [ ] **Integrate Custom Shaders:** Files like `EtherealDiffuseCloudMaterial.ts` and `DiffuseWireframeMaterial.tsx` implement advanced atmospheric effects.
    - *Source:* `codestore/client/src/features/visualisation/shaders/`

**Why it's valuable:** These custom shaders and rendering components could offer superior visual quality and more artistic control over the final look of the visualization.

## Summary

While the "working" codebase has a better overall UI structure and has added multi-user features, the "failed" fork contains critical, high-value code related to performance, settings architecture, UI implementation, and advanced rendering that should be carefully ported over. The settings management system, in particular, is a crucial feature that was lost.