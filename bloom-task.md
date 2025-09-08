You are absolutely right. My apologies for oversimplifying. Your clarification is crucial: the goal isn't to merge all effects into one, but to have **two distinct, independently controllable post-processing chains**—one for the foreground graph (nodes/edges) and one for the background environment (hologram).

The current implementation is indeed attempting this with its layer-based system (`bloomRegistry`, different bloom passes), but the multiple overlapping components and shaders obscure this intent and create conflicts.

With this nuance in mind, here is a revised, more precise analysis and a concrete plan to achieve your goal cleanly and efficiently.

---

### Refined Analysis: The Goal is a Dual-Pipeline System

The core requirement is a **dual-pipeline post-processing system** where:

1.  **Graph Pipeline (Foreground):** Affects nodes and edges (Layer 1). This might be a sharp, intense "bloom" effect to highlight data. It should be controlled by `settings.visualisation.bloom`.
2.  **Environment Pipeline (Background):** Affects the hologram rings, spheres, and particles (Layer 2). This is likely intended to be a soft, diffuse, atmospheric "glow". It should be controlled by `settings.visualisation.glow`.

The current problem is that this elegant goal is being implemented through several conflicting methods:
*   **The Correct Foundation:** `PostProcessingEffects.tsx` and `bloomRegistry.ts` are the right way to do this with modern R3F libraries. They correctly use layers to separate the two pipelines.
*   **The Legacy Conflict:** `CustomEffectsRenderer.tsx` and `DiffuseEffectsIntegration.tsx` are a separate, broken attempt at a global post-processor that conflicts with the modern one.
*   **The Shader Conflict:** The custom shaders in `visualisation/shaders/` (like `EtherealDiffuseCloudMaterial`) and layered meshes (`WireframeWithExtendedGlow`) are attempts to create the "diffuse glow" for the background *without* using a post-processing pass, which is redundant and inefficient when a proper pipeline exists.

The plan, therefore, is not to merge the two effects, but to **consolidate all rendering to use the single, modern, dual-pipeline post-processing component** and eliminate the other conflicting methods.

---

### The Way Forward: A Phased Refactoring Plan

This plan will solidify the dual-pipeline architecture, simplify the components that feed into it, and remove all redundant code.

#### Phase 1: Solidify the Dual-Pipeline Foundation

**Goal:** Establish `PostProcessingEffects.tsx` as the single, authoritative post-processing system in the application.

1.  **Deprecate and Delete Legacy Systems:**
    *   Delete the entire `src/rendering/` directory (`CustomEffectsRenderer`, `DiffuseEffectsIntegration`, `DiffuseWireframeMaterial`).
    *   Delete `features/visualisation/effects/AtmosphericGlow.tsx`.
    *   Delete `features/graph/components/SelectiveBloomPostProcessing.tsx` (it's a less complete version of `PostProcessingEffects`).

2.  **Elevate the Champion Component:**
    *   Rename `features/graph/components/PostProcessingEffects.tsx` to something more descriptive, like `DualBloomPipeline.tsx`.
    *   Move this file to a more central location, such as `src/rendering/DualBloomPipeline.tsx`.

3.  **Refine the Dual-Pipeline Logic:**
    *   Inside the new `DualBloomPipeline.tsx`, make the mapping explicit and clear.

    ```typescript
    // Inside the new DualBloomPipeline.tsx

    // ...
    const bloomSettings = settings?.visualisation?.bloom; // For Graph (Layer 1)
    const glowSettings = settings?.visualisation?.glow;   // For Environment (Layer 2)

    // ... in useMemo ...
    // Bloom pass for nodes/edges (layer 1)
    if (bloomSettings?.enabled) {
      const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(size.width, size.height),
        bloomSettings.strength,
        bloomSettings.radius,
        bloomSettings.threshold
      );
      // ... selective rendering logic for layer 1 ...
      composer.addPass(bloomPass);
    }

    // Glow pass for hologram/environment (layer 2)
    if (glowSettings?.enabled) {
      const glowPass = new UnrealBloomPass(
        new THREE.Vector2(size.width, size.height),
        glowSettings.intensity, // Use 'intensity' from glow settings
        glowSettings.radius,
        glowSettings.threshold
      );
      // ... selective rendering logic for layer 2 ...
      composer.addPass(glowPass);
    }
    // ...
    ```

4.  **Integrate into `GraphCanvas.tsx`:**
    *   Remove all old post-processing components.
    *   Directly render your new `<DualBloomPipeline />` component inside the `<Canvas>`.

**Result of Phase 1:** You now have a single, clean, and explicit dual-pipeline post-processing system. The foundation is solid.

#### Phase 2: Refactor Visual Components to Use the Pipeline

**Goal:** Simplify all hologram and particle components to be "dumb" meshes that just render themselves onto the correct layer, letting the pipeline create the visual effects.

1.  **Unify the Hologram Environment:**
    *   Delete `WorldClassHologram.tsx`, `EnhancedHologramSystem.tsx`, `WireframeCloudMesh.tsx`, and `WireframeWithExtendedGlow.tsx`.
    *   Create a new, definitive **`<HologramEnvironment />`** component in `features/visualisation/components/`.
    *   This component will be responsible for rendering all background elements: rings, spheres, domes, etc.
    *   **Crucially:** All meshes inside `<HologramEnvironment />` should use a simple, emissive material (like `BloomStandardMaterial` from `BloomHologramMaterial.ts`) and be assigned to the environment layer: `obj.layers.enable(2)`.

2.  **Achieve the "Diffuse Glow" via Post-Processing:**
    *   The soft, "ethereal cloud" look from the old shaders can now be achieved by tweaking the `glowSettings` that feed into the environment pipeline. A high `radius`, low `threshold`, and moderate `intensity` on the `UnrealBloomPass` for Layer 2 will create exactly this effect, but in a much more performant and controllable way.

3.  **Consolidate Particles:**
    *   Refactor `HologramMotes.tsx` into a general-purpose `<Particles />` component.
    *   Use this component inside your new `<HologramEnvironment />`.
    *   Ensure the particles are also assigned to the environment layer (`obj.layers.enable(2)`).

4.  **Verify Graph Components:**
    *   **`GraphManager.tsx`**: Ensure the instanced mesh for nodes is assigned to the graph layer (`obj.layers.enable(1)`). The `HologramNodeMaterial` is fine to keep, as it adds unique effects like scanlines that are not part of the bloom effect.
    *   **`FlowingEdges.tsx`**: Ensure the lines are assigned to the graph layer (`obj.layers.enable(1)`). Refactor it to use a simple emissive `LineBasicMaterial` whose color brightness is modulated by `edgeBloomStrength`.

**Result of Phase 2:** All visual components are now simplified. They no longer try to create their own glow effects. They simply declare "I am part of the graph" (Layer 1) or "I am part of the environment" (Layer 2), and the `DualBloomPipeline` handles the rest.

#### Phase 3: Final Cleanup and Shader Consolidation

**Goal:** Remove all now-unnecessary shader code and finalize the material library.

1.  **Delete Redundant Shaders:**
    *   Delete the entire `features/visualisation/shaders/` directory *except* for `BloomHologramMaterial.ts`. The complex cloud and diffuse shaders are now obsolete.

2.  **Organize Remaining Materials:**
    *   Create a new, central directory: `src/rendering/materials/`.
    *   Move `features/graph/shaders/HologramNodeMaterial.ts` to `src/rendering/materials/HologramNodeMaterial.ts`.
    *   From `features/visualisation/shaders/BloomHologramMaterial.ts`, extract the `BloomStandardMaterial` class into its own file at `src/rendering/materials/BloomStandardMaterial.ts`. This will be your standard tool for making things glow.
    *   Delete the now-empty `features/visualisation/shaders/` and `features/graph/shaders/` directories.

### Final Architecture

This refined plan achieves your nuanced goal perfectly:

*   **`GraphCanvas.tsx`**: The root, containing the scene and the `<DualBloomPipeline />`.
*   **`DualBloomPipeline.tsx`**: The single source of truth for post-processing. It runs two `UnrealBloomPass` instances:
    *   One for Layer 1, configured by `settings.visualisation.bloom`.
    *   One for Layer 2, configured by `settings.visualisation.glow`.
*   **Graph Components (`GraphManager`, `FlowingEdges`)**: Render their objects to **Layer 1**.
*   **Environment Components (`HologramEnvironment`, `Particles`)**: Render their objects to **Layer 2**.
*   **Materials (`HologramNodeMaterial`, `BloomStandardMaterial`)**: Are simple and focused. They make objects emissive so the bloom pipeline can pick them up, but they don't try to create fake glows themselves.

This structure is clean, performant, and directly maps your settings to the two independent visual effects you require, fully resolving the duplication and confusion.