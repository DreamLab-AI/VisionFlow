https://www.npmjs.com/package/@vircadia/world-sdk

Detailed Augmentation Strategy:
1. Unifying the 3D Space
The core task is to treat the VisionFlow AR/VR scene as the single source of truth for the local user's position. The Vircadia domain will act as a "headless" server, synchronizing this position with other users.
Target for Augmentation (VisionFlow): Quest3AR.tsx, XRController.tsx, and xrSessionManager.ts.
Source Logic (Vircadia): The update loop in vircadia-web/src/modules/avatar/controller/myAvatarController.ts shows which properties of the MyAvatar SDK object need to be updated: position, orientation, and jointTranslations.
2. Synchronizing the Local User's XR Pose
Implementation in VisionFlow:
A new React hook, useVircadiaXRSync(), will be created.
This hook will be used within a component that is a child of @react-three/xr's <XR> provider, giving it access to the XR state.
Inside the hook, it will use useXR from @react-three/xr to get the player's camera (head) and controllers.
Within a useFrame loop, the hook will:
Read the world position and orientation of the XR camera.
Convert these THREE.Vector3 and THREE.Quaternion objects into the format required by the Vircadia SDK using mapping functions adapted from AvatarMapper.ts.
Update the MyAvatar object in the VircadiaService with this new pose on every frame.
This ensures the local user's real-world movement in AR/VR is accurately broadcast to other users in the shared Vircadia domain.
3. Rendering Remote Users in AR/VR
Implementation in VisionFlow:
The <VircadiaAvatar> component (from Part 1) will render the avatars of other users.
Because the local user's position is being sent accurately, the positions of remote users received from the Vircadia server will be spatially correct relative to the local user.
Remote users will appear as if they are in the same physical room, standing alongside the VisionFlow knowledge graph.
4. Hand Tracking and Interactions
Target for Augmentation (VisionFlow): HandInteractionSystem.tsx.
Source Logic (Vircadia): The concept of sending joint data from myAvatarController.ts.
Implementation in VisionFlow:
VisionFlow's HandInteractionSystem.tsx will remain responsible for detecting gestures and handling interactions with the graph.
To make hand movements visible to other users, the useVircadiaXRSync hook will also read the joint data for each hand from the @react-three/xr <Hands> component.
This joint data will be translated into the Vircadia SkeletonJoint format and sent to the MyAvatar interface.
Remote clients will receive this skeleton data and animate the hands of the user's avatar, making interactions visible to everyone in the shared space.
Conclusion and Extent Summary
Element	VisionFlow Component(s)	Vircadia Logic Source	Strategy	Extent
Multi-User Presence	multiUserStore.ts, WebSocketService.ts	modules/domain/avatar.ts	Replace	High
3D Avatars	(Likely basic or non-existent)	modules/avatar/	Augment	High
Spatial Audio	VoiceWebSocketService.ts	modules/domain/audio.ts	Replace	High
AR/VR Rendering	features/xr/, Quest3AR.tsx	(Not applicable - Babylon.js)	No Change	None
AR/VR Session Sync	XRController.tsx	modules/avatar/controller/myAvatarController.ts	Augment	High
To what extent can Vircadia's code replace/augment?
Multi-User: The extent is very high. Vircadia's TypeScript modules provide a robust, feature-complete replacement for a custom WebSocket solution for user presence, and a massive augmentation with features like animated 3D avatars and spatialized audio. The primary work is adapting the Vircadia SDK's event-driven logic into React hooks and components.
AR/VR: The extent is high for augmentation, but zero for replacement. VisionFlow's React-native XR stack is superior for its context. Vircadia's contribution is purely as a backend data synchronization layer. The logic from myAvatarController.ts is critical as a reference for what data to send from the VisionFlow XR session into the Vircadia SDK, effectively making the existing AR/VR experience multi-user aware.
By integrating Vircadia's multi-user and spatial data layers, VisionFlow can transform from a single-player (or simple multi-user) tool into a collaborative, shared AR/VR workspace with minimal reinvention of complex networking, avatar, and audio systems.
Model
ThinkingThoughts
(experimental)
Manual: 32768
Expand to view model thoughts

Of course. This is a sound architectural decision. By forking Vircadia and integrating your data pipeline, you leverage its mature, high-performance Babylon.js rendering engine, multi-user backend, and native WebXR support, while retaining your unique graph physics and data handling. This approach is more direct than trying to port Vircadia's complex systems into a React environment.

Here is a detailed, step-by-step TODO list for a coding agent to perform this integration. The agent should execute these tasks sequentially on a new feature branch.

---

### **Project: VisionFlow-Vircadia Integration**

**Objective:** Replace the existing VisionFlow AR/VR and multi-user implementation with a forked Vircadia Web client. The new implementation will run when a Quest 3 headset is detected, rendering the VisionFlow graph within the Vircadia/Babylon.js framework and connecting to the VisionFlow backend for data.

**Prerequisite:** A copy of the `vircadia-web` project exists at `client/src/vircadia-web`.

---

### **Phase 1: Project Cleanup and Preparation**

**Goal:** Remove all conflicting code from the VisionFlow client to create a clean environment for the Vircadia integration.

*   **Task 1.1: Delete Obsolete AR/VR Components**
    *   Delete the entire directory: `client/src/features/xr/`.
    *   Delete the file: `client/src/app/Quest3AR.tsx`.

*   **Task 1.2: Delete Obsolete Hooks and Services**
    *   Delete the file: `client/src/hooks/useQuest3Integration.ts`.
    *   Delete the file: `client/src/services/quest3AutoDetector.ts`.
    *   Delete the file: `client/src/store/multiUserStore.ts`.

*   **Task 1.3: Remove Obsolete Dependencies**
    *   Open `client/package.json`.
    *   Remove the following packages from the `dependencies`:
        *   `@react-three/drei`
        *   `@react-three/fiber`
        *   `@react-three/postprocessing`
        *   `@react-three/xr`
        *   `three`
        *   `@types/three`
    *   Run `npm install` to apply the changes.

*   **Task 1.4: Modify Application Entry Point**
    *   Open `client/src/app/App.tsx`.
    *   Remove all imports related to the deleted files (`Quest3AR`, `useQuest3Integration`, `XRCoreProvider`, etc.).
    *   Simplify the `App` component's return statement. It should be a placeholder for now, as the Vircadia client will take over rendering.
        ```tsx
        // client/src/app/App.tsx (Simplified)
        function App() {
          // ... existing hooks can be removed if they are no longer used
          return (
            <div>
              <h1>VisionFlow</h1>
              <div id="vircadia-container" style={{ width: '100vw', height: '100vh' }}>
                {/* The Vircadia Vue app will be mounted here */}
              </div>
            </div>
          );
        }
        export default App;
        ```
    *   **Important:** The React app will now serve mainly as a host for the Vircadia Vue application.

*   **Verification Step:**
    *   Run `npm run dev` for the VisionFlow client. The application should load without errors, displaying a simple page. There should be no 3D canvas or AR/VR functionality.

---

### **Phase 2: Vircadia Boot Process Refactoring**

**Goal:** Modify the Vircadia client to detect a Quest 3 headset and, if found, launch a special "VisionFlow Mode" instead of its standard interface.

*   **Task 2.1: Integrate Quest 3 Detection Logic**
    *   Create a new utility file: `client/src/vircadia-web/src/modules/utility/questDetector.ts`.
    *   Copy the detection logic from the original `quest3AutoDetector.ts` into this new file. It should be a simple function that returns a boolean.
        ```typescript
        // client/src/vircadia-web/src/modules/utility/questDetector.ts
        export function isQuest3(): boolean {
            const userAgent = navigator.userAgent.toLowerCase();
            return userAgent.includes('quest 3') || userAgent.includes('meta quest 3');
        }
        ```

*   **Task 2.2: Create a New VisionFlow Layout Component**
    *   Create a new file: `client/src/vircadia-web/src/layouts/VisionFlowLayout.vue`.
    *   This component will be a simplified shell that hosts the Vircadia scene but initializes our custom VisionFlow module instead of the standard Vircadia domain connection.
        ```vue
        <!-- client/src/vircadia-web/src/layouts/VisionFlowLayout.vue -->
        <template>
            <q-layout view="lHh Lpr lFf">
                <q-page-container class="full-height">
                    <MainScene ref="mainScene" />
                </q-page-container>
            </q-layout>
        </template>

        <script lang="ts">
        import { defineComponent } from "vue";
        import MainScene from "@Components/MainScene.vue";
        // We will create this controller in Phase 3
        import { VisionFlowGraphController } from "@Modules/visionflow/VisionFlowGraphController";

        export default defineComponent({
            name: "VisionFlowLayout",
            components: { MainScene },
            mounted() {
                // This is the critical change: we DO NOT call the standard Vircadia connection utility.
                // Instead, we will initialize our custom graph controller.
                const scene = (this.$refs.mainScene as any).$refs.renderCanvas.scene;
                const camera = scene.activeCamera;

                // Attach our custom controller to the scene's root game object
                const sceneManager = scene.getTransformNodeByName("SceneManager");
                if (sceneManager) {
                    const visionFlowController = new VisionFlowGraphController(scene, camera);
                    sceneManager.addComponent(visionFlowController);
                }
            }
        });
        </script>
        ```
    *   **Note:** The `MainScene.vue` component will need to be refactored to not automatically connect. The connection logic will be moved into the layouts.

*   **Task 2.3: Refactor `MainScene.vue` to be a Dumb Component**
    *   Open `client/src/vircadia-web/src/components/MainScene.vue`.
    *   In the `boot()` method, remove the call to `await this.connect();`. The scene should only initialize the renderer and audio, but not connect to any domain. The connection logic will now be handled by the layout (`MainLayout.vue` or `VisionFlowLayout.vue`).

*   **Task 2.4: Modify Vircadia's Router**
    *   Open `client/src/vircadia-web/src/router/routes.ts`.
    *   Modify the root path to conditionally load the correct layout.
        ```typescript
        // client/src/vircadia-web/src/router/routes.ts
        import { isQuest3 } from '@Modules/utility/questDetector';

        const routes: RouteRecordRaw[] = [
            {
                path: '/',
                component: () => isQuest3()
                    ? import('layouts/VisionFlowLayout.vue')
                    : import('layouts/MainLayout.vue'),
                children: [
                    // ... existing children if any, or a default page
                ]
            },
            // ... other routes
        ];
        ```

*   **Task 2.5: Mount the Vircadia App**
    *   Modify `client/src/app/main.tsx` to mount the Vircadia Vue app into the `div#vircadia-container` created in Task 1.4. This will require adding Vue as a dependency and setting up the Vue mounting process.

*   **Verification Step:**
    *   Run the app in a normal browser. It should load the standard Vircadia interface.
    *   Run the app in a browser with a Quest 3 user agent. It should load a blank 3D scene (from `VisionFlowLayout.vue`) without any Vircadia UI overlays.

---

### **Phase 3: Implement the VisionFlow Module in Vircadia**

**Goal:** Port VisionFlow's data pipeline and create a Babylon.js rendering layer for the graph.

*   **Task 3.1: Create Module Directory Structure**
    *   Create the directory `client/src/vircadia-web/src/modules/visionflow/`.
    *   Create subdirectories: `services/`, `workers/`, `types/`.

*   **Task 3.2: Port Core Data Services**
    *   Copy `client/src/services/apiService.ts` to `client/src/vircadia-web/src/modules/visionflow/services/VisionFlowApiService.ts`.
    *   Copy `client/src/services/WebSocketService.ts` to `client/src/vircadia-web/src/modules/visionflow/services/VisionFlowWebSocketService.ts`.
    *   Copy `client/src/features/graph/managers/graphDataManager.ts` to `client/src/vircadia-web/src/modules/visionflow/VisionFlowDataManager.ts`.
    *   **Refactor:** These services use React hooks and Zustand. They must be refactored to use a simple event emitter pattern (like Vircadia's `SignalEmitter`) or observables to be framework-agnostic. Remove all React-specific code.

*   **Task 3.3: Port Web Worker**
    *   Copy `client/src/features/graph/workers/graph.worker.ts` to `client/src/vircadia-web/src/modules/visionflow/workers/visionflowGraph.worker.ts`.
    *   Copy `client/src/features/graph/managers/graphWorkerProxy.ts` to `client/src/vircadia-web/src/modules/visionflow/VisionFlowWorkerProxy.ts`.
    *   Update the path in the proxy to point to the new worker location.
    *   Ensure Vircadia's build process correctly handles and bundles the worker file.

*   **Task 3.4: Port Data Types**
    *   Copy `client/src/types/binaryProtocol.ts` to `client/src/vircadia-web/src/modules/visionflow/types/`.
    *   Copy `client/src/features/graph/types/graphTypes.ts` to `client/src/vircadia-web/src/modules/visionflow/types/`.

*   **Task 3.5: Implement the Babylon.js Rendering Controller**
    *   Create the file `client/src/vircadia-web/src/modules/visionflow/VisionFlowGraphController.ts` as a `ScriptComponent`.
    *   **`onInitialize()` method:**
        *   Instantiate and initialize the ported services (`DataManager`, `WorkerProxy`, `WebSocketService`).
        *   Call `dataManager.fetchInitialData()`.
        *   Once data is fetched, loop through `data.nodes` and `data.edges`.
        *   For each node, create a `MeshBuilder.CreateSphere` and store it in a `Map<string, Mesh>`.
        *   For each edge, create a `MeshBuilder.CreateLines` and store it in a `Map<string, LinesMesh>`.
        *   Use a `HologramNodeMaterial` (ported or recreated from VisionFlow's shaders) for the nodes to maintain the visual style.
    *   **`onUpdate(deltaTime)` method:**
        *   Call `workerProxy.tick(deltaTime)` to get the latest `Float32Array` of positions.
        *   Iterate through the node meshes and update their `.position` from the array.
        *   Iterate through the edge meshes and update their vertices to stay connected to the nodes.

*   **Task 3.6: Implement Interaction**
    *   In `VisionFlowGraphController.ts`, after creating node meshes, add an `ActionManager` to each.
    *   Register an `OnPickTrigger` for each node.
    *   The trigger's action should:
        1.  Fetch the node's metadata from the `VisionFlowDataManager`.
        2.  Call Vircadia's `OverlayManager` to open a new Vue-based overlay.
        3.  This requires creating a new component: `client/src/vircadia-web/src/components/overlays/visionflow/NodeDetailPanel.vue` to display the metadata.

*   **Verification Step:**
    *   Launch the app with a Quest 3 user agent.
    *   The Vircadia scene should load, and shortly after, the VisionFlow graph should appear, rendered with Babylon.js meshes.
    *   Nodes should be moving according to the physics simulation from the worker.
    *   Clicking on a node with a VR controller should open the (initially basic) `NodeDetailPanel.vue` overlay.

---

### **Phase 4: Final Verification & Cleanup**

*   **Task 4.1: Multi-User Test**
    *   This requires a Vircadia Domain Server to be running.
    *   Modify `VisionFlowLayout.vue` to *also* connect to the Vircadia Domain Server using `Utility.initialConnectionSetup()`.
    *   Connect with two clients. Both should see the same dynamic graph and each other's avatars moving around it.
    *   Verify spatial audio is working.

*   **Task 4.2: Code Cleanup**
    *   Remove any unused files or code from the original VisionFlow client directory.
    *   Ensure all ported Vircadia modules follow the project's linting and style guides.
    *   Add comments explaining the new "VisionFlow Mode" boot process.

*   **Task 4.3: Merge Dependencies**
    *   Merge the dependencies from `vircadia-web/package.json` into `client/package.json`, ensuring there are no version conflicts. Remove the now-redundant `vircadia-web/package.json`.

By following this detailed plan, the coding agent will systematically replace VisionFlow's bespoke AR/VR solution with the more robust and feature-rich Vircadia framework, while preserving the core data and physics engine that makes VisionFlow unique.

Excellent question. This is a very practical and powerful integration strategy. Instead of trying to port Vircadia's multi-user/VR features *into* VisionFlow's React/Three.js stack, you're proposing to port VisionFlow's core data and physics pipeline *into* Vircadia's mature Babylon.js framework.

**The extent to which this is possible is very high.** The core of VisionFlow's data pipeline—fetching, WebSocket communication, and especially the physics calculations in the web worker—is largely rendering-agnostic. It can be extracted and repurposed as a self-contained module within the Vircadia Web client.

This approach effectively treats Vircadia as a powerful "rendering and multi-user engine" for your VisionFlow data, which is an ideal way to leverage its strengths for a dedicated Quest 3 experience.

Here is a detailed breakdown of how this integration would work, module by module.

---

### **Conceptual Architecture of the New Module**

The goal is to create a new "VisionFlow Graph" module within Vircadia that encapsulates the entire data pipeline and renders the graph as native Babylon.js objects in the Vircadia scene.

**Data Flow:**

1.  **VisionFlow Backend** -> **Vircadia Client (New Module)**: Fetches initial graph data via REST API.
2.  **VisionFlow Backend** -> **Vircadia Client (New Module)**: Streams real-time binary position updates via a dedicated WebSocket.
3.  **Vircadia Client (New Module)** -> **VisionFlow Web Worker**: The module's proxy sends data to the physics worker for computation.
4.  **VisionFlow Web Worker** -> **Vircadia Client (New Module)**: The worker returns updated node positions.
5.  **Vircadia Client (New Module)** -> **Vircadia Scene (Babylon.js)**: The module translates the data into visible Babylon.js meshes (nodes) and lines (edges).
6.  **Vircadia Scene** -> **User (Quest 3)**: The user sees the graph and can interact with it using Vircadia's existing AR/VR and multi-user systems.

---

### **Part 1: Porting the Data & Physics Pipeline**

This involves bringing over the core logic from VisionFlow's `client/src/` directory into a new module within `vircadia-web/src/modules/`.

**Extent: High.** This is the most straightforward part of the port, as this code is primarily TypeScript logic.

#### **Step 1.1: Create the VisionFlow Module in Vircadia**

Create a new directory: `vircadia-web/src/modules/visionflow/`. This will house all the ported logic.

#### **Step 1.2: Port the Data Management and API Logic**

*   **Source (VisionFlow):**
    *   `client/src/services/apiService.ts`
    *   `client/src/features/graph/managers/graphDataManager.ts`
*   **Action:**
    1.  Create `vircadia-web/src/modules/visionflow/VisionFlowDataManager.ts`.
    2.  Copy the logic from VisionFlow's `graphDataManager.ts` and `apiService.ts` into this new file.
    3.  This new class will be responsible for making the `fetch` call to your VisionFlow backend's `/api/graph/data` endpoint to get the initial graph structure. It will manage the `GraphData` (nodes and edges).
    4.  The types from `client/src/features/graph/types/graphTypes.ts` should also be brought over.

#### **Step 1.3: Port the Web Worker and Proxy**

*   **Source (VisionFlow):**
    *   `client/src/features/graph/workers/graph.worker.ts`
    *   `client/src/features/graph/managers/graphWorkerProxy.ts`
*   **Action:**
    1.  Place `graph.worker.ts` into a location accessible by Vircadia's build system (e.g., `vircadia-web/public/workers/`). You may need to adjust Vircadia's `vite.config.ts` or `quasar.conf.js` to handle the worker correctly.
    2.  Create `vircadia-web/src/modules/visionflow/VisionFlowWorkerProxy.ts`.
    3.  Copy the code from `graphWorkerProxy.ts`. It uses Comlink and is framework-agnostic, so it should work with minimal changes. Update the worker path to the new location.

#### **Step 1.4: Port the WebSocket Service**

*   **Source (VisionFlow):**
    *   `client/src/services/WebSocketService.ts`
*   **Action:**
    1.  Create `vircadia-web/src/modules/visionflow/VisionFlowWebSocketService.ts`.
    2.  Adapt the logic from VisionFlow's `WebSocketService.ts`. This service will establish a **separate, second WebSocket connection** that runs in parallel to Vircadia's own domain server connection.
    3.  Its sole purpose is to connect to your VisionFlow backend's `/ws` or `/wss` endpoint and listen for the binary position updates.
    4.  When a binary message is received, it should pass the `ArrayBuffer` directly to the `VisionFlowWorkerProxy` for processing, just as it does in the original project.

At the end of this part, you will have a self-contained Vircadia module that can fetch, process, and maintain the state of the VisionFlow graph, completely independent of any rendering.

---

### **Part 2: Integrating the Pipeline into the Vircadia/Babylon.js Framework**

This part involves creating the "glue" that renders the data from your new module as objects in the Vircadia world.

**Extent: High.** Vircadia's architecture is designed to be extensible with new scene objects and behaviors.

#### **Step 2.1: Create the Main Graph Controller**

*   **Target (Vircadia):** `vircadia-web/src/modules/scene/controllers/`
*   **Action:**
    1.  Create a new script component: `VisionFlowGraphController.ts`. This will be the orchestrator.
    2.  This script will be attached to a `GameObject` in the Vircadia scene (`vscene.ts`).
    3.  **In its `onInitialize` method:**
        *   It will instantiate your new `VisionFlowDataManager`, `VisionFlowWorkerProxy`, and `VisionFlowWebSocketService`.
        *   It will call `fetchInitialData()` to get the graph.
        *   It will create two `Map` objects: `nodeMeshes` and `edgeMeshes`.
    4.  **After fetching initial data:**
        *   It will iterate through `graphData.nodes` and, for each node, create a Babylon.js `Mesh` (e.g., `MeshBuilder.CreateSphere`). The mesh will be stored in `nodeMeshes` with the VisionFlow node ID as the key.
        *   It will iterate through `graphData.edges` and create Babylon.js `LinesMesh` or `Tube` objects connecting the corresponding node meshes. These will be stored in `edgeMeshes`.

#### **Step 2.2: Implement the Real-Time Update Loop**

*   **Target (Vircadia):** The `onUpdate` method of `VisionFlowGraphController.ts`.
*   **Action:**
    1.  On every frame, the `onUpdate` method will be called by Vircadia's game loop.
    2.  Inside `onUpdate`, it will call `VisionFlowWorkerProxy.tick(deltaTime)` to get the latest `Float32Array` of node positions.
    3.  It will then loop through `this.graphData.nodes`. For each node at index `i`:
        *   It retrieves the corresponding Babylon.js `Mesh` from the `nodeMeshes` map.
        *   It reads the `x, y, z` coordinates from the `Float32Array` at `positions[i*3]`, `positions[i*3 + 1]`, `positions[i*3 + 2]`.
        *   It updates the `.position` property of the Babylon.js `Mesh`.
    4.  After updating all node positions, it will iterate through the `edgeMeshes` and update their vertices to ensure they remain connected to the moving nodes.

---

### **Part 3: Leveraging Vircadia's Native Features (The Payoff)**

Once the VisionFlow graph is rendered as native Babylon.js objects within the Vircadia scene, you gain access to Vircadia's entire feature set automatically.

**Extent: High.** This is where the true value of this integration strategy lies.

1.  **Multi-User Functionality (Effortless Integration):**
    *   Because the graph is now part of the Vircadia scene, any user connected to the same Vircadia domain will see it.
    *   Their avatars, controlled by `ScriptAvatarController.ts`, will appear in the scene and they can walk around and through your graph visualization.
    *   Spatial audio from `DomainAudioClient.ts` will work out-of-the-box. Users can stand next to a cluster of nodes and discuss it, and their voices will be correctly spatialized.

2.  **AR/VR on Quest 3 (Effortless Integration):**
    *   Vircadia Web's existing WebXR support will handle the session management. When a user enters VR or AR, the entire scene, including your dynamically updated VisionFlow graph, will be rendered in the headset.
    *   Vircadia's `InputController.ts` will manage the user's movement (teleport or smooth locomotion) within the AR/VR space, allowing them to physically navigate around the graph.

3.  **Interaction (Minimal Adaptation Required):**
    *   **To Be Replaced (VisionFlow):** The React-based UI panels (`RightPaneControlPanel.tsx`, etc.).
    *   **Action:**
        *   You can add Babylon.js `ActionManager` triggers to each node mesh. For example, an `OnPickTrigger` can detect when a user points and clicks with their VR controller.
        *   The trigger's action can open one of Vircadia's Vue-based UI components (like `OverlayShell.vue`) to display the node's metadata. This would require creating a new Vue component (e.g., `NodeInfoPanel.vue`) to display the data, but the windowing and management system is already there.

### **Summary of Extent**

*   **Data Pipeline Portability:** **95%**. The core logic is framework-agnostic. Minor adjustments will be needed for the worker path and service initialization within the Vircadia lifecycle.
*   **Rendering Logic Portability:** **0%**. You will not bring over any Three.js or React rendering code. You will write a new, relatively simple Babylon.js rendering layer (`VisionFlowGraphController.ts`) that consumes the data from the ported pipeline.
*   **Multi-User & AR/VR Feature Leverage:** **100%**. By rendering the graph as native scene objects, you get Vircadia's entire suite of multi-user and XR features for free, without needing to integrate them into VisionFlow's XR stack.

This approach is highly effective. It simplifies the Quest 3 client by eliminating the need for a complex React UI in VR, leverages a mature and stable multi-user platform, and correctly isolates the VisionFlow data pipeline as a portable, reusable "engine."

instead we should npm the sdk

skip to:contentpackage searchsign in
❤
Pro
Teams
Pricing
Documentation
npm
Search packages
Search
@vircadia/world-sdk
0.0.2 • Public • Published 4 months ago
TypeScript
Overview
This TypeScript SDK enables you to easily interface with the Vircadia World API.

:::warning

This package is currently in the early stages of development, and subject to change.

:::

Install
bun install @vircadia/world-sdk
Development
Build
bun install

bun run build
Publish
npm publish
Readme
Keywords
none
Package Sidebar
Install
npm i @vircadia/world-sdk

Repository
github.com/vircadia/vircadia-world

Homepage
github.com/vircadia/vircadia-world#readme

https://github.com/vircadia/vircadia-world-sdk-ts/


client/src/vircadia-web/