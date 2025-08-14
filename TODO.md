
### 1. Partially Completed & Disconnected Elements

This category includes features that are not fully implemented, UI components with placeholder data, and functionality that has been partially removed or is no longer connected to the backend.

*   **Claude Flow Actor:** The primary actor for multi-agent communication, `src/actors/claude_flow_actor_tcp.rs`, explicitly states that the WebSocket implementation has been removed and only a TCP version remains. However, parts of the TCP implementation are also incomplete, with `TODO`s and mock logic. This MUST be completed assuming the presence of the multi-agent system in another docker (this environment we are workign in)
*   **Bots Data Handling:** The `src/handlers/bots_handler.rs` file relies on mock data. The `fetch_hive_mind_agents` function returns an empty vector with a `warn!` message, and `initialize_swarm` returns a mock success response with a `TODO` to connect to the actual service. This indicates the entire "VisionFlow" agent visualization is disconnected from its intended live data source. This MUST be completed.
*   **Control Center UI Panels:**
    *   `client/src/app/components/RightPaneControlPanel.tsx`: The "Performance" section is entirely static, displaying hardcoded metrics like "FPS: 60" and "Memory: 245 MB". This is a placeholder for a real performance monitoring dashboard.
    *   `client/src/features/bots/components/BotsControlPanel.tsx`: The `handleAddAgent` function is not implemented and only logs to the console, making the UI buttons non-functional.

*   **Redundant API Endpoint:** `client/src/api/settingsApi.ts` defines an `updatePhysics` function that calls a dedicated `/api/settings/physics/{graph}` endpoint. However, the backend `src/handlers/settings_handler.rs` does not implement this route. Physics updates are handled by the main `POST /api/settings` endpoint, making the client's `updatePhysics` function disconnected. You need to assess the correct route based on the codebase and fully implement.

### 2. Hardcoded Variables & Paths

These are values that are fixed in the code but should be configurable, especially across different environments (development vs. production).

*   **GPU UUID in Production:** The `docker-compose.production.yml` file contains a hardcoded NVIDIA GPU UUID: `NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-GPU-553dc306-dab3-32e2-c69b-28175a6f4da6}`. This is extremely brittle and will only work on a specific machine. It should be a device index (e.g., `0`).
*   **Network Addresses & Ports:**

*   **File Paths:**
    *   **CUDA Kernels:** The Rust GPU code (`src/utils/unified_gpu_compute.rs`) and Dockerfiles contain multiple hardcoded search paths for the compiled `.ptx` kernel file, indicating potential for build failures if the path changes.
    *   **Health Check:** The health handler in `src/handlers/health_handler.rs` hardcodes the disk usage check to the `/workspace` path. workspace is a feature of this container not the project and it should be removed where ever it is found.


### 3. Hardcoded Multipliers & Magic Numbers

This category covers numeric constants used in calculations that affect the behavior of the physics simulation, animations, and UI components. These should be exposed as settings.

*   **CUDA Kernel Physics (`src/utils/visionflow_unified.cu`):** This file is a major source of hardcoded values that directly impact the simulation's behavior:
    *   `MIN_DISTANCE = 0.15f`: A critical value to prevent node collapse.
    *   `MAX_REPULSION_DIST = 50.0f`: A cutoff distance for repulsion calculations.
    *   `natural_length = 10.0f`: The ideal length for spring forces between nodes.
    *   `warmup = p.params.iteration / 200.0f`: A 200-iteration "warmup" period for physics stability.
    *   `extra_damping = 0.98f - 0.13f * warmup`: Magic numbers used in the warmup damping calculation.
    *   `temp_scale = ... / (1.0f + p.params.iteration * 0.0001f)`: A hardcoded cooling factor of `0.0001` for simulated annealing.
    *   `boundary_force_strength = 10.0f`: The strength of the force that pushes nodes back from the boundary.

### 4. Patch Fixes & Brittle Logic

This category includes code that appears to be a workaround for an underlying issue, logic that is not robust, or code that was refactored incompletely.

*   **Ignoring Hardcoded IP Address:** The `client/src/services/WebSocketService.ts` contains a specific check to ignore a problematic hardcoded IP address: `if (settings.system?.customBackendUrl?.includes('192.168.0.51'))`. This is a patch to prevent a bad configuration from breaking the application.
*   **Dual TCP Connection:** In `src/actors/claude_flow_actor_tcp.rs`, the `connect_to_claude_flow_tcp` function appears to connect to the same TCP server twice to create separate read and write streams. This is an unusual pattern and likely a workaround for a Tokio stream splitting issue.

*   **Legacy PTX Paths in Docker:** The `Dockerfile.dev` copies the compiled PTX kernel to two different locations for "legacy compatibility," indicating that some code may still be referencing an old, incorrect path. we are compiling the ptx in the docker build so we can remove the legacy path and the copy command.

### 5. Incomplete Routes & Connections

These are missing or improperly configured routes and communication channels between the frontend, backend, and other services.

*   **Incomplete GPU Physics Propagation:** In `src/handlers/settings_handler.rs`, when physics settings are updated, the call to `propagate_physics_to_gpu` is hardcoded to `"logseq"`. This means updates to the "visionflow" graph's physics from the UI will not be propagated to the GPU actor. MUST fix this.
*   **Missing Nginx MCP Relay Route:** The development Nginx configuration (`nginx.dev.conf`) defines a WebSocket proxy for `/ws/mcp-relay`, but the Rust API handler for it (`mcp_relay_handler.rs`) is not mounted under the `/api` scope in `src/handlers/api_handler/mod.rs`, making it unreachable.
*   **Legacy `dev-entrypoint.sh` Logic:** The `dev-entrypoint.sh` script has logic for starting services individually as a fallback but the primary method uses `supervisord`. The individual startup logic may be out of sync with the `supervisord.dev.conf` configuration.
