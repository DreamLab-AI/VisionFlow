This list is structured to tackle the most critical, system-breaking bugs first, then address major architectural flaws, perform necessary refactoring, implement new features, and finally, update the documentation.
Priority 1: Critical - Fix Broken Physics and Settings UI
These tasks are critical as they address a fundamental bug that makes the physics engine and its UI controls completely non-functional.
Resolve Settings Store Conflict:
Action: Delete the broken stub store file located at /ext/client/src/features/settings/store/settingsStore.ts.
Action: Perform a global search for from '@/features/settings/store/settingsStore' and replace all (approximately 53) incorrect import paths to point to the one correct, working store: from '@/store/settingsStore'.
Reference: CODEBASE_PARTIAL_REFACTORS_ANALYSIS.md, SETTINGS_SYSTEM_ANALYSIS.md, SETTINGS_GUIDE.md.
Re-enable Physics Controls:
Action: In the file /ext/client/src/features/physics/components/PhysicsEngineControls.tsx, remove the hardcoded const settings = null; and the stub const updatePhysics = async (update: any) => {};.
Action: Connect the component to the now-correctly-imported useSettingsStore hook to read settings and call the updateSettings (or updatePhysics) function.
Reference: CODEBASE_PARTIAL_REFACTORS_ANALYSIS.md, SETTINGS_SYSTEM_ANALYSIS.md.
Verify Full Physics Parameter Flow:
Action: Confirm that changes made in the UI's physics controls are successfully sent to the backend via the POST /api/settings REST endpoint.
Action: Verify the backend correctly propagates these settings from PhysicsSettings -> SimulationParams -> SimParams and finally to the GPUComputeActor and the unified CUDA kernel.
Action: Test the physics simulation to ensure it is stable, responds to UI changes, and that the fixes for nodes collapsing and incorrect GPU initialization are working as intended.
Reference: PHYSICS_PARAMETERS_FIX.md, CORRECTED_SETTINGS_GPU_FLOW.md, NODE_COLLAPSE_FIX.md.
Priority 2: High - Fix Core Architecture and Complete Stubs
These tasks fix major architectural flaws and complete key features that are currently disabled or incomplete.
Correct Frontend MCP Architecture:
Action: Remove all direct MCP WebSocket connection logic from the frontend, as it is architecturally incorrect.
Action: Delete the service file /client/src/features/bots/services/MCPWebSocketService.ts.
Action: Refactor BotsVisualization.tsx and related components to fetch agent metadata exclusively through the /api/bots/* REST endpoints. The frontend should not connect to MCP.
Reference: frontend-mcp-issue.md, mcp-integration.md.
Correct Backend MCP/Agent Connection:
Action: Ensure the backend's EnhancedClaudeFlowActor connects to the Claude Flow service (powerdev or multi-agent-container) exclusively via WebSocket, as specified in the corrected architecture documents. Disable and remove any fallback stdio or TCP logic.
Action: Fix and re-enable the BotsClient connection mentioned as "DISABLED" in main.rs. Ensure it uses the correct WebSocket protocol to communicate with the EnhancedClaudeFlowActor and the MCP service.
Action: Remove the mock data generation for agents and switch to using live data from the now-functional MCP connection.
Reference: mcp_connection.md, claude-flow-actor.md, CODEBASE_PARTIAL_REFACTORS_ANALYSIS.md.
Complete Backend Service Stubs:
Action: In agent_visualization_processor.rs, implement the //TODOs to fetch real CPU/memory usage from system metrics and other agent data from the MCP connection.
Action: In speech_service.rs, implement the OpenAI provider for Text-to-Speech and Speech-to-Text.
Action: In health_handler.rs, replace placeholder status checks with actual diagnostics for core services (GPU, MCP connection, Database, etc.).
Action: In edge_generation.rs, replace placeholder indices with logic to generate edges based on actual graph data relationships.
Reference: CODEBASE_PARTIAL_REFACTORS_ANALYSIS.md.
Priority 3: Medium - Refactoring and Code Cleanup
These tasks improve code health, maintainability, and performance by removing obsolete code.
Finalize CUDA Kernel Consolidation:
Action: Delete all legacy CUDA kernel (.cu) and compiled PTX (.ptx) files, keeping only the single unified kernel: visionflow_unified.cu and its compiled PTX.
Reference: CUDA_CONSOLIDATION_COMPLETE.md, UNIFIED_CUDA_COMPLETION.md.
Remove Deprecated Rust Modules:
Action: Delete the entire deprecated advanced_gpu_compute.rs module.
Action: Refactor any remaining code that references it to use unified_gpu_compute.rs exclusively.
Action: Ensure the Array of Structures (AoS) to Structure of Arrays (SoA) data conversion is handled correctly by the unified module, permanently resolving the issue from KERNEL_PARAMETER_FIX.md.
Reference: CODEBASE_PARTIAL_REFACTORS_ANALYSIS.md, UNIFIED_KERNEL_MIGRATION_COMPLETE.md.
Clean Up Legacy Settings Code:
Action: Once the settings system is confirmed stable, remove the legacy flat-field migration code from /src/config/mod.rs and any backward-compatibility layers in the frontend's settingsStore.ts.
Reference: CODEBASE_PARTIAL_REFACTORS_ANALYSIS.md.
