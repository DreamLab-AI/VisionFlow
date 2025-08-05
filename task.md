Here are high-level instructions for a coding agent to update the project documentation.
Overall Goal
The project's documentation has fallen significantly behind the codebase. Your task is to perform a comprehensive update to ensure all documentation accurately reflects the current architecture, component structure, and features. The most significant architectural changes to document are:
The migration of the Rust backend to an Actix Actor model, replacing the previous Arc<RwLock<T>> pattern for state management.
The introduction of a "Parallel Graphs" architecture, which allows for simultaneous visualization of the "Logseq" knowledge graph and the "VisionFlow" agent swarm, each with its own settings.
The full integration of the "Bots/VisionFlow/MCP" feature set, which is now a core part of the application.
Phase 1: Documentation Cleanup and Reorganization
Instruction: The docs/ directory is cluttered with numerous markdown files in its root. These files contain valuable but disorganized information about recent architectural changes. Your first task is to consolidate, de-duplicate, and move this information into the structured documentation directories (docs/architecture/, docs/client/, docs/server/).
Analyze and Consolidate:
Review all markdown files in the docs/ root (e.g., AGENTIC_FLOW_INTEGRATION.md, bots-implementation.md, claude-flow-*.md, VISIONFLOW_*.md, DOCKER_MCP_INTEGRATION.md, etc.).
Identify overlapping content. For example, there are multiple documents describing the bots/MCP integration and parallel graphs. Synthesize the information into a coherent whole.
Create new, consolidated documents or update existing ones in the appropriate subdirectories. For example:
Information about the bots/agent visualization should be moved into a new docs/client/features/bots.md and docs/server/features/bots.md.
High-level architectural concepts from AGENTIC_FLOW_INTEGRATION.md and VISIONFLOW_GPU_MIGRATION_ARCHITECTURE.md should be merged into docs/architecture/system-overview.md.
The two parallel-graphs.md files should be merged into a single, definitive version at docs/architecture/parallel-graphs.md and linked from the client/server docs.
Delete Old Files: Once the information has been successfully migrated into the structured documentation, delete the old markdown files from the docs/ root to eliminate redundancy and confusion.
Phase 2: Update Core Project and Architectural Documentation
Instruction: Update the main README.md and the highest-level architecture documents to reflect the project's current state and provide an accurate entry point for new developers.
Update README.md:
Project Name: Standardize the project name. The codebase and configuration suggest "VisionFlow" is the current primary name, not "LogseqXR". Update the title and references accordingly.
Modular Control Panel Section: The component client/src/app/TwoPaneLayout.tsx no longer exists. Update this section to refer to client/src/app/MainLayout.tsx and its use of react-resizable-panels. Correct the file paths for UI components (e.g., Tabs.tsx is now in features/design-system/components).
AR Features Section: The file client/src/features/xr/managers/xrInitializer.ts does not exist. Remove the reference and point to xrSessionManager.ts as the primary manager for XR sessions.
Diagrams: Briefly review the Mermaid diagrams. While they appear mostly correct, ensure component names like SettingsPanelRedesignOptimized are corrected to SettingsPanelRedesign to match the file system.
Update docs/architecture/system-overview.md:
This document is a good starting point. Ensure it clearly explains the parallel graph architecture (Logseq vs. VisionFlow) and the central role of the Actix Actor system in the backend.
Integrate the consolidated information from the root-level docs files to provide a complete and up-to-date overview.
Update docs/index.md:
Review the table of contents. Add new entries for the major features that are now documented (e.g., Bots/VisionFlow, Parallel Graphs). Ensure all links point to the correct, updated files.
Phase 3: Systematically Update Server Documentation
Instruction: The server documentation is severely outdated. Go through each file in docs/server/ and update it to reflect the migration to the Actix Actor model and the new settings structure.
docs/server/architecture.md (App State):
This is critically outdated. Rewrite the "Core Structure" section to explain that AppState now holds actor addresses (Addr<...Actor>) for managing state, not Arc<RwLock<T>>.
Update the AppState struct definition to match src/app_state.rs.
Explain that ClientManager is now an actor (ClientManagerActor).
Update the "Access Patterns" section to show examples of sending messages to actors (.send(...) or .do_send(...)) instead of locking a RwLock.
docs/server/models.md:
This document is also critically outdated. Rewrite it almost entirely.
Update the SimulationParams struct to match src/models/simulation_params.rs.
Replace the old settings documentation with the new multi-graph structure, explaining AppFullSettings, UserSettings, and UISettings. Reference the new visualisation.graphs.logseq and visualisation.graphs.visionflow structure.
Update the ProtectedSettings section to match the struct in src/models/protected_settings.rs.
Update the Metadata struct definition.
Remove all incorrect references to Arc<RwLock<T>> for state management and explain that actors are used instead.
docs/server/services.md:
Rewrite this document to describe services in the context of the actor model. GraphService is now GraphServiceActor, etc.
Remove the description of a monolithic AIService. Instead, create separate sections for RAGFlowService, PerplexityService, and SpeechService, explaining their individual roles.
Clarify that WhisperSttService does not exist as a separate service and that its functionality is handled within SpeechService.
docs/server/actors.md:
This document is a good foundation. Review it for accuracy.
Add a description for the new ClaudeFlowActor (EnhancedClaudeFlowActor), explaining its role in connecting to the MCP service and providing agent data.
Ensure the responsibilities and key messages for each actor are correct and up-to-date with the code in src/actors/.
docs/server/config.md:
Update the AppFullSettings struct definition to match src/config/mod.rs, paying close attention to the nested system and visualisation.graphs structures.
Explain how FeatureAccess is configured via environment variables as described in src/config/feature_access.rs.
Phase 4: Systematically Update Client Documentation
Instruction: The client documentation contains many incorrect file paths and outdated component names. Update the files in docs/client/ to reflect the current project structure and state management patterns.
docs/client/architecture.md & docs/client/components.md:
Correct all outdated file paths and component names.
TwoPaneLayout.tsx -> MainLayout.tsx
SettingsPanelRedesignOptimized.tsx -> SettingsPanelRedesign.tsx or SettingsPanelProgrammatic.tsx
api.ts -> apiService.ts
ui/* -> features/design-system/*
Remove references to non-existent files like xrInitializer.ts, SettingsObserver.ts, and eventEmitter.ts.
In components.md, update the WebSocketServiceInterface to show that onConnectionStatusChange receives a simple boolean, not an object.
docs/client/state.md:
This document is mostly accurate. Add a note clarifying that SettingsObserver.ts is no longer used and that Zustand's built-in subscription is the standard.
Confirm that the description of the settings validation (or lack thereof in the store) is correct.
docs/client/types.md:
Add a new section for the Bots/VisionFlow types, documenting BotsAgent, BotsEdge, and other related interfaces from client/src/features/bots/types/BotsTypes.ts.
docs/client/websocket.md:
Remove the reference to the non-existent binaryUtils.ts. Explain that decompression and parsing of binary data are handled within the graph.worker.ts.
Update the "Rate Limiting" section to clarify it refers to the server's dynamic update frequency and client-side processing throttle, not a strict message limit.
docs/client/xr.md:
Remove the reference to the non-existent xrInitializer.ts.
Update the "Settings Affecting XR Mode" section to use the correct setting paths from the new multi-graph structure (e.g., xr.enableHandTracking instead of xr.handTracking).
Phase 5: Final Review
Instruction: Perform a final pass over the entire docs directory to ensure consistency in terminology (e.g., "VisionFlow" vs. "LogseqXR"), check for broken links, and verify that all diagrams and code snippets are accurate.
