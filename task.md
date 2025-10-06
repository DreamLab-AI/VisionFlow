1. Architectural Anomalies & Duplicate Systems
This is the most significant area of concern, indicating potential architectural drift, incomplete refactoring, or multiple developers implementing similar features independently.
Multiple WebSocket Services: The project contains at least four overlapping WebSocket implementations, which creates confusion and maintenance overhead.
client/src/services/WebSocketService.ts: A generic, legacy-style WebSocket service which you should CAREFULLY remove only after confirming nothing else depends on it.

client/src/services/EnhancedWebSocketService.ts: A newer, more feature-rich implementation that seems intended to replace the legacy one but hasn't fully, complete this migration.

client/src/services/VoiceWebSocketService.ts: A specialized service for audio, which could potentially be a module within a unified WebSocket service rather than a standalone class.
You should verify if this is still needed given modern WebRTC capabilities. We have throttling of the data in response to voice interactions, but this might be confused with the actual voice data stream which is handled by WebRTC. check and refine and align the architecture here.


client/src/features/bots/services/BotsWebSocketIntegration.ts: Another specialized service that also contains deprecated methods and flags (useRestPolling), indicating a past or ongoing architectural shift. ensure this is fully migrated.

Recommendation: Consolidate into a single, modular WebSocket service (like EnhancedWebSocketService) that can be extended with plugins or handlers for different features (bots, voice, etc.).

Multiple Authentication Systems: There appear to be two separate authentication mechanisms.
client/src/features/auth/initializeAuthentication.ts: A standalone AuthenticationManager class that seems incomplete or abandoned. It uses a generic auth_token in localStorage and contains mock refresh logic. We use nostr for auth, so this seems obsolete.

client/src/services/nostrAuthService.ts & client/src/services/api/authInterceptor.ts: A more complete and integrated system using Nostr (NIP-07) for authentication, which automatically injects headers into API requests via the UnifiedApiClient.

Recommendation: Remove the unused AuthenticationManager in initializeAuthentication.ts to avoid confusion and rely solely on the Nostr-based implementation.

Parallel Vircadia Multi-User System: The client/src/services/vircadia/ directory contains a complete, parallel system for multi-user XR functionality, including its own client core (VircadiaClientCore.ts), entity management (EntitySyncManager.ts), and avatar system (AvatarManager.ts). While configured in .env.example, its integration with the primary "Bots" and "Graph" visualization systems is unclear, suggesting it may be a separate, poorly integrated feature or a duplicate effort. We should take a look around this code and build a plan for full integration as it's certainly needed. This element will require web searching. Document findings into the relevant section of the docs corpus along with detailed plans.


2. Development Artifacts, Mocks, and Stubs
These items are remnants of development or testing and should be cleaned up.
Backup File:
File: client/src/features/visualisation/components/IntegratedControlPanel.tsx.backup
Issue: This is a backup file left in the source tree. It should be removed.
Developer Overrides:


Mock Data Files:
Files: client/src/features/bots/services/mockAgentData.ts, client/src/features/bots/services/mockDataAdapter.ts remove all mocks.

Test-Specific Components:
File: client/src/features/graph/components/GraphCanvasTestMode.tsx
Issue: This component is designed for testing environments where WebGL is unavailable. The logic in GraphCanvasWrapper.tsx correctly isolates it. it should be removed.

Console Logs in Configuration:
File: client/vite.config.ts
Issue: The proxy configuration contains console.log statements for debugging requests and responses. These should be removed.

3. Hardcoded Variables & Magic Numbers

File: client/src/app/components/NarrativeGoldminePanel.tsx
src="https://narrativegoldmine.com//#/graph" is a hardcoded URL. this whole website opening system is quite old. When a node is double clicked on we should open the webpage as a new tab, not in an iframe inside the app. This is a security risk and a poor user experience.


The security check url.hostname.includes('narrativegoldmine.com') uses a hardcoded domain, which is correct for now but should be flagged for future configurability.

File: client/src/utils/iframeCommunication.ts replace with the new system for opening webpages in a new tab.


File: client/src/features/bots/components/BotsVisualizationFixed.tsx - Contains numerous magic numbers for scaling, animation speeds, and color calculations (e.g., lerpFactor = 0.15, pulseSpeed = 2 * tokenMultiplier * healthMultiplier). These are a major problem and should be flagged in the code for later refactoring.


File: client/src/features/visualisation/components/HolographicDataSphere.tsx - This file is filled with hardcoded numeric values for geometry, colors, opacity, and animation parameters.
Recommendation: Replace these numbers with named constants from the settings system for the hologram. There should be analogous settings for each hard coded value. Just make sure that each variable gets assigned to one of the disconnected settings variables in the settings system. This will make it easy to adjust the hologram's appearance and behavior without diving into the code. We can update the names in the UX later. Don't do that yet as the settings management system is brittle.

4. Other Technical Debt
Disabled Testing Framework:
File: client/package.json
Issue: All test scripts are disabled with the message: "echo 'Testing disabled due to supply chain attack - see SECURITY_ALERT.md'". This is a critical security and quality assurance issue. The block-test-packages.cjs script is a temporary workaround that has become technical debt.
Recommendation: This is the highest priority issue to resolve. The underlying security concern must be addressed, dependencies updated, and the testing framework re-enabled.


Legacy Binary Protocol Support:
File: client/src/types/binaryProtocol.ts
Issue: The code includes logic to handle a legacy PROTOCOL_V1 which had a known bug with node ID truncation. Supporting this legacy protocol adds complexity.
Recommendation: remove support for PROTOCOL_V1 entirely, simplifying the codebase and reducing potential bugs.

Deprecated Methods:
File: client/src/features/bots/services/BotsWebSocketIntegration.ts
Issue: Several methods like setPollingMode, startBotsGraphPolling, and requestInitialData are marked as @deprecated. remove them after confirming they are no longer in use.

Recommendation: Identify where these methods are still being used, refactor the code to use the new patterns (e.g., the useAgentPolling hook), and then remove the deprecated methods.
Outdated Documentation:

File: client/src/features/bots/docs/polling-system.md
Issue: The documentation describes a REST polling system. While this system exists, the architecture has evolved into a hybrid model with WebSockets. The documentation should be updated to reflect the current state, including the roles of both REST and WebSocket communication. legacy code should be removed