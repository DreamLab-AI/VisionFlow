# Undocumented Components - VisionFlow

This file lists all components that require documentation, prioritized by impact.

## CRITICAL - Block Onboarding & Production Debugging

### Actors (12 files - 52% missing)

#### High Impact
- [ ] `semantic_processor_actor.rs` - GPU integration point, visualization dependency
- [ ] `client_coordinator_actor.rs` - Client communication hub
- [ ] `metadata_actor.rs` - Core metadata management
- [ ] `task_orchestrator_actor.rs` - Task coordination
- [ ] `voice_commands.rs` - Voice feature implementation

#### Medium Impact
- [ ] `event_coordination.rs` - Event routing system
- [ ] `agent_monitor_actor.rs` - Agent health monitoring
- [ ] `multi_mcp_visualization_actor.rs` - MCP visualization
- [ ] `workspace_actor.rs` - Workspace operations
- [ ] `lifecycle.rs` - Actor lifecycle management
- [ ] `supervisor.rs` - Supervisor implementation
- [ ] `messaging/mod.rs` - Message passing infrastructure

**Total: 12 actors | Estimated effort: 60-80 hours**

---

### Handlers (15 files - 32% missing)

#### High Priority (API functionality undocumented)
- [ ] `semantic_handler.rs` - Semantic analysis endpoint
- [ ] `semantic_pathfinding_handler.rs` - Path finding algorithm
- [ ] `natural_language_query_handler.rs` - NLP query processing
- [ ] `schema_handler.rs` - Database schema operations
- [ ] `inference_handler.rs` - ML inference endpoints
- [ ] `validation_handler.rs` - Validation logic

#### Medium Priority (Infrastructure)
- [ ] `quic_transport_handler.rs` - QUIC protocol implementation
- [ ] `fastwebsockets_handler.rs` - WebSocket optimization
- [ ] `realtime_websocket_handler.rs` - Real-time updates
- [ ] `cypher_query_handler.rs` - Cypher database queries
- [ ] `collaborative_sync_handler.rs` - Collaborative features

#### Low Priority (Deprecated/Utilities)
- [ ] `solid_proxy_handler.rs` - Solid pod integration
- [ ] `workspace_handler.rs` - Workspace operations
- [ ] `pages_handler.rs` - Static pages
- [ ] `constraints_handler.rs` - Physics constraints

**Total: 15 handlers | Estimated effort: 40-50 hours**

---

## HIGH PRIORITY - Production Impact

### Critical Services (15 files)

**GPU Integration:**
- [ ] `src/gpu/` - CUDA implementation details, memory management, fallback logic
  - Missing: Kernel implementations, performance tuning, error handling

**Semantic Processing:**
- [ ] `semantic_analyzer.rs` - Graph analysis engine
- [ ] `semantic_type_registry.rs` - Type system implementation
- [ ] `semantic_pathfinding_service.rs` - Pathfinding algorithm

**Ontology Services (Logic Layer):**
- [ ] `ontology_reasoner.rs` - Reasoning algorithm
- [ ] `ontology_content_analyzer.rs` - Content analysis
- [ ] `ontology_converter.rs` - Format conversion
- [ ] `ontology_enrichment_service.rs` - Enrichment logic

**Data Services:**
- [ ] `graph_serialization.rs` - Serialization format and algorithm
- [ ] `natural_language_query_service.rs` - NLP implementation
- [ ] `jss_sync_service.rs` - JSS synchronization
- [ ] `jss_websocket_bridge.rs` - JSS WebSocket handling

**Estimated effort: 80-100 hours**

---

## MEDIUM PRIORITY - Feature Completeness

### Client Services (28 files - 93% missing)

**Core Infrastructure:**
- [ ] `SolidPodService.ts` - Solid pod integration
- [ ] `SpaceDriverService.ts` - Space driver implementation
- [ ] `remoteLogger.ts` - Remote logging service
- [ ] `interactionApi.ts` - Interaction API client

**Audio Services (3 files):**
- [ ] `AudioInputService.ts` - Audio input handling
- [ ] `AudioOutputService.ts` - Audio output streaming
- [ ] `AudioContextManager.ts` - Audio context lifecycle

**WebSocket/Network (3 files):**
- [ ] `VoiceWebSocketService.ts` - Voice WebSocket protocol
- [ ] `platformManager.ts` - Platform detection/management
- [ ] `authInterceptor.ts` - HTTP auth interceptor

**Vircadia Integration (7+ files):**
- [ ] `VircadiaClientCore.ts` - Core client implementation
- [ ] `GraphEntityMapper.ts` - Entity mapping
- [ ] `EntitySyncManager.ts` - Entity synchronization
- [ ] `ThreeJSAvatarRenderer.ts` - Avatar rendering
- [ ] `SpatialAudioManager.ts` - 3D audio
- [ ] `NetworkOptimizer.ts` - Network optimization
- [ ] `CollaborativeGraphSync.ts` - Collaborative sync
- [ ] `FeatureFlags.ts` - Feature flag system
- [ ] `AvatarManager.ts` - Avatar management
- [ ] `Quest3Optimizer.ts` - Meta Quest 3 optimization

**Bridge Services (2 files):**
- [ ] `GraphVircadiaBridge.ts` - Graph to Vircadia sync
- [ ] `BotsVircadiaBridge.ts` - Bots to Vircadia sync

**Estimated effort: 100-150 hours**

---

### React Hooks (20 files - 100% missing)

**Input/Control (4 hooks):**
- [ ] `useMouseControls.ts` - Mouse input handling
- [ ] `useKeyboardShortcuts.ts` - Keyboard shortcuts
- [ ] `useHeadTracking.ts` - Head tracking input
- [ ] `useOptimizedFrame.ts` - Frame optimization

**State/Features (6 hooks):**
- [ ] `useWorkspaces.ts` - Workspace management
- [ ] `useSelectiveSettingsStore.ts` - Settings state
- [ ] `useGraphSettings.ts` - Graph configuration
- [ ] `useControlCenterStatus.ts` - Status management
- [ ] `useAnalyticsControls.ts` - Analytics controls
- [ ] `useVoiceInteraction.ts` - Voice mode state

**Authentication/Storage (3 hooks):**
- [ ] `useSolidPod.ts` - Solid pod integration
- [ ] `useNostrAuth.ts` - Nostr authentication
- [ ] `useSolidResource.ts` - Solid resource handling

**Error/Logging (3 hooks):**
- [ ] `useErrorHandler.tsx` - Error handling
- [ ] `useWebSocketErrorHandler.ts` - WebSocket errors
- [ ] `useAnalytics.ts` - Analytics tracking

**UI/Integration (4 hooks):**
- [ ] `useToast.ts` - Toast notifications
- [ ] `useContainerSize.ts` - Container size tracking
- [ ] `useQuest3Integration.ts` - Meta Quest 3
- [ ] `useAutoBalanceNotifications.ts` - Physics notifications (partial - needs completion)

**Estimated effort: 40-60 hours**

---

## LOW PRIORITY - Completeness

### Remaining Services (20+ files)

- [ ] `agent_visualization_processor.rs`
- [ ] `agent_visualization_protocol.rs`
- [ ] `bots_client.rs`
- [ ] `edge_classifier.rs`
- [ ] `edge_generation.rs`
- [ ] `empty_graph_check.rs`
- [ ] `multi_mcp_agent_discovery.rs`
- [ ] `mcp_relay_manager.rs`
- [ ] `real_mcp_integration_bridge.rs`
- [ ] `settings_watcher.rs`
- [ ] `speech_voice_integration.rs`
- [ ] `streaming_sync_service.rs`
- [ ] `topology_visualization_engine.rs`
- [ ] `voice_context_manager.rs`
- [ ] `voice_tag_manager.rs`
- [ ] `owl_validator.rs` (logic - stubs exist)
- [ ] `nostr_service.rs` (logic details)
- [ ] `speech_service.rs` (logic details)
- [ ] `ragflow_service.rs` (logic details)
- [ ] `perplexity_service.rs` (logic details)

**Estimated effort: 60-80 hours**

---

## PROTOCOLS & SPECS (7 files - 87% missing)

- [ ] `binary_settings_protocol.rs` - Settings binary format
- [ ] Speech Protocol Details - In `speech_socket_handler`
- [ ] Voice Protocol Details - In `voice_commands.rs`
- [ ] MCP Protocol Extensions - In `mcp_relay_handler`
- [ ] Graph Delta Format - In `socket_flow_handler`
- [ ] Settings Delta Format - In `websocket_settings_handler`
- [ ] Nostr Integration Protocol - In `nostr_handler`

**Estimated effort: 30-40 hours**

---

## ARCHITECTURAL DOCUMENTATION (Missing)

**Cross-Component Patterns:**
- [ ] Client initialization sequence
- [ ] Hook composition patterns
- [ ] State management architecture
- [ ] Message flow diagrams
- [ ] Actor communication patterns
- [ ] Service dependency graph
- [ ] Client-server synchronization strategy
- [ ] Voice pipeline flow
- [ ] Vircadia integration flow

**Estimated effort: 40-50 hours**

---

## DEPRECATION CLEANUP

**Files to Remove (No Documentation):**
- [ ] `settings_handler.rs.bak` - Backup file
- [ ] `settings_handler.rs.temp` - Temporary file
- [ ] `pipeline_admin_handler.rs` - DEPRECATED (consolidate to admin handlers)
- [ ] `settings_handler.rs` - DEPRECATED (migrate to `api_handler::settings`)
- [ ] `graph_state_handler_refactored.rs` - Parallel version (consolidate)
- [ ] `owl_validator_stubs.rs` - Incomplete stubs

**Migration Documentation Required:**
- [ ] Settings handler migration guide
- [ ] Pipeline admin consolidation guide
- [ ] Graph state handler consolidation guide

---

## SUMMARY BY COMPONENT TYPE

| Component Type | Count | Documented | Coverage | Priority |
|---|---|---|---|---|
| Actors | 23 | 6 | 26% | CRITICAL |
| Handlers | 47 | 32 | 68% | HIGH |
| Services | 50+ | 15 | 30% | HIGH |
| Protocols | 8 | 1 | 13% | MEDIUM |
| Client Services | 30 | 2 | 7% | HIGH |
| Client Hooks | 20 | 0 | 0% | HIGH |
| **TOTAL** | **178+** | **56** | **31%** | **URGENT** |

---

## EFFORT ESTIMATION

| Category | Hours | Priority |
|---|---|---|
| Actors (12 remaining) | 60-80 | CRITICAL |
| Handlers (15 remaining) | 40-50 | HIGH |
| Services (15 critical) | 80-100 | HIGH |
| Client Services (28) | 100-150 | MEDIUM |
| Client Hooks (20) | 40-60 | MEDIUM |
| Protocols (7) | 30-40 | MEDIUM |
| Architecture Docs | 40-50 | HIGH |
| Cleanup/Migration | 20-30 | MEDIUM |
| **TOTAL** | **410-560 hours** | |
| **Full-Time Team (40hrs/week)** | **10-14 weeks** | |
| **Part-Time (10hrs/week)** | **41-56 weeks** | |

---

**Report Generated:** 2025-12-30
**Audit Tool:** Code Coverage Validator
**Status:** Complete

