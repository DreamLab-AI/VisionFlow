We should refactor the decument base (destructively, migrating completely in one shot) to reflect the current codebase.

There may be partial migrations and refactors in the code, and these should be clearly identified in their own section. The document refactor will provide us the primary source of truth on the state of the code.


This is the code

> tree src
src
├── actors
│   ├── agent_monitor_actor.rs
│   ├── client_coordinator_actor.rs
│   ├── gpu
│   │   ├── anomaly_detection_actor.rs
│   │   ├── clustering_actor.rs
│   │   ├── constraint_actor.rs
│   │   ├── cuda_stream_wrapper.rs
│   │   ├── force_compute_actor.rs
│   │   ├── gpu_manager_actor.rs
│   │   ├── gpu_resource_actor.rs
│   │   ├── mod.rs
│   │   ├── ontology_constraint_actor.rs
│   │   ├── shared.rs
│   │   └── stress_majorization_actor.rs
│   ├── graph_actor.rs
│   ├── graph_messages.rs
│   ├── graph_service_supervisor.rs
│   ├── graph_state_actor.rs
│   ├── messages.rs
│   ├── metadata_actor.rs
│   ├── mod.rs
│   ├── multi_mcp_visualization_actor.rs
│   ├── ontology_actor.rs
│   ├── optimized_settings_actor.rs
│   ├── physics_orchestrator_actor.rs
│   ├── protected_settings_actor.rs
│   ├── semantic_processor_actor.rs
│   ├── supervisor.rs
│   ├── task_orchestrator_actor.rs
│   ├── voice_commands.rs
│   └── workspace_actor.rs
├── adapters
│   ├── actor_graph_repository.rs
│   ├── gpu_semantic_analyzer.rs
│   ├── mod.rs
│   ├── sqlite_knowledge_graph_repository.rs
│   ├── sqlite_ontology_repository.rs
│   ├── sqlite_settings_repository.rs
│   └── whelk_inference_engine.rs
├── application
│   ├── graph
│   │   ├── mod.rs
│   │   ├── queries.rs
│   │   └── tests
│   │       ├── mod.rs
│   │       └── query_handler_tests.rs
│   ├── knowledge_graph
│   │   ├── directives.rs
│   │   ├── mod.rs
│   │   └── queries.rs
│   ├── mod.rs
│   ├── ontology
│   │   ├── directives.rs
│   │   ├── mod.rs
│   │   └── queries.rs
│   └── settings
│       ├── directives.rs
│       ├── mod.rs
│       └── queries.rs
├── app_state.rs
├── bin
│   ├── generate_types.rs
│   ├── test_mcp_connection.rs
│   └── test_tcp_connection_fixed.rs
├── client
│   ├── mcp_tcp_client.rs
│   ├── mod.rs
│   └── settings_cache_client.ts
├── config
│   ├── dev_config.rs
│   ├── feature_access.rs
│   ├── mod.rs
│   └── path_access.rs
├── errors
│   └── mod.rs
├── gpu
│   ├── dynamic_buffer_manager.rs
│   ├── hybrid_sssp
│   │   ├── adaptive_heap.rs
│   │   ├── communication_bridge.rs
│   │   ├── gpu_kernels.rs
│   │   ├── mod.rs
│   │   └── wasm_controller.rs
│   ├── mod.rs
│   ├── streaming_pipeline.rs
│   └── visual_analytics.rs
├── handlers
│   ├── api_handler
│   │   ├── analytics
│   │   │   ├── anomaly.rs
│   │   │   ├── clustering.rs
│   │   │   ├── community.rs
│   │   │   ├── mod.rs
│   │   │   ├── real_gpu_functions.rs
│   │   │   └── websocket_integration.rs
│   │   ├── bots
│   │   │   └── mod.rs
│   │   ├── files
│   │   │   └── mod.rs
│   │   ├── graph
│   │   │   ├── mod.rs
│   │   │   └── mod.rs.backup
│   │   ├── mod.rs
│   │   ├── ontology
│   │   │   └── mod.rs
│   │   ├── quest3
│   │   │   └── mod.rs
│   │   ├── settings_ws.rs
│   │   └── visualisation
│   │       └── mod.rs
│   ├── bots_handler.rs
│   ├── bots_visualization_handler.rs
│   ├── client_log_handler.rs
│   ├── client_messages_handler.rs
│   ├── clustering_handler.rs
│   ├── consolidated_health_handler.rs
│   ├── constraints_handler.rs
│   ├── graph_export_handler.rs
│   ├── graph_state_handler.rs
│   ├── mcp_relay_handler.rs
│   ├── mod.rs
│   ├── multi_mcp_websocket_handler.rs
│   ├── nostr_handler.rs
│   ├── ontology_handler.rs
│   ├── pages_handler.rs
│   ├── perplexity_handler.rs
│   ├── ragflow_handler.rs
│   ├── realtime_websocket_handler.rs
│   ├── settings_handler.rs
│   ├── settings_validation_fix.rs
│   ├── socket_flow_handler.rs
│   ├── speech_socket_handler.rs
│   ├── tests
│   │   ├── mod.rs
│   │   └── settings_tests.rs
│   ├── utils.rs
│   ├── validation_handler.rs
│   ├── websocket_settings_handler.rs
│   └── workspace_handler.rs
├── lib.rs
├── main.rs
├── middleware
│   ├── mod.rs
│   └── timeout.rs
├── models
│   ├── constraints.rs
│   ├── edge.rs
│   ├── graph_export.rs
│   ├── graph.rs
│   ├── graph_types.rs
│   ├── metadata.rs
│   ├── mod.rs
│   ├── node.rs
│   ├── pagination.rs
│   ├── protected_settings.rs
│   ├── ragflow_chat.rs
│   ├── simulation_params.rs
│   ├── user_settings.rs
│   └── workspace.rs
├── ontology
│   ├── actors
│   │   └── mod.rs
│   ├── mod.rs
│   ├── parser
│   │   ├── assembler.rs
│   │   ├── converter.rs
│   │   ├── mod.rs
│   │   └── parser.rs
│   ├── physics
│   │   └── mod.rs
│   └── services
│       ├── mod.rs
│       └── owl_validator.rs
├── performance
│   └── settings_benchmark.rs
├── physics
│   ├── integration_tests.rs
│   ├── mod.rs
│   ├── ontology_constraints.rs
│   ├── semantic_constraints.rs
│   └── stress_majorization.rs
├── ports
│   ├── gpu_semantic_analyzer.rs
│   ├── graph_repository.rs
│   ├── inference_engine.rs
│   ├── knowledge_graph_repository.rs
│   ├── mod.rs
│   ├── ontology_repository.rs
│   ├── physics_simulator.rs
│   ├── semantic_analyzer.rs
│   └── settings_repository.rs
├── protocols
│   └── binary_settings_protocol.rs
├── services
│   ├── agent_visualization_processor.rs
│   ├── agent_visualization_protocol.rs
│   ├── bots_client.rs
│   ├── database_service.rs
│   ├── edge_generation.rs
│   ├── empty_graph_check.rs
│   ├── file_service.rs
│   ├── github
│   │   ├── api.rs
│   │   ├── config.rs
│   │   ├── content_enhanced.rs
│   │   ├── mod.rs
│   │   ├── pr.rs
│   │   └── types.rs
│   ├── github_sync_service.rs
│   ├── graph_serialization.rs
│   ├── local_markdown_sync.rs
│   ├── management_api_client.rs
│   ├── mcp_relay_manager.rs
│   ├── mod.rs
│   ├── multi_mcp_agent_discovery.rs
│   ├── nostr_service.rs
│   ├── owl_validator.rs
│   ├── parsers
│   │   ├── knowledge_graph_parser.rs
│   │   ├── mod.rs
│   │   └── ontology_parser.rs
│   ├── perplexity_service.rs
│   ├── ragflow_service.rs
│   ├── real_mcp_integration_bridge.rs
│   ├── semantic_analyzer.rs
│   ├── settings_broadcast.rs
│   ├── settings_service.rs
│   ├── settings_watcher.rs
│   ├── speech_service.rs
│   ├── speech_voice_integration.rs
│   ├── topology_visualization_engine.rs
│   ├── voice_context_manager.rs
│   └── voice_tag_manager.rs
├── telemetry
│   ├── agent_telemetry.rs
│   ├── mod.rs
│   └── test_logging.rs
├── tests
│   └── voice_tag_integration_test.rs
├── types
│   ├── claude_flow.rs
│   ├── mcp_responses.rs
│   ├── mod.rs
│   ├── speech.rs
│   └── vec3.rs
└── utils
    ├── actor_timeout.rs
    ├── advanced_logging.rs
    ├── async_improvements.rs
    ├── audio_processor.rs
    ├── auth.rs
    ├── binary_protocol.rs
    ├── client_message_extractor.rs
    ├── cuda_error_handling.rs
    ├── dynamic_grid.cu
    ├── dynamic_grid.ptx
    ├── edge_data.rs
    ├── gpu_aabb_reduction.cu
    ├── gpu_aabb_reduction.ptx
    ├── gpu_clustering_kernels.cu
    ├── gpu_clustering_kernels.ptx
    ├── gpu_compute_tests.rs
    ├── gpu_diagnostics.rs
    ├── gpu_landmark_apsp.cu
    ├── gpu_landmark_apsp.ptx
    ├── gpu_memory.rs
    ├── gpu_safety.rs
    ├── handler_commons.rs
    ├── logging.rs
    ├── mcp_connection.rs
    ├── mcp_tcp_client.rs
    ├── memory_bounds.rs
    ├── mod.rs
    ├── network
    │   ├── circuit_breaker.rs
    │   ├── connection_pool.rs
    │   ├── graceful_degradation.rs
    │   ├── health_check.rs
    │   ├── mod.rs
    │   ├── retry.rs
    │   └── timeout.rs
    ├── ontology_constraints.cu
    ├── ptx
    │   ├── dynamic_grid.ptx
    │   ├── gpu_aabb_reduction.ptx
    │   ├── gpu_clustering_kernels.ptx
    │   ├── gpu_landmark_apsp.ptx
    │   ├── ontology_constraints.ptx
    │   ├── sssp_compact.ptx
    │   ├── visionflow_unified.ptx
    │   └── visionflow_unified_stability.ptx
    ├── ptx.rs
    ├── ptx_tests.rs
    ├── realtime_integration.rs
    ├── resource_monitor.rs
    ├── session_log_monitor.rs
    ├── socket_flow_constants.rs
    ├── socket_flow_messages.rs
    ├── sssp_compact.cu
    ├── sssp_compact.ptx
    ├── standard_websocket_messages.rs
    ├── unified_gpu_compute.rs
    ├── validation
    │   ├── errors.rs
    │   ├── middleware.rs
    │   ├── mod.rs
    │   ├── position_validator.rs
    │   ├── rate_limit.rs
    │   ├── sanitization.rs
    │   └── schemas.rs
    ├── visionflow_unified.cu
    ├── visionflow_unified.ptx
    ├── visionflow_unified_stability.cu
    ├── visionflow_unified_stability.ptx
    └── websocket_heartbeat.rs

47 directories, 270 files
> tree client/src
client/src
├── api
│   ├── analyticsApi.ts
│   ├── batchUpdateApi.ts
│   ├── exportApi.ts
│   ├── optimizationApi.ts
│   ├── settingsApi.ts
│   └── workspaceApi.ts
├── app
│   ├── AppInitializer.tsx
│   ├── App.tsx
│   ├── components
│   │   ├── ConversationPane.tsx
│   │   └── NarrativeGoldminePanel.tsx
│   ├── MainLayout.tsx
│   └── main.tsx
├── components
│   ├── AuthGatedVoiceButton.tsx
│   ├── AuthGatedVoiceIndicator.tsx
│   ├── BrowserSupportWarning.tsx
│   ├── ConnectionWarning.tsx
│   ├── DebugControlPanel.tsx
│   ├── ErrorBoundary.tsx
│   ├── error-handling
│   │   └── index.ts
│   ├── ErrorNotification.tsx
│   ├── KeyboardShortcutsModal.tsx
│   ├── settings
│   │   └── VircadiaSettings.tsx
│   ├── SettingsRetryStatus.tsx
│   ├── SpaceMouseStatus.tsx
│   ├── VoiceButton.tsx
│   ├── VoiceIndicator.tsx
│   └── VoiceStatusIndicator.tsx
├── contexts
│   ├── ApplicationModeContext.tsx
│   ├── VircadiaBridgesContext.tsx
│   └── VircadiaContext.tsx
├── features
│   ├── analytics
│   │   ├── components
│   │   │   ├── SemanticClusteringControls.tsx
│   │   │   └── ShortestPathControls.tsx
│   │   ├── index.ts
│   │   └── store
│   │       ├── analyticsStore.test.ts
│   │       └── analyticsStore.ts
│   ├── auth
│   ├── bots
│   │   ├── components
│   │   │   ├── ActivityLogPanel.tsx
│   │   │   ├── AgentDetailPanel.tsx
│   │   │   ├── AgentPollingStatus.tsx
│   │   │   ├── AgentTelemetryStream.tsx
│   │   │   ├── BotsControlPanel.tsx
│   │   │   ├── BotsVisualizationDebugInfo.tsx
│   │   │   ├── BotsVisualizationFixed.tsx
│   │   │   ├── index.ts
│   │   │   ├── MultiAgentInitializationPrompt.tsx
│   │   │   ├── ProgrammaticMonitorControl.tsx
│   │   │   └── SystemHealthPanel.tsx
│   │   ├── config
│   │   │   └── pollingConfig.ts
│   │   ├── contexts
│   │   │   └── BotsDataContext.tsx
│   │   ├── docs
│   │   │   └── polling-system.md
│   │   ├── hooks
│   │   │   ├── useAgentPolling.ts
│   │   │   └── useBotsWebSocketIntegration.ts
│   │   ├── index.ts
│   │   ├── services
│   │   │   ├── AgentPollingService.ts
│   │   │   ├── BotsWebSocketIntegration.ts
│   │   │   └── ConfigurationMapper.ts
│   │   ├── types
│   │   │   └── BotsTypes.ts
│   │   └── utils
│   │       ├── pollingPerformance.ts
│   │       └── programmaticMonitor.ts
│   ├── command-palette
│   │   ├── CommandRegistry.ts
│   │   ├── components
│   │   │   └── CommandPalette.tsx
│   │   ├── defaultCommands.ts
│   │   ├── hooks
│   │   │   └── useCommandPalette.ts
│   │   ├── index.ts
│   │   └── types.ts
│   ├── design-system
│   │   ├── animations.ts
│   │   ├── components
│   │   │   ├── Badge.tsx
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── Collapsible.tsx
│   │   │   ├── Dialog.tsx
│   │   │   ├── index.ts
│   │   │   ├── Input.tsx
│   │   │   ├── Label.tsx
│   │   │   ├── LoadingSkeleton.tsx
│   │   │   ├── LoadingSpinner.tsx
│   │   │   ├── Modal.tsx
│   │   │   ├── Progress.tsx
│   │   │   ├── RadioGroup.tsx
│   │   │   ├── ScrollArea.tsx
│   │   │   ├── SearchInput.tsx
│   │   │   ├── Select.tsx
│   │   │   ├── Separator.tsx
│   │   │   ├── Slider.tsx
│   │   │   ├── Switch.tsx
│   │   │   ├── Tabs.tsx
│   │   │   ├── Textarea.tsx
│   │   │   ├── Toast.tsx
│   │   │   └── Tooltip.tsx
│   │   └── patterns
│   │       └── MarkdownRenderer.tsx
│   ├── graph
│   │   ├── components
│   │   │   ├── FlowingEdges.tsx
│   │   │   ├── GraphCanvas.tsx
│   │   │   ├── GraphCanvasWrapper.tsx
│   │   │   ├── GraphManager_EventHandlers.ts
│   │   │   ├── GraphManager.tsx
│   │   │   ├── GraphViewport.tsx
│   │   │   ├── MetadataShapes.tsx
│   │   │   ├── NodeShaderToggle.tsx
│   │   │   └── PerformanceIntegration.tsx
│   │   ├── hooks
│   │   │   └── useGraphEventHandlers.ts
│   │   ├── innovations
│   │   │   └── index.ts
│   │   ├── managers
│   │   │   ├── graphDataManager.ts
│   │   │   └── graphWorkerProxy.ts
│   │   ├── services
│   │   │   ├── advancedInteractionModes.ts
│   │   │   ├── aiInsights.ts
│   │   │   ├── graphAnimations.ts
│   │   │   ├── graphComparison.ts
│   │   │   └── graphSynchronization.ts
│   │   ├── types
│   │   │   └── graphTypes.ts
│   │   └── workers
│   │       └── graph.worker.ts
│   ├── help
│   │   ├── components
│   │   │   ├── HelpProvider.tsx
│   │   │   └── HelpTooltip.tsx
│   │   ├── HelpRegistry.ts
│   │   ├── settingsHelp.ts
│   │   └── types.ts
│   ├── onboarding
│   │   ├── components
│   │   │   ├── OnboardingEventHandler.tsx
│   │   │   ├── OnboardingOverlay.tsx
│   │   │   └── OnboardingProvider.tsx
│   │   ├── flows
│   │   │   └── defaultFlows.ts
│   │   ├── hooks
│   │   │   └── useOnboarding.ts
│   │   ├── index.ts
│   │   └── types.ts
│   ├── ontology
│   │   ├── components
│   │   │   ├── ConstraintGroupControl.tsx
│   │   │   ├── index.ts
│   │   │   ├── OntologyMetrics.tsx
│   │   │   ├── OntologyModeToggle.tsx
│   │   │   ├── OntologyPanel.tsx
│   │   │   └── ValidationStatus.tsx
│   │   ├── hooks
│   │   │   └── useOntologyWebSocket.ts
│   │   ├── index.ts
│   │   └── store
│   │       └── useOntologyStore.ts
│   ├── physics
│   │   └── components
│   │       ├── ConstraintBuilderDialog.tsx
│   │       ├── PhysicsEngineControls.tsx
│   │       └── PhysicsPresets.tsx
│   ├── settings
│   │   ├── components
│   │   │   ├── BackendUrlSetting.tsx
│   │   │   ├── control-panel-context.tsx
│   │   │   ├── FloatingSettingsPanel.tsx
│   │   │   ├── GraphSelector.tsx
│   │   │   ├── LazySettingsSections.tsx
│   │   │   ├── LocalStorageSettingControl.tsx
│   │   │   ├── panels
│   │   │   │   ├── AgentControlPanel.tsx
│   │   │   │   ├── DashboardControlPanel.tsx
│   │   │   │   ├── index.ts
│   │   │   │   ├── PerformanceControlPanel.tsx
│   │   │   │   ├── SettingsPanelRedesign.backup.tsx
│   │   │   │   ├── SettingsPanelRedesign.tsx
│   │   │   │   └── SettingsPanelRedesign.tsx.backup
│   │   │   ├── PresetSelector.tsx
│   │   │   ├── SettingControlComponent.tsx
│   │   │   ├── SettingsSearch.tsx
│   │   │   ├── SettingsSection.tsx
│   │   │   ├── UndoRedoControls.tsx
│   │   │   └── VirtualizedSettingsGroup.tsx
│   │   ├── config
│   │   │   ├── debugSettingsUIDefinition.ts
│   │   │   ├── settingsConfig.ts
│   │   │   ├── settings.ts
│   │   │   ├── settingsUIDefinition.ts
│   │   │   ├── viewportSettings.ts
│   │   │   └── widgetTypes.ts
│   │   ├── hooks
│   │   │   ├── useSettingsHistory.ts
│   │   │   └── useSettingsPerformance.ts
│   │   └── presets
│   │       └── qualityPresets.ts
│   ├── visualisation
│   │   ├── components
│   │   │   ├── AgentNodesLayer.tsx
│   │   │   ├── AutoBalanceIndicator.tsx
│   │   │   ├── CameraController.tsx
│   │   │   ├── ControlPanel
│   │   │   │   ├── BotsStatusPanel.tsx
│   │   │   │   ├── config.ts
│   │   │   │   ├── ControlPanelHeader.tsx
│   │   │   │   ├── index.ts
│   │   │   │   ├── RestoredGraphTabs.tsx
│   │   │   │   ├── settingsConfig.ts
│   │   │   │   ├── SettingsTabContent.tsx
│   │   │   │   ├── SimpleGraphTabs.tsx
│   │   │   │   ├── SpacePilotStatus.tsx
│   │   │   │   ├── SystemInfo.tsx
│   │   │   │   ├── TabNavigation.tsx
│   │   │   │   └── types.ts
│   │   │   ├── dialogs
│   │   │   │   ├── ExportFormatDialog.tsx
│   │   │   │   ├── PublishGraphDialog.tsx
│   │   │   │   ├── ShareLinkManager.tsx
│   │   │   │   └── ShareSettingsDialog.tsx
│   │   │   ├── HeadTrackedParallaxController.tsx
│   │   │   ├── HolographicDataSphere.tsx
│   │   │   ├── IntegratedControlPanel.tsx
│   │   │   ├── MetadataVisualizer.tsx
│   │   │   ├── SpacePilotButtonPanel.tsx
│   │   │   ├── SpacePilotConnectButton.tsx
│   │   │   ├── SpacePilotOrbitControlsIntegration.tsx
│   │   │   ├── SpacePilotSimpleIntegration.tsx
│   │   │   ├── tabs
│   │   │   │   ├── GraphAnalysisTab.tsx
│   │   │   │   ├── GraphExportTab.tsx
│   │   │   │   ├── GraphInteractionTab.tsx
│   │   │   │   ├── GraphOptimisationTab.tsx
│   │   │   │   └── GraphVisualisationTab.tsx
│   │   │   └── WireframeCloudMesh.tsx
│   │   ├── controls
│   │   │   └── SpacePilotController.ts
│   │   ├── effects
│   │   │   └── AtmosphericGlow.tsx
│   │   └── hooks
│   │       ├── bloomRegistry.ts
│   │       ├── useGraphInteraction.ts
│   │       ├── useNodeInteraction.ts
│   │       └── useSpacePilot.ts
│   └── workspace
│       └── components
│           └── WorkspaceManager.tsx
├── hooks
│   ├── useAnalyticsControls.ts
│   ├── useAnalytics.ts
│   ├── useAutoBalanceNotifications.ts
│   ├── useContainerSize.ts
│   ├── useErrorHandler.tsx
│   ├── useGraphSettings.ts
│   ├── useHeadTracking.ts
│   ├── useHybridSystemStatus.ts
│   ├── useKeyboardShortcuts.ts
│   ├── useMouseControls.ts
│   ├── useOptimizedFrame.ts
│   ├── useQuest3Integration.ts
│   ├── useSelectiveSettingsStore.ts
│   ├── useSettingsWebSocket.ts
│   ├── useToast.ts
│   ├── useVoiceInteraction.ts
│   ├── useWebSocketErrorHandler.ts
│   └── useWorkspaces.ts
├── immersive
│   ├── babylon
│   │   ├── BabylonScene.ts
│   │   ├── GraphRenderer.ts
│   │   ├── VircadiaSceneBridge.ts
│   │   ├── XRManager.ts
│   │   └── XRUI.ts
│   ├── components
│   │   └── ImmersiveApp.tsx
│   └── hooks
│       └── useImmersiveData.ts
├── rendering
│   ├── index.ts
│   ├── materials
│   │   ├── BloomStandardMaterial.ts
│   │   ├── HologramNodeMaterial.ts
│   │   └── index.ts
│   └── SelectiveBloom.tsx
├── services
│   ├── api
│   │   ├── authInterceptor.ts
│   │   ├── index.ts
│   │   ├── README.md
│   │   └── UnifiedApiClient.ts
│   ├── AudioContextManager.ts
│   ├── AudioInputService.ts
│   ├── AudioOutputService.ts
│   ├── BinaryWebSocketProtocol.ts
│   ├── bridges
│   │   ├── BotsVircadiaBridge.ts
│   │   └── GraphVircadiaBridge.ts
│   ├── interactionApi.ts
│   ├── nostrAuthService.ts
│   ├── platformManager.ts
│   ├── quest3AutoDetector.ts
│   ├── remoteLogger.ts
│   ├── SpaceDriverService.ts
│   ├── __tests__
│   │   └── BinaryWebSocketProtocol.test.ts
│   ├── vircadia
│   │   ├── AvatarManager.ts
│   │   ├── CollaborativeGraphSync.ts
│   │   ├── EntitySyncManager.ts
│   │   ├── FeatureFlags.ts
│   │   ├── GraphEntityMapper.ts
│   │   ├── NetworkOptimizer.ts
│   │   ├── Quest3Optimizer.ts
│   │   ├── SpatialAudioManager.ts
│   │   └── VircadiaClientCore.ts
│   ├── VoiceWebSocketService.ts
│   └── WebSocketService.ts
├── shaders
├── store
│   ├── autoSaveManager.ts
│   ├── multiUserStore.ts
│   ├── settingsRetryManager.ts
│   └── settingsStore.ts
├── styles
│   ├── base.css
│   ├── index.css
│   └── tailwind-utilities.css
├── telemetry
│   ├── AgentTelemetry.ts
│   ├── DebugOverlay.tsx
│   ├── index.ts
│   ├── README.md
│   └── useTelemetry.ts
├── types
│   ├── binaryProtocol.ts
│   ├── generated
│   │   └── settings.ts
│   ├── getalby-sdk.d.ts
│   ├── lucide-react.d.ts
│   ├── nip07.d.ts
│   ├── node-env.d.ts
│   ├── ragflowTypes.ts
│   ├── react-syntax-highlighter.d.ts
│   ├── tailwind-merge.d.ts
│   ├── webhid.d.ts
│   └── websocketTypes.ts
├── utils
│   ├── accessibility.ts
│   ├── baseLogger.ts
│   ├── BatchQueue.ts
│   ├── classNameUtils.ts
│   ├── clientDebugState.ts
│   ├── console.ts
│   ├── debugConfig.ts
│   ├── downloadHelpers.ts
│   ├── dualGraphOptimizations.ts
│   ├── dualGraphPerformanceMonitor.ts
│   ├── loggerConfig.ts
│   ├── settingsSearch.ts
│   ├── three-geometries.ts
│   └── validation.ts
└── xr
    └── vircadia
        ├── components
        ├── hooks
        ├── services
        └── types

89 directories, 296 files


and this it the document pack

> tree docs
docs
├── 00-INDEX.md
├── api
│   ├── 01-authentication.md
│   ├── 02-endpoints.md
│   └── 03-websocket.md
├── API.md
├── architecture
│   ├── 00-ARCHITECTURE-OVERVIEW.md
│   ├── 01-ports-design.md
│   ├── 02-adapters-design.md
│   ├── 03-cqrs-application-layer.md
│   ├── 04-database-schemas.md
│   ├── 05-schema-implementation-summary.md
│   ├── ARCHITECTURE_INDEX.md
│   ├── code-examples.md
│   ├── components
│   │   └── websocket-protocol.md
│   ├── core
│   │   ├── client.md
│   │   ├── server.md
│   │   └── visualization.md
│   ├── cqrs-migration.md
│   ├── event-flow-diagrams.md
│   ├── github-sync-service-design.md
│   ├── gpu
│   │   ├── communication-flow.md
│   │   ├── optimizations.md
│   │   └── README.md
│   ├── gpu-stability.md
│   ├── hexagonal-cqrs-architecture.md
│   ├── interface.md
│   ├── migration-strategy.md
│   ├── overview.md
│   ├── phase3-ports-complete.md
│   ├── README.md
│   ├── security.md
│   ├── system-overview.md
│   ├── vircadia-integration-analysis.md
│   ├── vircadia-react-xr-integration.md
│   ├── voice-webrtc-migration-plan.md
│   └── xr-immersive-system.md
├── ARCHITECTURE.md
├── archive
│   ├── archive-summary.md
│   ├── gpu-actor-handlers-analysis.md
│   ├── gpu-handlers-deletion-guide.md
│   ├── gpu-handlers-summary.txt
│   ├── legacy-code-audit.md
│   ├── settings-handlers-consolidation-plan.md
│   └── whelk-rs-configuration-report.md
├── cargo-check-logs
│   ├── cargo_check_all_features.log
│   ├── cargo_check_default.log
│   ├── cargo_check_gpu.log
│   ├── cargo_check_ontology.log
│   └── README.md
├── code-examples
├── concepts
│   ├── agentic-workers.md
│   ├── data-flow.md
│   ├── decisions
│   │   ├── adr-001-unified-api-client.md
│   │   ├── adr-003-code-pruning-2025-10.md
│   │   └── index.md
│   ├── gpu-compute.md
│   ├── index.md
│   ├── networking-and-protocols.md
│   ├── ontology-and-validation.md
│   ├── README.md
│   ├── security-model.md
│   └── system-architecture.md
├── contributing.md
├── DATABASE.md
├── deployment
│   ├── 01-docker-deployment.md
│   ├── 02-configuration.md
│   ├── 03-monitoring.md
│   ├── 04-backup-restore.md
│   └── vircadia-docker-deployment.md
├── developer-guide
│   ├── 01-development-setup.md
│   ├── 02-project-structure.md
│   ├── 03-architecture.md
│   ├── 04-adding-features.md
│   ├── 05-testing.md
│   └── 06-contributing.md
├── DEVELOPER_GUIDE.md
├── development
│   └── deployment.md
├── getting-started
│   ├── 01-installation.md
│   └── 02-first-graph-and-agents.md
├── guides
│   ├── agent-orchestration.md
│   ├── configuration.md
│   ├── deployment.md
│   ├── development-workflow.md
│   ├── extending-the-system.md
│   ├── index.md
│   ├── ontology-parser.md
│   ├── orchestrating-agents.md
│   ├── README.md
│   ├── security.md
│   ├── telemetry-logging.md
│   ├── testing-guide.md
│   ├── troubleshooting.md
│   ├── vircadia-multi-user-guide.md
│   ├── working-with-gui-sandbox.md
│   └── xr-setup.md
├── hexagonal-migration
│   ├── AUDIT_SUMMARY.md
│   ├── dependency-audit-report.md
│   ├── dependency-diagram.txt
│   └── dependency-map.json
├── implementation
│   └── whelk-integration-summary.md
├── index.md
├── migration
│   ├── 00-QUEEN-COORDINATION-BRIEF.md
│   ├── 01-AGENT-CODE-AUDITOR-BRIEF.md
│   ├── 02-AGENT-ARCHITECTURE-PLANNER-BRIEF.md
│   ├── 03-AGENT-TEST-ANALYZER-BRIEF.md
│   ├── 04-AGENT-DEPRECATION-MARKER-BRIEF.md
│   ├── 05-AGENT-MIGRATION-EXECUTOR-BRIEF.md
│   ├── 06-AGENT-INTEGRATION-VALIDATOR-BRIEF.md
│   ├── 07-AGENT-LEGACY-REMOVER-BRIEF.md
│   ├── QUEEN-STATUS-REPORT.md
│   └── README.md
├── multi-agent-docker
│   ├── ARCHITECTURE.md
│   ├── DOCKER-ENVIRONMENT.md
│   ├── docs
│   │   ├── API_REFERENCE.md
│   │   ├── ARCHITECTURE.md
│   │   ├── archived
│   │   │   ├── claude-flow-integration.md
│   │   │   ├── COMPLETE_VALIDATION_SUMMARY.md
│   │   │   ├── docker-cli-validation.md
│   │   │   ├── DOCKER_MCP_VALIDATION.md
│   │   │   ├── docker-memory-coordination-status.md
│   │   │   ├── DOCKER_OPENROUTER_VALIDATION.md
│   │   │   ├── FASTMCP_COMPLETE.md
│   │   │   ├── FASTMCP_INTEGRATION_STATUS.md
│   │   │   ├── FINAL_SDK_VALIDATION.md
│   │   │   ├── FINAL_SYSTEM_VALIDATION.md
│   │   │   ├── FINAL_VALIDATION_SUMMARY.md
│   │   │   ├── FIXES-APPLIED-STATUS.md
│   │   │   ├── FLOW-NEXUS-COMPLETE.md
│   │   │   ├── HOTFIX_1.1.7.md
│   │   │   ├── INTEGRATION_CONFIRMED.md
│   │   │   ├── MCP_CLI_TOOLS_VALIDATION.md
│   │   │   ├── MCP_INTEGRATION_SUCCESS.md
│   │   │   ├── MCP_PROXY_VALIDATION.md
│   │   │   ├── mcp-validation-summary.md
│   │   │   ├── MODEL_VALIDATION_REPORT.md
│   │   │   ├── ONNX_ENV_VARS.md
│   │   │   ├── ONNX_FINAL_REPORT.md
│   │   │   ├── ONNX_IMPLEMENTATION_COMPLETE.md
│   │   │   ├── ONNX_IMPLEMENTATION_SUMMARY.md
│   │   │   ├── ONNX_INTEGRATION.md
│   │   │   ├── ONNX_OPTIMIZATION_SUMMARY.md
│   │   │   ├── ONNX_PHI4_RESEARCH.md
│   │   │   ├── ONNX_RUNTIME_INTEGRATION_PLAN.md
│   │   │   ├── ONNX_SUCCESS_REPORT.md
│   │   │   ├── ONNX_VS_CLAUDE_QUALITY.md
│   │   │   ├── OPENROUTER-FIX-VALIDATION.md
│   │   │   ├── OPENROUTER_ISSUES_AND_FIXES.md
│   │   │   ├── OPENROUTER_PROXY_COMPLETE.md
│   │   │   ├── OPENROUTER-SUCCESS-REPORT.md
│   │   │   ├── OPENROUTER_VALIDATION_COMPLETE.md
│   │   │   ├── OPTIMIZATION_SUMMARY.md
│   │   │   ├── PACKAGE-COMPLETE.md
│   │   │   ├── PHI4_HYPEROPTIMIZATION_PLAN.md
│   │   │   ├── PROVIDER_INSTRUCTION_OPTIMIZATION.md
│   │   │   ├── PROXY_VALIDATION.md
│   │   │   ├── quick-wins-validation.md
│   │   │   ├── README.md
│   │   │   ├── README_SDK_VALIDATION.md
│   │   │   ├── README_V1.1.11.md
│   │   │   ├── RELEASE-NOTES-v1.1.13.md
│   │   │   ├── RELEASE-SUMMARY-v1.1.14-beta.1.md
│   │   │   ├── RESEARCH_COMPLETE.txt
│   │   │   ├── ROUTER_VALIDATION.md
│   │   │   ├── SDK_INTEGRATION_COMPLETE.md
│   │   │   ├── SDK-SETUP-COMPLETE.md
│   │   │   ├── TOOL_INSTRUCTION_ENHANCEMENT.md
│   │   │   ├── V1.1.10_VALIDATION.md
│   │   │   ├── V1.1.11_COMPLETE_VALIDATION.md
│   │   │   ├── V1.1.11_MCP_PROXY_FIX.md
│   │   │   ├── V1.1.14-BETA-READY.md
│   │   │   ├── VALIDATION_COMPLETE.md
│   │   │   ├── VALIDATION-RESULTS.md
│   │   │   └── VALIDATION_SUMMARY.md
│   │   ├── CONFIGURATION.md
│   │   ├── DEPLOYMENT.md
│   │   ├── GETTING_STARTED.md
│   │   ├── guides
│   │   │   ├── DESKTOP_ENVIRONMENT.md
│   │   │   ├── GPU_CONFIGURATION.md
│   │   │   ├── MCP_TOOLS.md
│   │   │   ├── MULTI_MODEL_ROUTER.md
│   │   │   ├── README.md
│   │   │   └── TASK_MANAGEMENT.md
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── reference
│   │   │   ├── DOCKER.md
│   │   │   ├── ENVIRONMENT_VARIABLES.md
│   │   │   ├── QUICK_REFERENCE.md
│   │   │   └── SCRIPTS.md
│   │   ├── releases
│   │   │   ├── HOTFIX-v1.2.1.md
│   │   │   ├── NPM-PUBLISH-GUIDE-v1.2.0.md
│   │   │   ├── PUBLISH-COMPLETE-v1.2.0.md
│   │   │   ├── README.md
│   │   │   └── RELEASE-v1.2.0.md
│   │   └── TROUBLESHOOTING.md
│   ├── GOALIE-INTEGRATION.md
│   ├── PORT-CONFIGURATION.md
│   ├── README.md
│   ├── TOOLS.md
│   └── TROUBLESHOOTING.md
├── quality_reports
│   └── warnings_baseline.txt
├── README.md
├── reference
│   ├── agents
│   │   ├── analysis
│   │   │   ├── code-analyzer.md
│   │   │   └── code-review
│   │   │       └── analyze-code-quality.md
│   │   ├── architecture
│   │   │   └── system-design
│   │   │       └── arch-system-design.md
│   │   ├── base-template-generator.md
│   │   ├── consensus
│   │   │   ├── byzantine-coordinator.md
│   │   │   ├── crdt-synchronizer.md
│   │   │   ├── gossip-coordinator.md
│   │   │   ├── index.md
│   │   │   ├── performance-benchmarker.md
│   │   │   ├── quorum-manager.md
│   │   │   ├── raft-manager.md
│   │   │   ├── README.md
│   │   │   └── security-manager.md
│   │   ├── conventions.md
│   │   ├── core
│   │   │   ├── coder.md
│   │   │   ├── index.md
│   │   │   ├── planner.md
│   │   │   ├── researcher.md
│   │   │   ├── reviewer.md
│   │   │   └── tester.md
│   │   ├── data
│   │   │   └── ml
│   │   │       └── data-ml-model.md
│   │   ├── development
│   │   │   └── backend
│   │   │       └── dev-backend-api.md
│   │   ├── devops
│   │   │   └── ci-cd
│   │   │       └── ops-cicd-github.md
│   │   ├── documentation
│   │   │   └── api-docs
│   │   │       └── docs-api-openapi.md
│   │   ├── github
│   │   │   ├── code-review-swarm.md
│   │   │   ├── github-modes.md
│   │   │   ├── index.md
│   │   │   ├── issue-tracker.md
│   │   │   ├── multi-repo-swarm.md
│   │   │   ├── pr-manager.md
│   │   │   ├── project-board-sync.md
│   │   │   ├── release-manager.md
│   │   │   ├── release-swarm.md
│   │   │   ├── repo-architect.md
│   │   │   ├── swarm-issue.md
│   │   │   ├── swarm-pr.md
│   │   │   ├── sync-coordinator.md
│   │   │   └── workflow-automation.md
│   │   ├── index.md
│   │   ├── migration-summary.md
│   │   ├── optimization
│   │   │   ├── benchmark-suite.md
│   │   │   ├── index.md
│   │   │   ├── load-balancer.md
│   │   │   ├── performance-monitor.md
│   │   │   ├── README.md
│   │   │   ├── resource-allocator.md
│   │   │   └── topology-optimizer.md
│   │   ├── README.md
│   │   ├── sparc
│   │   │   ├── architecture.md
│   │   │   ├── index.md
│   │   │   ├── pseudocode.md
│   │   │   ├── refinement.md
│   │   │   └── specification.md
│   │   ├── specialized
│   │   │   └── mobile
│   │   │       └── spec-mobile-react-native.md
│   │   ├── swarm
│   │   │   ├── adaptive-coordinator.md
│   │   │   ├── hierarchical-coordinator.md
│   │   │   ├── index.md
│   │   │   ├── mesh-coordinator.md
│   │   │   └── README.md
│   │   ├── templates
│   │   │   ├── automation-smart-agent.md
│   │   │   ├── coordinator-swarm-init.md
│   │   │   ├── github-pr-manager.md
│   │   │   ├── implementer-sparc-coder.md
│   │   │   ├── index.md
│   │   │   ├── memory-coordinator.md
│   │   │   ├── migration-plan.md
│   │   │   ├── orchestrator-task.md
│   │   │   ├── performance-analyzer.md
│   │   │   └── sparc-coordinator.md
│   │   └── testing
│   │       ├── unit
│   │       │   └── tdd-london-swarm.md
│   │       └── validation
│   │           └── production-validator.md
│   ├── api
│   │   ├── binary-protocol.md
│   │   ├── client-api.md
│   │   ├── gpu-algorithms.md
│   │   ├── index.md
│   │   ├── mcp-protocol.md
│   │   ├── openapi-spec.yml
│   │   ├── rest-api.md
│   │   ├── voice-api.md
│   │   ├── websocket-api.md
│   │   └── websocket-protocol.md
│   ├── configuration.md
│   ├── cuda-parameters.md
│   ├── glossary.md
│   ├── index.md
│   ├── polling-system.md
│   ├── README.md
│   └── xr-api.md
├── research
│   ├── hexser-guide.md
│   ├── horned-owl-guide.md
│   ├── owl_rdf_ontology_integration_research.md
│   ├── whelk-rs-guide.md
│   └── whelk-rs-research-summary.json
├── specialized
│   └── ontology
│       ├── hornedowl.md
│       ├── MIGRATION_GUIDE.md
│       ├── ontology-api-reference.md
│       ├── ontology-integration-summary.md
│       ├── ontology-system-overview.md
│       ├── ontology-user-guide.md
│       ├── physics-integration.md
│       ├── protocol-design.md
│       ├── PROTOCOL_SUMMARY.md
│       └── README.md
├── tasks
│   ├── task-codeaudit.md
│   ├── task-gpu.md
│   └── task-hexagonal.md
└── user-guide
    ├── 01-getting-started.md
    ├── 02-installation.md
    ├── 03-basic-usage.md
    ├── 04-features-overview.md
    ├── 05-troubleshooting.md
    └── 06-faq.md

58 directories, 312 files

Category 1: Resolve Critical Contradictions

These tasks address the most confusing and conflicting information in the corpus.

    Task 1.1: Standardize Binary Protocol Specification

        Problem: There are multiple conflicting descriptions of the WebSocket binary protocol. API.md, ARCHITECTURE.md, and architecture/components/websocket-protocol.md specify a 36-byte V2 format. Other files, including README.md, architecture/core/client.md, architecture/core/server.md, architecture/interface.md and reference/api/websocket-api.md, refer to an older 34-byte format.

        Action:

            Confirm with the architecture/components/websocket-protocol.md and reference/api/binary-protocol.md as the source of truth for the 36-byte V2 protocol.

            Audit and update all documents that mention a 34-byte protocol to reflect the correct 36-byte V2 specification.

            Ensure the explanation for the upgrade from V1 (34-byte, u16 ID) to V2 (36-byte, u32 ID) is clear and consistent everywhere, especially in 00-INDEX.md.

            Correct the architecture/core/visualization.md file, which implies a u16 ID format.

    Task 1.2: Unify API Documentation

        Problem: The docs/api/ directory (01-authentication.md, 02-endpoints.md, 03-websocket.md) describes a legacy API on port 9090 with different endpoints and authentication methods than the current system documented in docs/API.md (port 8080).

        Action:

            Delete the entire docs/api/ directory.

            Establish docs/API.md as the single source of truth for the REST API.

            Rename docs/API.md to docs/reference/api/rest-api.md to match the structure in other reference docs and update all links.

            Consolidate all WebSocket documentation into docs/reference/api/websocket-protocol.md (deleting duplicates).

    Task 1.3: Resolve Conflicting Deployment Strategies

        Problem: The docs/deployment/ directory (01-docker-deployment.md, etc.) describes a system using PostgreSQL and RabbitMQ, which contradicts the three-SQLite-database architecture defined in ARCHITECTURE.md and DATABASE.md.

        Action:

            Delete the docs/deployment/ directory.

            Consolidate all valid deployment information from guides/deployment.md, development/deployment.md and deployment/vircadia-docker-deployment.md into a single, authoritative docs/deployment/README.md.

            Ensure the deployment guide accurately reflects the three-SQLite-database system and the Docker setup described in the main architecture documents.

    Task 1.4: Unify Developer Guides

        Problem: The docs/developer-guide/ directory contains numbered files (01-.. to 06-..) that describe a different project (using Vue.js) and contradict the main DEVELOPER_GUIDE.md which is aligned with the new architecture. Similarly, guides/development-workflow.md is extremely detailed and should be the primary source.

        Action:

            Delete the entire docs/developer-guide/ directory.

            Rename docs/DEVELOPER_GUIDE.md to docs/guides/developer-guide.md and establish it as the primary guide.

            Merge the highly detailed content from docs/guides/development-workflow.md into the new docs/guides/developer-guide.md to create a single, comprehensive document.

    Task 1.5: Reconcile Testing Procedures

        Problem: adr-003-code-pruning-2025-10.md and README.md state that automated testing was removed for security reasons, making testing manual-only. This contradicts developer-guide/05-testing.md and DEVELOPER_GUIDE.md, which provide instructions for running npm test and cargo test.

        Action:

            Determine the true state of testing in the project.

            If testing is manual-only, delete developer-guide/05-testing.md and remove test-running instructions from DEVELOPER_GUIDE.md.

            Elevate guides/testing-guide.md to be the single source of truth for manual testing procedures.

            If automated tests do exist, remove the contradictory statements from the ADR and README.md.

Category 2: Archive Development & Migration Artifacts

These files are working documents, not final documentation, and should be moved out of the main doc structure.

    Task 2.1: Archive Migration Plans

        Problem: The docs/migration/ directory contains detailed briefs for a migration swarm. The docs/architecture/ directory contains numerous migration plans (cqrs-migration.md, migration-strategy.md, hexagonal-cqrs-architecture.md). These are not user-facing documentation.

        Action:

            Move the entire docs/migration/ directory to docs/archive/migration-plan-2025-10/.

            Move the following files from docs/architecture/ to the same archive directory: cqrs-migration.md, migration-strategy.md, hexagonal-cqrs-architecture.md, phase3-ports-complete.md, github-sync-service-design.md, voice-webrtc-migration-plan.md.

    Task 2.2: Archive Architecture Planning Documents

        Problem: The docs/architecture/ directory contains numbered planning files (00-ARCHITECTURE-OVERVIEW.md through 05-schema-implementation-summary.md) and a README.md that points to them. These represent the plan for the new architecture, while ARCHITECTURE.md is the final output.

        Action:

            Move the numbered files (00- to 05-) and the README.md from docs/architecture/ into a new archive folder: docs/archive/hexagonal-planning-docs/.

    Task 2.3: Archive Other Working Documents

        Problem: Various status reports and summaries exist that are not final documentation.

        Action:

            Move implementation/whelk-integration-summary.md to docs/archive/implementation-notes/.

            Move research/whelk-rs-research-summary.json to docs/archive/research-notes/.

Category 3: Consolidate and De-duplicate Content

This category focuses on merging redundant files and creating single sources of truth.

    Task 3.1: Consolidate Index Files

        Problem: There are multiple root-level and architecture-level index files: 00-INDEX.md, index.md, ARCHITECTURE.md, README.md, architecture/ARCHITECTURE_INDEX.md.

        Action:

            Establish docs/00-INDEX.md as the single master index for the entire documentation.

            Delete docs/index.md, docs/ARCHITECTURE.md (the root file, not architecture/ARCHITECTURE.md), and docs/architecture/ARCHITECTURE_INDEX.md.

            Refactor the root docs/README.md to be a welcoming landing page that points to 00-INDEX.md and the getting-started guide, removing its redundant index content.

            Thoroughly audit all links in 00-INDEX.md and fix broken ones, ensuring it points to the newly consolidated and renamed files.

    Task 3.2: Consolidate Overlapping Overviews

        Problem: There are multiple high-level system overview documents: architecture/overview.md, architecture/system-overview.md, and concepts/system-architecture.md.

        Action:

            Merge the best content and diagrams from all three files into concepts/system-architecture.md.

            Delete architecture/overview.md and architecture/system-overview.md.

            Ensure the new consolidated overview is consistent with the final hexagonal architecture.

    Task 3.3: Consolidate guides directory

        Problem: The guides directory has both a README.md and an index.md file.

        Action: Merge the content into guides/README.md and delete guides/index.md.

    Task 3.4: Consolidate Ontology Documentation

        Problem: The specialized/ontology/ folder contains a mix of final documentation, working docs, and migration guides.

        Action:

            Move MIGRATION_GUIDE.md, PROTOCOL_SUMMARY.md, and protocol-design.md to an archive folder docs/archive/ontology-migration-notes/.

            Consolidate the content from the remaining files (hornedowl.md, ontology-api-reference.md, ontology-system-overview.md, ontology-user-guide.md, physics-integration.md, README.md) into the main concepts, guides, and reference sections to eliminate the specialized/ directory.

Category 4: Improve Content Quality and Structure

This category includes tasks to fix stubs, improve links, and convert diagrams.

    Task 4.1: Fix Stub User Guide Documents

        Problem: The files in docs/user-guide/ (01- to 06-) are generic templates with placeholder text like [brief description...].

        Action:

            Review the getting-started/ and guides/ directories for content that can be used to flesh out the user guide.

            Rewrite the user-guide/ files with actual, project-specific information. If no information is available, mark them clearly as stubs needing content and add this to a "Documentation TODO" section in the main README.md.

    Task 4.2: Fix Hyperlink Integration

        Problem: Many files, especially within reference/agents/, list filenames as "Related Topics" instead of creating proper markdown hyperlinks.

        Action:

            Create an agent to crawl the docs/ directory.

            Identify all occurrences of .md or file paths in plain text that should be hyperlinks.

            Convert them to correct relative markdown links. Example: [Agent Conventions](./conventions.md).

            Pay special attention to the massive, flat list of files in reference/agents/README.md. This list should be categorized and turned into a navigable, linked index.

    Task 4.3: Convert ASCII Diagrams to Mermaid

        Problem: The architecture/event-flow-diagrams.md file contains ASCII diagrams that would be more maintainable and readable as Mermaid diagrams.

        Action: Convert all ASCII diagrams in architecture/event-flow-diagrams.md to Mermaid graph or sequenceDiagram blocks. Integrate these new diagrams into the consolidated architecture documents.

    Task 4.4: Validate and Fix Mermaid Diagrams

        Problem: Some Mermaid diagrams may have syntax errors or might not render correctly.

        Action:

            Audit every ````mermaid` block in the entire corpus.

            Validate the syntax using a Mermaid renderer or linter.

            Fix any syntax errors found. The diagram in 00-INDEX.md appears complex and should be double-checked.

    Task 4.5: Clean up reference/agents directory

        Problem: The reference/agents directory has a very deep and wide structure that is hard to navigate. Many files are just agent prompts.

        Action:

            Consolidate the agent prompts into a more structured format. Consider creating tables or summary pages for different agent types (e.g., a single page for all consensus agents).

            Review and fix the chaotic linking within this section.

            Organize the directory into a more logical hierarchy. The existing subdirectories (core, github, consensus, etc.) are a good start but the file structure within them is flat and messy.

    Task 4.6: Remove Empty Files

        Problem: concepts/decisions/index.md is an empty file.

        Action: Delete the file or populate it with an index of the ADRs in that directory.