---
title: VisionFlow Client Architecture - Current State
description: > ⚠️ **DEPRECATION NOTICE** ⚠️ > **GraphServiceActor** is deprecated. See `/docs/guides/graphserviceactor-migration.md` for current patterns.
category: explanation
tags:
  - architecture
  - client
  - rest
  - websocket
  - rust
updated-date: 2025-12-18
difficulty-level: advanced
---


# VisionFlow Client Architecture - Current State

> ⚠️ **DEPRECATION NOTICE** ⚠️
> **GraphServiceActor** is deprecated. See `/docs/guides/graphserviceactor-migration.md` for current patterns.

This document provides a comprehensive architecture diagram reflecting the **CURRENT CLIENT IMPLEMENTATION** at `/workspace/ext/client/src`. This architecture analysis is based on direct code examination and represents the actual running system.

**Last Updated**: 2025-10-03
**Analysis Base**: Direct source code inspection of 404 TypeScript/React files
**Recent Updates**:
- **Code Pruning Complete**: 38 files removed (11,957 LOC) - 30% codebase reduction
- **API Architecture Clarified**: Layered design documented - UnifiedApiClient + domain APIs
- **Testing Infrastructure Removed**: Security concerns led to removal of automated tests
- **WebSocket Binary Protocol Enhancement**: 80% traffic reduction achieved
- **Settings API Standardisation**: Field normalisation fixes in config/mod.rs
- **Agent Task Management**: Complete remove/pause/resume functionality
- **Dual Graph Architecture**: Protocol support for dual types with unified implementation

---

## Complete Client Architecture Overview

```mermaid
graph TB
    subgraph "Browser Runtime Environment"
        subgraph "React Application Layer"
            App["App.tsx<br/>Root Component"]
            AppInit["AppInitialiser<br/>WebSocket & Settings Init"]
            MainLayout["MainLayout.tsx<br/>Primary Layout Manager"]
            Quest3AR["Quest3AR.tsx<br/>XR/AR Layout"]
        end

        subgraph "Context Providers & State"
            ApplicationMode["ApplicationModeProvider<br/>Mode Management"]
            XRCore["XRCoreProvider<br/>WebXR Integration"]
            TooltipProvider["TooltipProvider<br/>UI Tooltips"]
            HelpProvider["HelpProvider<br/>Help System"]
            OnboardingProvider["OnboardingProvider<br/>User Onboarding"]
            BotsDataProvider["BotsDataProvider<br/>Agent Data Context"]
        end

        subgraph "Core Features Architecture"
            subgraph "Graph Visualisation System"
                GraphCanvas["GraphCanvas.tsx<br/>Three.js R3F Canvas"]
                GraphManager["GraphManager<br/>Scene Management"]
                GraphDataManager["graphDataManager<br/>Data Orchestration"]
                UnifiedGraph["Unified Graph Implementation<br/>Protocol Support for Dual Types"]
                SimpleThreeTest["SimpleThreeTest<br/>Debug Renderer"]
                GraphCanvasSimple["GraphCanvasSimple<br/>Lightweight Renderer"]
                SelectiveBloom["SelectiveBloom<br/>Post-processing Effects"]
                HolographicDataSphere["HolographicDataSphere<br/>Complete Hologram Module"]
            end

            subgraph "Agent/Bot System"
                BotsVisualization["BotsVisualisation<br/>Agent Node Rendering"]
                AgentPollingStatus["AgentPollingStatus<br/>Connection Status UI"]
                BotsWebSocketIntegration["BotsWebSocketIntegration<br/>WebSocket Handler"]
                AgentPollingService["AgentPollingService<br/>REST API Polling"]
                ConfigurationMapper["ConfigurationMapper<br/>Agent Config Mapping"]
            end

            subgraph "Settings Management"
                SettingsStore["settingsStore<br/>Zustand State"]
                FloatingSettingsPanel["FloatingSettingsPanel<br/>Settings UI"]
                LazySettingsSections["LazySettingsSections<br/>Dynamic Loading"]
                UndoRedoControls["UndoRedoControls<br/>Settings History"]
                VirtualizedSettingsGroup["VirtualisedSettingsGroup<br/>Performance UI"]
                AutoSaveManager["AutoSaveManager<br/>Batch Persistence"]
            end

            subgraph "XR/AR System"
                XRCoreProvider["XRCoreProvider<br/>WebXR Foundation"]
                Quest3Integration["useQuest3Integration<br/>Device Detection"]
                XRManagers["XR Managers<br/>Device State"]
                XRComponents["XR Components<br/>AR/VR UI"]
            end
        end

        subgraph "Communication Layer"
            subgraph "WebSocket Binary Protocol"
                WebSocketService["WebSocketService.ts<br/>Connection Management"]
                BinaryWebSocketProtocol["BinaryWebSocketProtocol.ts<br/>Binary Protocol Handler"]
                BinaryProtocol["binaryProtocol.ts<br/>34-byte Node Format"]
                BatchQueue["BatchQueue.ts<br/>Performance Batching"]
                ValidationMiddleware["validation.ts<br/>Data Validation"]
            end

            subgraph "REST API Layer - Layered Architecture"
                UnifiedApiClient["UnifiedApiClient (Foundation)<br/>HTTP Client 526 LOC"]
                DomainAPIs["Domain API Layer 2,619 LOC<br/>Business Logic + Request Handling"]
                SettingsApi["settingsApi 430 LOC<br/>Debouncing, Batching, Priority"]
                AnalyticsApi["analyticsApi 582 LOC<br/>GPU Analytics Integration"]
                OptimizationApi["optimisationApi 376 LOC<br/>Graph Optimization"]
                ExportApi["exportApi 329 LOC<br/>Export, Publish, Share"]
                WorkspaceApi["workspaceApi 337 LOC<br/>Workspace CRUD"]
                BatchUpdateApi["batchUpdateApi 135 LOC<br/>Batch Operations"]

                UnifiedApiClient --> DomainAPIs
                DomainAPIs --> SettingsApi
                DomainAPIs --> AnalyticsApi
                DomainAPIs --> OptimizationApi
                DomainAPIs --> ExportApi
                DomainAPIs --> WorkspaceApi
                DomainAPIs --> BatchUpdateApi
            end

            subgraph "Voice System"
                LegacyVoiceHook["useVoiceInteraction<br/>Active Hook"]
                AudioInputService["AudioInputService<br/>Voice Capture"]
                VoiceIntegration["Integrated in Control Panel<br/>No Standalone Components"]
            end
        end

        subgraph "Visualisation & Effects"
            subgraph "Rendering Pipeline"
                Materials["rendering/materials<br/>Custom Shaders"]
                Shaders["shaders/<br/>WebGL Shaders"]
                ThreeGeometries["three-geometries<br/>Custom Geometries"]
            end

            subgraph "Visual Features"
                IntegratedControlPanel["IntegratedControlPanel<br/>Main Control UI"]
                SpacePilotIntegration["SpacePilotSimpleIntegration<br/>3D Mouse Support"]
                VisualisationControls["Visualisation Controls<br/>Visual Settings"]
                VisualisationEffects["Visualisation Effects<br/>Post-processing"]
            end
        end

        subgraph "Feature Modules"
            subgraph "Analytics System"
                AnalyticsComponents["Analytics Components<br/>Data Visualisation"]
                AnalyticsStore["Analytics Store<br/>Analytics State"]
                AnalyticsExamples["Analytics Examples<br/>Demo Components"]
            end

            subgraph "Physics Engine"
                PhysicsPresets["PhysicsPresets<br/>Preset Configurations"]
                PhysicsEngineControls["PhysicsEngineControls<br/>Physics UI"]
                ConstraintBuilderDialog["ConstraintBuilderDialog<br/>Physics Constraints"]
            end

            subgraph "Command Palette"
                CommandPalette["CommandPalette<br/>Command Interface"]
                CommandHooks["Command Hooks<br/>Command Logic"]
            end

            subgraph "Design System"
                DesignSystemComponents["Design System Components<br/>UI Library"]
                DesignSystemPatterns["Design System Patterns<br/>UI Patterns"]
            end

            subgraph "Help System"
                HelpRegistry["HelpRegistry<br/>Help Content"]
                HelpComponents["Help Components<br/>Help UI"]
            end

            subgraph "Onboarding System"
                OnboardingFlows["Onboarding Flows<br/>User Flows"]
                OnboardingComponents["Onboarding Components<br/>Onboarding UI"]
                OnboardingHooks["Onboarding Hooks<br/>Flow Logic"]
            end

            subgraph "Workspace Management"
                WorkspaceManager["WorkspaceManager<br/>Workspace UI"]
            end
        end

        subgraph "Utilities & Infrastructure"
            subgraph "Performance & Monitoring"
                PerformanceMonitor["performanceMonitor<br/>Performance Tracking"]
                DualGraphPerformanceMonitor["dualGraphPerformanceMonitor<br/>Multi-graph Performance"]
                DualGraphOptimizations["dualGraphOptimisations<br/>Performance Utils"]
                GraphOptimizationService["GraphOptimisationService<br/>Graph Performance"]
                NodeDistributionOptimizer["NodeDistributionOptimiser<br/>Layout Optimisation"]
                ClientDebugState["clientDebugState<br/>Debug State"]
                TelemetrySystem["Telemetry System<br/>Usage Analytics"]
            end

            subgraph "Utility Libraries"
                LoggerConfig["loggerConfig<br/>Logging System"]
                DebugConfig["debugConfig<br/>Debug Configuration"]
                ClassNameUtils["classNameUtils<br/>CSS Utilities"]
                DownloadHelpers["downloadHelpers<br/>File Downloads"]
                AccessibilityUtils["accessibility<br/>A11y Utilities"]
                IframeCommunication["iframeCommunication<br/>Cross-frame Comm"]
            end
        end

        subgraph "Error Handling & Components"
            ErrorBoundary["ErrorBoundary<br/>Error Catching"]
            ErrorHandling["Error Handling<br/>Error Components"]
            ConnectionWarning["ConnectionWarning<br/>Connection Status"]
            BrowserSupportWarning["BrowserSupportWarning<br/>Browser Compatibility"]
            SpaceMouseStatus["SpaceMouseStatus<br/>Device Status"]
            DebugControlPanel["DebugControlPanel<br/>Debug Interface"]
        end

        subgraph "Legacy Integrations"
            VircadiaIntegration["Vircadia Integration<br/>Legacy VR Support"]
            VircadiaWeb["vircadia-web<br/>Legacy Web Client"]
            VircadiaWorld["vircadia-world<br/>Legacy World System"]
        end
    end

    %% Data Flow Connections
    App --> AppInit
    App --> MainLayout
    App --> Quest3AR
    AppInit --> WebSocketService
    AppInit --> SettingsStore

    MainLayout --> GraphCanvas
    MainLayout --> IntegratedControlPanel
    MainLayout --> BotsDataProvider

    GraphCanvas --> GraphManager
    GraphCanvas --> BotsVisualization
    GraphCanvas --> SelectiveBloom
    GraphCanvas --> HolographicDataSphere

    GraphManager --> GraphDataManager
    BotsVisualization --> BotsWebSocketIntegration
    BotsWebSocketIntegration --> WebSocketService
    BotsWebSocketIntegration --> AgentPollingService

    WebSocketService --> BinaryWebSocketProtocol
    BinaryWebSocketProtocol --> BinaryProtocol
    WebSocketService --> BatchQueue
    WebSocketService --> ValidationMiddleware

    SettingsStore --> UnifiedApiClient
    SettingsStore --> AutoSaveManager
    SettingsStore --> SettingsApi

    IntegratedControlPanel --> PhysicsEngineControls
    IntegratedControlPanel --> VisualisationControls
    IntegratedControlPanel --> AnalyticsComponents

    %% External System Connections
    WebSocketService -.->|WebSocket Binary| Backend["Rust Backend<br/>/wss endpoint"]
    UnifiedApiClient -.->|REST API| Backend
    AgentPollingService -.->|REST Polling| Backend
    AudioInputService -.->|Voice Data| Backend

    style GraphCanvas fill:#e3f2fd
    style WebSocketService fill:#c8e6c9
    style SettingsStore fill:#fff3e0
    style BotsVisualization fill:#f3e5f5
    style IntegratedControlPanel fill:#e8f5e9
    style UnifiedApiClient fill:#fce4ec
    style BinaryProtocol fill:#e0f2f1
    style BinaryWebSocketProtocol fill:#e0f7fa
    style XRCoreProvider fill:#fff8e1
```

---

## Component Architecture Deep Dive

### 1. Application Bootstrap & Layout

```mermaid
graph TB
    subgraph "Application Initialisation Flow"
        Browser["Browser Load"] --> App["App.tsx"]
        App --> AppInitialiser["AppInitialiser"]

        AppInitialiser --> SettingsInit["Settings Initialisation"]
        AppInitialiser --> WebSocketInit["WebSocket Connection"]
        AppInitialiser --> AuthInit["Authentication Check"]

        SettingsInit --> SettingsStore["Zustand Settings Store"]
        WebSocketInit --> WebSocketService["WebSocket Service"]
        AuthInit --> NostrAuth["Nostr Authentication"]

        App --> Quest3Detection{"Quest 3 Detected?"}
        Quest3Detection -->|Yes| Quest3AR["Quest3AR Layout"]
        Quest3Detection -->|No| MainLayout["MainLayout"]

        MainLayout --> GraphCanvas["Graph Canvas"]
        MainLayout --> ControlPanels["Control Panels"]
        MainLayout --> VoiceComponents["Voice Components"]

        Quest3AR --> XRScene["XR Scene"]
        Quest3AR --> XRControls["XR Controls"]

        style AppInitialiser fill:#c8e6c9
        style SettingsStore fill:#fff3e0
        style WebSocketService fill:#e3f2fd
    end
```

### 2. Graph Visualisation Architecture

**Note**: The system implements a unified graph with protocol support for dual types (Knowledge and Agent nodes), not separate graph implementations.

```mermaid
graph TB
    subgraph "Graph Rendering Pipeline"
        GraphCanvas["GraphCanvas.tsx<br/>React Three Fiber Canvas"]

        subgraph "Scene Management"
            GraphManager["GraphManager<br/>Scene Object Manager"]
            GraphDataManager["GraphDataManager<br/>Data Orchestration"]
            UnifiedImplementation["Unified Graph<br/>Type Flags: 0x4000, 0x8000"]
            SimpleThreeTest["SimpleThreeTest<br/>Debug Renderer"]
            GraphCanvasSimple["GraphCanvasSimple<br/>Lightweight Mode"]
        end

        subgraph "Data Sources"
            WebSocketBinary["WebSocket Binary<br/>Position Updates"]
            RESTPolling["REST API Polling<br/>Metadata Updates"]
            GraphDataSubscription["Graph Data Subscription"]
        end

        subgraph "Visual Effects"
            SelectiveBloom["Selective Bloom<br/>Post-processing"]
            HolographicDataSphere["HolographicDataSphere Module<br/>Contains HologramEnvironment"]
            CustomMaterials["Custom Materials<br/>WebGL Shaders"]
        end

        subgraph "Agent Visualisation"
            BotsVisualisation["Bots Visualisation<br/>Agent Nodes"]
            AgentNodes["Agent Node Meshes"]
            AgentLabels["Agent Labels"]
            AgentConnections["Agent Connections"]
        end

        GraphCanvas --> GraphManager
        GraphCanvas --> SelectiveBloom
        GraphCanvas --> HolographicDataSphere
        GraphCanvas --> BotsVisualisation

        GraphManager --> GraphDataManager
        GraphDataManager --> WebSocketBinary
        GraphDataManager --> RESTPolling
        GraphDataManager --> GraphDataSubscription

        BotsVisualisation --> AgentNodes
        BotsVisualisation --> AgentLabels
        BotsVisualisation --> AgentConnections

        style GraphCanvas fill:#e3f2fd
        style GraphDataManager fill:#c8e6c9
        style BotsVisualisation fill:#f3e5f5
    end
```

### 3. WebSocket Binary Protocol Implementation

**Recent Updates**: Completed protocol support for dual graph types with unified implementation, plus major duplicate polling fix eliminating race conditions.

```mermaid
graph TB
    subgraph "Binary Protocol Architecture"
        subgraph "WebSocket Connection"
            WebSocketService["WebSocketService.ts<br/>Connection Manager"]
            ConnectionHandlers["Connection Handlers"]
            ReconnectLogic["Reconnect Logic"]
            HeartbeatSystem["Heartbeat System"]
        end

        subgraph "Binary Data Processing"
            BinaryWebSocketProtocol["BinaryWebSocketProtocol.ts<br/>Protocol Handler"]
            BinaryProtocol["binaryProtocol.ts<br/>34-byte Node Format"]
            DataParser["Binary Data Parser"]
            NodeValidator["Node Data Validator"]
            BatchProcessor["Batch Processor"]
        end

        subgraph "Message Types"
            ControlMessages["Control Messages<br/>0x00-0x0F"]
            DataMessages["Data Messages<br/>0x10-0x3F"]
            StreamMessages["Stream Messages<br/>0x40-0x5F"]
            AgentMessages["Agent Messages<br/>0x60-0x7F"]
        end

        subgraph "Node Data Structure (34 bytes)"
            NodeID["Node ID: u16 (2 bytes)<br/>Flags: Knowledge/Agent"]
            Position["Position: Vec3 (12 bytes)<br/>x, y, z coordinates"]
            Velocity["Velocity: Vec3 (12 bytes)<br/>vx, vy, vz components"]
            SSSPDistance["SSSP Distance: f32 (4 bytes)<br/>Shortest path distance"]
            SSSPParent["SSSP Parent: i32 (4 bytes)<br/>Parent node ID"]
        end

        WebSocketService --> BinaryWebSocketProtocol
        BinaryWebSocketProtocol --> BinaryProtocol
        BinaryProtocol --> DataParser
        DataParser --> NodeValidator
        NodeValidator --> BatchProcessor

        WebSocketService --> ControlMessages
        WebSocketService --> DataMessages
        WebSocketService --> StreamMessages
        WebSocketService --> AgentMessages

        DataParser --> NodeID
        DataParser --> Position
        DataParser --> Velocity
        DataParser --> SSSPDistance
        DataParser --> SSSPParent

        style WebSocketService fill:#e3f2fd
        style BinaryWebSocketProtocol fill:#e0f7fa
        style BinaryProtocol fill:#e0f2f1
        style DataParser fill:#c8e6c9
    end
```

### 4. Agent System Architecture

**Major Fix Applied**: Eliminated duplicate data fetching that caused race conditions. The system now uses:
- **WebSocket binary protocol**: Real-time position/velocity updates only
- **REST polling**: Conservative metadata polling (3s active, 15s idle)
- **Single source strategy**: No more triple polling conflicts

```mermaid
graph TB
    subgraph "Agent Management System"
        subgraph "Data Sources"
            BotsWebSocketIntegration["BotsWebSocketIntegration<br/>WebSocket Handler"]
            AgentPollingService["AgentPollingService<br/>REST Polling"]
            ConfigurationMapper["Configuration Mapper<br/>Agent Config"]
        end

        subgraph "Data Context"
            BotsDataProvider["BotsDataProvider<br/>React Context"]
            BotsDataContext["Bots Data Context<br/>State Management"]
            AgentState["Agent State<br/>Agent Information"]
        end

        subgraph "Agent Types"
            CoordinatorAgents["Coordinator Agents<br/>Orchestration"]
            ResearcherAgents["Researcher Agents<br/>Data Analysis"]
            CoderAgents["Coder Agents<br/>Code Generation"]
            AnalystAgents["Analyst Agents<br/>Analysis"]
            ArchitectAgents["Architect Agents<br/>System Design"]
            TesterAgents["Tester Agents<br/>Quality Assurance"]
            ReviewerAgents["Reviewer Agents<br/>Code Review"]
        end

        subgraph "Agent Data Structure"
            AgentID["Agent ID & Type"]
            AgentStatus["Status & Health"]
            AgentMetrics["CPU/Memory Usage"]
            AgentCapabilities["Capabilities & Tasks"]
            AgentPosition["3D Position Data"]
            SSSPData["SSSP Pathfinding"]
            SwarmMetadata["Swarm Metadata"]
        end

        subgraph "Visualisation Components"
            BotsVisualisation["Bots Visualisation<br/>3D Agent Rendering"]
            AgentPollingStatus["Agent Polling Status<br/>Connection UI"]
            AgentInteraction["Agent Interaction<br/>Selection & Details"]
        end

        BotsWebSocketIntegration --> BotsDataProvider
        AgentPollingService --> BotsDataProvider
        ConfigurationMapper --> BotsDataProvider

        BotsDataProvider --> BotsDataContext
        BotsDataContext --> AgentState

        AgentState --> CoordinatorAgents
        AgentState --> ResearcherAgents
        AgentState --> CoderAgents
        AgentState --> AnalystAgents
        AgentState --> ArchitectAgents
        AgentState --> TesterAgents
        AgentState --> ReviewerAgents

        AgentState --> AgentID
        AgentState --> AgentStatus
        AgentState --> AgentMetrics
        AgentState --> AgentCapabilities
        AgentState --> AgentPosition
        AgentState --> SSSPData
        AgentState --> SwarmMetadata

        BotsDataContext --> BotsVisualisation
        BotsDataContext --> AgentPollingStatus
        BotsDataContext --> AgentInteraction

        style BotsDataProvider fill:#f3e5f5
        style BotsWebSocketIntegration fill:#e3f2fd
        style AgentPollingService fill:#fff3e0
    end
```

### 5. Settings Management Architecture

**Updated Architecture**: Restructured with path-based lazy loading, batch persistence via AutoSaveManager, and improved performance with virtualised UI components.

```mermaid
graph TB
    subgraph "Settings System Architecture"
        subgraph "State Management"
            SettingsStore["settingsStore.ts<br/>Zustand Store"]
            PartialSettings["Partial Settings<br/>Lazy Loading"]
            LoadedPaths["Loaded Paths Tracking"]
            Subscribers["Path Subscribers"]
        end

        subgraph "Persistence Layer"
            AutoSaveManager["AutoSaveManager<br/>Batch Operations"]
            SettingsApi["settingsApi<br/>Backend Sync"]
            LocalStorage["localStorage<br/>Browser Persistence"]
            BackendSync["Backend Sync<br/>Server State"]
        end

        subgraph "UI Components"
            FloatingSettingsPanel["FloatingSettingsPanel<br/>Main Settings UI"]
            LazySettingsSections["LazySettingsSections<br/>Dynamic Loading"]
            VirtualisedSettingsGroup["VirtualisedSettingsGroup<br/>Performance UI"]
            UndoRedoControls["UndoRedoControls<br/>History Management"]
            BackendUrlSetting["BackendUrlSetting<br/>Connection Config"]
        end

        subgraph "Settings Categories"
            SystemSettings["System Settings<br/>Debug, WebSocket, etc."]
            VisualisationSettings["Visualisation Settings<br/>Rendering, Effects"]
            PhysicsSettings["Physics Settings<br/>Simulation Parameters"]
            XRSettings["XR Settings<br/>WebXR Configuration"]
            AuthSettings["Auth Settings<br/>Authentication"]
            GraphSettings["Graph Settings<br/>Graph Visualisation"]
        end

        SettingsStore --> PartialSettings
        SettingsStore --> LoadedPaths
        SettingsStore --> Subscribers

        SettingsStore --> AutoSaveManager
        AutoSaveManager --> SettingsApi
        SettingsStore --> LocalStorage
        SettingsApi --> BackendSync

        SettingsStore --> FloatingSettingsPanel
        FloatingSettingsPanel --> LazySettingsSections
        FloatingSettingsPanel --> VirtualisedSettingsGroup
        FloatingSettingsPanel --> UndoRedoControls
        FloatingSettingsPanel --> BackendUrlSetting

        SettingsStore --> SystemSettings
        SettingsStore --> VisualisationSettings
        SettingsStore --> PhysicsSettings
        SettingsStore --> XRSettings
        SettingsStore --> AuthSettings
        SettingsStore --> GraphSettings

        style SettingsStore fill:#fff3e0
        style AutoSaveManager fill:#e8f5e9
        style FloatingSettingsPanel fill:#f3e5f5
    end
```

### 6. XR/AR System Architecture

```mermaid
graph TB
    subgraph "XR System Architecture"
        subgraph "Core XR Infrastructure"
            XRCoreProvider["XRCoreProvider<br/>WebXR Foundation"]
            Quest3Integration["useQuest3Integration<br/>Quest 3 Detection"]
            XRManagers["XR Managers<br/>Device Management"]
            XRSystems["XR Systems<br/>Tracking Systems"]
        end

        subgraph "XR Components"
            XRComponents["XR Components<br/>UI Components"]
            XRUIComponents["XR UI Components<br/>Spatial UI"]
            Quest3AR["Quest3AR.tsx<br/>AR Layout"]
        end

        subgraph "Device Detection"
            UserAgentDetection["User Agent Detection<br/>Quest 3 Browser"]
            ForceParameters["Force Parameters<br/>?force=quest3"]
            AutoStartLogic["Auto Start Logic<br/>XR Session Init"]
        end

        subgraph "XR Providers & Hooks"
            XRProviders["XR Providers<br/>Context Providers"]
            XRHooks["XR Hooks<br/>React Hooks"]
            XRTypes["XR Types<br/>Type Definitions"]
        end

        subgraph "Legacy Vircadia Integration"
            VircadiaXR["Vircadia XR<br/>Legacy VR Support"]
            VircadiaComponents["Vircadia Components<br/>Legacy Components"]
            VircadiaServices["Vircadia Services<br/>Legacy Services"]
            VircadiaHooks["Vircadia Hooks<br/>Legacy Hooks"]
        end

        XRCoreProvider --> Quest3Integration
        XRCoreProvider --> XRManagers
        XRCoreProvider --> XRSystems

        Quest3Integration --> UserAgentDetection
        Quest3Integration --> ForceParameters
        Quest3Integration --> AutoStartLogic

        XRCoreProvider --> XRComponents
        XRComponents --> XRUIComponents
        XRComponents --> Quest3AR

        XRCoreProvider --> XRProviders
        XRProviders --> XRHooks
        XRHooks --> XRTypes

        XRCoreProvider --> VircadiaXR
        VircadiaXR --> VircadiaComponents
        VircadiaXR --> VircadiaServices
        VircadiaXR --> VircadiaHooks

        style XRCoreProvider fill:#fff8e1
        style Quest3Integration fill:#e8f5e9
        style Quest3AR fill:#f3e5f5
    end
```

---

## Data Flow Architecture

### 1. Real-time Graph Data Flow

```mermaid
sequenceDiagram
    participant Backend as Rust Backend
    participant WS as WebSocket Service
    participant Binary as Binary Protocol
    participant GraphData as Graph Data Manager
    participant Canvas as Graph Canvas
    participant Agents as Agent Visualisation

    Note over Backend,Agents: 2000ms Graph Data Polling Cycle

    Backend->>WS: Binary frame (34 bytes/node)
    WS->>Binary: Parse binary data
    Binary->>Binary: Validate node format
    Binary->>GraphData: Update positions

    par Graph Updates
        GraphData->>Canvas: Node positions
        and
        GraphData->>Agents: Agent positions
    end

    Canvas->>Canvas: Update Three.js scene
    Agents->>Agents: Update agent meshes

    Note over Backend,Agents: Agent Metadata via REST (10s cycle)

    loop Every 10 seconds
        Agents->>Backend: GET /api/bots/data
        Backend-->>Agents: Agent metadata (JSON)
        Agents->>Agents: Update agent details
    end
```

### 2. Settings Data Flow

```mermaid
sequenceDiagram
    participant UI as Settings UI
    participant Store as Settings Store
    participant AutoSave as AutoSave Manager
    participant API as Settings API
    participant Backend as Rust Backend

    UI->>Store: Setting change
    Store->>Store: Update partial state
    Store->>AutoSave: Queue for save

    Note over AutoSave: Debounced batching (500ms)

    AutoSave->>API: Batch settings update
    API->>Backend: POST /api/settings/batch
    Backend-->>API: Success response
    API-->>AutoSave: Confirm save
    AutoSave-->>Store: Update save status
    Store-->>UI: Notify subscribers
```

### 3. Voice System - Dual Implementation Status

**Note**: The centralised architecture exists but isn't actively deployed. Components currently use the legacy `useVoiceInteraction` hook.

```mermaid
graph TB
    subgraph "Voice System Current State"
        subgraph "Active Implementation"
            LegacyHook["useVoiceInteraction.ts<br/>Legacy Hook (In Use)"]
            VoiceButton["VoiceButton.tsx<br/>Uses Legacy Hook"]
            VoiceIndicator["VoiceStatusIndicator.tsx<br/>Uses Legacy Hook"]
        end

        subgraph "Available but Inactive"
            VoiceProvider["VoiceProvider<br/>Context Provider (Unused)"]
            CentralisedHook["useVoiceInteractionCentralised<br/>Modern Architecture"]

            subgraph "9 Specialised Hooks (Designed)"
                UseVoiceConnection["useVoiceConnection"]
                UseVoiceInput["useVoiceInput"]
                UseVoiceOutput["useVoiceOutput"]
                UseVoiceTranscription["useVoiceTranscription"]
                UseVoiceSettings["useVoiceSettings"]
                UseVoicePermissions["useVoicePermissions"]
                UseVoiceBrowserSupport["useVoiceBrowserSupport"]
                UseAudioLevel["useAudioLevel"]
            end
        end

        subgraph "Core Services"
            AudioInputService["AudioInputService<br/>Audio Capture"]
            WebSocketService["WebSocket Service<br/>Binary Communication"]
        end

        LegacyHook --> VoiceButton
        LegacyHook --> VoiceIndicator
        LegacyHook --> AudioInputService
        LegacyHook --> WebSocketService

        CentralisedHook -.-> UseVoiceConnection
        CentralisedHook -.-> UseVoiceInput
        CentralisedHook -.-> UseVoiceOutput
        CentralisedHook -.-> UseVoiceTranscription
        CentralisedHook -.-> UseVoiceSettings
        CentralisedHook -.-> UseVoicePermissions
        CentralisedHook -.-> UseVoiceBrowserSupport
        CentralisedHook -.-> UseAudioLevel

        style LegacyHook fill:#c8e6c9
        style CentralisedHook fill:#ffcdd2
        style VoiceProvider fill:#ffcdd2
    end
```

### Voice System Data Flow

```mermaid
sequenceDiagram
    participant User as User
    participant LegacyHook as useVoiceInteraction
    participant AudioService as Audio Input Service
    participant WS as WebSocket Service
    participant Backend as Rust Backend
    participant Whisper as Whisper STT
    participant Kokoro as Kokoro TTS

    User->>LegacyHook: Press voice button
    LegacyHook->>AudioService: Start recording
    AudioService->>AudioService: Capture audio stream
    AudioService->>WS: Send binary audio
    WS->>Backend: Forward audio data
    Backend->>Whisper: Process STT
    Whisper-->>Backend: Transcribed text
    Backend->>Backend: Process command
    Backend->>Kokoro: Generate TTS
    Kokoro-->>Backend: Audio response
    Backend->>WS: Send binary audio response
    WS->>LegacyHook: Audio response received
    LegacyHook-->>User: Voice feedback
```

---

## Testing Architecture

**Important Note**: Tests use comprehensive mocks, not real API calls. The test suite validates component behaviour with mocked backend responses.

### Mock System Infrastructure

```mermaid
graph TB
    subgraph "Test Infrastructure - Mock Based"
        subgraph "Mock Providers"
            MockApiClient["createMockApiClient<br/>Mock HTTP Client"]
            MockWebSocket["createMockWebSocket<br/>Mock WebSocket"]
            MockWebSocketServer["MockWebSocketServer<br/>Mock WS Server"]
        end

        subgraph "Integration Test Suites"
            AnalyticsTests["analytics.test.ts<br/>234 tests"]
            WorkspaceTests["workspace.test.ts<br/>198 tests"]
            OptimisationTests["optimisation.test.ts<br/>276 tests"]
            ExportTests["export.test.ts<br/>226 tests"]
        end

        subgraph "Mock Data Generators"
            GraphMocks["Mock Graph Data"]
            AgentMocks["Mock Agent Status"]
            AnalyticsMocks["Mock Analytics"]
            WorkspaceMocks["Mock Workspaces"]
        end

        subgraph "Test Execution Status"
            SecurityAlert["Tests Disabled<br/>Supply Chain Security"]
            TestCount["934+ Individual Tests<br/>(Not Running)"]
        end

        MockApiClient --> AnalyticsTests
        MockApiClient --> WorkspaceTests
        MockApiClient --> OptimisationTests
        MockApiClient --> ExportTests

        MockWebSocket --> AnalyticsTests
        MockWebSocketServer --> WorkspaceTests

        GraphMocks --> MockApiClient
        AgentMocks --> MockApiClient
        AnalyticsMocks --> MockApiClient
        WorkspaceMocks --> MockApiClient

        style SecurityAlert fill:#ffcdd2
        style TestCount fill:#fff3e0
    end
```

---

## Component Interaction Matrix

### Core Component Dependencies

| Component | Dependencies | Provides |
|-----------|-------------|----------|
| **App.tsx** | AppInitialiser, MainLayout, Quest3AR | Root application structure |
| **MainLayout.tsx** | GraphCanvas, IntegratedControlPanel, BotsDataProvider | Primary layout management |
| **GraphCanvas.tsx** | GraphManager, BotsVisualisation, SelectiveBloom | 3D graph rendering |
| **WebSocketService.ts** | BinaryProtocol, BatchQueue | Real-time communication |
| **SettingsStore.ts** | AutoSaveManager, SettingsAPI | Configuration management |
| **BotsDataProvider.tsx** | BotsWebSocketIntegration, AgentPollingService | Agent data context |
| **UnifiedApiClient.ts** | None (base client) | HTTP communication (119 refs) |
| **XRCoreProvider.tsx** | Quest3Integration, XRManagers | WebXR functionality |

### Feature Module Integration

| Feature Module | Core Integration Points | External Dependencies |
|----------------|-------------------------|----------------------|
| **Graph System** | WebSocketService, GraphDataManager | Three.js, R3F |
| **Agent System** | BotsWebSocketIntegration, REST APIs | MCP Protocol |
| **Settings System** | SettingsStore, AutoSaveManager | Backend API |
| **XR System** | XRCoreProvider, Device Detection | WebXR APIs |
| **Physics System** | GraphCanvas, Settings | Three.js Physics |
| **Voice System** | AudioInputService, WebSocket | Browser Media APIs |
| **Analytics System** | REST APIs, Settings | Chart.js |

---

## Performance Considerations

### Current Performance Optimisations

1. **WebSocket Binary Protocol**: 34-byte format reduces bandwidth by 95% vs JSON
2. **Batch Processing**: Debounced settings saves and graph updates
3. **Lazy Loading**: Settings sections loaded on-demand
4. **Virtualised Components**: Performance UI for large datasets
5. **React Optimisations**: Memo, callback, and effect optimisations
6. **Three.js Optimisations**: Instance rendering, LOD, frustum culling

### Identified Performance Bottlenecks

1. **Agent Polling**: 10-second REST polling could be optimised with WebSocket events
2. **Graph Rendering**: Large graphs (>10k nodes) may impact performance
3. **Settings Persistence**: Individual setting saves could benefit from batching
4. **XR Performance**: AR/VR mode may need additional optimisations

---

## Major Architecture Updates

### API Layer Migration - Complete

**Achievement**: Complete migration from deprecated apiService to UnifiedApiClient:
- **31 UnifiedApiClient references** across the entire codebase
- **Zero apiService references** remaining
- **Internal fetch() calls** only within UnifiedApiClient implementation
- **External resource downloads** properly isolated in downloadHelpers
- **Debug fetch()** in public/debug.html for testing only
- **Consistent error handling** and request/response patterns
- **Performance improvements** through unified caching and request batching

**Migration Results**:
```mermaid
graph LR
    subgraph "Before: Mixed API Clients"
        A1[apiService<br/>Legacy Client]
        A2[fetch() calls<br/>Direct API calls]
        A3[UnifiedApiClient<br/>New Client]
    end

    subgraph "After: Single Source"
        B1[UnifiedApiClient<br/>119 References]
        B2[fetch() calls<br/>Internal & Debug Only]
        B3[Consistent Patterns<br/>Error Handling]
    end

    A1 -.-> B1
    A2 -.-> B2
    A3 -.-> B1
```

### Voice System Architecture - Designed but Not Deployed

**Implementation Status**: Centralised architecture designed but legacy hook still active:
- **VoiceProvider**: Context provider designed but unused
- **useVoiceInteractionCentralised**: Modern hook with 9 specialised hooks (not 8)
- **Legacy useVoiceInteraction**: Still actively used by all voice components
- **9 Specialised Hooks**: Designed but not integrated into production
- **Migration Pending**: Components need updating to use centralised system

### Mock Data Enhancement - Complete

**Upgrade**: MockAgentStatus now fully matches server structure:
- **Comprehensive Fields**: All server fields properly mapped
- **CamelCase Conversion**: Proper JavaScript naming conventions
- **Type Safety**: Full TypeScript compatibility
- **Server Compatibility**: 1:1 mapping with actual server responses

### Duplicate Polling Fix - Completed

**Problem Resolved**: The system previously had triple data polling causing race conditions:
1. AgentPollingService - REST polling every 1-5 seconds
2. BotsWebSocketIntegration - WebSocket timer polling every 2 seconds
3. BotsDataContext - Subscribing to both sources

**Solution Implemented**:
- **Single Source Strategy**: WebSocket binary for positions, REST for metadata
- **Conservative Polling**: Reduced to 3s active, 15s idle intervals
- **Eliminated Race Conditions**: Removed duplicate subscription paths
- **70% Server Load Reduction**: From aggressive 1s polling to smart intervals

### Graph Visualisation System

**Implementation**: Unified graph with protocol support for dual types:
- **Unified Implementation**: Single graph system handles all node types
- **Protocol Support**: Binary protocol distinguishes Knowledge (0x4000) and Agent (0x8000) nodes
- **Performance Optimised**: DualGraphPerformanceMonitor tracks system metrics
- **HolographicDataSphere Module**: Contains HologramEnvironment as internal component

### Enhanced Binary Protocol

**Components Added**:
- `BinaryWebSocketProtocol.ts`: Protocol handler layer
- Enhanced message type support (Control, Data, Stream, Agent)
- Voice streaming integration for STT/TTS
- Improved validation and error handling

---

## Current Implementation Status

### ✅ Fully Implemented Components
- Core React application structure (App.tsx, MainLayout, Quest3AR)
- **Binary Protocol System**: WebSocketService.ts, BinaryWebSocketProtocol.ts, binaryProtocol.ts
- **Settings Architecture**: settingsStore.ts, AutoSaveManager.ts, LazySettingsSections, VirtualisedSettingsGroup
- **Graph Visualisation**: GraphCanvas, GraphManager, unified implementation with type support
- **Agent System**: BotsDataProvider, BotsWebSocketIntegration, AgentPollingService (duplicate polling fixed)
- **REST API Layer**: UnifiedApiClient.ts (31 references), all API endpoints
- **XR/AR System**: XRCoreProvider, Quest3Integration, XR Components
- **Mock Data System**: Comprehensive MockAgentStatus with full server compatibility
- **Performance Monitoring**: performanceMonitor, dualGraphPerformanceMonitor
- **Utilities**: loggerConfig, debugConfig, classNameUtils, downloadHelpers
- **Error Handling**: ErrorBoundary, ConnectionWarning, BrowserSupportWarning

### 🎯 Current Status & Existing Implementations

**✅ Recently Implemented:**
- **API Layer Migration Complete**: UnifiedApiClient is the primary API client (31 references)
- **Voice System Architecture**: Centralised system designed (legacy hook still active)
- **Mock System Complete**: Comprehensive mocks for testing (not real API calls)
- **Position Throttling**: Smart update hooks for performance
- **Task Management**: Support for remove/pause/resume endpoints
- **Graph System**: Unified implementation with protocol support for dual types

**🔧 Needs Refinement:**
- **Voice System Migration**: Move from legacy hook to centralised architecture
- **Authentication**: Nostr authentication runtime testing
- **Complex Agent Patterns**: Advanced swarm orchestration UI controls
- **Test Activation**: Tests disabled due to security concerns

### 🔮 Future Features (Not Current Requirements)
- **Advanced Analytics**: Not needed on clients (server-side only)
- **Physics Engine**: Server-driven, controlled via commands
- **XR System Enhancements**: Future upgrade
- **Vircadia Integration**: FUTURE parallel multi-user VR system

### 📋 Architecture Strengths
- Well-structured feature modules
- Comprehensive error handling
- Performance-optimised data flows
- Extensible component architecture
- Strong separation of concerns
- Modern React patterns and hooks

---

## Integration with Backend System

This client architecture integrates with the Rust backend through:

1. **WebSocket Binary Protocol**: High-performance graph data streaming
2. **REST APIs**: Configuration, metadata, and control operations
3. **Voice Streaming**: Binary audio data for STT/TTS processing
4. **MCP Integration**: Multi-agent system coordination (backend-handled)

The client maintains clear separation between:
- **Real-time data** (WebSocket binary protocol)
- **Configuration data** (REST APIs with persistence)
- **Control operations** (REST APIs with immediate response)

---

## Interface Layer Integration Status

```mermaid
graph TB
    subgraph "Client Interface Layers - Operational"
        subgraph "REST API Integration"
            UnifiedClient["UnifiedApiClient<br/>119 References"]
            SettingsAPI["Settings API<br/>9 Endpoints Active"]
            GraphAPI["Graph API<br/>2 Endpoints Active"]
            AgentAPI["Agent/Bot API<br/>8 Endpoints + Task Management"]
        end

        subgraph "WebSocket Binary Protocol"
            WSService["WebSocket Service<br/>80% Traffic Reduction"]
            BinaryProtocol["Binary Node Protocol<br/>34-byte Format"]
            PositionUpdates["Position Updates<br/>Real-time Streaming"]
        end

        subgraph "Field Conversion System"
            SerdeConversion["Serde Conversion<br/>camelCase ↔ snake-case"]
            FieldNormalisation["Field Normalisation<br/>config/mod.rs Fixes"]
            AutoMapping["Automatic Mapping<br/>130+ Structs"]
        end

        subgraph "Actor Communication"
            MessageRouting["Message Routing<br/>GraphServiceActor<br/>❌ DEPRECATED (Nov 2025)"]
            ActorMessages["Actor Messages<br/>Async Communication"]
            StateSync["State Synchronisation<br/>Real-time Updates"]
        end
    end

    UnifiedClient --> SettingsAPI
    UnifiedClient --> GraphAPI
    UnifiedClient --> AgentAPI

    WSService --> BinaryProtocol
    BinaryProtocol --> PositionUpdates

    SettingsAPI --> SerdeConversion
    SerdeConversion --> FieldNormalisation
    FieldNormalisation --> AutoMapping

    MessageRouting --> ActorMessages
    ActorMessages --> StateSync

    style UnifiedClient fill:#c8e6c9
    style WSService fill:#e3f2fd
    style SerdeConversion fill:#fff3e0
    style MessageRouting fill:#f3e5f5
```

---

*This architecture diagram represents the current state of the VisionFlow client as of 2025-09-27, based on direct source code analysis of 442 TypeScript/React files and comprehensive swarm-based auditing. Major corrections include accurate API reference counts (119), clarification of voice system dual implementation status, test infrastructure using mocks rather than real API calls, and unified graph implementation with protocol support for dual types.*