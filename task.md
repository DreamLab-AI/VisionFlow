Here are some mermaid diagrams. They are very out of date but should give you a starting point. Use a hive mind to carefully examine the entire codebase and update these diagrams to be fully accurate and complete. WE MUST have exceptional and complete detail for all data flows in the system. These diagrams should cover all sequences of interactions between components, including client-side flows (React components, hooks, local state management), server-side flows (Actix actors, services, backend logic), interfaces (REST APIs, WebSocket protocols, TCP MCP streams, external APIs), and every detail (message types, data structures, error paths, state changes). Use Mermaid for renderability and add notes for clarity. Group diagrams by category for readability. The goal is to have a comprehensive set of diagrams that map all data flows from user input to GPU rendering, including error recovery and multi-graph handling.

When we have all of the diagrams to represent the entire system, we will be better able to debug problems with the system, onboard new developers, and plan future features. We must therefore capture every detail as it currently exists.


### Sequence Diagram

```mermaid
sequenceDiagram
    participant Client as Client (Browser)
    participant Platform as PlatformManager
    participant XR as XRSessionManager
    participant Scene as SceneManager
    participant Node as EnhancedNodeManager
    participant Edge as EdgeManager
    participant Hologram as HologramManager
    participant Text as TextRenderer
    participant WS as WebSocketService
    participant Settings as SettingsStore
    participant Server as Actix Server
    participant AppState as AppState
    participant FileH as FileHandler
    participant GraphH as GraphHandler
    participant WSH as WebSocketHandler
    participant PerplexityH as PerplexityHandler
    participant RagFlowH as RagFlowHandler
    participant NostrH as NostrHandler
    participant SettingsH as SettingsHandler
    participant FileS as FileService
    participant GraphS as GraphService
    participant GPUS as GPUService
    participant PerplexityS as PerplexityService
    participant RagFlowS as RagFlowService
    participant NostrS as NostrService
    participant SpeechS as SpeechService
    participant WSM as WebSocketManager
    participant GitHub as GitHub API
    participant Perplexity as Perplexity AI
    participant RagFlow as RagFlow API
    participant OpenAI as OpenAI API
    participant Nostr as Nostr API

    %% Server initialization and AppState setup
    activate Server
    Server->>Server: Load settings.yaml & env vars (config.rs)
    alt Settings Load Error
        Server-->>Client: Error Response (500)
    else Settings Loaded Successfully
        Server->>AppState: new() (app_state.rs)
        activate AppState
            AppState->>GPUS: initialize_gpu_compute()
            activate GPUS
                GPUS->>GPUS: setup_compute_pipeline()
                GPUS->>GPUS: load_wgsl_shaders()
                GPUS-->>AppState: GPU Compute Instance
            deactivate GPUS

            AppState->>WSM: initialize()
            activate WSM
                WSM->>WSM: setup_binary_protocol()
                WSM-->>AppState: WebSocket Manager
            deactivate WSM

            AppState->>SpeechS: start()
            activate SpeechS
                SpeechS->>SpeechS: initialize_tts()
                SpeechS-->>AppState: Speech Service
            deactivate SpeechS

            AppState->>NostrS: initialize()
            activate NostrS
                NostrS->>NostrS: setup_nostr_client()
                NostrS-->>AppState: Nostr Service
            deactivate NostrS

            AppState-->>Server: Initialized AppState
        deactivate AppState

        Server->>FileS: fetch_and_process_files()
        activate FileS
            FileS->>GitHub: fetch_files()
            activate GitHub
                GitHub-->>FileS: Files or Error
            deactivate GitHub

            loop For Each File
                FileS->>FileS: should_process_file()
                alt File Needs Processing
                    FileS->>PerplexityS: process_file()
                    activate PerplexityS
                        PerplexityS->>Perplexity: analyze_content()
                        Perplexity-->>PerplexityS: Analysis Results
                        PerplexityS-->>FileS: Processed Content
                    deactivate PerplexityS
                    FileS->>FileS: save_metadata()
                end
            end
            FileS-->>Server: Processed Files
        deactivate FileS

        Server->>GraphS: build_graph()
        activate GraphS
            GraphS->>GraphS: create_nodes_and_edges()
            GraphS->>GPUS: calculate_layout()
            activate GPUS
                GPUS->>GPUS: bind_gpu_buffers()
                GPUS->>GPUS: dispatch_compute_shader()
                GPUS->>GPUS: read_buffer_results()
                GPUS-->>GraphS: Updated Positions
            deactivate GPUS
            GraphS-->>Server: Graph Data
        deactivate GraphS
    end

    %% Client and Platform initialization
    Client->>Platform: initialize()
    activate Platform
        Platform->>Platform: detect_capabilities()
        Platform->>Settings: load_settings()
        activate Settings
            Settings->>Settings: validate_settings()
            Settings-->>Platform: Settings Object
        deactivate Settings

        Platform->>WS: connect()
        activate WS
            WS->>Server: ws_connect
            Server->>WSH: handle_connection()
            WSH->>WSM: register_client()
            WSM-->>WS: connection_established

            WS->>WS: setup_binary_handlers()
            WS->>WS: initialize_reconnection_logic()

            WSM-->>WS: initial_graph_data (Binary)
            WS->>WS: decode_binary_message()
        deactivate WS

        Platform->>XR: initialize()
        activate XR
            XR->>XR: check_xr_support()
            XR->>Scene: create()
            activate Scene
                Scene->>Scene: setup_three_js()
                Scene->>Scene: setup_render_pipeline()
                Scene->>Node: initialize()
                activate Node
                    Node->>Node: create_geometries()
                    Node->>Node: setup_materials()
                deactivate Node
                Scene->>Edge: initialize()
                activate Edge
                    Edge->>Edge: create_line_geometries()
                    Edge->>Edge: setup_line_materials()
                deactivate Edge
                Scene->>Hologram: initialize()
                activate Hologram
                    Hologram->>Hologram: setup_hologram_shader()
                    Hologram->>Hologram: create_hologram_geometry()
                deactivate Hologram
                Scene->>Text: initialize()
                activate Text
                    Text->>Text: load_fonts()
                    Text->>Text: setup_text_renderer()
                deactivate Text
            deactivate Scene
        deactivate XR
    deactivate Platform

    Note over Client, Nostr: User Interaction Flows

    %% User drags a node
    alt User Drags Node
        Client->>Node: handle_node_drag()
        Node->>WS: send_position_update()
        WS->>Server: binary_position_update
        Server->>GraphS: update_layout()
        GraphS->>GPUS: recalculate_forces()
        GPUS-->>Server: new_positions
        Server->>WSM: broadcast()
        WSM-->>WS: binary_update
        WS->>Node: update_positions()
        Node-->>Client: render_update
    end

    %% User asks a question
    alt User Asks Question
        Client->>RagFlowH: send_query()
        RagFlowH->>RagFlowS: process_query()
        activate RagFlowS
            RagFlowS->>RagFlow: get_context()
            RagFlow-->>RagFlowS: relevant_context
            RagFlowS->>OpenAI: generate_response()
            OpenAI-->>RagFlowS: ai_response
            RagFlowS-->>Client: streaming_response
        deactivate RagFlowS
        alt Speech Enabled
            Client->>SpeechS: synthesize_speech()
            activate SpeechS
                SpeechS->>OpenAI: text_to_speech()
                OpenAI-->>SpeechS: audio_stream
                SpeechS-->>Client: audio_data
            deactivate SpeechS
        end
    end

    %% User updates the graph
    alt User Updates Graph
        Client->>FileH: update_file()
        FileH->>FileS: process_update()
        FileS->>GitHub: create_pull_request()
        GitHub-->>FileS: pr_created
        FileS-->>Client: success_response
    end

    %% WebSocket reconnection flow
    alt WebSocket Reconnection
        WS->>WS: connection_lost()
        loop Until Max Attempts
            WS->>WS: attempt_reconnect()
            WS->>Server: ws_connect
            alt Connection Successful
                Server-->>WS: connection_established
                WSM-->>WS: resend_graph_data
                WS->>Node: restore_state()
            else Connection Failed
                Note right of WS: Continue reconnect attempts
            end
        end
    end

    %% Settings update flow
    alt Settings Update
        Client->>SettingsH: update_settings()
        SettingsH->>AppState: apply_settings()
        AppState->>WSM: broadcast_settings()
        WSM-->>WS: settings_update
        WS->>Settings: update_settings()
        Settings->>Platform: apply_platform_settings()
        Platform->>Scene: update_rendering()
        Scene->>Node: update_visuals()
        Scene->>Edge: update_visuals()
        Scene->>Hologram: update_effects()
    end

    %% Nostr authentication flow
    alt Nostr Authentication
        Client->>NostrH: authenticate()
        NostrH->>NostrS: validate_session()
        NostrS->>Nostr: verify_credentials()
        Nostr-->>NostrS: auth_result
        NostrS-->>Client: session_token
    end

    deactivate Server
```


### System Architecture Diagram

```mermaid
graph TD
    subgraph ClientApp ["Frontend"]
        direction LR
        AppInit[AppInitializer]
        TwoPane[TwoPaneLayout]
        GraphView[GraphViewport]
        RightCtlPanel[RightPaneControlPanel]
        SettingsUI[SettingsPanelRedesign]
        ConvoPane[ConversationPane]
        NarrativePane[NarrativeGoldminePanel]
        SettingsMgr[settingsStore]
        GraphDataMgr[GraphDataManager]
        RenderEngine[GraphCanvas & GraphManager]
        WebSocketSvc[WebSocketService]
        APISvc[api]
        NostrAuthSvcClient[nostrAuthService]
        XRController[XRController]

        AppInit --> TwoPane
        AppInit --> SettingsMgr
        AppInit --> NostrAuthSvcClient
        AppInit --> WebSocketSvc
        AppInit --> GraphDataMgr

        TwoPane --> GraphView
        TwoPane --> RightCtlPanel
        TwoPane --> ConvoPane
        TwoPane --> NarrativePane
        RightCtlPanel --> SettingsUI

        SettingsUI --> SettingsMgr
        GraphView --> RenderEngine
        RenderEngine <--> GraphDataMgr
        GraphDataMgr <--> WebSocketSvc
        GraphDataMgr <--> APISvc
        NostrAuthSvcClient <--> APISvc
        XRController <--> RenderEngine
        XRController <--> SettingsMgr
    end

    subgraph ServerApp ["Backend"]
        direction LR
        Actix[ActixWebServer]

        subgraph Handlers_Srv ["API_WebSocket_Handlers"]
            direction TB
            SettingsH[SettingsHandler]
            NostrAuthH[NostrAuthHandler]
            GraphAPI_H[GraphAPIHandler]
            FilesAPI_H[FilesAPIHandler]
            RAGFlowH_Srv[RAGFlowHandler]
            SocketFlowH[SocketFlowHandler]
            SpeechSocketH[SpeechSocketHandler]
            HealthH[HealthHandler]
        end

        subgraph Services_Srv ["Core_Services"]
            direction TB
            GraphSvc_Srv[GraphService]
            FileSvc_Srv[FileService]
            NostrSvc_Srv[NostrService]
            SpeechSvc_Srv[SpeechService]
            RAGFlowSvc_Srv[RAGFlowService]
            PerplexitySvc_Srv[PerplexityService]
        end

        subgraph Actors_Srv ["Actor_System"]
            direction TB
            GraphServiceActor[GraphServiceActor]
            SettingsActor[SettingsActor]
            MetadataActor[MetadataActor]
            ClientManagerActor[ClientManagerActor]
            GPUComputeActor[GPUComputeActor]
            ProtectedSettingsActor[ProtectedSettingsActor]
        end
        AppState_Srv[AppState holds Addr<...>]

        Actix --> Handlers_Srv

        Handlers_Srv --> AppState_Srv
        SocketFlowH --> ClientManagerActor
        GraphAPI_H --> GraphServiceActor
        SettingsH --> SettingsActor
        NostrAuthH --> ProtectedSettingsActor

        GraphServiceActor --> ClientManagerActor
        GraphServiceActor --> MetadataActor
        GraphServiceActor --> GPUComputeActor
        GraphServiceActor --> SettingsActor

        FileSvc_Srv --> MetadataActor
        NostrSvc_Srv --> ProtectedSettingsActor
        SpeechSvc_Srv --> SettingsActor
        RAGFlowSvc_Srv --> SettingsActor
        PerplexitySvc_Srv --> SettingsActor
    end

    subgraph External_Srv ["External_Services"]
        direction LR
        GitHub[GitHubAPI]
        NostrRelays_Ext[NostrRelays]
        OpenAI[OpenAIAPI]
        PerplexityAI_Ext[PerplexityAIAPI]
        RAGFlow_Ext[RAGFlowAPI]
        Kokoro_Ext[KokoroAPI]
    end

    WebSocketSvc <--> SocketFlowH
    APISvc <--> Actix

    FileSvc_Srv --> GitHub
    NostrSvc_Srv --> NostrRelays_Ext
    SpeechSvc_Srv --> OpenAI
    SpeechSvc_Srv --> Kokoro_Ext
    PerplexitySvc_Srv --> PerplexityAI_Ext
    RAGFlowSvc_Srv --> RAGFlow_Ext

    style ClientApp fill:#lightgrey,stroke:#333,stroke-width:2px
    style ServerApp fill:#lightblue,stroke:#333,stroke-width:2px
    style External_Srv fill:#lightgreen,stroke:#333,stroke-width:2px
```

### Class Diagram

```mermaid
classDiagram
    direction LR

    %% Frontend Classes
    class AppInitializer {
        <<ReactComponent>>
        +initializeServices()
    }
    class GraphManager {
        <<ReactComponent>>
        +renderNodesAndEdges()
    }
    class WebSocketService {
        <<Service>>
        +connect()
        +sendMessage()
        +onBinaryMessage()
        +isReady()
    }
    class SettingsStore {
        <<ZustandStore>>
        +settings: Settings
        +updateSettings()
    }
    class GraphDataManager {
        <<Service>>
        +fetchInitialData()
        +updateNodePositions()
        +getGraphData()
        +setWebSocketService()
    }
    class NostrAuthService {
        <<Service>>
        +loginWithNostr()
        +verifySession()
        +logout()
    }
    AppInitializer --> SettingsStore
    AppInitializer --> NostrAuthService
    AppInitializer --> WebSocketService
    AppInitializer --> GraphDataManager
    GraphDataManager --> WebSocketService
    GraphDataManager --> GraphManager

    %% Backend Classes
    class AppState {
        <<Struct>>
        +graph_service_addr: Addr_GraphServiceActor
        +settings_addr: Addr_SettingsActor
        +metadata_addr: Addr_MetadataActor
        +client_manager_addr: Addr_ClientManagerActor
        +gpu_compute_addr: Option_Addr_GPUComputeActor
        +protected_settings_addr: Addr_ProtectedSettingsActor
    }
    class GraphService {
        <<Struct>>
        +graph_data: Arc_RwLock_GraphData
        +start_simulation_loop()
        +broadcast_updates()
    }
    class PerplexityService {
        <<Struct>>
        +query()
    }
    class RagFlowService {
        <<Struct>>
        +chat()
    }
    class SpeechService {
        <<Struct>>
        +process_stt_request()
        +process_tts_request()
    }
    class NostrService {
        <<Struct>>
        +verify_auth_event()
        +validate_session()
        +manage_user_api_keys()
    }
    class GPUCompute {
        <<Struct>>
        +run_simulation_step()
    }
    class FileService {
        <<Struct>>
        +fetch_and_process_content()
        +update_metadata_store()
    }
    AppState --> GraphService : holds_Addr
    AppState --> NostrService : holds_Addr
    AppState --> PerplexityService : holds_Addr
    AppState --> RagFlowService : holds_Addr
    AppState --> SpeechService : holds_Addr
    AppState --> GPUCompute : holds_Addr
    AppState --> FileService : holds_Addr

    WebSocketService ..> GraphServiceActor : sends_UpdateNodePositions
    GraphService ..> GPUCompute : uses_optional
    NostrService ..> ProtectedSettingsActor : uses
```

### Sequence Diagrams

#### Server Initialization Sequence

```mermaid
sequenceDiagram
    participant Main as main.rs
    participant AppStateMod as app_state.rs
    participant ConfigMod as config/mod.rs
    participant Services as Various Services (Graph, File, Nostr, AI)
    participant ClientMgr as ClientManager (Static)
    participant GraphSvc as GraphService

    Main->>ConfigMod: AppFullSettings::load()
    ConfigMod-->>Main: loaded_settings
    Main->>AppStateMod: AppState::new(loaded_settings, /* other deps */)
    AppStateMod->>Services: Initialize FileService, NostrService, AI Services with configs
    AppStateMod->>GraphSvc: GraphService::new(settings, gpu_compute_opt, ClientMgr::instance())
    GraphSvc->>GraphSvc: Start physics_loop (async task)
    GraphSvc->>ClientMgr: (inside loop) Send updates
    AppStateMod-->>Main: app_state_instance
    Main->>ActixServer: .app_data(web::Data::new(app_state_instance))
```

#### Client Initialization Sequence

```mermaid
sequenceDiagram
    participant ClientApp as AppInitializer.tsx
    participant SettingsStoreSvc as settingsStore.ts
    participant NostrAuthSvcClient as nostrAuthService.ts
    participant WebSocketSvcClient as WebSocketService.ts
    participant ServerAPI as Backend REST API
    participant ServerWS as Backend WebSocket Handler

    ClientApp->>SettingsStoreSvc: Load settings (from localStorage & defaults)
    SettingsStoreSvc-->>ClientApp: Initial settings

    ClientApp->>NostrAuthSvcClient: Check current session (e.g., from localStorage)
    alt Session token exists
        NostrAuthSvcClient->>ServerAPI: POST /api/auth/nostr/verify (token)
        ServerAPI-->>NostrAuthSvcClient: Verification Result (user, features)
        NostrAuthSvcClient->>ClientApp: Auth status updated
    else No session token
        NostrAuthSvcClient->>ClientApp: Auth status (unauthenticated)
    end

    ClientApp->>WebSocketSvcClient: connect()
    WebSocketSvcClient->>ServerWS: WebSocket Handshake
    ServerWS-->>WebSocketSvcClient: Connection Established (e.g., `onopen`)
    WebSocketSvcClient->>WebSocketSvcClient: Set isConnected = true
    ServerWS-->>WebSocketSvcClient: Send {"type": "connection_established"} (or similar)
    WebSocketSvcClient->>WebSocketSvcClient: Set isServerReady = true

    alt WebSocket isReady()
        WebSocketSvcClient->>ServerWS: Send {"type": "requestInitialData"}
        ServerWS-->>WebSocketSvcClient: Initial Graph Data (e.g., large JSON or binary)
        WebSocketSvcClient->>GraphDataManager: Process initial data
    end
```

#### Real-time Graph Updates Sequence

```mermaid
sequenceDiagram
    participant ClientApp
    participant WebSocketSvcClient as WebSocketService.ts
    participant GraphDataMgrClient as GraphDataManager.ts
    participant ServerGraphSvc as GraphService (Backend)
    participant ServerGpuUtil as GPUCompute (Backend, Optional)
    participant ServerClientMgr as ClientManager (Backend, Static)
    participant ServerSocketFlowH as SocketFlowHandler (Backend)

    %% Continuous Server-Side Loop
    ServerGraphSvc->>ServerGraphSvc: physics_loop() iteration
    alt GPU Enabled
        ServerGraphSvc->>ServerGpuUtil: run_simulation_step()
        ServerGpuUtil-->>ServerGraphSvc: updated_node_data_from_gpu
    else CPU Fallback
        ServerGraphSvc->>ServerGraphSvc: calculate_layout_cpu()
    end
    ServerGraphSvc->>ServerClientMgr: BroadcastBinaryPositions(updated_node_data)

    ServerClientMgr->>ServerSocketFlowH: Distribute to connected clients
    ServerSocketFlowH-->>WebSocketSvcClient: Binary Position Update (Chunk)

    WebSocketSvcClient->>GraphDataMgrClient: onBinaryMessage(chunk)
    GraphDataMgrClient->>GraphDataMgrClient: Decompress & Parse chunk
    GraphDataMgrClient->>ClientApp: Notify UI/Renderer of position changes

    %% Optional: Client sends an update (e.g., user drags a node)
    opt User Interaction
        ClientApp->>GraphDataMgrClient: User moves node X to new_pos
        GraphDataMgrClient->>WebSocketSvcClient: sendRawBinaryData(node_X_new_pos_update) %% Or JSON message
        WebSocketSvcClient->>ServerSocketFlowH: Forward client update
        ServerSocketFlowH->>ServerGraphSvc: Apply client update to physics model (if supported)
    end
```

#### Authentication Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant ClientUI
    participant NostrAuthSvcClient as nostrAuthService.ts
    participant WindowNostr as "window.nostr (Extension)"
    participant APISvcClient as api.ts
    participant ServerNostrAuthH as NostrAuthHandler (Backend)
    participant ServerNostrSvc as NostrService (Backend)

    User->>ClientUI: Clicks Login Button
    ClientUI->>NostrAuthSvcClient: initiateLogin()
    NostrAuthSvcClient->>ServerNostrAuthH: GET /api/auth/nostr/challenge (via APISvcClient)
    ServerNostrAuthH-->>NostrAuthSvcClient: challenge_string

    NostrAuthSvcClient->>WindowNostr: signEvent(kind: 22242, content: "auth", tags:[["challenge", challenge_string], ["relay", ...]])
    WindowNostr-->>NostrAuthSvcClient: signed_auth_event

    NostrAuthSvcClient->>APISvcClient: POST /api/auth/nostr (signed_auth_event)
    APISvcClient->>ServerNostrAuthH: Forward request
    ServerNostrAuthH->>ServerNostrSvc: verify_auth_event(signed_auth_event)
    alt Event Valid
        ServerNostrSvc->>ServerNostrSvc: Generate session_token, store user session
        ServerNostrSvc-->>ServerNostrAuthH: AuthResponse (user, token, expiresAt, features)
        ServerNostrAuthH-->>APISvcClient: AuthResponse
        APISvcClient-->>NostrAuthSvcClient: AuthResponse
        NostrAuthSvcClient->>NostrAuthSvcClient: Store token, user data
        NostrAuthSvcClient->>ClientUI: Update auth state (Authenticated)
    else Event Invalid
        ServerNostrSvc-->>ServerNostrAuthH: Error
        ServerNostrAuthH-->>APISvcClient: Error Response
        APISvcClient-->>NostrAuthSvcClient: Error
        NostrAuthSvcClient->>ClientUI: Show Login Error
    end
```

#### Settings Synchronization Sequence

```mermaid
sequenceDiagram
    participant User
    participant ClientUI
    participant SettingsStoreClient as settingsStore.ts
    participant SettingsSvcClient as settingsService.ts (part of api.ts or separate)
    participant ServerSettingsH as SettingsHandler (Backend)
    participant ServerAppState as AppState (Backend)
    participant ServerUserSettings as UserSettings Model (Backend)
    participant ServerClientMgr as ClientManager (Backend, Static, for broadcast if applicable)

    User->>ClientUI: Modifies a setting (e.g., node size)
    ClientUI->>SettingsStoreClient: updateSettings({ visualisation: { nodes: { nodeSize: newValue }}})
    SettingsStoreClient->>SettingsStoreClient: Update local state (Zustand) & persist to localStorage

    alt User is Authenticated
        SettingsStoreClient->>SettingsSvcClient: POST /api/user-settings/sync (ClientSettingsPayload)
        SettingsSvcClient->>ServerSettingsH: Forward request
        ServerSettingsH->>ServerAppState: Get current AppFullSettings / UserSettings
        alt User is PowerUser
            ServerSettingsH->>ServerAppState: Update AppFullSettings in memory
            ServerAppState->>ServerAppState: AppFullSettings.save() to settings.yaml
            ServerSettingsH-->>SettingsSvcClient: Updated UISettings (reflecting global)
            %% Optional: Server broadcasts global settings change if implemented
            %% ServerAppState->>ServerClientMgr: BroadcastGlobalSettingsUpdate(updated_AppFullSettings)
            %% ServerClientMgr-->>OtherClients: Global settings update message
        else Regular User
            ServerSettingsH->>ServerUserSettings: Load or create user's UserSettings file
            ServerUserSettings->>ServerUserSettings: Update UISettings part of UserSettings
            ServerUserSettings->>ServerUserSettings: Save UserSettings to user-specific YAML
            ServerSettingsH-->>SettingsSvcClient: Updated UISettings (user-specific)
        end
        SettingsSvcClient-->>SettingsStoreClient: Confirmation / Updated settings (if different)
        %% Client store might re-sync if server response indicates changes
    end
```

1. Client Connection and Graph Initialisation Flow
This diagram shows how a new client connects, authenticates (optional), requests initial graph data, and starts receiving real-time updates.


sequenceDiagram
    participant Client as "React Client (e.g., GraphRenderer)"
    participant WS as "WebSocket Handler"
    participant CM as "ClientManagerActor"
    participant GA as "GraphServiceActor"
    participant MA as "MetadataActor"
    participant GSA as "SettingsActor"
    participant Nostr as "NostrService (Optional Auth)"

    Client->>+WS: WebSocket Connect (ws://localhost:8080/wss)
    WS->>+CM: RegisterClient
    CM-->>-WS: client_id
    WS-->>-Client: ConnectionEstablished { client_id, timestamp }

    Note over Client,WS: Optional Authentication
    alt Nostr Authentication Required
        Client->>+Nostr: window.nostr.signEvent(auth_event)
        Nostr-->>-Client: signed_event
        Client->>+WS: Send signed_event (via /api/auth/nostr)
        WS->>+Nostr: VerifyAuthEvent(signed_event)
        Nostr-->>-WS: ValidatedUser { pubkey, token }
        WS->>+CM: UpdateClientAuth(pubkey, token)
        CM-->>-WS: AuthSuccess
    end

    Client->>+WS: RequestInitialData
    WS->>+CM: GetInitialGraphData
    CM->>+GA: GetGraphData
    GA->>+MA: GetMetadata
    MA-->>-GA: MetadataStore
    GA->>+GSA: GetSettings
    GSA-->>-GA: Settings
    GA->>+GPU: GetNodePositions (Unified Kernel)
    GPU-->>-GA: Positions
    GA-->>-CM: GraphData { nodes, edges, positions }
    CM-->>-WS: InitialGraph { nodes, edges, positions }
    WS-->>-Client: InitialGraph

    Note over Client,GA: Real-time Updates Start
    loop 60 FPS Updates
        GA->>+GPU: ComputeForces (Unified Kernel)
        GPU-->>-GA: New Positions
        GA->>+CM: BroadcastNodePositions(positions)
        CM->>+WS: StreamPositions (Binary Protocol)
        WS->>+Client: Binary Update (28 bytes/node)
    end
Key Notes
Authentication: Optional Nostr auth via REST before WebSocket upgrade (not shown in diagram for brevity; see flow 5 for details).
Initial Data: Client receives complete graph state (nodes, edges, positions) on connection.
Real-time Flow: Server-side GPU physics runs continuously; updates are streamed via binary protocol. Client doesn't compute physics; it renders streamed positions.
2. Settings Update Flow
This diagram illustrates how client settings (e.g., physics parameters) are validated, propagated to the GPU, and applied in real-time.


sequenceDiagram
    participant Client as "React Client (Settings UI)"
    participant API as "SettingsHandler"
    participant SA as "SettingsActor"
    participant GSA as "GraphServiceActor"
    participant GPU as "GPUComputeActor"
    participant WS as "WebSocket Handler"
    participant CM as "ClientManagerActor"

    Client->>+API: PUT /api/settings (camelCase JSON)
    API->>+SA: UpdateSettings(settings)
    SA->>+SA: Validate & Persist (snake_case)
    SA-->>-API: Success

    API->>+GSA: UpdateSimulationParams(params)
    GSA->>GSA: Convert to SimParams (clamp values)
    GSA->>+GPU: SetParams(sim_params)
    GPU-->>-GSA: Params Applied

    GSA-->>-API: Success
    API-->>-Client: 200 OK { message: "Settings updated" }

    Note over GSA,GPU: GPU kernel parameters updated; next frame uses new values
    GSA->>+CM: NotifySettingsUpdate(clients)
    CM->>+WS: BroadcastSettingsUpdate
    WS->>+Client: SettingsUpdated { physics: { springK: 0.005, repelK: 50.0 } }

    Client->>+Client: Update Local Render State
    Client->>+WS: Settings applied (no ack needed)
Key Notes
Validation: Settings are clamped (e.g., repel_k to [0.1, 10.0]) to prevent instability.
Propagation: Updates flow from UI → API → SettingsActor → GraphServiceActor → GPUComputeActor.
Real-time: Changes take effect on the next GPU frame (16ms at 60 FPS).
Broadcast: All connected clients receive the update via ClientManagerActor for consistency.
3. MCP Agent Integration Flow
This diagram shows the backend's interaction with Claude Flow MCP for spawning and managing agents.


sequenceDiagram
    participant API as "REST API (e.g., /api/bots/initialize-multi-agent)"
    participant CFA as "ClaudeFlowActorTcp"
    participant MCP as "Claude Flow MCP (TCP)"
    participant GSA as "GraphServiceActor"
    participant GPU as "GPUComputeActor"
    participant CM as "ClientManagerActor"

    API->>+CFA: InitializeMultiAgent(config)
    CFA->>+CFA: GenerateRequestId & StoreSender
    CFA->>+MCP: JSON-RPC "multi-agent.initialize" (TCP)
    MCP-->>-CFA: Response { swarmId: "swarm-123", agents: [...] }
    CFA->>CFA: Parse & Store Swarm State
    CFA-->>-API:





Client-side flows: React components, hooks, and local state management (e.g., GraphDataManager, SettingsStore).
Server-side flows: Actix actors (e.g., GraphServiceActor, ClaudeFlowActorTcp), services (e.g., RAGFlowService, SemanticAnalyzer), and backend logic.
Interfaces: REST APIs, WebSocket binary/JSON protocols, TCP MCP streams, and external APIs (e.g., GitHub, Perplexity).
Every detail: Each diagram shows participants, message types, data structures (e.g., BinaryNodeData, SimulationParams), error paths, and state changes. I've used Mermaid for renderability, with notes for clarity.
The diagrams are grouped by category for readability. They collectively map all data flows, from user input to GPU rendering, including error recovery and multi-graph handling.

1. Client-Server Connection & Initialisation Flow
This covers client connection, optional Nostr auth, initial graph data fetch, and real-time update subscription.


sequenceDiagram
    participant Client as "React Client (App.tsx)"
    participant AuthUI as "AuthUI.tsx (Nostr Auth)"
    participant WS as "WebSocketHandler (wss://localhost:8080/wss)"
    participant CM as "ClientManagerActor"
    participant Nostr as "NostrService (Auth)"
    participant GSA as "GraphServiceActor"
    participant MA as "MetadataActor"
    participant SA as "SettingsActor"
    participant GPU as "GPUComputeActor"

    Client->>+AuthUI: User clicks "Login" or "Connect"
    AuthUI->>AuthUI: Check localStorage for session
    alt No Session
        AuthUI->>+Nostr: window.nostr.getPublicKey()
        Nostr-->>-AuthUI: pubkey (if available)
        alt Nostr Extension Available
            AuthUI->>+Nostr: window.nostr.signEvent(authEvent)
            Nostr->>Nostr: Sign NIP-42 auth event
            Nostr-->>-AuthUI: signedEvent {id, pubkey, content, sig}
            AuthUI->>+Client: POST /api/auth/nostr (signedEvent)
            Client->>+Nostr: VerifyAuthEvent(signedEvent)
            Nostr->>Nostr: Validate signature & challenge
            Nostr-->>-Client: {user: {pubkey, npub, isPowerUser}, token}
            Client->>AuthUI: Store in localStorage (pubkey, token)
            AuthUI-->>-Client: AuthSuccess {features}
        end
    end

    Client->>+WS: WebSocket Connect (wss://localhost:8080/wss)
    WS->>+CM: RegisterClient {ws}
    CM-->>-WS: client_id
    WS-->>-Client: ConnectionEstablished {client_id, timestamp}

    Client->>+WS: RequestInitialData
    WS->>+CM: GetInitialGraphData {client_id}
    CM->>+GSA: GetGraphData
    GSA->>+MA: GetMetadata
    MA-->>-GSA: MetadataStore (HashMap<String, Metadata>)
    GSA->>+SA: GetSettings
    SA-->>-GSA: Settings (AppFullSettings)
    GSA->>+GPU: GetNodePositions (Unified Kernel)
    GPU-->>-GSA: Positions (Vec<Vec3>)
    GSA->>GSA: Build GraphData {nodes: Vec<Node>, edges: Vec<Edge>, positions}
    GSA-->>-CM: GraphData {nodes, edges, positions}
    CM-->>-WS: InitialGraph {nodes, edges, positions}
    WS-->>-Client: InitialGraph

    Note over Client,GPU: Real-Time Loop Starts (60 FPS)
    loop Physics Updates
        GSA->>+GPU: ComputeForces (SimulationParams)
        GPU-->>-GSA: New Positions (Vec<Vec3>)
        GSA->>+CM: BroadcastNodePositions {positions, timestamp}
        CM->>+WS: StreamPositions (BinaryNodeData array)
        WS->>+Client: Binary Update (28 bytes/node)
        Client->>Client: Update GraphDataManager positions
        Client->>Client: Render frame (Three.js)
    end

    Note over Client,GPU: User Interaction (Optional)
    Client->>+WS: UpdateNodePosition {node_id, position, velocity}
    WS->>+CM: UpdateClientPosition {client_id, node_id, position, velocity}
    CM->>+GSA: UpdateNodePosition {node_id, position, velocity}
    GSA->>GSA: Apply to graph_data.nodes
    GSA->>+GPU: UpdateNodeOnGPU {node_id, position, velocity}
    GPU-->>-GSA: Updated
    GSA-->>-CM: PositionUpdated
    CM->>+WS: BroadcastNodePosition {node_id, position, velocity}
    WS->>+Client: Binary Update (single node)