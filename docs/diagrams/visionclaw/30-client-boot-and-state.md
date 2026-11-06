---
id: VC-30
title: React client boot sequence and state layer
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2074, ADR-2077, ADR-2139, ADR-2141, ADR-2142, ADR-2181, ADR-2189]
sources:
  - ../project/client/src/app/main.tsx
  - ../project/client/src/app/App.tsx
  - ../project/client/src/app/AppInitializer.tsx
  - ../project/client/src/app/MainLayout.tsx
  - ../project/client/src/contexts/ApplicationModeContext.tsx
  - ../project/client/src/store/settingsStore.ts
  - ../project/client/src/store/settings/coreSlice.ts
  - ../project/client/src/store/settings/physicsSlice.ts
  - ../project/client/src/store/settings/persistenceSlice.ts
  - ../project/client/src/store/settings/subscriberTrie.ts
  - ../project/client/src/store/settings/retiredPaths.ts
  - ../project/client/src/store/websocketStore.ts
  - ../project/client/src/store/websocket/storeState.ts
  - ../project/client/src/store/timelineStore.ts
  - ../project/client/src/store/transientBeamStore.ts
  - ../project/client/src/store/workerErrorStore.ts
  - ../project/client/src/store/autoSaveManager.ts
  - ../project/client/src/store/settingsRetryManager.ts
  - ../project/client/src/store/settingsSyncStatus.ts
  - ../project/client/src/api/settings/schemaMappings.ts
  - ../project/client/src/features/graph/hooks/useFpsMonitor.ts
  - ../project/client/src/hooks/useSelectiveSettingsStore.ts
  - ../project/client/src/services/WebSocketRegistry.ts
  - ../project/client/src/services/WebSocketEventBus.ts
  - ../project/client/src/services/VoiceWebSocketService.ts
  - ../project/client/src/services/solidPod/podNotifications.ts
  - ../project/client/src/store/websocket/index.ts
  - ../project/client/src/store/websocket/textMessageHandler.ts
  - ../project/client/src/store/websocket/solidWebSocket.ts
  - ../project/client/src/features/bots/services/BotsWebSocketIntegration.ts
  - ../project/client/src/services/platformManager.ts
  - ../project/client/src/services/remoteLogger.ts
  - ../project/client/src/services/api/UnifiedApiClient.ts
  - ../project/client/src/services/api/authInterceptor.ts
  - ../project/client/src/api/settingsApi.ts
  - ../project/client/src/api/settings/endpoints.ts
  - ../project/client/src/api/settings/physicsPatch.ts
  - ../project/client/src/services/powerUserStatus.ts
  - ../project/client/src/api/analyticsApi.ts
  - ../project/client/src/api/constraintsApi.ts
  - ../project/client/src/api/graphExpandApi.ts
  - ../project/client/src/api/layoutApi.ts
  - ../project/client/src/api/workspaceApi.ts
  - ../project/client/src/features/graph/managers/graphDataManager.ts
  - ../project/client/src/features/graph/managers/dataManager/restClient.ts
  - ../project/client/src/features/graph/managers/dataManager/nodeUtils.ts
  - ../project/client/src/features/graph/managers/dataManager/topology.ts
  - ../project/client/src/features/graph/managers/dataManager/wsClient.ts
  - ../project/client/src/features/graph/hooks/useGraphDataSubscription.ts
  - ../project/client/src/features/graph/components/GraphManager.tsx
  - ../project/client/src/features/graph/managers/graphWorkerProxy.ts
  - ../project/client/src/features/graph/workers/graph.worker.ts
  - ../project/client/src/features/graph/__tests__/sabOrTransferDetection.test.ts
  - ../project/client/src/features/graph/components/GraphCanvas.tsx
  - ../project/client/src/features/graph/components/NodeDetailPanel.tsx
  - ../project/client/src/features/graph/components/TimelineScrubber.tsx
  - ../project/client/src/features/control-center/ControlCenter.tsx
  - ../project/client/src/features/control-center/status/StatusSurface.tsx
  - ../project/client/src/features/control-center/hooks/useSettingField.ts
  - ../project/client/src/features/bots/components/AgentOverlay.tsx
  - ../project/client/src/features/bots/agentMarkerPass.ts
  - ../project/client/src/store/agentTargetStore.ts
  - ../project/client/src/features/visualisation/components/VirtualAgentMarkers.tsx
  - ../project/client/src/features/graph/contexts/NodePositionContext.tsx
  - ../project/client/src/components/WorkerErrorModal.tsx
  - ../project/client/src/features/visualisation/components/EmbeddingCloudLayer.tsx
  - ../project/client/src/features/visualisation/hooks/useTransientBeams.ts
  - ../project/client/src/features/graph/components/GemNodes.tsx
  - ../project/client/src/features/graph/components/GlassEdges.tsx
  - ../project/client/src/features/visualisation/components/TransientBeamsLayer.tsx
  - ../project/client/src/store/websocket/binaryProtocol.ts
  - ../project/client/src/features/control-center/hooks/useRevealSetting.ts
  - ../project/client/src/features/control-center/primitives/NostrAuthControl.tsx
  - ../project/client/src/features/control-center/status/StatusFlyout.tsx
  - ../project/client/src/features/control-center/drawer/RightDrawer.tsx
  - ../project/client/src/services/nostrAuthService.ts
verified_commit: {visionclaw: 809a859aae9e3d96d4d6b99690378018e48e3ad8}
---
## For developers

The React client boots through `AppInitializer`: graph worker first, then the settings store, then the graph socket and the REST topology in parallel; the stores below it (settings, websocket, timeline, beams, worker errors) are what every surface reads. Three mechanisms changed at ec785afa9: every acknowledged binary chunk is applied first (ADR-2141, VC-30.10), the worker's position buffer grows with the graph (ADR-2181 D4, VC-30.11), and retired settings are stripped on both load paths (ADR-2189, VC-30.3, VC-30.8). At 809a859 the settings-integrity merge (`7cde2f00c`) inverted load precedence: the server document is the base and a value the server stores beats this browser's copy, except a change this browser could not save (VC-30.13). Changes made before load are held rather than dropped, and every lost write (not ready, sync off, failed) lands in one persisted record that the status chip counts (VC-30.3, VC-30.4, VC-30.12). The dead `exportApi.ts` is deleted (`90f1bb1c0`, VC-30.8).

## For the business

Opening the app loads the graph engine, the person's saved preferences and the live graph together. Every live position update is drawn before the server is told it arrived, and very large graphs no longer silently lose their outer nodes. Preferences now follow the server: a change made on another device, by another viewer or by the assistant shows up on the next load instead of being overwritten by an old copy in this browser. A change that could not be saved is no longer lost quietly; the status chip says how many are unsaved and why. Some unsaved changes never clear on their own (see the register), so the chip can stay lit.

## VC-30.1 Provider nesting and top-level render states
```mermaid
sequenceDiagram
    autonumber
    participant Main as main.tsx<br/>client/src/app/main.tsx:31
    participant App as App<br/>client/src/app/App.tsx:30
    participant Auth as useNostrAuth<br/>client/src/app/App.tsx:36
    participant Tip as TooltipProvider<br/>client/src/app/App.tsx:167
    participant Help as HelpProvider<br/>client/src/app/App.tsx:168
    participant Onb as OnboardingProvider<br/>client/src/app/App.tsx:169
    participant EB as ErrorBoundary<br/>client/src/app/App.tsx:170
    participant AMP as ApplicationModeProvider<br/>client/src/contexts/ApplicationModeContext.tsx:41
    participant AI as AppInitializer<br/>client/src/app/AppInitializer.tsx:70
    participant ML as MainLayout<br/>client/src/app/MainLayout.tsx:120

    Main->>App: ReactDOM.createRoot().render(<StrictMode><App/></StrictMode>)<br/>main.tsx:31-35
    App->>Auth: useNostrAuth()<br/>App.tsx:36
    alt isAuthLoading true
        App-->>Main: return LoadingScreen "Checking authentication..."<br/>App.tsx:121-123
    end
    Note over App: skipAuth = VITE_PUBLIC_DEMO hostname junkiejarvis.com OR (dev AND ?skipAuth=true|?test=visual)<br/>App.tsx:127-133
    alt !authenticated && !skipAuth
        App-->>Main: return OnboardingWizard<br/>App.tsx:136-141
    end
    App->>Tip: mount delayDuration=300 skipDelayDuration=100<br/>App.tsx:167
    Tip->>Help: mount
    Help->>Onb: mount
    Onb->>EB: mount
    EB->>AMP: mount
    AMP->>AMP: renderContent() switch(initializationState)<br/>App.tsx:143-162
    alt initializationState==loading
        AMP-->>App: LoadingScreen "Connecting to server..."<br/>App.tsx:145-146
        AMP->>AI: mount AppInitializer(onInitialized,onError)<br/>App.tsx:175-177
    else initializationState==error
        AMP-->>App: div Error Initializing Application + Retry button<br/>App.tsx:147-154
    else initializationState==initialized
        AMP->>ML: mount BotsDataProvider > MainLayout<br/>App.tsx:157-159
        AMP->>AMP: mount ConnectionWarning, CommandPalette,<br/>DebugControlPanel, WorkerErrorModal<br/>App.tsx:179-184
    end
    Note over App: useEffect(authenticated,user): setAuthenticated/setUser on settingsStore,<br/>solidPodService.connectWebSocket() else disconnect()<br/>App.tsx:44-75
    Note over App: when the user is not already flagged isPowerUser, fetchPowerUserStatus(pubkey) asks<br/>GET /api/admin/rbac/whoami and calls setUser({isPowerUser,pubkey}) only if the store still holds<br/>the same pubkey, an answer for another key or any error yields null and is ignored (ER-26)<br/>App.tsx:53-64, powerUserStatus.ts:26-39
    Note over App: useEffect(initialized): initializeCommandPalette, registerSettingsCommands,<br/>registerSettingsHelp, registerOnboardingCommands - first-visit onboarding<br/>dispatched after 1000ms via localStorage hasVisited flag<br/>App.tsx:85-103
```
## VC-30.2 AppInitializer — worker, settings, websocket and data fan-out
```mermaid
sequenceDiagram
    autonumber
    participant AI as AppInitializer.initApp<br/>client/src/app/AppInitializer.tsx:74
    participant LS as loadServices<br/>client/src/app/AppInitializer.tsx:12
    participant IM as innovationManager<br/>client/src/app/AppInitializer.tsx:26
    participant WP as graphWorkerProxy<br/>client/src/app/AppInitializer.tsx:7
    participant GDM as graphDataManager<br/>client/src/app/AppInitializer.tsx:8
    participant WES as useWorkerErrorStore<br/>client/src/app/AppInitializer.tsx:5
    participant Set as useSettingsStore.initialize<br/>client/src/app/AppInitializer.tsx:71
    participant WS as webSocketService<br/>client/src/app/AppInitializer.tsx:6
    participant App as onInitialized/onError<br/>client/src/app/App.tsx:105-118

    AI->>LS: loadServices().catch(...)<br/>fire-and-forget, AppInitializer.tsx:78
    LS->>IM: innovationManager.initialize({enableSync,enableComparison,<br/>enableAnimations,enableAdvancedInteractions,performanceMode:'balanced'})<br/>AppInitializer.tsx:26-32
    par race
        IM-->>LS: init resolves
    and
        Note over LS: timeout 5000ms rejects "Innovation Manager initialization timeout"<br/>AppInitializer.tsx:34-36
    end
    LS->>LS: Promise.allSettled(serviceLoaders) - rejection logged as warn, non-fatal<br/>AppInitializer.tsx:49-60

    AI->>WP: graphWorkerProxy.initialize()<br/>AppInitializer.tsx:86
    AI->>GDM: graphDataManager.ensureWorkerReady()<br/>AppInitializer.tsx:89
    alt worker init throws or workerReady false
        AI->>WES: setWorkerError('graph visualization engine failed to initialize', details)<br/>AppInitializer.tsx:108-111
        Note over AI: details narrows to "SharedArrayBuffer is not available" when<br/>typeof SharedArrayBuffer === 'undefined' - AppInitializer.tsx:102-103
        AI->>AI: continue without fully initialized worker (non-fatal)<br/>AppInitializer.tsx:113-114
    end
    AI->>WES: setRetryHandler(() => initializeWorker())<br/>AppInitializer.tsx:118-123
    AI->>Set: await initialize()<br/>AppInitializer.tsx:130
    Note over AI: applies system.debug settings: debugState.enableDebug/enableDataDebug/<br/>enablePerformanceDebug - AppInitializer.tsx:134-147
    par WebSocket connect (non-blocking of data fetch)
        AI->>WS: initializeWebSocket(settings)<br/>AppInitializer.tsx:156
        alt websocketService.connect() throws
            Note over AI: caught, logged, non-fatal - AppInitializer.tsx:158-160
        end
    and Initial data fetch
        AI->>GDM: graphDataManager.fetchInitialData()<br/>AppInitializer.tsx:168
        alt fetch throws
            AI->>GDM: setGraphData({nodes:[],edges:[]})<br/>AppInitializer.tsx:173
        end
    end
    AI->>AI: await Promise.all([wsPromise,dataPromise])<br/>AppInitializer.tsx:178
    alt outer try succeeds
        AI->>App: onInitialized()<br/>AppInitializer.tsx:181
    else outer try throws (e.g. Set.initialize rejects)
        AI->>App: onError(error)<br/>AppInitializer.tsx:186
    end
    Note over AI: INVARIANT (ADR-2141): the client sends NO subscribe message. The connection-status handler only<br/>enables binary updates, the server streams position chunks to every registered session and<br/>sends the next chunk when this client acknowledges the last (binaryProtocol.ts:404-408).<br/>The hasSubscribedToPositions flag and subscribe_position_updates are deleted<br/>AppInitializer.tsx:260-278
```
## VC-30.3 Zustand store catalogue
```mermaid
classDiagram
    class useSettingsStore {
      client/src/store/settingsStore.ts:27
      persist middleware name="graph-viz-settings-v2"
      storage localStorage via createJSONStorage settingsStore.ts:36
      partialize authenticated,user,isPowerUser,partialSettings settingsStore.ts:37-46
      merge() strips ADR-2189 retired paths then restores persisted.partialSettings settingsStore.ts:47-73
    }
    class CoreSlice {
      partialSettings DeepPartial~Settings~ coreSlice.ts:70
      settings DeepPartial~Settings~ coreSlice.ts:71
      loadedPaths Set~string~ coreSlice.ts:72
      loadingSections Set~string~ coreSlice.ts:73
      initialized bool coreSlice.ts:74
      authenticated bool coreSlice.ts:75
      settingsSyncEnabled bool coreSlice.ts:78
      +initialize() coreSlice.ts:85
      +get(path) coreSlice.ts:200
      +set(path,value,skipServerSync) coreSlice.ts:232
      +subscribe(path,cb,immediate) coreSlice.ts:261
      +ensureLoaded(paths) coreSlice.ts:275
      +loadSection(section) coreSlice.ts:335
      +updateSettings(updater) coreSlice.ts:371
      +updateComputeMode(mode) coreSlice.ts:442
      +updateConstraints(constraints) coreSlice.ts:453
    }
    class PhysicsSlice {
      +updatePhysics(graphName,params) physicsSlice.ts:19
      +updateWarmupSettings(settings) physicsSlice.ts:134
      updatePhysics clamps streamChunkNodes 100-20000, streamMoveThreshold 0-100, streamAckTimeoutMs 20-10000 (ADR-2141) physicsSlice.ts:80-89
      +notifyPhysicsUpdate() no-op since 2026-06-03 physicsSlice.ts:145-155
      +updateTweening(graphName,params) params enabled,responseMs 5-1000,snapThreshold,maxDivergence physicsSlice.ts:157,172-174
      +notifyTweeningUpdate(params) dispatches window CustomEvent tweeningSettingsUpdated physicsSlice.ts:204
    }
    class PersistenceSlice {
      +getByPath(path) persistenceSlice.ts:36
      +setByPath(path,value) persistenceSlice.ts:47
      +batchUpdate(updates) persistenceSlice.ts:60
      +flushPendingUpdates() persistenceSlice.ts:84
      +resetSettings() persistenceSlice.ts:88
      +exportSettings() persistenceSlice.ts:107
      +importSettings(json) persistenceSlice.ts:128
    }
    class SubscriberTrie {
      subscriberTrieRoot SubscriberTrieNode subscriberTrie.ts:13
      +getOrCreateTrieNode(path,create) subscriberTrie.ts:18
      +collectDescendants(node,out) subscriberTrie.ts:34
      +collectMatchedCallbacks(changedPaths) subscriberTrie.ts:41
      +scheduleNotify(callbacks) requestAnimationFrame batch subscriberTrie.ts:78
    }
    class useWebSocketStore {
      client/src/store/websocketStore.ts:9-16 re-export shim (36 lines)
      real impl client/src/store/websocket/index.ts:515 lines
      socket WebSocket~null~ storeState.ts:48
      isConnected bool storeState.ts:49
      isServerReady bool storeState.ts:50
      connectionState ConnectionState storeState.ts:51
      solidSubscriptions Map~string,Set~ storeState.ts:55
      nodeTypeMap Map~number,NodeType~ storeState.ts:56
      messageQueue QueuedMessage[] storeState.ts:57
      statistics WebSocketStatistics storeState.ts:31-44
      reconnectInterval=1000 maxReconnectAttempts=10 storeState.ts:69-71
    }
    class useTimelineStore {
      client/src/store/timelineStore.ts:87
      active bool timelineStore.ts:90
      diffMode bool timelineStore.ts:91
      t,t1,t2 string~null~ timelineStore.ts:92-94
      domainMinMs,domainMaxMs number timelineStore.ts:95-96
      validSubjects,addedSubjects,retractedSubjects,knownSubjects Set~string~ timelineStore.ts:97-100
      version number, bumped on every derived-set change timelineStore.ts:103,181
      +loadStateAt(t,recordedAsOf) timelineStore.ts:105
      +loadDiff(t1,t2,recordedAsOf) timelineStore.ts:133
      +setDiffMode(on) timelineStore.ts:175
      +reset() timelineStore.ts:190
      DEFAULT_DOMAIN_LOOKBACK_MS = 30d timelineStore.ts:34
    }
    class useTransientBeamStore {
      client/src/store/transientBeamStore.ts:65
      beams TransientBeam[] transientBeamStore.ts:66
      +pushBeams(events) transientBeamStore.ts:68
      +pruneExpired() transientBeamStore.ts:94
      +clear() transientBeamStore.ts:107
      DEFAULT_BEAM_DURATION_MS=1500 MAX_TRANSIENT_BEAMS=256 MIN_BEAM_DURATION_MS=400<br/>transientBeamStore.ts:24-28
    }
    class useWorkerErrorStore {
      client/src/store/workerErrorStore.ts:23
      hasWorkerError bool workerErrorStore.ts:24
      transientErrorCount number workerErrorStore.ts:27
      retryWorkerInit fn~null~ workerErrorStore.ts:28
      +setWorkerError(msg,details) workerErrorStore.ts:30
      +recordTransientError(context) workerErrorStore.ts:36
      +resetTransientErrors() workerErrorStore.ts:50
      +setRetryHandler(handler) workerErrorStore.ts:74
      TRANSIENT_THRESHOLD=30 workerErrorStore.ts:21
    }
    class AutoSaveManager {
      client/src/store/autoSaveManager.ts:15 singleton autoSaveManager.ts:149
      pendingChanges Map~string,any~ autoSaveManager.ts:16
      DEBOUNCE_DELAY=500ms autoSaveManager.ts:20
      CLIENT_ONLY_PATHS auth.nostr.connected,auth.nostr.publicKey autoSaveManager.ts:38-41
      +setSyncEnabled(false) marks pending sync-off then clears autoSaveManager.ts:24-35
      +setInitialized(true) flushes changes held before load autoSaveManager.ts:50-55
      -markUnsaved(paths,reason) server-bucket paths only autoSaveManager.ts:58-63
      +queueChange(path,value) autoSaveManager.ts:65
      +queueChanges(changes) autoSaveManager.ts:70
      +forceFlush() autoSaveManager.ts:99
      -flushPendingChanges() calls settingsApi.updateSettingsByPaths autoSaveManager.ts:108
    }
    class useSettingsSyncStatus {
      client/src/store/settingsSyncStatus.ts:29
      unsaved Record~path,UnsavedReason~ settingsSyncStatus.ts:21
      UnsavedReason not-ready,sync-off,failed settingsSyncStatus.ts:17
      persisted key graph-viz-settings-unsynced settingsSyncStatus.ts:27,54
      +markUnsaved(paths,reason) settingsSyncStatus.ts:33
      +markSaved(paths) settingsSyncStatus.ts:39
      +isUnsyncedPath(path) ancestor or descendant match settingsSyncStatus.ts:62-68
      +describeUnsaved(unsaved) chip tooltip settingsSyncStatus.ts:72
    }
    class SettingsRetryManager {
      client/src/store/settingsRetryManager.ts:21 singleton settingsRetryManager.ts:207
      maxRetries=3 baseRetryDelay=1000ms maxRetryDelay=30000ms settingsRetryManager.ts:24-26
      +addFailedUpdate(path,value,error) settingsRetryManager.ts:33
      -processRetryQueue() polled every 5000ms settingsRetryManager.ts:147,56
      -calculateRetryDelay(attempts) exp backoff + jitter settingsRetryManager.ts:134
      dispatches window CustomEvent settings-retry-failed on maxRetries exhausted settingsRetryManager.ts:126-128
    }
    useSettingsStore *-- CoreSlice
    useSettingsStore *-- PhysicsSlice
    useSettingsStore *-- PersistenceSlice
    CoreSlice --> SubscriberTrie : subscribe and updateSettings walk trie
    CoreSlice --> AutoSaveManager : set and updateSettings queueChange coreSlice L253 L401
    PersistenceSlice --> useSettingsSyncStatus : setByPath and batchUpdate bypass the queue, recordWrite persistenceSlice L8 L54 L78
    AutoSaveManager --> SettingsRetryManager : flush failure delegates autoSaveManager L127-133
    AutoSaveManager --> useSettingsSyncStatus : markUnsaved on hold or failure, markSaved on success
    SettingsRetryManager --> useSettingsSyncStatus : markSaved on retry success
    CoreSlice ..> useSettingsSyncStatus : initialize keeps an unsynced local value
    useTimelineStore ..> useTransientBeamStore : independent, no direct coupling
```
## VC-30.4 Settings subscriber trie, selective hooks and autosave/retry pipeline
```mermaid
sequenceDiagram
    autonumber
    participant Comp as Component<br/>useSelectiveSetting/useSettingsSubscription<br/>client/src/hooks/useSelectiveSettingsStore.ts:14,102
    participant Store as useSettingsStore.updateSettings<br/>client/src/store/settings/coreSlice.ts:371
    participant Trie as subscriberTrie<br/>client/src/store/settings/subscriberTrie.ts
    participant ASM as autoSaveManager<br/>client/src/store/autoSaveManager.ts:15
    participant API as settingsApi.updateSettingsByPaths<br/>client/src/store/autoSaveManager.ts:120
    participant Retry as settingsRetryManager<br/>client/src/store/settingsRetryManager.ts:21
    participant SS as useSettingsSyncStatus<br/>client/src/store/settingsSyncStatus.ts:29

    Comp->>Store: useSettingsStore(state => state.get(path))<br/>useSelectiveSettingsStore.ts:22-25 (Zustand selector memoisation)
    Comp->>Store: subscribe(path, callback, immediate=false)<br/>useSelectiveSettingsStore.ts:131
    Store->>Trie: getOrCreateTrieNode(path).subscribers.add(callback)<br/>coreSlice.ts:262, subscriberTrie.ts:18
    Note over Comp,Store: immediate=true fires callback synchronously against get().initialized<br/>coreSlice.ts:264-265

    Store->>Store: produce(partialSettings,updater) via immer, findChangedPaths()<br/>coreSlice.ts:374-376
    alt changedPaths.length == 0
        Store-->>Comp: no-op return
    end
    Store->>ASM: autoSaveManager.queueChanges(batchChanges)<br/>coreSlice.ts:401
    alt path in CLIENT_ONLY_PATHS (auth.nostr.*)
        ASM-->>Store: filtered out, never leaves client<br/>autoSaveManager.ts:71-72
    else sync off (local-only mode)
        ASM->>SS: markUnsaved(paths,'sync-off'), not queued<br/>autoSaveManager.ts:73-76
    else !isInitialized
        ASM->>ASM: pendingChanges.set(path,value), HELD until setInitialized(true)<br/>autoSaveManager.ts:77-81, :54
        ASM->>SS: markUnsaved(paths,'not-ready')<br/>autoSaveManager.ts:80
    else
        ASM->>ASM: pendingChanges.set(path,value), scheduleFlush()<br/>debounce 500ms via clearTimeout/setTimeout<br/>autoSaveManager.ts:77,84,88-95
    end
    Store->>Trie: collectMatchedCallbacks(changedPaths)<br/>coreSlice.ts:439, subscriberTrie.ts:41
    Trie->>Trie: walk ancestor prefixes + collectDescendants(exactNode)<br/>subscriberTrie.ts:44-59
    Store->>Trie: scheduleNotify(matched) - adds to pendingNotifyCallbacks,<br/>one requestAnimationFrame per batch<br/>subscriberTrie.ts:78-83
    Trie->>Comp: flushNotifyCallbacks() invokes each callback, errors caught+logged<br/>subscriberTrie.ts:67-75

    Note over ASM: after DEBOUNCE_DELAY=500ms, flushPendingChanges() fires<br/>autoSaveManager.ts:93-95,108
    ASM->>API: settingsApi.updateSettingsByPaths(updates)<br/>autoSaveManager.ts:120
    Note over API: a path under visualisation.graphs.*.physics. becomes a NESTED partial via physicsPatchEntry<br/>(physics.semantic.dag.levelAttraction becomes {semantic:{dag:{levelAttraction}}}) and mergeInto<br/>deep-merges entries, so two sliders of one group in the same debounce window both survive,<br/>then updatePhysics GETs and PUTs the merged document<br/>endpoints.ts:337-346, :361-365, :390 and physicsPatch.ts:10-18
    alt API resolves
        API-->>ASM: SUCCESS - pendingChanges already cleared pre-send<br/>autoSaveManager.ts:114-121
        ASM->>SS: markSaved(paths)<br/>autoSaveManager.ts:122
    else API rejects
        ASM->>SS: markUnsaved(paths,'failed')<br/>autoSaveManager.ts:124
        ASM->>Retry: settingsRetryManager.addFailedUpdate(path,value,error.message)<br/>per failed update, autoSaveManager.ts:127-133
        Retry->>Retry: retryQueue.set(path,{attempts,lastAttempt,error})<br/>settingsRetryManager.ts:41-47
        Retry->>Retry: startRetryProcessor() window.setInterval 5000ms<br/>settingsRetryManager.ts:144-151
        loop every 5000ms while retryQueue non-empty
            Retry->>Retry: calculateRetryDelay(attempts)=min(1000*2^(attempts-1),30000)+jitter<br/>settingsRetryManager.ts:134-140
            alt now - lastAttempt >= delay
                Retry->>API: settingsApi.updateSettingsByPaths (batch) or updateSettingByPath (single)<br/>settingsRetryManager.ts:87,108
                Note over API: updateSettingByPath now rethrows (endpoints.ts:329-331), so a failed<br/>single retry is no longer counted as a success
                alt retry succeeds
                    Retry->>Retry: retryQueue.delete(path)<br/>settingsRetryManager.ts:92,109
                    Retry->>SS: markSaved(paths)<br/>settingsRetryManager.ts:90,110
                else retry throws
                    Note over Retry: OPEN: only lastAttempt and error are updated, attempts is not<br/>incremented (settingsRetryManager.ts:99-102, :113-114), so the<br/>give-up check below is never reached by retries alone
                end
            end
            opt attempts >= maxRetries=3
                Retry->>Retry: retryQueue.delete, dispatchEvent 'settings-retry-failed' {path,value,error}<br/>settingsRetryManager.ts:120-128
            end
        end
        Retry->>Retry: stopRetryProcessor() when retryQueue.size==0<br/>settingsRetryManager.ts:58-60
    end
```
## VC-30.5 WebSocketRegistry and WebSocketEventBus fan-out across three sockets
```mermaid
sequenceDiagram
    autonumber
    participant Graph as websocket/index.ts connect()<br/>client/src/store/websocket/index.ts:154
    participant Voice as VoiceWebSocketService<br/>client/src/services/VoiceWebSocketService.ts:89
    participant Solid as solidWebSocket store adapter<br/>client/src/store/websocket/solidWebSocket.ts:71
    participant Pod as podNotificationManager<br/>client/src/services/solidPod/podNotifications.ts:92
    participant Reg as webSocketRegistry<br/>client/src/services/WebSocketRegistry.ts:22
    participant Bus as webSocketEventBus<br/>client/src/services/WebSocketEventBus.ts:39
    participant Bots as BotsWebSocketIntegration<br/>client/src/features/bots/services/BotsWebSocketIntegration.ts:53

    Note over Reg: connections Map~string,RegistryEntry~ keyed by name - WebSocketRegistry.ts:23
    Graph->>Reg: register("graph", url, socket)<br/>websocket/index.ts:178
    Voice->>Reg: register(REGISTRY_NAME="voice", url, socket)<br/>VoiceWebSocketService.ts:14,89
    Solid->>Pod: connect() - the adapter registers nothing of its own<br/>solidWebSocket.ts:81
    Pod->>Reg: register(REGISTRY_NAME="solid-pod", url, socket)<br/>podNotifications.ts:20,92
    alt name already registered
        Reg->>Reg: unregister(name) first - removes old listeners, prevents orphaned handlers<br/>WebSocketRegistry.ts:33-35
    end
    Reg->>Reg: attach open/close/error listeners updating state label<br/>readyStateLabel() connecting|open|closing|closed|unknown<br/>WebSocketRegistry.ts:37-58, :121-134
    Reg->>Bus: emit("registry:registered",{name,url})<br/>WebSocketRegistry.ts:63

    par each socket also emits directly
        Graph->>Bus: emit("connection:open",{name:"graph",url})<br/>websocket/index.ts:179
    and
        Voice->>Bus: emit("connection:open",{name:"voice",url})<br/>VoiceWebSocketService.ts:90
    and
        Pod->>Bus: emit("connection:open",{name:"solid-pod",url})<br/>podNotifications.ts:93
        Pod-->>Solid: onLifecycle{type:"open"} -> isSolidConnected, emit("solid-connected")<br/>podNotifications.ts:94, solidWebSocket.ts:36-41
    end
    Note over Solid,Pod: INVARIANT: ADR-2100 - the store adapter is NOT a fourth registrant. It opened its own<br/>socket to the same VITE_JSS_WS_URL and registered as "solid-store" with a 10-attempt ladder -<br/>both are deleted. One socket, one registry entry ("solid-pod"), one 5-attempt ladder - see VC-32.16

    Bots->>Bus: emit("message:bots",{data:message})<br/>BotsWebSocketIntegration.ts:53
    Voice->>Bus: emit("message:voice",{data:message})<br/>VoiceWebSocketService.ts:178
    Pod->>Bus: emit("message:pod",{data:msg})<br/>podNotifications.ts:99
    Note over Bus: RESOLVED ADR-2077: the message:graph event type is DELETED. It was<br/>declared in the union and payload map but never emitted, so a switch over<br/>the bus events is now exhaustive over events that actually fire.

    Bus->>Bus: on(event,handler) adds to handlers Map~string,Set~Handler~~<br/>WebSocketEventBus.ts:45-57, returns unsubscribe closure calling off() :62-73
    Bus->>Bus: emit(event,data) iterates handlers Set, try/catch per handler,<br/>logs and continues on throw - WebSocketEventBus.ts:78-92

    alt connection error (any socket)
        Voice->>Bus: emit("connection:error",{name,error})<br/>VoiceWebSocketService.ts:116
        Graph->>Bus: emit("connection:error",{name:"graph",error:errorMessage})<br/>websocket/index.ts:246
    end
    alt connection close (any socket)
        Graph->>Reg: unregister("graph")<br/>websocket/index.ts:217,276
        Reg->>Reg: removeEventListener for each tracked listener, delete from connections<br/>WebSocketRegistry.ts:74-79
        Reg->>Bus: emit("registry:unregistered",{name})<br/>WebSocketRegistry.ts:81
        Graph->>Bus: emit("connection:close",{name,code,reason})<br/>websocket/index.ts:218
    end
    Note over Reg: RESOLVED ADR-2077: closeAll() is DELETED - it had zero callers.<br/>register, unregister, get, getEntry, getAll, size and readyStateLabel<br/>remain. Nothing can mass-close sockets. no caller ever wanted that.
```
## VC-30.6 platformManager — UA sniff, XR capability probe, event dispatch
```mermaid
sequenceDiagram
    autonumber
    participant Caller as PlatformManager.initialize<br/>client/src/services/platformManager.ts:348
    participant Store as usePlatformStore<br/>client/src/services/platformManager.ts:61
    participant Nav as navigator.xr<br/>client/src/services/platformManager.ts:92
    participant Win as window resize listener<br/>client/src/services/platformManager.ts:133-137

    Caller->>Store: initialize()<br/>platformManager.ts:85
    Store->>Store: detectPlatform()<br/>call platformManager.ts:89, fn :154
    Store->>Store: userAgent = navigator.userAgent<br/>platformManager.ts:155
    alt userAgent includes "Quest"
        alt includes "Quest 3"
            Store->>Store: platform="quest3" xrDeviceType="quest"<br/>platformManager.ts:162,168
        else includes "Quest 2"
            Store->>Store: platform="quest2"<br/>platformManager.ts:163-164
        else
            Store->>Store: platform="quest"<br/>platformManager.ts:165-166
        end
    else userAgent includes "Pico"|"PICO"
        Store->>Store: platform="pico" xrDeviceType="pico"<br/>platformManager.ts:171-173
    else /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i test<br/>platformManager.ts:176
        Store->>Store: platform="mobile" xrDeviceType="mobile-xr"<br/>platformManager.ts:177-178
    else
        Store->>Store: platform="desktop" xrDeviceType="desktop-xr"<br/>platformManager.ts:182-183
    end
    Note over Store: performanceTier/maxTextureSize/memoryLimited per platform<br/>quest3: high,4096,true - quest2: medium,2048,true - quest: low,2048,true<br/>pico: medium,2048,true - mobile: low,2048,true - desktop: high,8192,false<br/>platformManager.ts:191-222
    Store->>Store: hasTouchscreen via navigator.maxTouchPoints or ontouchstart<br/>platformManager.ts:225-226
    alt prevPlatform !== platform
        Store->>Store: dispatchEvent("platformchange",{platform})<br/>platformManager.ts:247-248
    end

    alt navigator.xr exists
        Caller->>Nav: navigator.xr.isSessionSupported("immersive-vr")<br/>platformManager.ts:95
        Caller->>Nav: navigator.xr.isSessionSupported("immersive-ar")<br/>platformManager.ts:97
        Nav-->>Store: vrSupported, arSupported
        Store->>Store: capabilities.xrSupported = vrSupported||arSupported<br/>platformManager.ts:99-107
        alt isSessionSupported throws
            Note over Caller: caught, logged, capabilities keep prior defaults<br/>platformManager.ts:110-111
        end
        Store->>Store: capabilities.handTrackingSupported = isQuest()<br/>platformManager.ts:116-129
    else navigator.xr undefined
        Note over Store: isWebXRSupported is !!navigator.xr - false when absent -<br/>capabilities.xrSupported keeps its initial false<br/>platformManager.ts:66,79
    end

    Caller->>Win: window.addEventListener("resize", () => detectPlatform())<br/>platformManager.ts:133-137
    Caller->>Store: set({initialized:true, isWebXRSupported})<br/>platformManager.ts:140-145

    Note over Store: DOC-DRIFT (fixed in this pass): this diagram used to describe<br/>setXRMode/setXRSessionState dispatching "xrmodechange"/"xrsessionstatechange".<br/>RESOLVED ADR-2081 (see VC-37.2) deleted both methods, the XRSessionState type<br/>and the isXRMode/xrSessionState store fields - they had zero callers. The<br/>listener/dispatchEvent mechanism below is generic and still used for<br/>"platformchange" only.
    Store->>Store: dispatchEvent(event,data) iterates listeners.get(event) Set,<br/>try/catch per callback<br/>platformManager.ts:275-286
    Note over Caller: PlatformManager class (platformManager.ts:332) is a backwards-compat<br/>singleton adapter (getInstance) wrapping usePlatformStore.getState() calls -<br/>exported as platformManager, platformManager.ts:394
```
## VC-30.7 remoteLogger — console interception, buffering, batch POST, failure path
```mermaid
sequenceDiagram
    autonumber
    participant Mod as module load<br/>client/src/services/remoteLogger.ts:314
    participant RL as RemoteLogger<br/>client/src/services/remoteLogger.ts:14
    participant Con as console.log/debug/info/warn/error<br/>client/src/services/remoteLogger.ts:49
    participant Timer as flushTimer setInterval<br/>client/src/services/remoteLogger.ts:139
    participant Srv as POST serverEndpoint<br/>client/src/services/remoteLogger.ts:179

    Mod->>RL: new RemoteLogger()<br/>remoteLogger.ts:24,314
    RL->>RL: enabled = !VITE_REMOTE_LOGGING_DISABLED<br/>remoteLogger.ts:26-27
    RL->>RL: serverEndpoint = VITE_API_URL + "/api/client-logs"<br/>remoteLogger.ts:31-33
    alt enabled
        RL->>Timer: startFlushTimer() setInterval every flushInterval=1000ms<br/>remoteLogger.ts:35-36,139-147
        RL->>Con: interceptConsole() wraps log/debug/info/warn/error,<br/>calls originalConsole then this.log(level,'console',...)<br/>remoteLogger.ts:37,49-83
    end
    RL->>RL: window.addEventListener("beforeunload", () => flush(true))<br/>remoteLogger.ts:41-45

    Con->>RL: log(level,'console',formatArgs(args)[,extractStack(args) for error])<br/>remoteLogger.ts:59-82
    RL->>RL: buffer.push(entry) - entry carries userAgent,url,timestamp,stack,data<br/>remoteLogger.ts:110-130
    alt buffer.length >= maxBufferSize=50
        RL->>RL: flush() immediately<br/>remoteLogger.ts:133-135,17
    end

    loop every flushInterval=1000ms while buffer non-empty
        Timer->>RL: flush()<br/>remoteLogger.ts:142-146
    end
    RL->>RL: logs = copy of buffer, buffer = [], payload={logs,sessionId,timestamp}<br/>remoteLogger.ts:161-169
    alt sync=true (beforeunload path)
        RL->>Srv: navigator.sendBeacon(serverEndpoint, blob) if available<br/>remoteLogger.ts:173-176
    else async flush
        RL->>Srv: fetch(serverEndpoint, POST, Content-Type application-json)<br/>remoteLogger.ts:179-185
        alt response not ok
            RL->>RL: consecutiveFailures++<br/>remoteLogger.ts:188
            alt status==404 && consecutiveFailures >= maxConsecutiveFailures=3
                RL->>RL: setEnabled(false) - auto-disable, endpoint unavailable<br/>remoteLogger.ts:189-192,220-228
            else status!=404
                RL->>RL: console.warn "Failed to send logs" - the INTERCEPTED console.warn, so the warning is buffered<br/>and posted on the next flush (remoteLogger.ts:193-194, wrapper :74-77). The failed batch itself is dropped, not re-queued
            end
        else response ok
            RL->>RL: consecutiveFailures = 0<br/>remoteLogger.ts:197
        end
        alt fetch throws
            RL->>RL: buffer = logs.concat(buffer) - re-queue for next flush<br/>remoteLogger.ts:200-206
        end
    end
    Note over RL,Srv: DEBT: while the endpoint keeps answering non-404 errors, each failed flush buffers one more warning about itself<br/>(remoteLogger.ts:194 through the wrapper :74-77 into log() :110-135), so the logger posts at least one entry a second<br/>indefinitely. setEnabled(false) stops log() (:117) but never restores the original console methods (:220-228).
    Note over RL: sessionId persisted in sessionStorage key remote-logger-session,<br/>format session-{Date.now()}-{random36}<br/>remoteLogger.ts:210-217

    Mod->>RL: setTimeout(() => logXRInfo(), 1000) on module load<br/>remoteLogger.ts:320-324
    RL->>RL: logXRInfo(): webXRSupported='xr' in navigator, isQuestDevice via<br/>/OculusBrowser|Quest/i test, questVersion via /Quest\s*(\d+)?/i match<br/>remoteLogger.ts:271-309
    opt navigator.xr present
        RL->>RL: isSessionSupported('immersive-vr'/'immersive-ar') logged as<br/>separate 'xr-capabilities' entries<br/>remoteLogger.ts:281-295
    end
```
## VC-30.8 api layer — two parallel HTTP stacks and their interceptor chains
```mermaid
sequenceDiagram
    autonumber
    participant Mod as main.tsx bootstrap<br/>client/src/app/main.tsx:28-29
    participant UAC as unifiedApiClient<br/>client/src/services/api/UnifiedApiClient.ts:68,466
    participant AuthI as authInterceptor<br/>client/src/services/api/authInterceptor.ts:80,51
    participant Nostr as nostrAuth<br/>client/src/services/nostrAuthService.ts
    participant Fetch as window.fetch<br/>UnifiedApiClient.ts:226
    participant Ax as axios (global instance)<br/>client/src/api/settings/endpoints.ts:4,42

    Mod->>UAC: initializeAuthInterceptor(unifiedApiClient)<br/>main.tsx:28, authInterceptor.ts:131-138
    UAC->>UAC: setInterceptors({onRequest:authRequestInterceptor,<br/>onResponse:authResponseInterceptor}) - no onError<br/>authInterceptor.ts:132-135, UnifiedApiClient.ts:116-118

    Note over UAC: request(method,url,data,config) sets body via JSON.stringify<br/>for POST/PUT/PATCH unless string/ArrayBuffer/FormData<br/>UnifiedApiClient.ts:316-343
    UAC->>UAC: executeWithRetry(url,config,attempt=0)<br/>UnifiedApiClient.ts:151
    UAC->>AuthI: interceptors.onRequest(finalConfig,fullUrl)<br/>UnifiedApiClient.ts:186-188
    AuthI->>AuthI: headers['X-Request-ID']=generateRequestId()<br/>authInterceptor.ts:89-90
    alt nostrAuth.isAuthenticated()
        AuthI->>AuthI: computeAuthHeaders(fullUrl,method,body)<br/>authInterceptor.ts:99 calling :51-78 (RESOLVED ADR-2074 -<br/>the single shared decision point, see the Ax note below)
        alt isDevMode()
            AuthI->>AuthI: Authorization: Bearer dev-session-token<br/>+ X-Nostr-Pubkey if known<br/>authInterceptor.ts:62-70
        else not dev mode, pubkey known
            AuthI->>Nostr: signRequest(fullUrl,method,body) NIP-98<br/>authInterceptor.ts:76
            Nostr-->>AuthI: token
            AuthI->>AuthI: Authorization: Nostr {token}<br/>authInterceptor.ts:77
        else no pubkey
            AuthI-->>AuthI: {} - no auth headers<br/>authInterceptor.ts:72-74
        end
    else not authenticated
        Note over AuthI: computeAuthHeaders returns {} immediately<br/>authInterceptor.ts:56-58 - no auth headers sent
    end
    UAC->>Fetch: fetch(fullUrl,{headers,signal:abortController.signal})<br/>UnifiedApiClient.ts:211-226
    Note over UAC: per-request AbortController, setTimeout(timeoutMs=DEFAULT_TIMEOUT=30000ms)<br/>aborts on timeout<br/>UnifiedApiClient.ts:202-209
    alt fetch throws AbortError
        UAC->>UAC: createApiError('Request timeout or cancelled',0,'Timeout')<br/>UnifiedApiClient.ts:231-236
    else fetch throws other
        UAC->>UAC: createApiError('Network error: ...',0,'Network Error')<br/>UnifiedApiClient.ts:239-244
    end
    alt !response.ok
        UAC->>UAC: createApiError(message,status) - interceptors.onError is unset, so it is thrown as is<br/>UnifiedApiClient.ts:276-296
        alt retryCondition true (attempt<3, status>=500 or 0, not 401/403)
            UAC->>UAC: delay = retryDelay(1000) * 2^attempt, retry<br/>UnifiedApiClient.ts:47-58, :161-173
        else
            UAC-->>Mod: throw apiError
        end
    else response.ok
        UAC->>AuthI: interceptors.onResponse(apiResponse) - no config argument<br/>UnifiedApiClient.ts:301-303
        AuthI->>AuthI: authResponseInterceptor: warnIfServerInReleaseMode(status,sentDevToken)<br/>authInterceptor.ts:121-129, :27-38
        UAC-->>Mod: ApiResponse~T~{data,status,statusText,headers}
    end
    Note over UAC,AuthI: DEBT: the one-shot "server is a release build" warning (ADR-06 sD1, authInterceptor.ts:27-38) can never fire.<br/>It needs status 401 and a sent dev token, but authResponseInterceptor is registered as onResponse, which runs only<br/>on response.ok (UnifiedApiClient.ts:276-303), and is called without config, so sentDevToken is always false (authInterceptor.ts:125-126).

    rect rgb(252,242,225)
    Note over Ax: RESOLVED ADR-2074: settingsApi still uses its own global axios instance<br/>(a transport choice), but its interceptor no longer reimplements the auth<br/>branch - it calls the shared computeAuthHeaders from authInterceptor.ts:51.<br/>All four HTTP transports now share one signing helper. endpoints.ts:42
    Ax->>Ax: axios.interceptors.request.use: await computeAuthHeaders(fullUrl,method,body),<br/>Object.assign onto config.headers<br/>endpoints.ts:42-57
    Ax->>Ax: axios.get/put(`${API_BASE}/api/settings/...`)<br/>API_BASE='' - relative, proxied by Vite/nginx<br/>endpoints.ts:36,91,96
    Ax->>Ax: getSettingsByPaths strips ADR-2189 retired paths from /api/settings/all before caching<br/>so a server document still carrying them never re-enters the store<br/>endpoints.ts:233,243-248
    end

    Note over Mod: Endpoint modules on UnifiedApiClient: analyticsApi.ts (AnalyticsAPI class, includes getCycles),<br/>graphExpandApi.ts (fetchNodeRelations/expandNode), exportApi.ts deleted as dead (no importer, a route that never existed),<br/>workspaceApi.ts (WorkspaceApiError): client/src/api/analyticsApi.ts:134,409,<br/>graphExpandApi.ts:69,84, workspaceApi.ts:62,80,136. constraintsApi.ts and layoutApi.ts import the global axios<br/>instead (constraintsApi.ts:5,86, layoutApi.ts:5,66). layoutApi.setMode posts the engine's own LayoutMode name with no<br/>remapping layer - toEngineLayoutMode is gone (ADR-2140, layoutApi.ts:47-60, :70-77)
```
**Drift (retiredPaths comment vs code):** `retiredPaths.ts:11-12` places the server-load strip in the settings store and on a `/visual` response; it runs in `getSettingsByPaths` on `/api/settings/all` (`endpoints.ts:243-247`).
**Invariant:** every settings load goes through `getSettingsByPaths` (`coreSlice.ts:103`, `coreSlice.ts:284`, `persistenceSlice.ts:116`), so a retired path cannot reach the store from the server (`endpoints.ts:247`).

## VC-30.9 graphDataManager — fetchInitialData, setGraphData, id maps, cache
```mermaid
sequenceDiagram
    autonumber
    participant Caller as AppInitializer<br/>client/src/app/AppInitializer.tsx:168
    participant GDM as GraphDataManager<br/>client/src/features/graph/managers/graphDataManager.ts:25
    participant Rest as fetchGraphData<br/>client/src/features/graph/managers/dataManager/restClient.ts
    participant Gate as dropLinkedPageStubs<br/>client/src/features/graph/managers/dataManager/nodeUtils.ts
    participant Topo as buildNodeIdMaps/setDataAndNotify<br/>client/src/features/graph/managers/dataManager/topology.ts
    participant WP as graphWorkerProxy<br/>client/src/features/graph/managers/graphDataManager.ts:6
    participant WSH as handleInitialGraphLoad<br/>client/src/store/websocket/textMessageHandler.ts:229

    Caller->>GDM: fetchInitialData() - counts restLoadsInFlight++ then runs _fetchInitialDataInner<br/>graphDataManager.ts:202-206,217
    GDM->>GDM: includeLinkedPages = settings.nodeFilter.includeLinkedPages ?? false<br/>graphDataManager.ts:220-221
    GDM->>Rest: fetchGraphData(graphType,graphTypeFilter,!includeLinkedPages)<br/>graphDataManager.ts:222-226
    Rest->>Rest: normalise edge.source/target via String(), node.id via String()<br/>restClient.ts:107,121
    Rest-->>GDM: GraphData{nodes,edges}
    GDM->>GDM: setGraphData(validatedData)<br/>graphDataManager.ts:228,291
    alt currentData.nodes.length == 0
        GDM->>GDM: scheduleEmptyDataRetry(1,retryTimeout,fetchInitialData,...)<br/>T6 fix - backend up but Oxigraph empty<br/>graphDataManager.ts:234-241
    end
    GDM-->>Caller: return currentData (lastGraphData ?? validatedData)<br/>graphDataManager.ts:230,244
    Note over GDM: finally: restLoadsInFlight-- and, when it returns to 0, resolveHeldServerLoad(restSucceeded)<br/>graphDataManager.ts:209-214

    rect rgb(235,245,255)
    Note over WSH,GDM: INVARIANT: REST topology is authoritative during the startup race. A server-pushed initialGraphLoad<br/>(capped at connect time) is held while a REST load is in flight and never seeds or replaces the topology first<br/>textMessageHandler.ts:235-243, graphDataManager.ts:58-64
    WSH->>GDM: offerServerGraphLoad(data,{isFilterResponse})<br/>textMessageHandler.ts:319-321, graphDataManager.ts:258
    alt isRestLoadInFlight()
        GDM-->>WSH: deferred - the latest load is kept in heldServerLoad<br/>graphDataManager.ts:262-268
        Note over GDM: when REST settles the held load is DROPPED on REST success unless it answers our own filter_update,<br/>and APPLIED through setGraphData on REST failure<br/>graphDataManager.ts:274-287
    else no REST load pending
        GDM->>GDM: setGraphData(data) and return applied<br/>graphDataManager.ts:270-271
    end
    end

    Note over GDM: setGraphData(data) - population gate + id-map build + cache + worker delivery<br/>graphDataManager.ts:291
    GDM->>Gate: dropLinkedPageStubs(data,includeLinkedPages)<br/>strips ~14.7k of 17.1k linked_page wikilink stubs<br/>graphDataManager.ts:304
    GDM->>GDM: nodes.map(ensureNodeHasValidPosition)<br/>graphDataManager.ts:308-312
    GDM->>Topo: buildNodeIdMaps(nodes,nodeIdMap,reverseNodeIdMap)<br/>graphDataManager.ts:320
    GDM->>Topo: setDataAndNotify(validatedData,lastGraphDataHash,listeners,setter)<br/>ADR-03 D5 single cached delivery path<br/>graphDataManager.ts:323-331
    Topo-->>GDM: onGraphDataChange listeners fire with validatedData
    alt graphWorkerProxy.isReady()
        GDM->>WP: setGraphTopology(validatedData)<br/>ADR-03 D7 - worker gets topology directly from the manager,<br/>NOT from any React state<br/>graphDataManager.ts:334-336
        alt setGraphTopology throws
            Note over GDM: caught, logged warn, topology delivery to worker skipped this pass<br/>graphDataManager.ts:337-339
        end
    end

    Note over GDM: mergeGraphData(newNodes,newEdges,anchorNodeId,anchorPosition) - additive expansion,<br/>seeds new nodes on golden-angle ring around LIVE anchor position (worker/SAB,<br/>not stale cached node.position), dedups nodes/edges by String()-coerced id<br/>graphDataManager.ts:397-470
    Note over GDM: getCachedGraphData() always returns null by design - worker data is<br/>async-only, callers must use fallback positioning<br/>graphDataManager.ts:134-136
```
## VC-30.10 graphDataManager binary path and the dataWithPositions forwarding invariant
```mermaid
sequenceDiagram
    autonumber
    participant WS as websocketService.onBinaryMessage<br/>client/src/app/AppInitializer.tsx:201
    participant GDM as GraphDataManager.updateNodePositions<br/>client/src/features/graph/managers/graphDataManager.ts:497
    participant Heal as maybeSelfHealUnknownNodes<br/>client/src/features/graph/managers/graphDataManager.ts:515
    participant Frame as handleBinaryFrame<br/>client/src/features/graph/managers/dataManager/wsClient.ts:19
    participant WP as graphWorkerProxy.processBinaryFrame<br/>client/src/features/graph/managers/dataManager/wsClient.ts:42
    participant Sub as useGraphDataSubscription<br/>client/src/features/graph/hooks/useGraphDataSubscription.ts:23
    participant GM as GraphManager<br/>client/src/features/graph/components/GraphManager.tsx:326
    participant BP as handleLegacyBinaryData<br/>client/src/store/websocket/binaryProtocol.ts:352

    WS->>GDM: updateNodePositions(positionData: ArrayBuffer)<br/>graphDataManager.ts:497
    GDM->>Heal: maybeSelfHealUnknownNodes(positionData) - sampled every 120 frames<br/>graphDataManager.ts:499,515-516
    opt sample frame && nodeIdMap non-empty && no opted-in serverRemove quality/authority filter
        Heal->>Heal: parseBinaryNodeData, check reverseNodeIdMap.has(getActualNodeId(id))<br/>graphDataManager.ts:528-529
        alt unknown ids found && cooldown (30000ms) elapsed
            Heal->>GDM: fetchInitialData() - topology hash makes false-positive refetch a no-op,<br/>skipped only when nodeFilter.serverRemove narrowed the graph (grey-out keeps every id)<br/>graphDataManager.ts:524-535
        end
    end
    GDM->>Frame: handleBinaryFrame(positionData,lastBinaryUpdateTime,onUpdateTime)<br/>graphDataManager.ts:500-504
    Frame->>Frame: onUpdateTime(Date.now()) - no throttle, every ack-paced chunk is applied<br/>wsClient.ts:26-27
        Frame->>WP: processBinaryFrame(new Uint8Array(positionData)) - zero-copy transfer,<br/>frames wait in the proxy FIFO in arrival order and none<br/>is collapsed (ADR-2141), the promise resolves once the worker<br/>has applied this frame<br/>wsClient.ts:41-42, graphWorkerProxy.ts:227-241
        Frame->>Frame: resetTransientErrors() on success<br/>wsClient.ts:43
        alt processBinaryFrame throws
            Frame->>Frame: recordTransientError('updateNodePositions')<br/>wsClient.ts:63
        end

    rect rgb(255,245,230)
    Note over GDM,BP: ADR-2141 acknowledgement: processBinaryData routes<br/>V3 and V5 frames to handleLegacyBinaryData first, which awaits<br/>updateNodePositions and only then sends BroadcastAck naming the<br/>sequence the frame carried. A failed apply returns without an ack,<br/>so the server rebuilds the chunk after its timeout, and a frame<br/>with no sequence is not acknowledged<br/>binaryProtocol.ts:469-471, :393-409
    BP->>GDM: await updateNodePositions(data)<br/>binaryProtocol.ts:394
    BP->>BP: sendPositionAck(get, frame.broadcastSequence, parsedNodes.length)<br/>binaryProtocol.ts:407-408, sendPositionAck :127-139
    Note over Frame,BP: INVARIANT: the 16ms skip is gone, so a chunk resolves updateNodePositions<br/>only after processBinaryFrame applied it and only then is acknowledged<br/>(wsClient.ts:39-42, binaryProtocol.ts:394,408), as ADR-2141 decision 5 requires
    end

    rect rgb(230,250,230)
    Note over Sub,GM: INVARIANT (historical bug, now fixed): useGraphDataSubscription's<br/>handleGraphUpdate builds dataWithPositions (String()-coerced ids,<br/>getPositionForNode() fallback for (0,0,0) nodes, edge source/target<br/>recovery) and the caller MUST forward THAT object, not the original<br/>data, onward - useGraphDataSubscription.ts:69-100,117
    Sub->>Sub: onGraphData(dataWithPositions)<br/>useGraphDataSubscription.ts:117
    Sub->>GM: setGraphData(dataWithPositions) via onGraphData callback<br/>GraphManager.tsx:326-327
    Note over Sub: this React graphData state feeds RENDER only - the worker's topology<br/>comes from GraphDataManager.setGraphData -> graphWorkerProxy.setGraphTopology<br/>directly (VC-30.9), an entirely separate path from this hook
    end
    Note over Sub: 5000ms fallback timer seeds 3 synthetic nodes if no real data<br/>arrived by then (lastProcessedGraphRef still null)<br/>useGraphDataSubscription.ts:141-155
```
**Debt (loose end):** with the 16 ms throttle gone, `handleBinaryFrame` keeps an unused `_lastUpdateTime` parameter (`wsClient.ts:21`) and `updateNodePositions` still passes `lastBinaryUpdateTime` into it (`graphDataManager.ts:502`).

## VC-30.11 graph worker handoff — proxy init, SAB-vs-Comlink capability, dispose
```mermaid
sequenceDiagram
    autonumber
    participant AI as AppInitializer<br/>client/src/app/AppInitializer.tsx:86
    participant Proxy as GraphWorkerProxy<br/>client/src/features/graph/managers/graphWorkerProxy.ts:128
    participant Comlink as comlink wrap/transfer<br/>client/src/features/graph/managers/graphWorkerProxy.ts:37,182,270
    participant W as GraphWorker<br/>client/src/features/graph/workers/graph.worker.ts:17,580-581
    participant Test as sabOrTransferDetection.test.ts<br/>client/src/features/graph/__tests__/sabOrTransferDetection.test.ts:16

    Note over Proxy: WORKER_USES_SAB = SAB_CAPABLE && !FORCE_COMLINK, computed ONCE at<br/>module load - SAB_CAPABLE = typeof SharedArrayBuffer!=='undefined' &&<br/>self.crossOriginIsolated===true, FORCE_COMLINK = VITE_FORCE_COMLINK==='1'<br/>graphWorkerProxy.ts:78-89
    Test->>Proxy: import('../managers/graphWorkerProxy') under mocked globals<br/>via vi.resetModules() per case<br/>sabOrTransferDetection.test.ts:17-19,36,45,58
    alt crossOriginIsolated=false
        Test->>Test: expect(WORKER_USES_SAB).toBe(false)<br/>sabOrTransferDetection.test.ts:30-38
    else SharedArrayBuffer undefined
        Test->>Test: expect(WORKER_USES_SAB).toBe(false)<br/>sabOrTransferDetection.test.ts:40-50
    else both SAB and crossOriginIsolated true
        Test->>Test: expect(WORKER_USES_SAB).toBe(true)<br/>sabOrTransferDetection.test.ts:52-60
    end

    AI->>Proxy: graphWorkerProxy.initialize()<br/>graphWorkerProxy.ts:160
    Proxy->>Comlink: new Worker(new URL('../workers/graph.worker.ts',import.meta.url),<br/>{type:'module'})<br/>graphWorkerProxy.ts:173-176
    Proxy->>Comlink: workerApi = wrap~GraphWorkerType~(worker)<br/>graphWorkerProxy.ts:182
    Proxy->>W: workerApi.initialize() - handshake, resolves immediately<br/>graphWorkerProxy.ts:185, graph.worker.ts:103-106
    alt WORKER_USES_SAB
        Proxy->>Proxy: sharedBuffer = new SharedArrayBuffer(POSITION_BUFFER_BYTES)<br/>INITIAL_NODES=50000 * POSITION_FLOATS_PER_NODE=4 * 4 bytes, an initial size not a ceiling<br/>graphWorkerProxy.ts:111-114,193-194
        Proxy->>W: workerApi.setupSharedPositions(sharedBuffer)<br/>graphWorkerProxy.ts:195, graph.worker.ts:199-205
        alt SAB attach throws
            Proxy->>Proxy: degrade to Comlink transfer path, sharedBuffer=null<br/>graphWorkerProxy.ts:199-206
        end
    end
    Proxy->>Proxy: isInitialized=true<br/>graphWorkerProxy.ts:209

    Note over Proxy: D7 surface is EXACTLY 4 methods - processBinaryFrame, getPositions,<br/>setGraphTopology, dispose. Legacy methods (tick,getGraphData,<br/>getAnalyticsBuffer,reheatSimulation,updateForcePhysicsSettings) throw<br/>deprecation shims - graphWorkerProxy.ts:1-15,399-472
    AI->>Proxy: setGraphTopology(graph) - the caller is GraphDataManager.setGraphData (graphDataManager.ts:336), not AppInitializer<br/>graphWorkerProxy.ts:330
    Proxy->>Proxy: ensureSharedCapacity(nodes.length) - ADR-2181 D4, grow the SAB to the next power of two<br/>via sharedCapacityFor and publishNodeBufferReport(total,drawn,truncated) for the in-canvas notice<br/>graphWorkerProxy.ts:334,121-126,344-364
    Proxy->>W: setupSharedPositions(grown buffer) - the worker refills it at once<br/>graphWorkerProxy.ts:352-353, graph.worker.ts:203-204
    Proxy->>W: workerApi.setGraphData(graph)<br/>graphWorkerProxy.ts:335, graph.worker.ts:115
    W->>W: ensureNodeHasValidPosition per node, String()-coerce node.id,<br/>build nodeIdMap/reverseNodeIdMap/nodeIndexMap, edge adjacency maps<br/>graph.worker.ts:117,132-157
    W->>W: initPositionBuffers preserving old positions by nodeIndexMap match<br/>graph.worker.ts:159-166
    W->>W: syncToSharedBuffer(), graphDataLoaded=true<br/>graph.worker.ts:185,188
    opt pendingBinaryFrames queued (FIX 4 race guard)
        W->>W: replay each queued frame via processBinaryData()<br/>graph.worker.ts:189-196,273-279
    end

    rect rgb(235,235,252)
    Note over Proxy,W: processBinaryFrame ordered single-flight (ADR-03 D2, amended by ADR-2141): at most one Comlink call in flight,<br/>frames arriving meanwhile wait in _frameQueue in arrival order and NONE is collapsed, each promise resolves when its own<br/>frame has been applied, a backstop of 256 queued frames rejects and drops the oldest (MAX_QUEUED_FRAMES)<br/>graphWorkerProxy.ts:44,143-145,227-263
    Proxy->>Comlink: transfer(frame.buffer,[frame.buffer]) - neuters caller's ArrayBuffer<br/>graphWorkerProxy.ts:270
    alt WORKER_USES_SAB
        Proxy->>W: workerApi.processBinaryFrame(transferable) - returns void,<br/>renderer reads SAB view directly<br/>graphWorkerProxy.ts:272-277, graph.worker.ts:216-220
    else Comlink transfer mode
        Proxy->>W: workerApi.processBinaryFrame(transferable)<br/>graphWorkerProxy.ts:281-283
        W-->>Proxy: transfer(out,[out]) - full stride-3 currentPositions,<br/>NOT the stride-4 [nodeId,x,y,z] update array<br/>graph.worker.ts:221-225
        Proxy->>Proxy: lastTransferredView = new Float32Array(returned)<br/>graphWorkerProxy.ts:284-286
    end
    loop _frameQueue not empty
        Proxy->>Proxy: _drain shifts the next job, awaits its dispatch, resolves it, then yields a microtask<br/>so the render loop gets a turn<br/>graphWorkerProxy.ts:243-263
    end
    end

    Note over Proxy: getPositionsSync() - SAB mode returns sharedPositionView directly,<br/>Comlink mode returns lastTransferredView - called from useFrame per-frame<br/>render, avoids Promise allocation<br/>graphWorkerProxy.ts:316-320
    Note over Proxy,W: DEBT: the worker takes tweening settings (responseMs, default 120) only through setTweeningSettings, reached from its<br/>updateSettings (graph.worker.ts:44-47,264,484-489), the proxy exposes setTweeningSettings as a throwing shim<br/>(graphWorkerProxy.ts:523-528) and the tweeningSettingsUpdated event dispatched by the settings store has no listener in client/src<br/>(physicsSlice.ts:200-209)
    AI->>Proxy: dispose() - no production caller in client/src at this revision, only a unit test tears it down<br/>graphWorkerProxy.ts:368
    Proxy->>W: worker.terminate() - no graceful worker-side RPC, hard terminate<br/>graphWorkerProxy.ts:369-372
    Proxy->>Proxy: clear sharedBuffer/sharedPositionView/lastTransferredView,<br/>reject every still-queued frame job with "worker disposed", reset isInitialized/initPromise<br/>graphWorkerProxy.ts:373-382
```
## VC-30.12 Store to component dependency graph
```mermaid
flowchart LR
    settingsStore["useSettingsStore<br/>client/src/store/settingsStore.ts:27"]
    websocketStore["useWebSocketStore<br/>client/src/store/websocketStore.ts:9"]
    timelineStore["useTimelineStore<br/>client/src/store/timelineStore.ts:87"]
    beamStore["useTransientBeamStore<br/>client/src/store/transientBeamStore.ts:65"]
    agentTargetStore["useAgentTargetStore<br/>client/src/store/agentTargetStore.ts:43"]
    nodePosCtx["NodePositionContext module singleton<br/>client/src/features/graph/contexts/NodePositionContext.tsx:8<br/>written by GraphManager.tsx:447"]
    workerErrStore["useWorkerErrorStore<br/>client/src/store/workerErrorStore.ts:23"]

    GraphManager["GraphManager<br/>client/src/features/graph/components/GraphManager.tsx:57-74"]
    GraphCanvas["GraphCanvas<br/>client/src/features/graph/components/GraphCanvas.tsx:180,194-200"]
    NodeDetailPanel["NodeDetailPanel embedded in the drawer Node tab<br/>client/src/features/graph/components/NodeDetailPanel.tsx:65<br/>mounted at RightDrawer.tsx:317, no longer by MainLayout"]
    GemNodes["GemNodes<br/>client/src/features/graph/components/GemNodes.tsx"]
    GlassEdges["GlassEdges<br/>client/src/features/graph/components/GlassEdges.tsx"]
    TimelineScrubber["TimelineScrubber<br/>diffMode written through useSettingField provenance.diffMode, ADR-2189<br/>client/src/features/graph/components/TimelineScrubber.tsx:77-87"]
    ControlCenter["ControlCenter<br/>client/src/features/control-center/ControlCenter.tsx:37"]
    SettingField["useSettingField hook<br/>client/src/features/control-center/hooks/useSettingField.ts:41"]
    RevealSetting["useRevealSetting hook<br/>client/src/features/control-center/hooks/useRevealSetting.ts"]
    NostrAuthControl["NostrAuthControl<br/>client/src/features/control-center/primitives/NostrAuthControl.tsx"]
    StatusFlyout["StatusFlyout<br/>layout mode readout and voice row moved here, ADR-2189<br/>client/src/features/control-center/status/StatusFlyout.tsx:156-158,269-275"]
    AgentOverlay["AgentOverlay<br/>nameplates, trails, collab edges, payment labels<br/>client/src/features/bots/components/AgentOverlay.tsx:23<br/>mounted at GraphCanvas.tsx:381"]
    VirtualAgentMarkers["VirtualAgentMarkers<br/>client/src/features/visualisation/components/VirtualAgentMarkers.tsx:35<br/>mounted at GraphManager.tsx:752"]
    MarkerPass["agentMarkers - the one agent-position writer<br/>client/src/features/bots/agentMarkerPass.ts:71,144<br/>stepped by GraphManager.tsx:470, reads agentTargetStore"]
    WorkerErrorModal["WorkerErrorModal<br/>client/src/components/WorkerErrorModal.tsx:7"]
    AppInitializer["AppInitializer<br/>client/src/app/AppInitializer.tsx:108,118"]
    EmbeddingCloudLayer["EmbeddingCloudLayer<br/>client/src/features/visualisation/components/EmbeddingCloudLayer.tsx:147,332"]
    useTransientBeams["useTransientBeams hook<br/>client/src/features/visualisation/hooks/useTransientBeams.ts:25-26"]
    TransientBeamsLayer["TransientBeamsLayer<br/>client/src/features/visualisation/components/TransientBeamsLayer.tsx"]
    binaryProtocol["binaryProtocol.ts handleAgentActionTagged<br/>client/src/store/websocket/binaryProtocol.ts:427, routed at :476"]

    settingsStore --> GraphManager
    settingsStore --> GraphCanvas
    settingsStore --> NodeDetailPanel
    SettingField --> TimelineScrubber
    settingsStore --> StatusFlyout
    settingsStore --> ControlCenter
    settingsStore --> SettingField
    settingsStore --> RevealSetting
    settingsStore --> NostrAuthControl
    settingsStore --> AgentOverlay
    settingsStore --> VirtualAgentMarkers
    settingsStore --> AppInitializer

    websocketStore --> EmbeddingCloudLayer
    websocketStore -.->|main.tsx dev-mode window expose only| AppInitializer

    timelineStore --> TimelineScrubber

    beamStore --> useTransientBeams
    useTransientBeams --> TransientBeamsLayer
    binaryProtocol -->|pushTransientBeams| beamStore
    binaryProtocol -->|pushActions| agentTargetStore
    GraphCanvas -->|"mounts AgentOverlay, GraphCanvas.tsx:381"| AgentOverlay
    agentTargetStore -->|"seenAt ids, AgentOverlay.tsx:27"| AgentOverlay
    agentTargetStore --> VirtualAgentMarkers
    nodePosCtx -->|"drawn positions and id map"| EmbeddingCloudLayer
    MarkerPass -->|"agentMarkers positions"| VirtualAgentMarkers
    MarkerPass -->|"agentMarkerPosition per wire id"| AgentOverlay
    PassNote["ADR-2142: agents are not graph nodes. Once a frame, after<br/>GraphManager copies the drawn-positions array, the marker pass<br/>walks agentTargetStore and fills one wire-id to world-position map<br/>that markers, beams, nameplates, trails, collab edges and payment<br/>labels all read - client/src/features/bots/agentMarkerPass.ts:1-19<br/>RESOLVED: BotsVisualization, BotsNode and their per-agent nudge<br/>are deleted (VisionClaw 15acb9219, 6cc5c153d)"]
    MarkerPass --- PassNote
    NudgeDebt["DEBT: GraphManager publishes the drawn-positions array and<br/>the id map through module-level mutable exports rather than a<br/>React context, so nothing invalidates a stale reader after a remount.<br/>client/src/features/graph/contexts/NodePositionContext.tsx:7-19"]
    nodePosCtx --- NudgeDebt

    workerErrStore --> WorkerErrorModal
    workerErrStore --> AppInitializer

    subgraph GraphRenderTree["Graph render tree"]
        GraphManager --> GemNodes
        GraphManager --> GlassEdges
        GraphManager --> VirtualAgentMarkers
    end

    subgraph ControlCenterTree["Control Center tree"]
        ControlCenter --> SettingField
        ControlCenter --> RevealSetting
        ControlCenter -->|"ControlCenter.tsx:134"| RightDrawer
        RightDrawer --> NodeDetailPanel
    end
    RightDrawer["RightDrawer<br/>client/src/features/control-center/drawer/RightDrawer.tsx:86"]
    StatusSurface["StatusSurface<br/>client/src/features/control-center/status/StatusSurface.tsx:34,57<br/>N unsaved chip, StatusSurface.tsx:47-48,92-100<br/>right corner row, MainLayout.tsx:97-103"]
    StatusSurface --> StatusFlyout
    syncStatus["useSettingsSyncStatus<br/>client/src/store/settingsSyncStatus.ts:29"]
    syncStatus --> StatusSurface
    syncStatus -.->|"coreSlice.ts:129"| settingsStore
```
## VC-30.13 Settings load precedence and where an unsaved change surfaces
```mermaid
flowchart TB
    subgraph Load["coreSlice.initialize - server wins for what it stores"]
        GET["getSettingsByPaths(ESSENTIAL_PATHS) fetches /api/settings/all,<br/>strips retired paths, records storedClientPaths<br/>client/src/api/settings/endpoints.ts:243-248"]
        Base["base = server document, overlay = persisted browser copy<br/>deepMergeSettings(essentialSettings, partialSettings)<br/>client/src/store/settings/coreSlice.ts:116-119"]
        Win{"browser path stored on the server, has a server bucket<br/>and is NOT unsynced?<br/>client/src/store/settings/coreSlice.ts:128-129"}
        SrvVal["server value overwrites the browser copy<br/>client/src/store/settings/coreSlice.ts:130-131"]
        Keep["browser value kept: browser-only, never stored,<br/>or a change this browser could not save"]
        Phys["physics and clientTweening replaced whole from the server (ADR-2140)<br/>client/src/store/settings/coreSlice.ts:134-146"]
        Loaded["loadedPaths = every path the server document carries,<br/>so get() returns the server value, not undefined<br/>client/src/store/settings/coreSlice.ts:154-158"]
        Ready["autoSaveManager.setInitialized(true) flushes held changes<br/>client/src/store/settings/coreSlice.ts:166, autoSaveManager.ts:50-55"]
        GET --> Base --> Win
        Win -->|yes| SrvVal --> Phys
        Win -->|no| Keep --> Phys
        Phys --> Loaded --> Ready
    end
    subgraph Write["Write paths and the unsaved record"]
        Auto["autoSave queue: not-ready, sync-off or failed<br/>client/src/store/autoSaveManager.ts:58-63"]
        Direct["setByPath and batchUpdate bypass the queue,<br/>recordWrite marks failed or saved<br/>client/src/store/settings/persistenceSlice.ts:8-14,54,78"]
        Retry["retry success marks saved<br/>client/src/store/settingsRetryManager.ts:90,110,193"]
        Rec["unsaved path to reason, persisted in its own localStorage key<br/>client/src/store/settingsSyncStatus.ts:21,27"]
        Chip["status chip: N unsaved, tooltip describeUnsaved<br/>client/src/features/control-center/status/StatusSurface.tsx:92-100"]
        Auto --> Rec
        Direct --> Rec
        Retry --> Rec
        Rec --> Chip
    end
    Rec -.->|"isUnsyncedPath, client/src/store/settingsSyncStatus.ts:62"| Win
    Note1["OPEN: nothing calls clear() and setSyncEnabled(true) re-sends nothing,<br/>so a sync-off or a failed setByPath entry stays in the record,<br/>keeps the chip lit and keeps beating the server value on every load<br/>client/src/store/autoSaveManager.ts:24-26, persistenceSlice.ts:54-55"]
    Rec --- Note1
```
**What it shows:** the load rule since the settings-integrity fix (`7cde2f00c`): the server document is the base, and for any path the server actually stores its value replaces this browser's copy unless the record of unsaved changes holds that path (`coreSlice.ts:128-131`). Every write path that can lose a change now feeds one persisted record, and the status chip counts it (`StatusSurface.tsx:47-48`).
**Why it is this way:** the audit found the browser copy won everywhere (F5) and changes were dropped with a debug log (F7); the fix inverted precedence and made every loss visible (`settingsSyncStatus.ts:1-11`).

**Open (risk):** a `sync-off` entry is never re-queued: `setSyncEnabled(true)` only flips the flag (`autoSaveManager.ts:24-26`), `setByPath` failures go to the record but not to the retry queue (`persistenceSlice.ts:54-55`), and the store's `clear` (`settingsSyncStatus.ts:51`) has no caller, so such a path keeps the chip lit and keeps overriding the server on every load (`coreSlice.ts:129`).
**Open (risk):** a failed retry updates only `lastAttempt` (`settingsRetryManager.ts:99-102`, `settingsRetryManager.ts:113-114`); `attempts` grows only in `addFailedUpdate` (`settingsRetryManager.ts:44`), so a path that keeps failing is retried every poll and the `settings-retry-failed` give-up (`settingsRetryManager.ts:120-128`) never fires from retries alone.
**Tension (parallel):** two write routers disagree on physics: the batch path nests `visualisation.graphs.*.physics.*` through `physicsPatchEntry` (`endpoints.ts:362-364`), while the single-path `updateSettingByPath` used by one-item retries and `setByPath` flattens it to its last segment (`endpoints.ts:303-304`), so a nested physics leaf retried alone lands at the top of the physics document.
**Open (risk):** the FPS auto-degrade writes `visualisation.sceneEffects.*` through `updateSettings` (`useFpsMonitor.ts:117-127`), which queues them into the shared visual bucket (`coreSlice.ts:401`), so one slow machine lowers effects for every viewer when `qualityGates.autoAdjust` is on (`useFpsMonitor.ts:39-41`).
**Debt (loose end):** `getSettingsByPaths` builds a per-path `result` and then returns the whole document (`endpoints.ts:253-261`), so its `paths` argument selects nothing.
**Debt (loose end):** `exportSettings` fetches the full document only when `loadedPaths.size === ESSENTIAL_PATHS.length` (`persistenceSlice.ts:112`), which no longer holds once `initialize` marks every server path loaded (`coreSlice.ts:154-158`).
**Debt (loose end):** the retired-path list cites an ephemeral scratchpad file as its source (`retiredPaths.ts:22`) rather than a tracked document.
