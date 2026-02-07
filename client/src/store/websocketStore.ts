import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { createLogger, createErrorMetadata } from '../utils/loggerConfig';
import { debugState } from '../utils/clientDebugState';
import { useSettingsStore } from './settingsStore';
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import { parseBinaryNodeData, isAgentNode, BinaryNodeData } from '../types/binaryProtocol';
import { NodePositionBatchQueue, createWebSocketBatchProcessor } from '../utils/BatchQueue';
import { validateNodePositions, createValidationMiddleware } from '../utils/validation';
import {
  WebSocketMessage,
  WebSocketConnectionState,
  WebSocketStatistics,
} from '../types/websocketTypes';
import { binaryProtocol, MessageType, GraphTypeFlag, AgentActionEvent } from '../services/BinaryWebSocketProtocol';
import { nostrAuth } from '../services/nostrAuthService';

const logger = createLogger('WebSocketStore');

// WebSocketAdapter interface for components that need to send binary data
export interface WebSocketAdapter {
  send: (data: ArrayBuffer) => void;
  isReady: () => boolean;
}

// Re-export types for backward compatibility
export interface WebSocketErrorFrame {
  code: string;
  message: string;
  category: 'validation' | 'server' | 'protocol' | 'auth' | 'rate_limit';
  details?: unknown;
  retryable: boolean;
  retryAfter?: number;
  affectedPaths?: string[];
  timestamp: number;
}

export interface QueuedMessage {
  type: 'text' | 'binary';
  data: string | ArrayBuffer;
  timestamp: number;
  retries: number;
}

export interface ConnectionState {
  status: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'failed';
  lastConnected?: number;
  lastError?: string;
  reconnectAttempts: number;
}

export interface SolidNotification {
  type: 'pub' | 'ack';
  url: string;
}

export type SolidNotificationCallback = (notification: SolidNotification) => void;

type LegacyMessageHandler = (message: WebSocketMessage) => void;
type BinaryMessageHandler = (data: ArrayBuffer) => void;
type ConnectionStatusHandler = (connected: boolean) => void;
type ConnectionStateHandler = (state: ConnectionState) => void;
type EventHandler = (data: unknown) => void;

// Store state interface
export interface WebSocketState {
  // Connection state
  socket: WebSocket | null;
  isConnected: boolean;
  isServerReady: boolean;
  connectionState: ConnectionState;
  url: string;

  // Solid WebSocket state
  solidSocket: WebSocket | null;
  isSolidConnected: boolean;
  solidSubscriptions: Map<string, Set<SolidNotificationCallback>>;

  // Message queuing
  messageQueue: QueuedMessage[];

  // Statistics
  statistics: WebSocketStatistics;

  // Configuration
  reconnectInterval: number;
  maxReconnectAttempts: number;
  reconnectAttempts: number;

  // Actions
  connect: () => Promise<void>;
  disconnect: () => void;
  close: () => void;
  sendMessage: (type: string, data?: unknown) => void;
  sendRawBinaryData: (data: ArrayBuffer) => void;
  sendFilterUpdate: (filter: FilterUpdateParams) => void;
  forceRefreshFilter: () => Promise<void>;
  forceReconnect: () => void;
  clearMessageQueue: () => void;
  setCustomBackendUrl: (backendUrl: string | null) => void;

  // Node position updates
  sendNodePositionUpdates: (updates: NodePositionUpdate[]) => void;
  flushPositionUpdates: () => Promise<void>;
  getPositionQueueMetrics: () => ReturnType<NodePositionBatchQueue['getMetrics']> | null;

  // Event handling
  onMessage: (handler: LegacyMessageHandler) => () => void;
  onBinaryMessage: (handler: BinaryMessageHandler) => () => void;
  onConnectionStatusChange: (handler: ConnectionStatusHandler) => () => void;
  onConnectionStateChange: (handler: ConnectionStateHandler) => () => void;
  on: (eventName: string, handler: EventHandler) => () => void;
  emit: (eventName: string, data: unknown) => void;

  // Solid WebSocket
  connectSolid: () => void;
  disconnectSolid: () => void;
  subscribeSolidResource: (resourceUrl: string, callback: SolidNotificationCallback) => () => void;
  unsubscribeSolidResource: (resourceUrl: string) => void;
  isSolidWebSocketConnected: () => boolean;
  getSolidSubscriptions: () => string[];

  // Utility methods
  isReady: () => boolean;
  getConnectionState: () => ConnectionState;
  getQueuedMessageCount: () => number;

  // Testing/cleanup
  _reset: () => void;
  _getInternals: () => WebSocketInternals;
}

interface FilterUpdateParams {
  enabled?: boolean;
  qualityThreshold?: number;
  authorityThreshold?: number;
  filterByQuality?: boolean;
  filterByAuthority?: boolean;
  filterMode?: string;
}

interface NodePositionUpdate {
  nodeId: number;
  position: { x: number; y: number; z: number };
  velocity?: { x: number; y: number; z: number };
}

interface WebSocketInternals {
  messageHandlers: LegacyMessageHandler[];
  binaryMessageHandlers: BinaryMessageHandler[];
  connectionStatusHandlers: ConnectionStatusHandler[];
  connectionStateHandlers: ConnectionStateHandler[];
  eventHandlers: Map<string, EventHandler[]>;
  positionBatchQueue: NodePositionBatchQueue | null;
  heartbeatInterval: number | null;
  reconnectTimeout: number | null;
}

// Internal state not exposed in the store (non-serializable)
let messageHandlers: LegacyMessageHandler[] = [];
let binaryMessageHandlers: BinaryMessageHandler[] = [];
let connectionStatusHandlers: ConnectionStatusHandler[] = [];
let connectionStateHandlers: ConnectionStateHandler[] = [];
let eventHandlers: Map<string, EventHandler[]> = new Map();
let positionBatchQueue: NodePositionBatchQueue | null = null;
let heartbeatInterval: number | null = null;
let heartbeatTimeout: number | null = null;
let reconnectTimeout: number | null = null;
let binaryMessageCount = 0;
let positionUpdateSequence = 0;
let lastAckSentSequence = 0;
let filterSubscriptionSet = false;
// P2 PERFORMANCE FIX: Track individual filter fields instead of serializing
// the entire settings tree on every state change.
interface FilterSnapshot {
  enabled?: boolean;
  qualityThreshold?: number;
  authorityThreshold?: number;
  filterByQuality?: boolean;
  filterByAuthority?: boolean;
  filterMode?: string;
}
let lastFilterSnapshot: FilterSnapshot | null = null;
let solidReconnectTimeout: number | null = null;
let solidReconnectAttempts = 0;

const ACK_BATCH_SIZE = 10;
const HEARTBEAT_INTERVAL_MS = 30000;
const HEARTBEAT_TIMEOUT_MS = 45000;  // Allow for server load
const MAX_QUEUE_SIZE = 100;
const MAX_RECONNECT_DELAY = 30000;
const SOLID_MAX_RECONNECT_ATTEMPTS = 10;
const SOLID_RECONNECT_DELAY = 1000;

function determineWebSocketUrl(): string {
  // SSR safety check
  if (typeof window === 'undefined') {
    return 'ws://localhost:3001/wss';
  }

  const isDev = import.meta.env.DEV;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname;
  const port = isDev ? '3001' : window.location.port;
  const baseUrl = `${protocol}//${host}:${port}`;
  const wsUrl = `${baseUrl}/wss`;

  if (debugState.isEnabled()) {
    logger.info(`Determined WebSocket URL (${isDev ? 'dev' : 'prod'}): ${wsUrl}`);
  }

  return wsUrl;
}

function getUrlFromSettings(): string {
  const state = useSettingsStore.getState();
  const settings = state.settings;
  let newUrl = determineWebSocketUrl();

  if (settings?.system?.customBackendUrl &&
      settings.system.customBackendUrl.trim() !== '') {
    const customUrl = settings.system.customBackendUrl.trim();
    const protocol = customUrl.startsWith('https://') ? 'wss://' : 'ws://';
    const hostAndPath = customUrl.replace(/^(https?:\/\/)?/, '');
    newUrl = `${protocol}${hostAndPath.replace(/\/$/, '')}/wss`;
    if (debugState.isEnabled()) {
      logger.info(`Using custom backend WebSocket URL: ${newUrl}`);
    }
  }

  return newUrl;
}

function getSolidWebSocketUrl(): string | null {
  return import.meta.env.VITE_JSS_WS_URL || null;
}

// SSR-safe window check for clearTimeout/setInterval
function safeSetTimeout(fn: () => void, delay: number): number | null {
  if (typeof window !== 'undefined') {
    return window.setTimeout(fn, delay);
  }
  return null;
}

function safeClearTimeout(id: number | null): void {
  if (typeof window !== 'undefined' && id !== null) {
    window.clearTimeout(id);
  }
}

function safeSetInterval(fn: () => void, delay: number): number | null {
  if (typeof window !== 'undefined') {
    return window.setInterval(fn, delay);
  }
  return null;
}

function safeClearInterval(id: number | null): void {
  if (typeof window !== 'undefined' && id !== null) {
    window.clearInterval(id);
  }
}

export const useWebSocketStore = create<WebSocketState>()(
  subscribeWithSelector((set, get) => {
    // Helper functions that need access to set/get
    const updateConnectionState = (
      status: ConnectionState['status'],
      lastError?: string,
      lastConnected?: number
    ) => {
      set(state => ({
        connectionState: {
          ...state.connectionState,
          status,
          lastError,
          lastConnected,
          reconnectAttempts: state.reconnectAttempts
        }
      }));
      notifyConnectionStateHandlers();
    };

    const notifyConnectionStatusHandlers = (connected: boolean) => {
      connectionStatusHandlers.forEach(handler => {
        try {
          handler(connected);
        } catch (error) {
          logger.error('Error in connection status handler:', createErrorMetadata(error));
        }
      });
    };

    const notifyConnectionStateHandlers = () => {
      const state = get();
      connectionStateHandlers.forEach(handler => {
        try {
          handler(state.connectionState);
        } catch (error) {
          logger.error('Error in connection state handler:', createErrorMetadata(error));
        }
      });
    };

    const emit = (eventName: string, data: unknown) => {
      const handlers = eventHandlers.get(eventName);
      if (handlers) {
        handlers.forEach(handler => {
          try {
            handler(data);
          } catch (error) {
            logger.error(`Error in event handler for ${eventName}:`, createErrorMetadata(error));
          }
        });
      }
    };

    const queueMessage = (type: 'text' | 'binary', data: string | ArrayBuffer) => {
      set(state => {
        const newQueue = [...state.messageQueue];
        if (newQueue.length >= MAX_QUEUE_SIZE) {
          newQueue.shift();
          logger.warn('Message queue full, removed oldest message');
        }
        newQueue.push({
          type,
          data,
          timestamp: Date.now(),
          retries: 0
        });
        return { messageQueue: newQueue };
      });
    };

    const processMessageQueue = async () => {
      const state = get();
      if (!state.isConnected || !state.socket || state.messageQueue.length === 0) {
        return;
      }

      const messagesToProcess = [...state.messageQueue];
      set({ messageQueue: [] });

      for (const queuedMessage of messagesToProcess) {
        try {
          if (queuedMessage.type === 'text') {
            state.socket.send(queuedMessage.data as string);
          } else {
            state.socket.send(queuedMessage.data as ArrayBuffer);
          }

          if (debugState.isDataDebugEnabled()) {
            logger.debug(`Processed queued ${queuedMessage.type} message`);
          }
        } catch (error) {
          queuedMessage.retries++;
          if (queuedMessage.retries < 3) {
            set(s => ({ messageQueue: [...s.messageQueue, queuedMessage] }));
            logger.warn(`Failed to send queued message, retry ${queuedMessage.retries}/3`);
          } else {
            logger.error('Failed to send queued message after 3 retries, dropping:', createErrorMetadata(error));
          }
        }
      }
    };

    const startHeartbeat = () => {
      stopHeartbeat();
      heartbeatInterval = window.setInterval(() => {
        sendHeartbeat();
      }, HEARTBEAT_INTERVAL_MS);
    };

    const stopHeartbeat = () => {
      if (heartbeatInterval) {
        window.clearInterval(heartbeatInterval);
        heartbeatInterval = null;
      }
      if (heartbeatTimeout) {
        window.clearTimeout(heartbeatTimeout);
        heartbeatTimeout = null;
      }
    };

    const sendHeartbeat = () => {
      const state = get();
      if (!state.isConnected || !state.socket) {
        return;
      }

      try {
        state.socket.send('ping');

        heartbeatTimeout = window.setTimeout(() => {
          logger.warn('Heartbeat timeout - server not responding');
          handleHeartbeatTimeout();
        }, HEARTBEAT_TIMEOUT_MS);

        if (debugState.isDataDebugEnabled()) {
          logger.debug('Sent heartbeat ping');
        }
      } catch (error) {
        logger.error('Error sending heartbeat:', createErrorMetadata(error));
        handleHeartbeatTimeout();
      }
    };

    const handleHeartbeatResponse = () => {
      if (heartbeatTimeout) {
        window.clearTimeout(heartbeatTimeout);
        heartbeatTimeout = null;
      }

      if (debugState.isDataDebugEnabled()) {
        logger.debug('Received heartbeat pong');
      }
    };

    const handleHeartbeatTimeout = () => {
      logger.warn('Heartbeat timeout detected, connection may be dead');
      const state = get();
      if (state.socket) {
        state.socket.close(4000, 'Heartbeat timeout');
      }
    };

    const initializeBatchQueue = () => {
      if (positionBatchQueue) {
        positionBatchQueue.destroy();
      }

      const validationMiddleware = createValidationMiddleware({
        maxNodes: 10000,
        maxCoordinate: 10000,
        minCoordinate: -10000,
        maxVelocity: 1000
      });

      const batchProcessor = createWebSocketBatchProcessor((data: ArrayBuffer) => {
        const state = get();
        if (!state.isConnected || !state.socket) {
          logger.warn('Cannot send batch: WebSocket not connected');
          return;
        }

        try {
          state.socket.send(data);

          if (debugState.isDataDebugEnabled()) {
            logger.debug(`Sent binary batch: ${data.byteLength} bytes`);
          }
        } catch (error) {
          logger.error('Error sending batch:', createErrorMetadata(error));
          throw error;
        }
      });

      positionBatchQueue = new NodePositionBatchQueue({
        processBatch: async (batch: BinaryNodeData[]) => {
          const validatedBatch = validationMiddleware(batch);

          if (validatedBatch.length === 0) {
            logger.warn('All nodes in batch failed validation');
            return;
          }

          await batchProcessor.processBatch(validatedBatch);
        },
        onError: batchProcessor.onError,
        onSuccess: batchProcessor.onSuccess
      });

      logger.info('Position batch queue initialized');
    };

    const sendPositionAck = (sequenceId: number, nodesReceived: number) => {
      const state = get();
      if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
        return;
      }

      try {
        const ackMessage = binaryProtocol.createBroadcastAck(sequenceId, nodesReceived);
        state.socket.send(ackMessage);

        if (debugState.isDataDebugEnabled() && sequenceId % 100 === 0) {
          logger.debug(`Sent BroadcastAck: seq=${sequenceId}, nodes=${nodesReceived}`);
        }
      } catch (error) {
        logger.error('Error sending position ACK:', createErrorMetadata(error));
      }
    };

    const detectBotsData = (data: ArrayBuffer): boolean => {
      try {
        const allNodes = parseBinaryNodeData(data);
        return allNodes.some(node => isAgentNode(node.nodeId));
      } catch (error) {
        logger.error('Error detecting bots data:', createErrorMetadata(error));
        return false;
      }
    };

    const validateMessage = (message: unknown): message is WebSocketMessage => {
      return (
        message !== null &&
        typeof message === 'object' &&
        typeof (message as WebSocketMessage).type === 'string' &&
        (message as WebSocketMessage).type.length > 0 &&
        (message as WebSocketMessage).type.length <= 100
      );
    };

    const validateBinaryData = (data: ArrayBuffer): boolean => {
      // P2 PERFORMANCE: Only do cheap size checks here.
      // Full parsing happens once in processBinaryData — no redundant decode.
      if (!data || data.byteLength === 0) {
        return false;
      }

      if (data.byteLength > 50 * 1024 * 1024) {
        logger.warn(`Binary data too large: ${data.byteLength} bytes`);
        return false;
      }

      // P1 BUG FIX: Previously returned true on parse failure, bypassing validation.
      // Now we only perform lightweight structural checks (size bounds).
      // The binary protocol header is validated inside processBinaryData.
      return true;
    };

    const handleErrorFrame = (error: WebSocketErrorFrame) => {
      logger.error('Received error frame from server:', error);

      emit('error-frame', error);

      switch (error.category) {
        case 'validation':
          if (error.affectedPaths && error.affectedPaths.length > 0) {
            emit('validation-error', {
              paths: error.affectedPaths,
              message: error.message
            });
          }
          break;

        case 'rate_limit':
          if (error.retryAfter) {
            logger.warn(`Rate limited. Retry after ${error.retryAfter}ms`);
            emit('rate-limit', {
              retryAfter: error.retryAfter,
              message: error.message
            });
          }
          break;

        case 'auth':
          emit('auth-error', {
            code: error.code,
            message: error.message
          });
          break;

        case 'server':
          if (error.retryable && error.retryAfter) {
            setTimeout(() => {
              processMessageQueue();
            }, error.retryAfter);
          }
          break;

        case 'protocol':
          logger.error('Protocol error - considering reconnection');
          if (error.code === 'PROTOCOL_VERSION_MISMATCH') {
            get().forceReconnect();
          }
          break;
      }
    };

    const handleGraphUpdate = async (data: ArrayBuffer, header: ReturnType<typeof binaryProtocol.parseHeader>) => {
      if (!header) return;

      const graphTypeFlag = header.graphTypeFlag as GraphTypeFlag;
      const currentMode = useSettingsStore.getState().get<'knowledge_graph' | 'ontology'>('visualisation.graphs.mode') || 'knowledge_graph';

      const shouldProcess =
        (currentMode === 'knowledge_graph' && graphTypeFlag === GraphTypeFlag.KNOWLEDGE_GRAPH) ||
        (currentMode === 'ontology' && graphTypeFlag === GraphTypeFlag.ONTOLOGY);

      if (!shouldProcess) {
        if (debugState.isDataDebugEnabled()) {
          logger.debug(`Skipping graph update - mode mismatch: current=${currentMode}, flag=${graphTypeFlag}`);
        }
        return;
      }

      const payload = binaryProtocol.extractPayload(data, header);

      emit('graph-update', {
        graphType: graphTypeFlag === GraphTypeFlag.ONTOLOGY ? 'ontology' : 'knowledge_graph',
        data: payload
      });

      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Processed graph update: mode=${currentMode}, size=${payload.byteLength}`);
      }
    };

    const handleVoiceData = async (data: ArrayBuffer, header: ReturnType<typeof binaryProtocol.parseHeader>) => {
      if (!header) return;

      const payload = binaryProtocol.extractPayload(data, header);

      emit('voice-data', payload);

      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Processed voice data: size=${payload.byteLength}`);
      }
    };

    const handlePositionUpdate = async (data: ArrayBuffer, header: ReturnType<typeof binaryProtocol.parseHeader>) => {
      if (!header) return;

      const payload = binaryProtocol.extractPayload(data, header);
      // Binary protocol: 28 bytes per node (matches Rust BinaryNodeData)
      // u32 node_id (4) + f32 x,y,z (12) + f32 vx,vy,vz (12) = 28 bytes
      const estimatedNodeCount = Math.floor(payload.byteLength / 28);

      // P2 PERFORMANCE FIX: Parse binary nodes once, reuse for both bot detection
      // and position updates instead of parsing multiple times.
      let parsedNodes: BinaryNodeData[] | null = null;
      try {
        parsedNodes = parseBinaryNodeData(payload);
      } catch (error) {
        logger.error('Error parsing binary node data:', createErrorMetadata(error));
        return;
      }

      const hasBotsData = parsedNodes.some(node => isAgentNode(node.nodeId));

      if (hasBotsData) {
        emit('bots-position-update', payload);
        if (debugState.isDataDebugEnabled()) {
          logger.debug('Emitted bots-position-update event');
        }
      }

      const graphType = graphDataManager.getGraphType();

      binaryMessageCount = (binaryMessageCount || 0) + 1;
      if (binaryMessageCount % 100 === 1) {
        logger.debug('Position update received', { graphType, dataSize: payload.byteLength, msgCount: binaryMessageCount });
      }

      if (graphType === 'logseq') {
        try {
          await graphDataManager.updateNodePositions(payload);
          if (binaryMessageCount % 100 === 1) {
            logger.debug('Node positions updated successfully');
          }
        } catch (error) {
          logger.error('[WebSocketStore] Error updating positions:', createErrorMetadata(error));
          logger.error('Error processing position data in graphDataManager:', createErrorMetadata(error));
        }
      } else if (binaryMessageCount % 100 === 1) {
        logger.debug('Skipping position update - graph type is', { graphType });
      }

      positionUpdateSequence++;
      if (positionUpdateSequence - lastAckSentSequence >= ACK_BATCH_SIZE) {
        sendPositionAck(positionUpdateSequence, estimatedNodeCount);
        lastAckSentSequence = positionUpdateSequence;
      }
    };

    const handleLegacyBinaryData = async (data: ArrayBuffer) => {
      const estimatedNodeCount = Math.floor(data.byteLength / 28);

      // P2 PERFORMANCE FIX: Parse once and reuse for bot detection.
      let parsedNodes: BinaryNodeData[] | null = null;
      try {
        parsedNodes = parseBinaryNodeData(data);
      } catch (error) {
        logger.error('Error parsing legacy binary data:', createErrorMetadata(error));
        return;
      }

      const hasBotsData = parsedNodes.some(node => isAgentNode(node.nodeId));

      if (hasBotsData) {
        emit('bots-position-update', data);
        if (debugState.isDataDebugEnabled()) {
          logger.debug('Emitted bots-position-update event (legacy)');
        }
      }

      const graphType = graphDataManager.getGraphType();
      binaryMessageCount = (binaryMessageCount || 0) + 1;

      if (binaryMessageCount % 100 === 1) {
        logger.debug('Legacy binary data received', { graphType, dataSize: data.byteLength, msgCount: binaryMessageCount });
      }

      if (graphType === 'logseq') {
        try {
          await graphDataManager.updateNodePositions(data);
          if (binaryMessageCount % 100 === 1) {
            logger.debug('Node positions updated successfully (legacy)');
          }
        } catch (error) {
          logger.error('Error processing legacy binary data:', createErrorMetadata(error));
        }
      }

      positionUpdateSequence++;
      if (positionUpdateSequence - lastAckSentSequence >= ACK_BATCH_SIZE) {
        sendPositionAck(positionUpdateSequence, estimatedNodeCount);
        lastAckSentSequence = positionUpdateSequence;
      }
    };

    const handleAgentAction = async (data: ArrayBuffer, header: ReturnType<typeof binaryProtocol.parseHeader>) => {
      if (!header) return;

      const payload = binaryProtocol.extractPayload(data, header);

      // Decode single action or batch
      const actions = payload.byteLength >= 15
        ? binaryProtocol.decodeAgentActions(payload)
        : [];

      if (actions.length > 0) {
        emit('agent-action', actions);

        if (debugState.isDataDebugEnabled()) {
          logger.debug(`Processed ${actions.length} agent action(s)`);
        }
      }
    };

    const processBinaryData = async (data: ArrayBuffer) => {
      try {
        if (debugState.isDataDebugEnabled()) {
          logger.debug(`Processing binary data: ${data.byteLength} bytes`);
        }

        const header = binaryProtocol.parseHeader(data);
        if (!header) {
          logger.error('Failed to parse binary message header');
          return;
        }

        switch (header.type) {
          case MessageType.GRAPH_UPDATE:
            await handleGraphUpdate(data, header);
            break;

          case MessageType.VOICE_DATA:
            await handleVoiceData(data, header);
            break;

          case MessageType.POSITION_UPDATE:
          case MessageType.AGENT_POSITIONS:
            await handlePositionUpdate(data, header);
            break;

          case MessageType.AGENT_ACTION:
            await handleAgentAction(data, header);
            break;

          default:
            await handleLegacyBinaryData(data);
            break;
        }

        binaryMessageHandlers.forEach(handler => {
          try {
            handler(data);
          } catch (error) {
            logger.error('Error in binary message handler:', createErrorMetadata(error));
          }
        });
      } catch (error) {
        logger.error('Error processing binary data:', createErrorMetadata(error));
      }
    };

    const setupFilterSubscription = () => {
      if (filterSubscriptionSet) return;
      filterSubscriptionSet = true;

      const filterPaths = [
        'nodeFilter.enabled',
        'nodeFilter.qualityThreshold',
        'nodeFilter.authorityThreshold',
        'nodeFilter.filterByQuality',
        'nodeFilter.filterByAuthority',
        'nodeFilter.filterMode',
      ] as const;

      filterPaths.forEach(path => {
        // Use type assertion to handle the settings store subscribe signature
        const store = useSettingsStore.getState();
        if (store.subscribe) {
          store.subscribe(path as Parameters<typeof store.subscribe>[0], () => {
            handleFilterChange();
          });
        }
      });

      useSettingsStore.subscribe((state) => {
        const nodeFilter = state.settings?.nodeFilter;
        const wsState = get();
        if (nodeFilter && wsState.isConnected) {
          // P2 PERFORMANCE FIX: Shallow field comparison instead of JSON.stringify
          // on the entire settings tree every state change.
          const prev = lastFilterSnapshot;
          if (
            !prev ||
            prev.enabled !== nodeFilter.enabled ||
            prev.qualityThreshold !== nodeFilter.qualityThreshold ||
            prev.authorityThreshold !== nodeFilter.authorityThreshold ||
            prev.filterByQuality !== nodeFilter.filterByQuality ||
            prev.filterByAuthority !== nodeFilter.filterByAuthority ||
            prev.filterMode !== nodeFilter.filterMode
          ) {
            lastFilterSnapshot = {
              enabled: nodeFilter.enabled,
              qualityThreshold: nodeFilter.qualityThreshold,
              authorityThreshold: nodeFilter.authorityThreshold,
              filterByQuality: nodeFilter.filterByQuality,
              filterByAuthority: nodeFilter.filterByAuthority,
              filterMode: nodeFilter.filterMode,
            };
            wsState.sendFilterUpdate(lastFilterSnapshot);
          }
        }
      });

      logger.info('Filter subscription set up - changes will sync to server');
    };

    const handleFilterChange = () => {
      const state = get();
      if (!state.isConnected) return;

      const nodeFilter = useSettingsStore.getState().settings?.nodeFilter;
      if (nodeFilter) {
        state.sendFilterUpdate({
          enabled: nodeFilter.enabled,
          qualityThreshold: nodeFilter.qualityThreshold,
          authorityThreshold: nodeFilter.authorityThreshold,
          filterByQuality: nodeFilter.filterByQuality,
          filterByAuthority: nodeFilter.filterByAuthority,
          filterMode: nodeFilter.filterMode,
        });
      }
    };

    const attemptReconnect = () => {
      if (reconnectTimeout) {
        window.clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
      }

      const state = get();
      if (state.reconnectAttempts < state.maxReconnectAttempts) {
        set(s => ({ reconnectAttempts: s.reconnectAttempts + 1 }));

        const baseDelay = 1000;
        const exponentialDelay = baseDelay * Math.pow(2, state.reconnectAttempts);
        const delay = Math.min(exponentialDelay, MAX_RECONNECT_DELAY);

        updateConnectionState('reconnecting', `Reconnecting in ${delay}ms`);

        if (debugState.isEnabled()) {
          logger.info(`Attempting to reconnect in ${delay}ms (attempt ${state.reconnectAttempts + 1}/${state.maxReconnectAttempts})`);
        }

        reconnectTimeout = window.setTimeout(() => {
          get().connect().catch(error => {
            logger.error('Reconnect attempt failed:', createErrorMetadata(error));
            attemptReconnect();
          });
        }, delay);
      } else {
        logger.error(`Maximum reconnect attempts (${state.maxReconnectAttempts}) reached. Giving up.`);
        updateConnectionState('failed', 'Maximum reconnect attempts reached');
      }
    };

    const handleSolidMessage = (msg: string) => {
      if (msg.startsWith('protocol ')) {
        const protocol = msg.slice(9);
        logger.debug('Solid WebSocket protocol handshake complete', { protocol });

        const state = get();
        for (const url of state.solidSubscriptions.keys()) {
          state.solidSocket?.send(`sub ${url}`);
          logger.debug('Resubscribed to Solid resource', { url });
        }

        emit('solid-protocol', { protocol });
      } else if (msg.startsWith('ack ')) {
        const url = msg.slice(4);
        logger.debug('Solid subscription acknowledged', { url });
        notifySolidSubscribers(url, { type: 'ack', url });
      } else if (msg.startsWith('pub ')) {
        const url = msg.slice(4);
        logger.debug('Solid resource changed', { url });
        notifySolidSubscribers(url, { type: 'pub', url });
        emit('solid-resource-changed', { url });
      } else if (msg.startsWith('error ')) {
        const errorMsg = msg.slice(6);
        logger.error('Solid WebSocket error message', { error: errorMsg });
        emit('solid-error', { message: errorMsg });
      }
    };

    const notifySolidSubscribers = (url: string, notification: SolidNotification) => {
      const state = get();
      const callbacks = state.solidSubscriptions.get(url);
      callbacks?.forEach((cb) => {
        try {
          cb(notification);
        } catch (error) {
          logger.error('Error in Solid notification callback', { url, error });
        }
      });

      const containerUrl = url.substring(0, url.lastIndexOf('/') + 1);
      if (containerUrl !== url) {
        const containerCallbacks = state.solidSubscriptions.get(containerUrl);
        containerCallbacks?.forEach((cb) => {
          try {
            cb(notification);
          } catch (error) {
            logger.error('Error in Solid container notification callback', { containerUrl, error });
          }
        });
      }
    };

    const attemptSolidReconnect = () => {
      if (solidReconnectTimeout) {
        window.clearTimeout(solidReconnectTimeout);
        solidReconnectTimeout = null;
      }

      if (solidReconnectAttempts >= SOLID_MAX_RECONNECT_ATTEMPTS) {
        logger.warn('Max Solid WebSocket reconnect attempts reached');
        return;
      }

      solidReconnectAttempts++;
      const delay = SOLID_RECONNECT_DELAY * Math.pow(2, solidReconnectAttempts - 1);

      logger.info(`Solid WebSocket reconnecting in ${delay}ms (attempt ${solidReconnectAttempts})`);

      solidReconnectTimeout = window.setTimeout(() => {
        get().connectSolid();
      }, delay);
    };

    // Initialize URL from settings subscription
    let previousCustomBackendUrl = useSettingsStore.getState().settings?.system?.customBackendUrl;
    useSettingsStore.subscribe((state) => {
      const newCustomBackendUrl = state.settings?.system?.customBackendUrl;
      if (newCustomBackendUrl !== previousCustomBackendUrl) {
        if (debugState.isEnabled()) {
          logger.info(`customBackendUrl setting changed from "${previousCustomBackendUrl}" to "${newCustomBackendUrl}", re-evaluating WebSocket URL.`);
        }
        previousCustomBackendUrl = newCustomBackendUrl;
        const wsState = get();
        set({ url: getUrlFromSettings() });
        if (wsState.isConnected || (wsState.socket && wsState.socket.readyState === WebSocket.CONNECTING)) {
          logger.info('Reconnecting WebSocket due to customBackendUrl change.');
          wsState.close();
          setTimeout(() => {
            get().connect().catch(error => {
              logger.error('Failed to reconnect WebSocket after URL change:', createErrorMetadata(error));
            });
          }, 100);
        }
      }
    });

    return {
      // Initial state
      socket: null,
      isConnected: false,
      isServerReady: false,
      connectionState: {
        status: 'disconnected',
        reconnectAttempts: 0
      },
      url: getUrlFromSettings(),
      solidSocket: null,
      isSolidConnected: false,
      solidSubscriptions: new Map(),
      messageQueue: [],
      statistics: {
        messagesReceived: 0,
        messagesSent: 0,
        bytesReceived: 0,
        bytesSent: 0,
        connectionTime: 0,
        reconnections: 0,
        averageLatency: 0,
        messagesByType: {},
        errors: 0,
        lastActivity: Date.now()
      },
      reconnectInterval: 1000,
      maxReconnectAttempts: 10,
      reconnectAttempts: 0,

      // Actions
      connect: async () => {
        const state = get();
        if (state.socket && (state.socket.readyState === WebSocket.CONNECTING || state.socket.readyState === WebSocket.OPEN)) {
          return;
        }

        try {
          if (debugState.isEnabled()) {
            logger.info(`Connecting to WebSocket at ${state.url}`);
          }

          const socket = new WebSocket(state.url);
          socket.binaryType = 'arraybuffer';

          socket.onopen = () => {
            // Guard against stale socket from a prior connect() cycle
            if (get().socket !== null && get().socket !== socket) return;

            set({ socket, isConnected: true, reconnectAttempts: 0 });
            updateConnectionState('connected', undefined, Date.now());

            if (debugState.isEnabled()) {
              logger.info('WebSocket connection established');
            }

            // P1 SECURITY: Send auth token as first message after connect,
            // NOT as a URL query parameter (which leaks in logs/proxies).
            const token = nostrAuth.getSessionToken();
            const user = nostrAuth.getCurrentUser();
            if (token && user) {
              socket.send(JSON.stringify({ type: 'AUTH', token, pubkey: user.pubkey }));
            }

            const currentFilter = useSettingsStore.getState().settings?.nodeFilter;
            if (currentFilter) {
              get().sendMessage('filter_update', {
                enabled: currentFilter.enabled,
                quality_threshold: currentFilter.qualityThreshold,
                authority_threshold: currentFilter.authorityThreshold,
                filter_by_quality: currentFilter.filterByQuality,
                filter_by_authority: currentFilter.filterByAuthority,
                filter_mode: currentFilter.filterMode,
              });
            }

            initializeBatchQueue();
            setupFilterSubscription();
            notifyConnectionStatusHandlers(true);
            startHeartbeat();
            processMessageQueue();
          };

          socket.onmessage = (event: MessageEvent) => {
            // P0 STALE CLOSURE FIX: Discard messages from a replaced socket.
            if (get().socket !== socket) return;

            if (event.data === 'pong') {
              handleHeartbeatResponse();
              return;
            }

            if (event.data instanceof Blob) {
              if (debugState.isDataDebugEnabled()) {
                logger.debug('Received binary blob data');
              }

              event.data.arrayBuffer().then(buffer => {
                if (validateBinaryData(buffer)) {
                  processBinaryData(buffer);
                } else {
                  logger.warn('Invalid binary data received, skipping processing');
                }
              }).catch(error => {
                logger.error('Error converting Blob to ArrayBuffer:', createErrorMetadata(error));
              });
              return;
            }

            if (event.data instanceof ArrayBuffer) {
              if (debugState.isDataDebugEnabled()) {
                logger.debug(`Received binary ArrayBuffer data: ${event.data.byteLength} bytes`);
              }
              if (validateBinaryData(event.data)) {
                processBinaryData(event.data);
              } else {
                logger.warn('Invalid binary data received, skipping processing');
              }
              return;
            }

            try {
              if (typeof event.data !== 'string' || event.data.trim() === '') {
                logger.warn('Received empty or invalid message data');
                return;
              }

              const message = JSON.parse(event.data) as WebSocketMessage;

              if (!validateMessage(message)) {
                logger.warn('Received malformed message, skipping processing');
                return;
              }

              if (debugState.isDataDebugEnabled()) {
                logger.debug(`Received WebSocket message: ${message.type}`, (message as unknown as Record<string, unknown>).data);
              }

              if (message.type === 'connection_established') {
                set({ isServerReady: true });
                if (debugState.isEnabled()) {
                  logger.info('Server connection established and ready');
                }
              }

              if (message.type === 'error' && (message as unknown as Record<string, unknown>).error) {
                handleErrorFrame((message as unknown as Record<string, unknown>).error as WebSocketErrorFrame);
                return;
              }

              if (message.type === 'filter_confirmed') {
                if (debugState.isEnabled()) {
                  logger.info(`Filter applied: ${message.data?.visible_nodes}/${message.data?.total_nodes} nodes visible`);
                }
                emit('filterApplied', {
                  visibleNodes: message.data?.visible_nodes,
                  totalNodes: message.data?.total_nodes
                });
              }

              if (message.type === 'initialGraphLoad') {
                const msgData = message as unknown as { nodes?: unknown[]; edges?: unknown[] };
                const nodes = msgData.nodes || [];
                const edges = msgData.edges || [];
                logger.info(`[WebSocket] Received initialGraphLoad with ${nodes.length} nodes, ${edges.length} edges - updating graph`);

                const transformedNodes = nodes.map((node: unknown) => {
                  const n = node as Record<string, unknown>;
                  return {
                    id: String(n.id),
                    label: String(n.label || n.name || n.id),
                    position: (n.position as { x: number; y: number; z: number }) || { x: Number(n.x) || 0, y: Number(n.y) || 0, z: Number(n.z) || 0 },
                    metadata: {
                      ...(n.metadata as Record<string, unknown>),
                      quality_score: n.quality_score ?? (n.metadata as Record<string, unknown>)?.quality_score,
                      authority_score: n.authority_score ?? (n.metadata as Record<string, unknown>)?.authority_score,
                    },
                    color: n.color as string | undefined,
                    size: n.size as number | undefined,
                  };
                });

                const transformedEdges = edges.map((edge: unknown) => {
                  const e = edge as Record<string, unknown>;
                  return {
                    id: String(e.id || `${e.source}-${e.target}`),
                    source: String(e.source),
                    target: String(e.target),
                    weight: e.weight as number | undefined,
                    label: e.label as string | undefined,
                  };
                });

                graphDataManager.setGraphData({
                  nodes: transformedNodes,
                  edges: transformedEdges,
                }).then(() => {
                  logger.info(`[WebSocket] Graph updated with ${transformedNodes.length} nodes from server filter`);
                  emit('graphDataUpdated', {
                    nodeCount: transformedNodes.length,
                    edgeCount: transformedEdges.length,
                    source: 'websocket_filter'
                  });
                }).catch(error => {
                  logger.error('[WebSocket] Failed to update graph data from initialGraphLoad:', createErrorMetadata(error));
                });
              }

              messageHandlers.forEach(handler => {
                try {
                  handler(message);
                } catch (error) {
                  logger.error('Error in message handler:', createErrorMetadata(error));
                }
              });
            } catch (error) {
              logger.error('Error parsing WebSocket message:', createErrorMetadata(error));
            }
          };

          socket.onclose = (event: CloseEvent) => {
            // P0 STALE CLOSURE FIX: If a newer socket replaced this one, ignore the event.
            if (get().socket !== null && get().socket !== socket) return;

            set({ isConnected: false, isServerReady: false });
            stopHeartbeat();

            if (debugState.isEnabled()) {
              logger.info(`WebSocket connection closed: ${event.code} ${event.reason}`);
            }

            notifyConnectionStatusHandlers(false);

            const isNormalClosure = event.code === 1000 || event.code === 1001;
            const wasCleanShutdown = event.wasClean;

            if (!isNormalClosure || !wasCleanShutdown) {
              updateConnectionState('reconnecting', event.reason);
              attemptReconnect();
            } else {
              updateConnectionState('disconnected');
            }
          };

          socket.onerror = (event: Event) => {
            // P0 STALE CLOSURE FIX: Ignore errors from a replaced socket.
            if (get().socket !== null && get().socket !== socket) return;

            const errorMessage = event instanceof ErrorEvent ? event.message : 'Unknown WebSocket error';
            logger.error('WebSocket error:', { event, message: errorMessage });
            updateConnectionState('failed', errorMessage);
          };

          // Store socket reference immediately so handlers can guard against staleness.
          // The definitive set happens inside onopen after the staleness check.
          set({ socket });

          return new Promise<void>((resolve, reject) => {
            socket.addEventListener('open', () => resolve(), { once: true });
            socket.addEventListener('error', () => {
              if (socket.readyState !== WebSocket.OPEN) {
                reject(new Error('WebSocket connection failed'));
              }
            }, { once: true });
          });
        } catch (error) {
          logger.error('Error establishing WebSocket connection:', createErrorMetadata(error));
          throw error;
        }
      },

      disconnect: () => {
        get().close();
        get().disconnectSolid();
      },

      close: () => {
        const state = get();
        if (state.socket) {
          if (reconnectTimeout) {
            window.clearTimeout(reconnectTimeout);
            reconnectTimeout = null;
          }

          stopHeartbeat();

          if (positionBatchQueue) {
            positionBatchQueue.destroy();
            positionBatchQueue = null;
          }

          try {
            state.socket.close(1000, 'Normal closure');
            if (debugState.isEnabled()) {
              logger.info('WebSocket connection closed by client');
            }
          } catch (error) {
            logger.error('Error closing WebSocket:', createErrorMetadata(error));
          } finally {
            set({
              socket: null,
              isConnected: false,
              isServerReady: false,
              reconnectAttempts: 0,
              messageQueue: []
            });
            updateConnectionState('disconnected');
            notifyConnectionStatusHandlers(false);
          }
        }
      },

      sendMessage: (type: string, data?: unknown) => {
        const state = get();
        const message = { type, data };
        const messageStr = JSON.stringify(message);

        if (!state.isConnected || !state.socket) {
          queueMessage('text', messageStr);
          logger.warn(`Message queued: ${type} (WebSocket not connected)`);
          return;
        }

        try {
          state.socket.send(messageStr);

          if (debugState.isDataDebugEnabled()) {
            logger.debug(`Sent message: ${type}`);
          }
        } catch (error) {
          logger.error('Error sending WebSocket message:', createErrorMetadata(error));
          queueMessage('text', messageStr);
        }
      },

      sendRawBinaryData: (data: ArrayBuffer) => {
        const state = get();
        if (!state.isConnected || !state.socket) {
          queueMessage('binary', data);
          logger.warn(`Binary data queued: ${data.byteLength} bytes (WebSocket not connected)`);
          return;
        }

        try {
          state.socket.send(data);

          if (debugState.isDataDebugEnabled()) {
            logger.debug(`Sent binary data: ${data.byteLength} bytes`);
          }
        } catch (error) {
          logger.error('Error sending binary data:', createErrorMetadata(error));
          queueMessage('binary', data);
        }
      },

      sendFilterUpdate: (filter: FilterUpdateParams) => {
        const state = get();
        if (!state.isConnected) {
          logger.warn('Cannot send filter update: WebSocket not connected');
          return;
        }

        state.sendMessage('filter_update', {
          enabled: filter.enabled,
          quality_threshold: filter.qualityThreshold,
          authority_threshold: filter.authorityThreshold,
          filter_by_quality: filter.filterByQuality,
          filter_by_authority: filter.filterByAuthority,
          filter_mode: filter.filterMode,
        });

        logger.info('Filter update sent to server', filter);
      },

      forceRefreshFilter: async () => {
        const state = get();
        if (!state.isConnected) {
          logger.warn('Cannot force refresh filter: WebSocket not connected');
          return;
        }

        const nodeFilter = useSettingsStore.getState().settings?.nodeFilter;
        if (nodeFilter) {
          lastFilterSnapshot = null;

          logger.info('[Refresh] Clearing local graph and requesting fresh filtered data', nodeFilter);

          await graphDataManager.setGraphData({ nodes: [], edges: [] });
          logger.info('[Refresh] Local graph cleared, awaiting server response...');

          state.sendFilterUpdate({
            enabled: nodeFilter.enabled,
            qualityThreshold: nodeFilter.qualityThreshold,
            authorityThreshold: nodeFilter.authorityThreshold,
            filterByQuality: nodeFilter.filterByQuality,
            filterByAuthority: nodeFilter.filterByAuthority,
            filterMode: nodeFilter.filterMode,
          });
        } else {
          logger.warn('No nodeFilter settings found in store');
        }
      },

      forceReconnect: () => {
        logger.info('Forcing WebSocket reconnection');
        const state = get();
        if (state.socket) {
          state.socket.close(4001, 'Forced reconnection');
        }
      },

      clearMessageQueue: () => {
        const queueSize = get().messageQueue.length;
        set({ messageQueue: [] });
        if (queueSize > 0) {
          logger.info(`Cleared ${queueSize} messages from queue`);
        }
      },

      setCustomBackendUrl: (backendUrl: string | null) => {
        if (!backendUrl) {
          set({ url: determineWebSocketUrl() });
          if (debugState.isEnabled()) {
            logger.info(`Reset to default WebSocket URL: ${get().url}`);
          }
          return;
        }

        const protocol = backendUrl.startsWith('https://') ? 'wss://' : 'ws://';
        const hostWithProtocol = backendUrl.replace(/^(https?:\/\/)?/, '');
        const newUrl = `${protocol}${hostWithProtocol}/wss`;

        set({ url: newUrl });

        if (debugState.isEnabled()) {
          logger.info(`Set custom WebSocket URL: ${newUrl}`);
        }

        const state = get();
        if (state.isConnected && state.socket) {
          if (debugState.isEnabled()) {
            logger.info('Reconnecting with new WebSocket URL');
          }
          state.close();
          state.connect().catch(error => {
            logger.error('Failed to reconnect with new URL:', createErrorMetadata(error));
          });
        }
      },

      sendNodePositionUpdates: (updates: NodePositionUpdate[]) => {
        if (!positionBatchQueue) {
          logger.warn('Position batch queue not initialized');
          return;
        }

        try {
          const binaryNodes: BinaryNodeData[] = updates.map(update => ({
            nodeId: update.nodeId,
            position: update.position,
            velocity: update.velocity || { x: 0, y: 0, z: 0 },
            ssspDistance: 0,
            ssspParent: -1
          }));

          const validation = validateNodePositions(binaryNodes, {
            maxNodes: updates.length + 100
          });

          if (!validation.valid) {
            logger.error('Position updates failed validation:', validation.errors);
            return;
          }

          binaryNodes.forEach(node => {
            const priority = isAgentNode(node.nodeId) ? 10 : 0;
            positionBatchQueue!.enqueuePositionUpdate(node, priority);
          });

          if (debugState.isDataDebugEnabled()) {
            logger.debug(`Queued ${updates.length} position updates for batching`);
          }
        } catch (error) {
          logger.error('Error queuing position updates:', createErrorMetadata(error));
        }
      },

      flushPositionUpdates: () => {
        if (positionBatchQueue) {
          return positionBatchQueue.flush();
        }
        return Promise.resolve();
      },

      getPositionQueueMetrics: () => {
        if (positionBatchQueue) {
          return positionBatchQueue.getMetrics();
        }
        return null;
      },

      onMessage: (handler: LegacyMessageHandler) => {
        messageHandlers.push(handler);
        return () => {
          messageHandlers = messageHandlers.filter(h => h !== handler);
        };
      },

      onBinaryMessage: (handler: BinaryMessageHandler) => {
        binaryMessageHandlers.push(handler);
        return () => {
          binaryMessageHandlers = binaryMessageHandlers.filter(h => h !== handler);
        };
      },

      onConnectionStatusChange: (handler: ConnectionStatusHandler) => {
        connectionStatusHandlers.push(handler);
        handler(get().isConnected);
        return () => {
          connectionStatusHandlers = connectionStatusHandlers.filter(h => h !== handler);
        };
      },

      onConnectionStateChange: (handler: ConnectionStateHandler) => {
        connectionStateHandlers.push(handler);
        handler(get().connectionState);
        return () => {
          connectionStateHandlers = connectionStateHandlers.filter(h => h !== handler);
        };
      },

      on: (eventName: string, handler: EventHandler) => {
        if (!eventHandlers.has(eventName)) {
          eventHandlers.set(eventName, []);
        }
        eventHandlers.get(eventName)!.push(handler);

        return () => {
          const handlers = eventHandlers.get(eventName);
          if (handlers) {
            const index = handlers.indexOf(handler);
            if (index > -1) {
              handlers.splice(index, 1);
            }
          }
        };
      },

      emit,

      connectSolid: () => {
        const wsUrl = getSolidWebSocketUrl();

        if (!wsUrl) {
          logger.warn('JSS WebSocket URL not configured (VITE_JSS_WS_URL)');
          return;
        }

        const state = get();
        if (state.solidSocket?.readyState === WebSocket.OPEN) {
          logger.debug('Solid WebSocket already connected');
          return;
        }

        try {
          logger.info(`Connecting to JSS WebSocket at ${wsUrl}`);
          const solidSocket = new WebSocket(wsUrl);

          solidSocket.onopen = () => {
            logger.info('JSS WebSocket connected');
            set({ isSolidConnected: true });
            solidReconnectAttempts = 0;
            emit('solid-connected', { url: wsUrl });
          };

          solidSocket.onmessage = (event) => {
            const msg = event.data.toString().trim();
            handleSolidMessage(msg);
          };

          solidSocket.onerror = (error) => {
            logger.error('JSS WebSocket error', { error });
            emit('solid-error', { error });
          };

          solidSocket.onclose = (event) => {
            logger.info('JSS WebSocket disconnected', { code: event.code, reason: event.reason });
            set({ isSolidConnected: false });
            emit('solid-disconnected', { code: event.code, reason: event.reason });
            attemptSolidReconnect();
          };

          set({ solidSocket });
        } catch (error) {
          logger.error('Failed to connect Solid WebSocket', { error });
        }
      },

      disconnectSolid: () => {
        if (solidReconnectTimeout) {
          window.clearTimeout(solidReconnectTimeout);
          solidReconnectTimeout = null;
        }

        const state = get();
        if (state.solidSocket) {
          try {
            state.solidSocket.close(1000, 'Normal closure');
            logger.info('Solid WebSocket disconnected by client');
          } catch (error) {
            logger.error('Error closing Solid WebSocket:', createErrorMetadata(error));
          }
        }

        set({
          solidSocket: null,
          isSolidConnected: false,
          solidSubscriptions: new Map()
        });
        solidReconnectAttempts = 0;
      },

      subscribeSolidResource: (resourceUrl: string, callback: SolidNotificationCallback) => {
        set(state => {
          const newSubscriptions = new Map(state.solidSubscriptions);
          if (!newSubscriptions.has(resourceUrl)) {
            newSubscriptions.set(resourceUrl, new Set());

            if (state.solidSocket?.readyState === WebSocket.OPEN) {
              state.solidSocket.send(`sub ${resourceUrl}`);
              logger.debug('Subscribed to Solid resource', { url: resourceUrl });
            }
          }

          newSubscriptions.get(resourceUrl)!.add(callback);
          return { solidSubscriptions: newSubscriptions };
        });

        return () => {
          set(state => {
            const newSubscriptions = new Map(state.solidSubscriptions);
            const callbacks = newSubscriptions.get(resourceUrl);
            if (callbacks) {
              callbacks.delete(callback);

              if (callbacks.size === 0) {
                if (state.solidSocket?.readyState === WebSocket.OPEN) {
                  state.solidSocket.send(`unsub ${resourceUrl}`);
                  logger.debug('Unsubscribed from Solid resource', { url: resourceUrl });
                }
                newSubscriptions.delete(resourceUrl);
              }
            }
            return { solidSubscriptions: newSubscriptions };
          });
        };
      },

      unsubscribeSolidResource: (resourceUrl: string) => {
        set(state => {
          const newSubscriptions = new Map(state.solidSubscriptions);
          if (newSubscriptions.has(resourceUrl)) {
            if (state.solidSocket?.readyState === WebSocket.OPEN) {
              state.solidSocket.send(`unsub ${resourceUrl}`);
              logger.debug('Unsubscribed from Solid resource (all callbacks)', { url: resourceUrl });
            }
            newSubscriptions.delete(resourceUrl);
          }
          return { solidSubscriptions: newSubscriptions };
        });
      },

      isSolidWebSocketConnected: () => {
        const state = get();
        return state.isSolidConnected && state.solidSocket?.readyState === WebSocket.OPEN;
      },

      getSolidSubscriptions: () => {
        return Array.from(get().solidSubscriptions.keys());
      },

      isReady: () => {
        const state = get();
        return state.isConnected && state.isServerReady;
      },

      getConnectionState: () => {
        return { ...get().connectionState };
      },

      getQueuedMessageCount: () => {
        return get().messageQueue.length;
      },

      // Testing/cleanup methods
      _reset: () => {
        const state = get();
        state.close();
        state.disconnectSolid();

        // Reset all internal state
        messageHandlers = [];
        binaryMessageHandlers = [];
        connectionStatusHandlers = [];
        connectionStateHandlers = [];
        eventHandlers = new Map();
        positionBatchQueue = null;
        heartbeatInterval = null;
        heartbeatTimeout = null;
        reconnectTimeout = null;
        binaryMessageCount = 0;
        positionUpdateSequence = 0;
        lastAckSentSequence = 0;
        filterSubscriptionSet = false;
        lastFilterSnapshot = null;
        solidReconnectTimeout = null;
        solidReconnectAttempts = 0;

        set({
          socket: null,
          isConnected: false,
          isServerReady: false,
          connectionState: {
            status: 'disconnected',
            reconnectAttempts: 0
          },
          url: getUrlFromSettings(),
          solidSocket: null,
          isSolidConnected: false,
          solidSubscriptions: new Map(),
          messageQueue: [],
          statistics: {
            messagesReceived: 0,
            messagesSent: 0,
            bytesReceived: 0,
            bytesSent: 0,
            connectionTime: 0,
            reconnections: 0,
            averageLatency: 0,
            messagesByType: {},
            errors: 0,
            lastActivity: Date.now()
          },
          reconnectAttempts: 0
        });
      },

      _getInternals: () => ({
        messageHandlers,
        binaryMessageHandlers,
        connectionStatusHandlers,
        connectionStateHandlers,
        eventHandlers,
        positionBatchQueue,
        heartbeatInterval,
        reconnectTimeout
      })
    };
  })
);

// Backward compatibility: create a service-like object that wraps the store
// This allows gradual migration without breaking existing code
class WebSocketServiceCompat {
  private static _instance: WebSocketServiceCompat | null = null;

  public static getInstance(): WebSocketServiceCompat {
    if (!WebSocketServiceCompat._instance) {
      WebSocketServiceCompat._instance = new WebSocketServiceCompat();
    }
    return WebSocketServiceCompat._instance;
  }

  // Reset for testing
  public static resetInstance(): void {
    if (WebSocketServiceCompat._instance) {
      useWebSocketStore.getState()._reset();
    }
    WebSocketServiceCompat._instance = null;
  }

  connect = () => useWebSocketStore.getState().connect();
  disconnect = () => useWebSocketStore.getState().disconnect();
  close = () => useWebSocketStore.getState().close();
  sendMessage = (type: string, data?: unknown) => useWebSocketStore.getState().sendMessage(type, data);
  sendRawBinaryData = (data: ArrayBuffer) => useWebSocketStore.getState().sendRawBinaryData(data);
  sendFilterUpdate = (filter: FilterUpdateParams) => useWebSocketStore.getState().sendFilterUpdate(filter);
  forceRefreshFilter = () => useWebSocketStore.getState().forceRefreshFilter();
  forceReconnect = () => useWebSocketStore.getState().forceReconnect();
  clearMessageQueue = () => useWebSocketStore.getState().clearMessageQueue();
  setCustomBackendUrl = (url: string | null) => useWebSocketStore.getState().setCustomBackendUrl(url);
  sendNodePositionUpdates = (updates: NodePositionUpdate[]) => useWebSocketStore.getState().sendNodePositionUpdates(updates);
  flushPositionUpdates = () => useWebSocketStore.getState().flushPositionUpdates();
  getPositionQueueMetrics = () => useWebSocketStore.getState().getPositionQueueMetrics();
  onMessage = (handler: LegacyMessageHandler) => useWebSocketStore.getState().onMessage(handler);
  onBinaryMessage = (handler: BinaryMessageHandler) => useWebSocketStore.getState().onBinaryMessage(handler);
  onConnectionStatusChange = (handler: ConnectionStatusHandler) => useWebSocketStore.getState().onConnectionStatusChange(handler);
  onConnectionStateChange = (handler: ConnectionStateHandler) => useWebSocketStore.getState().onConnectionStateChange(handler);
  on = (eventName: string, handler: EventHandler) => useWebSocketStore.getState().on(eventName, handler);
  emit = (eventName: string, data: unknown) => useWebSocketStore.getState().emit(eventName, data);
  connectSolid = () => useWebSocketStore.getState().connectSolid();
  disconnectSolid = () => useWebSocketStore.getState().disconnectSolid();
  subscribeSolidResource = (url: string, cb: SolidNotificationCallback) => useWebSocketStore.getState().subscribeSolidResource(url, cb);
  unsubscribeSolidResource = (url: string) => useWebSocketStore.getState().unsubscribeSolidResource(url);
  isSolidWebSocketConnected = () => useWebSocketStore.getState().isSolidWebSocketConnected();
  getSolidSubscriptions = () => useWebSocketStore.getState().getSolidSubscriptions();
  isReady = () => useWebSocketStore.getState().isReady();
  getConnectionState = () => useWebSocketStore.getState().getConnectionState();
  getQueuedMessageCount = () => useWebSocketStore.getState().getQueuedMessageCount();

  // Getter for isConnected property
  get isConnected(): boolean {
    return useWebSocketStore.getState().isConnected;
  }

  // Send error frame to server (backward compatibility)
  sendErrorFrame = (error: Partial<WebSocketErrorFrame>) => {
    const errorFrame: WebSocketErrorFrame = {
      code: error.code || 'CLIENT_ERROR',
      message: error.message || 'Unknown client error',
      category: error.category || 'protocol',
      retryable: error.retryable ?? false,
      timestamp: Date.now(),
      ...error
    };

    useWebSocketStore.getState().sendMessage('error', { error: errorFrame });
  };

  // WebSocketAdapter interface compatibility
  send = (data: ArrayBuffer) => useWebSocketStore.getState().sendRawBinaryData(data);
}

// Export singleton for backward compatibility
export const webSocketService = WebSocketServiceCompat.getInstance();

// Export class for testing
export { WebSocketServiceCompat };

// Utility hooks for common selections
export const useWebSocketConnection = () => useWebSocketStore(state => ({
  isConnected: state.isConnected,
  isServerReady: state.isServerReady,
  connectionState: state.connectionState
}));

export const useWebSocketActions = () => useWebSocketStore(state => ({
  connect: state.connect,
  disconnect: state.disconnect,
  sendMessage: state.sendMessage
}));

export default useWebSocketStore;
