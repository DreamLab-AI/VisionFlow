import { createLogger } from '../utils/loggerConfig';
import { createErrorMetadata } from '../utils/loggerConfig';
import { debugState } from '../utils/clientDebugState';
import { useSettingsStore } from '../store/settingsStore';
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import { parseBinaryNodeData, isAgentNode, createBinaryNodeData, BinaryNodeData } from '../types/binaryProtocol';
import { NodePositionBatchQueue, createWebSocketBatchProcessor } from '../utils/BatchQueue';
import { validateNodePositions, createValidationMiddleware } from '../utils/validation';
import {
  WebSocketMessage,
  WebSocketEventHandlers,
  WebSocketConfig,
  WebSocketConnectionState,
  WebSocketError,
  WebSocketStatistics,
  Subscription,
  SubscriptionFilters,
  MessageHandler
} from '../types/websocketTypes';
import { binaryProtocol, MessageType, GraphTypeFlag } from './BinaryWebSocketProtocol';
import { nostrAuth } from './nostrAuthService';

const logger = createLogger('WebSocketService');

export interface WebSocketAdapter {
  send: (data: ArrayBuffer) => void;
  isReady: () => boolean;
}

// Legacy interface for backward compatibility
export interface LegacyWebSocketMessage {
  type: string;
  data?: any;
  error?: WebSocketErrorFrame;
}

export interface WebSocketErrorFrame {
  code: string;
  message: string;
  category: 'validation' | 'server' | 'protocol' | 'auth' | 'rate_limit';
  details?: any;
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

// Legacy interface - replaced by WebSocketConnectionState from websocketTypes
export interface ConnectionState {
  status: 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'failed';
  lastConnected?: number;
  lastError?: string;
  reconnectAttempts: number;
}

// Solid notification types (solid-0.1 protocol)
export interface SolidNotification {
  type: 'pub' | 'ack';
  url: string;
}

export type SolidNotificationCallback = (notification: SolidNotification) => void;

// Legacy types for backward compatibility
type LegacyMessageHandler = (message: LegacyWebSocketMessage) => void;
type BinaryMessageHandler = (data: ArrayBuffer) => void;
type ConnectionStatusHandler = (connected: boolean) => void;
type ConnectionStateHandler = (state: ConnectionState) => void;
type EventHandler = (data: any) => void;

class WebSocketService {
  private static instance: WebSocketService;
  private socket: WebSocket | null = null;
  private messageHandlers: LegacyMessageHandler[] = [];
  private binaryMessageHandlers: BinaryMessageHandler[] = [];
  private connectionStatusHandlers: ConnectionStatusHandler[] = [];
  private eventHandlers: Map<string, EventHandler[]> = new Map();


  private subscriptions: Map<string, Subscription> = new Map();
  private subscriptionCounter: number = 0;
  private statistics: WebSocketStatistics;
  private config: WebSocketConfig;
  private reconnectInterval: number = 1000;
  private maxReconnectAttempts: number = 10;
  private reconnectAttempts: number = 0;
  private maxReconnectDelay: number = 30000;
  private reconnectTimeout: number | null = null;
  private isConnected: boolean = false;
  private isServerReady: boolean = false;
  private url: string;
  private messageQueue: QueuedMessage[] = [];
  private maxQueueSize: number = 100;
  private heartbeatInterval: number | null = null;
  private heartbeatTimeout: number | null = null;
  private heartbeatIntervalMs: number = 30000;
  private heartbeatTimeoutMs: number = 10000;
  private connectionState: ConnectionState = {
    status: 'disconnected',
    reconnectAttempts: 0
  };
  private connectionStateHandlers: ConnectionStateHandler[] = [];
  private positionBatchQueue: NodePositionBatchQueue | null = null;
  private binaryMessageCount: number = 0;

  // Backpressure flow control - client-side sequence tracking
  // Used to send ACKs back to server after processing position updates
  private positionUpdateSequence: number = 0;
  private lastAckSentSequence: number = 0;
  private ackBatchSize: number = 10; // Send ACK every N position updates

  // JSS/Solid WebSocket for notifications (solid-0.1 protocol)
  private solidSocket: WebSocket | null = null;
  private solidSubscriptions: Map<string, Set<SolidNotificationCallback>> = new Map();
  private solidReconnectAttempts: number = 0;
  private solidMaxReconnectAttempts: number = 5;
  private solidReconnectDelay: number = 1000;
  private solidReconnectTimeout: number | null = null;
  private isSolidConnected: boolean = false;


  private enhancedConnectionState: WebSocketConnectionState = {
    status: 'disconnected',
    reconnectAttempts: 0,
    serverFeatures: []
  };

  private constructor() {

    this.statistics = {
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
    };


    this.config = {
      reconnect: {
        maxAttempts: 10,
        baseDelay: 1000,
        maxDelay: 30000,
        backoffFactor: 2
      },
      heartbeat: {
        interval: 30000,
        timeout: 10000
      },
      compression: true,
      binaryProtocol: true
    };


    this.url = this.determineWebSocketUrl();


    this.updateFromSettings();


    let previousCustomBackendUrl = useSettingsStore.getState().settings?.system?.customBackendUrl;
    useSettingsStore.subscribe((state) => {
      const newCustomBackendUrl = state.settings?.system?.customBackendUrl;
      if (newCustomBackendUrl !== previousCustomBackendUrl) {
        if (debugState.isEnabled()) {
          logger.info(`customBackendUrl setting changed from "${previousCustomBackendUrl}" to "${newCustomBackendUrl}", re-evaluating WebSocket URL.`);
        }
        previousCustomBackendUrl = newCustomBackendUrl;
        this.updateFromSettings();
        if (this.isConnected || (this.socket && this.socket.readyState === WebSocket.CONNECTING)) {
          logger.info('Reconnecting WebSocket due to customBackendUrl change.');
          this.close();
          setTimeout(() => {
            this.connect().catch(error => {
              logger.error('Failed to reconnect WebSocket after URL change:', createErrorMetadata(error));
            });
          }, 100);
        }
      }
    });
  }

  private updateFromSettings(): void {
    const state = useSettingsStore.getState();
    const settings = state.settings;
    let newUrl = this.determineWebSocketUrl();

    if (settings?.system?.websocket) {
      this.reconnectInterval = settings?.system?.websocket?.reconnectDelay || 2000;
      this.maxReconnectAttempts = settings.system.websocket.reconnectAttempts || 10;
    }


    if (settings?.system?.customBackendUrl &&
        settings.system.customBackendUrl.trim() !== '') {
      const customUrl = settings.system.customBackendUrl.trim();
      const protocol = customUrl.startsWith('https://') ? 'wss://' : 'ws://';
      const hostAndPath = customUrl.replace(/^(https?:\/\/)?/, '');
      newUrl = `${protocol}${hostAndPath.replace(/\/$/, '')}/wss`;
      if (debugState.isEnabled()) {
        logger.info(`Using custom backend WebSocket URL: ${newUrl}`);
      }
    } else {
      if (debugState.isEnabled()) {
        logger.info(`Using default WebSocket URL: ${newUrl}`);
      }
    }
    this.url = newUrl;
  }

  public static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService();
    }
    return WebSocketService.instance;
  }

  /**
   * Reset singleton instance for test isolation.
   * Disconnects any active connection and clears all handlers.
   * @deprecated Prefer using websocketStore from '../store/websocketStore' for new code
   */
  public static resetInstance(): void {
    if (WebSocketService.instance) {
      WebSocketService.instance.disconnect();
      WebSocketService.instance.messageHandlers = [];
      WebSocketService.instance.binaryMessageHandlers = [];
      WebSocketService.instance.connectionStatusHandlers = [];
      WebSocketService.instance.connectionStateHandlers = [];
      WebSocketService.instance.eventHandlers.clear();
      WebSocketService.instance.solidSubscriptions.clear();
      WebSocketService.instance.messageQueue = [];
    }
    WebSocketService.instance = undefined as unknown as WebSocketService;
  }

  private determineWebSocketUrl(): string {
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


  public setCustomBackendUrl(backendUrl: string | null): void {
    if (!backendUrl) {

      this.url = this.determineWebSocketUrl();
      if (debugState.isEnabled()) {
        logger.info(`Reset to default WebSocket URL: ${this.url}`);
      }
      return;
    }


    const protocol = backendUrl.startsWith('https://') ? 'wss://' : 'ws://';

    const hostWithProtocol = backendUrl.replace(/^(https?:\/\/)?/, '');

    this.url = `${protocol}${hostWithProtocol}/wss`;

    if (debugState.isEnabled()) {
      logger.info(`Set custom WebSocket URL: ${this.url}`);
    }


    if (this.isConnected && this.socket) {
      if (debugState.isEnabled()) {
        logger.info('Reconnecting with new WebSocket URL');
      }
      this.close();
      this.connect().catch(error => {
        logger.error('Failed to reconnect with new URL:', createErrorMetadata(error));
      });
    }
  }

  public async connect(): Promise<void> {

    if (this.socket && (this.socket.readyState === WebSocket.CONNECTING || this.socket.readyState === WebSocket.OPEN)) {
      return;
    }

    try {
      if (debugState.isEnabled()) {
        logger.info(`Connecting to WebSocket at ${this.url}`);
      }

      // Get auth token if available
      const token = nostrAuth.getSessionToken();
      const wsUrl = token ? `${this.url}?token=${token}` : this.url;

      this.socket = new WebSocket(wsUrl);


      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
      this.socket.onerror = this.handleError.bind(this);


      return new Promise<void>((resolve, reject) => {
        if (!this.socket) {
          reject(new Error('Socket initialization failed'));
          return;
        }


        this.socket.addEventListener('open', () => resolve(), { once: true });


        this.socket.addEventListener('error', (event) => {

          if (this.socket && this.socket.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket connection failed'));
          }
        }, { once: true });
      });
    } catch (error) {
      logger.error('Error establishing WebSocket connection:', createErrorMetadata(error));
      throw error;
    }
  }

  private handleOpen(event: Event): void {
    this.isConnected = true;
    this.reconnectAttempts = 0;
    this.updateConnectionState('connected', undefined, new Date().getTime());

    if (debugState.isEnabled()) {
      logger.info('WebSocket connection established');
    }

    // Send authentication message if token is available
    const token = nostrAuth.getSessionToken();
    const user = nostrAuth.getCurrentUser();
    if (token && user) {
      this.sendMessage('authenticate', {
        token,
        pubkey: user.pubkey
      });
    }

    // Send initial filter settings
    const currentFilter = useSettingsStore.getState().settings?.nodeFilter;
    if (currentFilter) {
      this.sendMessage('filter_update', {
        enabled: currentFilter.enabled,
        quality_threshold: currentFilter.qualityThreshold,
        authority_threshold: currentFilter.authorityThreshold,
        filter_by_quality: currentFilter.filterByQuality,
        filter_by_authority: currentFilter.filterByAuthority,
        filter_mode: currentFilter.filterMode,
      });
      if (debugState.isEnabled()) {
        logger.info('Initial filter settings sent to server');
      }
    }

    this.initializeBatchQueue();

    // Set up filter subscription to sync UI changes to server
    this.setupFilterSubscription();

    this.notifyConnectionStatusHandlers(true);
    this.startHeartbeat();
    this.processMessageQueue();
  }

  private handleMessage(event: MessageEvent): void {

    if (event.data === 'pong') {
      this.handleHeartbeatResponse();
      return;
    }


    if (event.data instanceof Blob) {
      if (debugState.isDataDebugEnabled()) {
        logger.debug('Received binary blob data');
      }

      event.data.arrayBuffer().then(buffer => {
        if (this.validateBinaryData(buffer)) {
          this.processBinaryData(buffer);
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
      if (this.validateBinaryData(event.data)) {
        this.processBinaryData(event.data);
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


      if (!this.validateMessage(message)) {
        logger.warn('Received malformed message, skipping processing');
        return;
      }

      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Received WebSocket message: ${message.type}`, (message as any).data);
      }


      if (message.type === 'connection_established') {
        this.isServerReady = true;
        if (debugState.isEnabled()) {
          logger.info('Server connection established and ready');
        }
      }


      if (message.type === 'error' && (message as any).error) {
        this.handleErrorFrame((message as any).error as WebSocketErrorFrame);
        return;
      }

      // Handle filter confirmation from server
      if (message.type === 'filter_confirmed') {
        if (debugState.isEnabled()) {
          logger.info(`Filter applied: ${message.data?.visible_nodes}/${message.data?.total_nodes} nodes visible`);
        }
        this.emit('filterApplied', {
          visibleNodes: message.data?.visible_nodes,
          totalNodes: message.data?.total_nodes
        });
      }

      // Handle initialGraphLoad - this is sent when server provides new graph data (e.g., after filtering)
      if (message.type === 'initialGraphLoad') {
        const nodes = message.nodes || [];
        const edges = message.edges || [];
        logger.info(`[WebSocket] Received initialGraphLoad with ${nodes.length} nodes, ${edges.length} edges - updating graph`);

        // Transform server node format to client format
        const transformedNodes = nodes.map((node: any) => ({
          id: String(node.id),
          label: node.label || node.name || String(node.id),
          position: node.position || { x: node.x || 0, y: node.y || 0, z: node.z || 0 },
          metadata: {
            ...node.metadata,
            quality_score: node.quality_score ?? node.metadata?.quality_score,
            authority_score: node.authority_score ?? node.metadata?.authority_score,
          },
          color: node.color,
          size: node.size,
        }));

        const transformedEdges = edges.map((edge: any) => ({
          id: edge.id || `${edge.source}-${edge.target}`,
          source: String(edge.source),
          target: String(edge.target),
          weight: edge.weight,
          label: edge.label,
        }));

        // Update the graph data manager with the new filtered data
        graphDataManager.setGraphData({
          nodes: transformedNodes,
          edges: transformedEdges,
        }).then(() => {
          logger.info(`[WebSocket] Graph updated with ${transformedNodes.length} nodes from server filter`);
          this.emit('graphDataUpdated', {
            nodeCount: transformedNodes.length,
            edgeCount: transformedEdges.length,
            source: 'websocket_filter'
          });
        }).catch(error => {
          logger.error('[WebSocket] Failed to update graph data from initialGraphLoad:', createErrorMetadata(error));
        });
      }

      this.messageHandlers.forEach(handler => {
        try {
          handler(message as any);
        } catch (error) {
          logger.error('Error in message handler:', createErrorMetadata(error));
        }
      });
    } catch (error) {
      logger.error('Error parsing WebSocket message:', createErrorMetadata(error));
    }
  }


  private async processBinaryData(data: ArrayBuffer): Promise<void> {
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
          await this.handleGraphUpdate(data, header);
          break;

        case MessageType.VOICE_DATA:
          await this.handleVoiceData(data, header);
          break;

        case MessageType.POSITION_UPDATE:
        case MessageType.AGENT_POSITIONS:
          await this.handlePositionUpdate(data, header);
          break;

        default:

          await this.handleLegacyBinaryData(data);
          break;
      }


      this.binaryMessageHandlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          logger.error('Error in binary message handler:', createErrorMetadata(error));
        }
      });
    } catch (error) {
      logger.error('Error processing binary data:', createErrorMetadata(error));
    }
  }

  private async handleGraphUpdate(data: ArrayBuffer, header: any): Promise<void> {
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


    this.emit('graph-update', {
      graphType: graphTypeFlag === GraphTypeFlag.ONTOLOGY ? 'ontology' : 'knowledge_graph',
      data: payload
    });

    if (debugState.isDataDebugEnabled()) {
      logger.debug(`Processed graph update: mode=${currentMode}, size=${payload.byteLength}`);
    }
  }

  private async handleVoiceData(data: ArrayBuffer, header: any): Promise<void> {
    const payload = binaryProtocol.extractPayload(data, header);


    this.emit('voice-data', payload);

    if (debugState.isDataDebugEnabled()) {
      logger.debug(`Processed voice data: size=${payload.byteLength}`);
    }
  }

  private async handlePositionUpdate(data: ArrayBuffer, header: any): Promise<void> {
    const payload = binaryProtocol.extractPayload(data, header);

    // Estimate node count from payload (48 bytes per node in Protocol V3)
    const estimatedNodeCount = Math.floor(payload.byteLength / 48);

    const hasBotsData = this.detectBotsData(payload);

    if (hasBotsData) {

      this.emit('bots-position-update', payload);
      if (debugState.isDataDebugEnabled()) {
        logger.debug('Emitted bots-position-update event');
      }
    }


    const graphType = graphDataManager.getGraphType();


    this.binaryMessageCount = (this.binaryMessageCount || 0) + 1;
    if (this.binaryMessageCount % 100 === 1) {
      logger.debug('Position update received', { graphType, dataSize: payload.byteLength, msgCount: this.binaryMessageCount });
    }

    if (graphType === 'logseq') {
      try {
        await graphDataManager.updateNodePositions(payload);
        if (this.binaryMessageCount % 100 === 1) {
          logger.debug('Node positions updated successfully');
        }
      } catch (error) {
        console.error('[WebSocketService] Error updating positions:', error);
        logger.error('Error processing position data in graphDataManager:', createErrorMetadata(error));
      }
    } else if (this.binaryMessageCount % 100 === 1) {
      logger.debug('Skipping position update - graph type is', { graphType });
      if (debugState.isDataDebugEnabled()) {
        logger.debug('Skipping position data processing - not a Logseq graph');
      }
    }

    // Send backpressure ACK to server after processing position update
    // Batched to reduce ACK traffic (every ackBatchSize updates)
    this.positionUpdateSequence++;
    if (this.positionUpdateSequence - this.lastAckSentSequence >= this.ackBatchSize) {
      this.sendPositionAck(this.positionUpdateSequence, estimatedNodeCount);
      this.lastAckSentSequence = this.positionUpdateSequence;
    }
  }

  /**
   * Send backpressure acknowledgement to server after processing position updates
   * This enables true end-to-end flow control vs queue-only confirmation
   */
  private sendPositionAck(sequenceId: number, nodesReceived: number): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      const ackMessage = binaryProtocol.createBroadcastAck(sequenceId, nodesReceived);
      this.socket.send(ackMessage);

      if (debugState.isDataDebugEnabled() && sequenceId % 100 === 0) {
        logger.debug(`Sent BroadcastAck: seq=${sequenceId}, nodes=${nodesReceived}`);
      }
    } catch (error) {
      logger.error('Error sending position ACK:', createErrorMetadata(error));
    }
  }

  private async handleLegacyBinaryData(data: ArrayBuffer): Promise<void> {
    // Estimate node count from data (28 bytes per node in legacy format)
    const estimatedNodeCount = Math.floor(data.byteLength / 28);

    const hasBotsData = this.detectBotsData(data);

    if (hasBotsData) {
      this.emit('bots-position-update', data);
      if (debugState.isDataDebugEnabled()) {
        logger.debug('Emitted bots-position-update event (legacy)');
      }
    }

    const graphType = graphDataManager.getGraphType();
    this.binaryMessageCount = (this.binaryMessageCount || 0) + 1;

    if (this.binaryMessageCount % 100 === 1) {
      logger.debug('Legacy binary data received', { graphType, dataSize: data.byteLength, msgCount: this.binaryMessageCount });
    }

    if (graphType === 'logseq') {
      try {
        await graphDataManager.updateNodePositions(data);
        if (this.binaryMessageCount % 100 === 1) {
          logger.debug('Node positions updated successfully (legacy)');
        }
      } catch (error) {
        logger.error('Error processing legacy binary data:', createErrorMetadata(error));
      }
    }

    // Send backpressure ACK to server after processing (same as handlePositionUpdate)
    this.positionUpdateSequence++;
    if (this.positionUpdateSequence - this.lastAckSentSequence >= this.ackBatchSize) {
      this.sendPositionAck(this.positionUpdateSequence, estimatedNodeCount);
      this.lastAckSentSequence = this.positionUpdateSequence;
    }
  }

  private detectBotsData(data: ArrayBuffer): boolean {
    try {

      const allNodes = parseBinaryNodeData(data);
      return allNodes.some(node => isAgentNode(node.nodeId));
    } catch (error) {
      logger.error('Error detecting bots data:', createErrorMetadata(error));
      return false;
    }
  }

  private handleClose(event: CloseEvent): void {
    this.isConnected = false;
    this.isServerReady = false;
    this.stopHeartbeat();

    if (debugState.isEnabled()) {
      logger.info(`WebSocket connection closed: ${event.code} ${event.reason}`);
    }

    this.notifyConnectionStatusHandlers(false);


    const isNormalClosure = event.code === 1000 || event.code === 1001;
    const wasCleanShutdown = event.wasClean;

    if (!isNormalClosure || !wasCleanShutdown) {
      this.updateConnectionState('reconnecting', event.reason);
      this.attemptReconnect();
    } else {
      this.updateConnectionState('disconnected');
    }
  }

  private handleError(event: Event): void {
    const errorMessage = event instanceof ErrorEvent ? event.message : 'Unknown WebSocket error';
    logger.error('WebSocket error:', { event, message: errorMessage });
    this.updateConnectionState('failed', errorMessage);

  }

  private attemptReconnect(): void {

    if (this.reconnectTimeout) {
      window.clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;


      const baseDelay = 1000;
      const exponentialDelay = baseDelay * Math.pow(2, this.reconnectAttempts - 1);
      const delay = Math.min(exponentialDelay, this.maxReconnectDelay);

      this.updateConnectionState('reconnecting', `Reconnecting in ${delay}ms`);

      if (debugState.isEnabled()) {
        logger.info(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      }

      this.reconnectTimeout = window.setTimeout(() => {
        this.connect().catch(error => {
          logger.error('Reconnect attempt failed:', createErrorMetadata(error));

          this.attemptReconnect();
        });
      }, delay);
    } else {
      logger.error(`Maximum reconnect attempts (${this.maxReconnectAttempts}) reached. Giving up.`);
      this.updateConnectionState('failed', 'Maximum reconnect attempts reached');
    }
  }

  public sendMessage(type: string, data?: any): void {
    const message = { type, data } as any;
    const messageStr = JSON.stringify(message);

    if (!this.isConnected || !this.socket) {

      this.queueMessage('text', messageStr);
      logger.warn(`Message queued: ${type} (WebSocket not connected)`);
      return;
    }

    try {
      this.socket.send(messageStr);

      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Sent message: ${type}`);
      }
    } catch (error) {
      logger.error('Error sending WebSocket message:', createErrorMetadata(error));

      this.queueMessage('text', messageStr);
    }
  }

  public sendRawBinaryData(data: ArrayBuffer): void {
    if (!this.isConnected || !this.socket) {

      this.queueMessage('binary', data);
      logger.warn(`Binary data queued: ${data.byteLength} bytes (WebSocket not connected)`);
      return;
    }

    try {
      this.socket.send(data);

      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Sent binary data: ${data.byteLength} bytes`);
      }
    } catch (error) {
      logger.error('Error sending binary data:', createErrorMetadata(error));

      this.queueMessage('binary', data);
    }
  }

  /**
   * Send filter settings update to server
   * Called when user changes filter settings in the UI
   */
  public sendFilterUpdate(filter: {
    enabled?: boolean;
    qualityThreshold?: number;
    authorityThreshold?: number;
    filterByQuality?: boolean;
    filterByAuthority?: boolean;
    filterMode?: string;
  }): void {
    if (!this.isConnected) {
      logger.warn('Cannot send filter update: WebSocket not connected');
      return;
    }

    this.sendMessage('filter_update', {
      enabled: filter.enabled,
      quality_threshold: filter.qualityThreshold,
      authority_threshold: filter.authorityThreshold,
      filter_by_quality: filter.filterByQuality,
      filter_by_authority: filter.filterByAuthority,
      filter_mode: filter.filterMode,
    });

    logger.info('Filter update sent to server', filter);
  }

  /**
   * Subscribe to settings store filter changes and sync to server
   */
  private filterSubscriptionSet = false;
  private lastFilterState: any = null;

  public setupFilterSubscription(): void {
    if (this.filterSubscriptionSet) return;
    this.filterSubscriptionSet = true;

    // Use the store's custom subscribe method for specific nodeFilter paths
    const filterPaths = [
      'nodeFilter.enabled',
      'nodeFilter.qualityThreshold',
      'nodeFilter.authorityThreshold',
      'nodeFilter.filterByQuality',
      'nodeFilter.filterByAuthority',
      'nodeFilter.filterMode',
    ] as const;

    // Subscribe to each filter path
    filterPaths.forEach(path => {
      useSettingsStore.getState().subscribe(path as any, () => {
        this.handleFilterChange();
      });
    });

    // Also use zustand's basic subscribe as a fallback
    useSettingsStore.subscribe((state) => {
      const nodeFilter = state.settings?.nodeFilter;
      if (nodeFilter && this.isConnected) {
        const current = JSON.stringify(nodeFilter);
        if (current !== this.lastFilterState) {
          this.lastFilterState = current;
          this.sendFilterUpdate({
            enabled: nodeFilter.enabled,
            qualityThreshold: nodeFilter.qualityThreshold,
            authorityThreshold: nodeFilter.authorityThreshold,
            filterByQuality: nodeFilter.filterByQuality,
            filterByAuthority: nodeFilter.filterByAuthority,
            filterMode: nodeFilter.filterMode,
          });
        }
      }
    });

    logger.info('Filter subscription set up - changes will sync to server');
  }

  private handleFilterChange(): void {
    if (!this.isConnected) return;

    const nodeFilter = useSettingsStore.getState().settings?.nodeFilter;
    if (nodeFilter) {
      this.sendFilterUpdate({
        enabled: nodeFilter.enabled,
        qualityThreshold: nodeFilter.qualityThreshold,
        authorityThreshold: nodeFilter.authorityThreshold,
        filterByQuality: nodeFilter.filterByQuality,
        filterByAuthority: nodeFilter.filterByAuthority,
        filterMode: nodeFilter.filterMode,
      });
    }
  }

  /**
   * Force refresh filter - clears local graph, then requests fresh filtered data from server
   * Called by the "Refresh Graph" button to force a complete graph reload with current filter
   */
  public async forceRefreshFilter(): Promise<void> {
    if (!this.isConnected) {
      logger.warn('Cannot force refresh filter: WebSocket not connected');
      return;
    }

    const nodeFilter = useSettingsStore.getState().settings?.nodeFilter;
    if (nodeFilter) {
      // Reset last filter state to force sending even if values haven't changed
      this.lastFilterState = null;

      logger.info('[Refresh] Clearing local graph and requesting fresh filtered data', nodeFilter);

      // Step 1: Clear the local graph completely - don't try to fill gaps
      await graphDataManager.setGraphData({ nodes: [], edges: [] });
      logger.info('[Refresh] Local graph cleared, awaiting server response...');

      // Step 2: Send filter update to server - server will respond with filtered initialGraphLoad
      this.sendFilterUpdate({
        enabled: nodeFilter.enabled,
        qualityThreshold: nodeFilter.qualityThreshold,
        authorityThreshold: nodeFilter.authorityThreshold,
        filterByQuality: nodeFilter.filterByQuality,
        filterByAuthority: nodeFilter.filterByAuthority,
        filterMode: nodeFilter.filterMode,
      });

      // The server will respond with initialGraphLoad containing the filtered, metadata-rich sparse dataset
      // The initialGraphLoad handler will populate the graph - no gap filling
    } else {
      logger.warn('No nodeFilter settings found in store');
    }
  }


  private initializeBatchQueue(): void {
    if (this.positionBatchQueue) {
      this.positionBatchQueue.destroy();
    }


    const validationMiddleware = createValidationMiddleware({
      maxNodes: 10000,
      maxCoordinate: 10000,
      minCoordinate: -10000,
      maxVelocity: 1000
    });


    const batchProcessor = createWebSocketBatchProcessor((data: ArrayBuffer) => {
      if (!this.isConnected || !this.socket) {
        logger.warn('Cannot send batch: WebSocket not connected');
        return;
      }

      try {
        this.socket.send(data);

        if (debugState.isDataDebugEnabled()) {
          logger.debug(`Sent binary batch: ${data.byteLength} bytes`);
        }
      } catch (error) {
        logger.error('Error sending batch:', createErrorMetadata(error));
        throw error;
      }
    });


    this.positionBatchQueue = new NodePositionBatchQueue({
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
  }


  public sendNodePositionUpdates(updates: Array<{nodeId: number, position: {x: number, y: number, z: number}, velocity?: {x: number, y: number, z: number}}>): void {
    if (!this.positionBatchQueue) {
      logger.warn('Position batch queue not initialized');
      return;
    }

    try {

      const binaryNodes: BinaryNodeData[] = updates.map(update => ({
        nodeId: update.nodeId,
        position: update.position,
        velocity: update.velocity || {x: 0, y: 0, z: 0},
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
        this.positionBatchQueue!.enqueuePositionUpdate(node, priority);
      });

      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Queued ${updates.length} position updates for batching`);
      }
    } catch (error) {
      logger.error('Error queuing position updates:', createErrorMetadata(error));
    }
  }


  public flushPositionUpdates(): Promise<void> {
    if (this.positionBatchQueue) {
      return this.positionBatchQueue.flush();
    }
    return Promise.resolve();
  }


  public getPositionQueueMetrics() {
    if (this.positionBatchQueue) {
      return this.positionBatchQueue.getMetrics();
    }
    return null;
  }

  public onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.push(handler as any);
    return () => {
      this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
    };
  }

  public onBinaryMessage(handler: BinaryMessageHandler): () => void {
    this.binaryMessageHandlers.push(handler);
    return () => {
      this.binaryMessageHandlers = this.binaryMessageHandlers.filter(h => h !== handler);
    };
  }

  public onConnectionStatusChange(handler: ConnectionStatusHandler): () => void {
    this.connectionStatusHandlers.push(handler);

    handler(this.isConnected);
    return () => {
      this.connectionStatusHandlers = this.connectionStatusHandlers.filter(h => h !== handler);
    };
  }

  public onConnectionStateChange(handler: ConnectionStateHandler): () => void {
    this.connectionStateHandlers.push(handler);

    handler(this.connectionState);
    return () => {
      this.connectionStateHandlers = this.connectionStateHandlers.filter(h => h !== handler);
    };
  }

  private notifyConnectionStatusHandlers(connected: boolean): void {
    this.connectionStatusHandlers.forEach(handler => {
      try {
        handler(connected);
      } catch (error) {
        logger.error('Error in connection status handler:', createErrorMetadata(error));
      }
    });
  }

  private notifyConnectionStateHandlers(): void {
    this.connectionStateHandlers.forEach(handler => {
      try {
        handler(this.connectionState);
      } catch (error) {
        logger.error('Error in connection state handler:', createErrorMetadata(error));
      }
    });
  }

  private updateConnectionState(
    status: ConnectionState['status'],
    lastError?: string,
    lastConnected?: number
  ): void {
    this.connectionState = {
      ...this.connectionState,
      status,
      lastError,
      lastConnected,
      reconnectAttempts: this.reconnectAttempts
    };
    this.notifyConnectionStateHandlers();
  }

  public isReady(): boolean {
    return this.isConnected && this.isServerReady;
  }

  public getConnectionState(): ConnectionState {
    return { ...this.connectionState };
  }

  public getQueuedMessageCount(): number {
    return this.messageQueue.length;
  }

  public emit(eventName: string, data: any): void {
    const handlers = this.eventHandlers.get(eventName);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          logger.error(`Error in event handler for ${eventName}:`, createErrorMetadata(error));
        }
      });
    }
  }

  public on(eventName: string, handler: EventHandler): () => void {
    if (!this.eventHandlers.has(eventName)) {
      this.eventHandlers.set(eventName, []);
    }
    this.eventHandlers.get(eventName)!.push(handler);

    return () => {
      const handlers = this.eventHandlers.get(eventName);
      if (handlers) {
        const index = handlers.indexOf(handler);
        if (index > -1) {
          handlers.splice(index, 1);
        }
      }
    };
  }

  public close(): void {
    if (this.socket) {

      if (this.reconnectTimeout) {
        window.clearTimeout(this.reconnectTimeout);
        this.reconnectTimeout = null;
      }

      this.stopHeartbeat();


      if (this.positionBatchQueue) {
        this.positionBatchQueue.destroy();
        this.positionBatchQueue = null;
      }

      try {

        this.socket.close(1000, 'Normal closure');
        if (debugState.isEnabled()) {
          logger.info('WebSocket connection closed by client');
        }
      } catch (error) {
        logger.error('Error closing WebSocket:', createErrorMetadata(error));
      } finally {
        this.socket = null;
        this.isConnected = false;
        this.isServerReady = false;
        this.reconnectAttempts = 0;
        this.messageQueue = [];
        this.updateConnectionState('disconnected');
        this.notifyConnectionStatusHandlers(false);
      }
    }
  }


  public disconnect(): void {
    this.close();
    this.disconnectSolid();
  }


  private validateMessage(message: any): message is WebSocketMessage {
    return (
      message &&
      typeof message === 'object' &&
      typeof message.type === 'string' &&
      message.type.length > 0 &&
      message.type.length <= 100
    );
  }

  private validateBinaryData(data: ArrayBuffer): boolean {
    try {

      if (!data || data.byteLength === 0) {
        return false;
      }


      if (data.byteLength > 50 * 1024 * 1024) {
        logger.warn(`Binary data too large: ${data.byteLength} bytes`);
        return false;
      }


      try {
        parseBinaryNodeData(data);
        return true;
      } catch (error) {

        logger.warn('Binary data parsing validation failed, but allowing through:', createErrorMetadata(error));
        return true;
      }
    } catch (error) {
      logger.error('Error validating binary data:', createErrorMetadata(error));
      return false;
    }
  }


  private queueMessage(type: 'text' | 'binary', data: string | ArrayBuffer): void {

    if (this.messageQueue.length >= this.maxQueueSize) {

      const removed = this.messageQueue.shift();
      logger.warn('Message queue full, removed oldest message');
    }

    const queuedMessage: QueuedMessage = {
      type,
      data,
      timestamp: Date.now(),
      retries: 0
    };

    this.messageQueue.push(queuedMessage);
  }

  private async processMessageQueue(): Promise<void> {
    if (!this.isConnected || !this.socket || this.messageQueue.length === 0) {
      return;
    }

    const messagesToProcess = [...this.messageQueue];
    this.messageQueue = [];

    for (const queuedMessage of messagesToProcess) {
      try {
        if (queuedMessage.type === 'text') {
          this.socket.send(queuedMessage.data as string);
        } else {
          this.socket.send(queuedMessage.data as ArrayBuffer);
        }

        if (debugState.isDataDebugEnabled()) {
          logger.debug(`Processed queued ${queuedMessage.type} message`);
        }
      } catch (error) {

        queuedMessage.retries++;
        if (queuedMessage.retries < 3) {
          this.messageQueue.push(queuedMessage);
          logger.warn(`Failed to send queued message, retry ${queuedMessage.retries}/3`);
        } else {
          logger.error('Failed to send queued message after 3 retries, dropping:', createErrorMetadata(error));
        }
      }
    }
  }


  private startHeartbeat(): void {
    this.stopHeartbeat();

    this.heartbeatInterval = window.setInterval(() => {
      this.sendHeartbeat();
    }, this.heartbeatIntervalMs);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      window.clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    if (this.heartbeatTimeout) {
      window.clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = null;
    }
  }

  private sendHeartbeat(): void {
    if (!this.isConnected || !this.socket) {
      return;
    }

    try {
      this.socket.send('ping');


      this.heartbeatTimeout = window.setTimeout(() => {
        logger.warn('Heartbeat timeout - server not responding');
        this.handleHeartbeatTimeout();
      }, this.heartbeatTimeoutMs);

      if (debugState.isDataDebugEnabled()) {
        logger.debug('Sent heartbeat ping');
      }
    } catch (error) {
      logger.error('Error sending heartbeat:', createErrorMetadata(error));
      this.handleHeartbeatTimeout();
    }
  }

  private handleHeartbeatResponse(): void {
    if (this.heartbeatTimeout) {
      window.clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = null;
    }

    if (debugState.isDataDebugEnabled()) {
      logger.debug('Received heartbeat pong');
    }
  }

  private handleHeartbeatTimeout(): void {
    logger.warn('Heartbeat timeout detected, connection may be dead');


    if (this.socket) {
      this.socket.close(4000, 'Heartbeat timeout');
    }
  }


  public forceReconnect(): void {
    logger.info('Forcing WebSocket reconnection');
    if (this.socket) {
      this.socket.close(4001, 'Forced reconnection');
    }

  }


  public clearMessageQueue(): void {
    const queueSize = this.messageQueue.length;
    this.messageQueue = [];
    if (queueSize > 0) {
      logger.info(`Cleared ${queueSize} messages from queue`);
    }
  }


  private handleErrorFrame(error: WebSocketErrorFrame): void {
    logger.error('Received error frame from server:', error);


    this.emit('error-frame', error);


    switch (error.category) {
      case 'validation':

        if (error.affectedPaths && error.affectedPaths.length > 0) {
          this.emit('validation-error', {
            paths: error.affectedPaths,
            message: error.message
          });
        }
        break;

      case 'rate_limit':

        if (error.retryAfter) {
          logger.warn(`Rate limited. Retry after ${error.retryAfter}ms`);
          this.emit('rate-limit', {
            retryAfter: error.retryAfter,
            message: error.message
          });
        }
        break;

      case 'auth':

        this.emit('auth-error', {
          code: error.code,
          message: error.message
        });
        break;

      case 'server':

        if (error.retryable && error.retryAfter) {
          setTimeout(() => {
            this.processMessageQueue();
          }, error.retryAfter);
        }
        break;

      case 'protocol':

        logger.error('Protocol error - considering reconnection');
        if (error.code === 'PROTOCOL_VERSION_MISMATCH') {
          this.forceReconnect();
        }
        break;
    }
  }


  public sendErrorFrame(error: Partial<WebSocketErrorFrame>): void {
    const errorFrame: WebSocketErrorFrame = {
      code: error.code || 'CLIENT_ERROR',
      message: error.message || 'Unknown client error',
      category: error.category || 'protocol',
      retryable: error.retryable ?? false,
      timestamp: Date.now(),
      ...error
    };

    this.sendMessage('error', { error: errorFrame });
  }

  // ============================================
  // JSS/Solid WebSocket Notifications (solid-0.1)
  // ============================================

  /**
   * Get the JSS WebSocket URL from environment
   */
  private getSolidWebSocketUrl(): string | null {
    return import.meta.env.VITE_JSS_WS_URL || null;
  }

  /**
   * Connect to JSS WebSocket for real-time Solid notifications
   * Uses solid-0.1 protocol for resource change notifications
   */
  public connectSolid(): void {
    const wsUrl = this.getSolidWebSocketUrl();

    if (!wsUrl) {
      logger.warn('JSS WebSocket URL not configured (VITE_JSS_WS_URL)');
      return;
    }

    if (this.solidSocket?.readyState === WebSocket.OPEN) {
      logger.debug('Solid WebSocket already connected');
      return;
    }

    try {
      logger.info(`Connecting to JSS WebSocket at ${wsUrl}`);
      this.solidSocket = new WebSocket(wsUrl);

      this.solidSocket.onopen = () => {
        logger.info('JSS WebSocket connected');
        this.isSolidConnected = true;
        this.solidReconnectAttempts = 0;

        // Emit connection event
        this.emit('solid-connected', { url: wsUrl });
      };

      this.solidSocket.onmessage = (event) => {
        const msg = event.data.toString().trim();
        this.handleSolidMessage(msg);
      };

      this.solidSocket.onerror = (error) => {
        logger.error('JSS WebSocket error', { error });
        this.emit('solid-error', { error });
      };

      this.solidSocket.onclose = (event) => {
        logger.info('JSS WebSocket disconnected', { code: event.code, reason: event.reason });
        this.isSolidConnected = false;
        this.emit('solid-disconnected', { code: event.code, reason: event.reason });
        this.attemptSolidReconnect();
      };
    } catch (error) {
      logger.error('Failed to connect Solid WebSocket', { error });
    }
  }

  /**
   * Handle incoming solid-0.1 protocol messages
   */
  private handleSolidMessage(msg: string): void {
    if (msg.startsWith('protocol ')) {
      // Protocol handshake complete (e.g., "protocol solid-0.1")
      const protocol = msg.slice(9);
      logger.debug('Solid WebSocket protocol handshake complete', { protocol });

      // Resubscribe to all tracked resources
      for (const url of this.solidSubscriptions.keys()) {
        this.solidSocket?.send(`sub ${url}`);
        logger.debug('Resubscribed to Solid resource', { url });
      }

      this.emit('solid-protocol', { protocol });
    } else if (msg.startsWith('ack ')) {
      // Subscription acknowledged
      const url = msg.slice(4);
      logger.debug('Solid subscription acknowledged', { url });
      this.notifySolidSubscribers(url, { type: 'ack', url });
    } else if (msg.startsWith('pub ')) {
      // Resource changed notification
      const url = msg.slice(4);
      logger.debug('Solid resource changed', { url });
      this.notifySolidSubscribers(url, { type: 'pub', url });

      // Also emit as general event for components to listen
      this.emit('solid-resource-changed', { url });
    } else if (msg.startsWith('error ')) {
      // Error from server
      const errorMsg = msg.slice(6);
      logger.error('Solid WebSocket error message', { error: errorMsg });
      this.emit('solid-error', { message: errorMsg });
    }
  }

  /**
   * Notify all subscribers for a given resource URL
   */
  private notifySolidSubscribers(url: string, notification: SolidNotification): void {
    // Notify exact URL subscribers
    const callbacks = this.solidSubscriptions.get(url);
    callbacks?.forEach((cb) => {
      try {
        cb(notification);
      } catch (error) {
        logger.error('Error in Solid notification callback', { url, error });
      }
    });

    // Also notify container subscribers (parent directory)
    const containerUrl = url.substring(0, url.lastIndexOf('/') + 1);
    if (containerUrl !== url) {
      const containerCallbacks = this.solidSubscriptions.get(containerUrl);
      containerCallbacks?.forEach((cb) => {
        try {
          cb(notification);
        } catch (error) {
          logger.error('Error in Solid container notification callback', { containerUrl, error });
        }
      });
    }
  }

  /**
   * Attempt to reconnect Solid WebSocket with exponential backoff
   */
  private attemptSolidReconnect(): void {
    if (this.solidReconnectTimeout) {
      window.clearTimeout(this.solidReconnectTimeout);
      this.solidReconnectTimeout = null;
    }

    if (this.solidReconnectAttempts >= this.solidMaxReconnectAttempts) {
      logger.warn('Max Solid WebSocket reconnect attempts reached');
      return;
    }

    this.solidReconnectAttempts++;
    const delay = this.solidReconnectDelay * Math.pow(2, this.solidReconnectAttempts - 1);

    logger.info(`Solid WebSocket reconnecting in ${delay}ms (attempt ${this.solidReconnectAttempts})`);

    this.solidReconnectTimeout = window.setTimeout(() => {
      this.connectSolid();
    }, delay);
  }

  /**
   * Subscribe to notifications for a Solid resource
   * @param resourceUrl The URL of the resource to subscribe to
   * @param callback Callback function called when resource changes
   * @returns Unsubscribe function
   */
  public subscribeSolidResource(resourceUrl: string, callback: SolidNotificationCallback): () => void {
    if (!this.solidSubscriptions.has(resourceUrl)) {
      this.solidSubscriptions.set(resourceUrl, new Set());

      // Send subscription if already connected
      if (this.solidSocket?.readyState === WebSocket.OPEN) {
        this.solidSocket.send(`sub ${resourceUrl}`);
        logger.debug('Subscribed to Solid resource', { url: resourceUrl });
      }
    }

    this.solidSubscriptions.get(resourceUrl)!.add(callback);

    // Return unsubscribe function
    return () => {
      this.unsubscribeSolidResourceCallback(resourceUrl, callback);
    };
  }

  /**
   * Unsubscribe a specific callback from a Solid resource
   */
  private unsubscribeSolidResourceCallback(resourceUrl: string, callback: SolidNotificationCallback): void {
    const callbacks = this.solidSubscriptions.get(resourceUrl);
    if (callbacks) {
      callbacks.delete(callback);

      // If no more callbacks for this URL, unsubscribe from server
      if (callbacks.size === 0) {
        if (this.solidSocket?.readyState === WebSocket.OPEN) {
          this.solidSocket.send(`unsub ${resourceUrl}`);
          logger.debug('Unsubscribed from Solid resource', { url: resourceUrl });
        }
        this.solidSubscriptions.delete(resourceUrl);
      }
    }
  }

  /**
   * Unsubscribe from all callbacks for a Solid resource
   */
  public unsubscribeSolidResource(resourceUrl: string): void {
    if (this.solidSubscriptions.has(resourceUrl)) {
      if (this.solidSocket?.readyState === WebSocket.OPEN) {
        this.solidSocket.send(`unsub ${resourceUrl}`);
        logger.debug('Unsubscribed from Solid resource (all callbacks)', { url: resourceUrl });
      }
      this.solidSubscriptions.delete(resourceUrl);
    }
  }

  /**
   * Disconnect Solid WebSocket and clear subscriptions
   */
  public disconnectSolid(): void {
    if (this.solidReconnectTimeout) {
      window.clearTimeout(this.solidReconnectTimeout);
      this.solidReconnectTimeout = null;
    }

    if (this.solidSocket) {
      try {
        this.solidSocket.close(1000, 'Normal closure');
        logger.info('Solid WebSocket disconnected by client');
      } catch (error) {
        logger.error('Error closing Solid WebSocket:', createErrorMetadata(error));
      }
      this.solidSocket = null;
    }

    this.solidSubscriptions.clear();
    this.isSolidConnected = false;
    this.solidReconnectAttempts = 0;
  }

  /**
   * Check if Solid WebSocket is connected
   */
  public isSolidWebSocketConnected(): boolean {
    return this.isSolidConnected && this.solidSocket?.readyState === WebSocket.OPEN;
  }

  /**
   * Get list of currently subscribed Solid resources
   */
  public getSolidSubscriptions(): string[] {
    return Array.from(this.solidSubscriptions.keys());
  }
}

// Create and export singleton instance
export const webSocketService = WebSocketService.getInstance();

export default WebSocketService;
