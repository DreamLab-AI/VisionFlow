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

      
      this.socket = new WebSocket(this.url);

      
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
    
    
    this.initializeBatchQueue();
    
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
        logger.debug(`Received WebSocket message: ${message.type}`, message.data);
      }

      
      if (message.type === 'connection_established') {
        this.isServerReady = true;
        if (debugState.isEnabled()) {
          logger.info('Server connection established and ready');
        }
      }
      
      
      if (message.type === 'error' && message.error) {
        this.handleErrorFrame(message.error);
        return;
      }

      
      
      

      
      this.messageHandlers.forEach(handler => {
        try {
          handler(message);
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
  }

  private async handleLegacyBinaryData(data: ArrayBuffer): Promise<void> {
    
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
    const message: WebSocketMessage = { type, data };
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
        velocity: update.velocity || {x: 0, y: 0, z: 0}
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
    this.messageHandlers.push(handler);
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
}

// Create and export singleton instance
export const webSocketService = WebSocketService.getInstance();

export default WebSocketService;
