import { createLogger, createErrorMetadata } from '../utils/logger';
import { debugState } from '../utils/clientDebugState';
import { useSettingsStore } from '../store/settingsStore'; // Keep alias here for now, fix later if needed
import { graphDataManager } from '../features/graph/managers/graphDataManager';
import { parseBinaryNodeData, isAgentNode, createBinaryNodeData } from '../types/binaryProtocol';

const logger = createLogger('WebSocketService');

export interface WebSocketAdapter {
  send: (data: ArrayBuffer) => void;
  isReady: () => boolean;
}

export interface WebSocketMessage {
  type: string;
  data?: any;
}

type MessageHandler = (message: WebSocketMessage) => void;
type BinaryMessageHandler = (data: ArrayBuffer) => void;
type ConnectionStatusHandler = (connected: boolean) => void;
type EventHandler = (data: any) => void;

class WebSocketService {
  private static instance: WebSocketService;
  private socket: WebSocket | null = null;
  private messageHandlers: MessageHandler[] = [];
  private binaryMessageHandlers: BinaryMessageHandler[] = [];
  private connectionStatusHandlers: ConnectionStatusHandler[] = [];
  private eventHandlers: Map<string, EventHandler[]> = new Map();
  private reconnectInterval: number = 2000;
  private maxReconnectAttempts: number = 10;
  private reconnectAttempts: number = 0;
  private reconnectTimeout: number | null = null;
  private isConnected: boolean = false;
  private isServerReady: boolean = false;
  private url: string;

  private constructor() {
    // Default WebSocket URL
    this.url = this.determineWebSocketUrl();

    // Update URL when settings change
    this.updateFromSettings();

    // Subscribe to store changes and manually check customBackendUrl
    let previousCustomBackendUrl = useSettingsStore.getState().settings?.system?.customBackendUrl;
    useSettingsStore.subscribe((state) => {
      const newCustomBackendUrl = state.settings?.system?.customBackendUrl;
      if (newCustomBackendUrl !== previousCustomBackendUrl) {
        if (debugState.isEnabled()) {
          logger.info(`customBackendUrl setting changed from "${previousCustomBackendUrl}" to "${newCustomBackendUrl}", re-evaluating WebSocket URL.`);
        }
        previousCustomBackendUrl = newCustomBackendUrl; // Update for next comparison
        this.updateFromSettings(); // Sets this.url based on the latest state
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
    let newUrl = this.determineWebSocketUrl(); // Default to relative path

    if (settings?.system?.websocket) {
      this.reconnectInterval = settings.system.websocket.reconnectDelay || 2000;
      this.maxReconnectAttempts = settings.system.websocket.reconnectAttempts || 10;
    }

    // Only use custom backend URL if it's not the problematic hardcoded IP
    if (settings?.system?.customBackendUrl && 
        settings.system.customBackendUrl.trim() !== '' &&
        !settings.system.customBackendUrl.includes('192.168.0.51')) {
      const customUrl = settings.system.customBackendUrl.trim();
      const protocol = customUrl.startsWith('https://') ? 'wss://' : 'ws://';
      const hostAndPath = customUrl.replace(/^(https?:\/\/)?/, '');
      newUrl = `${protocol}${hostAndPath.replace(/\/$/, '')}/wss`; // Ensure /wss and handle trailing slash
      if (debugState.isEnabled()) {
        logger.info(`Using custom backend WebSocket URL: ${newUrl}`);
      }
    } else {
      if (debugState.isEnabled()) {
        if (settings?.system?.customBackendUrl?.includes('192.168.0.51')) {
          logger.warn('Ignoring problematic hardcoded IP address 192.168.0.51, using default URL instead');
        }
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
    // Check if we're in development mode (Vite dev server)
    if (import.meta.env.DEV) {
      // In development, connect through the Vite dev server proxy
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname;
      const port = window.location.port || '3001'; // Default to 3001 if no port
      const url = `${protocol}//${host}:${port}/wss`;
      if (debugState.isEnabled()) {
        logger.info(`Determined WebSocket URL (dev): ${url}`);
      }
      return url;
    } else {
      // In production, use relative path for proper routing
      const url = '/wss';
      if (debugState.isEnabled()) {
        logger.info(`Determined WebSocket URL (production): ${url}`);
      }
      return url;
    }
  }

  /**
   * Set a custom backend URL for WebSocket connections
   * @param backendUrl The backend URL (e.g., 'http://192.168.0.51:8000' or just '192.168.0.51:8000')
   */
  public setCustomBackendUrl(backendUrl: string | null): void {
    if (!backendUrl) {
      // Reset to default URL
      this.url = this.determineWebSocketUrl();
      if (debugState.isEnabled()) {
        logger.info(`Reset to default WebSocket URL: ${this.url}`);
      }
      return;
    }

    // Determine protocol (ws or wss)
    const protocol = backendUrl.startsWith('https://') ? 'wss://' : 'ws://';
    // Extract host and port
    const hostWithProtocol = backendUrl.replace(/^(https?:\/\/)?/, '');
    // Set the WebSocket URL
    this.url = `${protocol}${hostWithProtocol}/wss`; // Main backend WebSocket

    if (debugState.isEnabled()) {
      logger.info(`Set custom WebSocket URL: ${this.url}`);
    }

    // If already connected, reconnect with new URL
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
    // Don't try to connect if already connecting or connected
    if (this.socket && (this.socket.readyState === WebSocket.CONNECTING || this.socket.readyState === WebSocket.OPEN)) {
      return;
    }

    try {
      if (debugState.isEnabled()) {
        logger.info(`Connecting to WebSocket at ${this.url}`);
      }

      // Create a new WebSocket connection
      this.socket = new WebSocket(this.url);

      // Handle WebSocket events
      this.socket.onopen = this.handleOpen.bind(this);
      this.socket.onmessage = this.handleMessage.bind(this);
      this.socket.onclose = this.handleClose.bind(this);
      this.socket.onerror = this.handleError.bind(this);

      // Create a promise that resolves when the connection opens or rejects on error
      return new Promise<void>((resolve, reject) => {
        if (!this.socket) {
          reject(new Error('Socket initialization failed'));
          return;
        }

        // Resolve when the socket successfully opens
        this.socket.addEventListener('open', () => resolve(), { once: true });

        // Reject if there's an error before the socket opens
        this.socket.addEventListener('error', (event) => {
          // Only reject if the socket hasn't opened yet
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
    if (debugState.isEnabled()) {
      logger.info('WebSocket connection established');
    }
    this.notifyConnectionStatusHandlers(true);
  }

  private handleMessage(event: MessageEvent): void {
    // Check for binary data first
    if (event.data instanceof Blob) {
      if (debugState.isDataDebugEnabled()) {
        logger.debug('Received binary blob data');
      }
      // Convert Blob to ArrayBuffer
      event.data.arrayBuffer().then(buffer => {
        // Process the ArrayBuffer, with possible decompression
        this.processBinaryData(buffer);
      }).catch(error => {
        logger.error('Error converting Blob to ArrayBuffer:', createErrorMetadata(error));
      });
      return;
    }

    if (event.data instanceof ArrayBuffer) {
      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Received binary ArrayBuffer data: ${event.data.byteLength} bytes`);
      }
      // Process the ArrayBuffer directly, with possible decompression
      this.processBinaryData(event.data);
      return;
    }

    // If not binary, try to parse as JSON
    try {
      const message = JSON.parse(event.data) as WebSocketMessage;

      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Received WebSocket message: ${message.type}`, message.data);
      }

      // Special handling for connection_established message
      if (message.type === 'connection_established') {
        this.isServerReady = true;
        if (debugState.isEnabled()) {
          logger.info('Server connection established and ready');
        }
      }

      // BREADCRUMB: Physics settings are updated via REST API, not WebSocket
      // Position and velocity data comes through WebSocket binary protocol
      // Settings changes should be handled by the settings store after REST updates

      // Notify all message handlers
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

  // Make the function async to handle graphDataManager processing
  private async processBinaryData(data: ArrayBuffer): Promise<void> {
    try {
      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Processing binary data: ${data.byteLength} bytes`);
      }

      // Check if binary data contains bots nodes (agent flag 0x80)
      const hasBotsData = this.detectBotsData(data);
      
      if (hasBotsData) {
        // Emit bots-position-update event for bots visualization
        this.emit('bots-position-update', data);
        if (debugState.isDataDebugEnabled()) {
          logger.debug('Emitted bots-position-update event');
        }
      }

      // Only process binary data for Logseq graphs (check graph type)
      if (graphDataManager.getGraphType() === 'logseq') {
        try {
          await graphDataManager.updateNodePositions(data);
        } catch (error) {
          logger.error('Error processing binary data in graphDataManager:', createErrorMetadata(error));
        }
      } else {
        if (debugState.isDataDebugEnabled()) {
          logger.debug('Skipping binary data processing - not a Logseq graph');
        }
      }

      // Notify binary message handlers
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

  private detectBotsData(data: ArrayBuffer): boolean {
    try {
      // Use the standardized binary protocol detection
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

    if (debugState.isEnabled()) {
      logger.info(`WebSocket connection closed: ${event.code} ${event.reason}`);
    }

    this.notifyConnectionStatusHandlers(false);

    // Attempt to reconnect if it wasn't a normal closure
    if (event.code !== 1000 && event.code !== 1001) {
      this.attemptReconnect();
    }
  }

  private handleError(event: Event): void {
    logger.error('WebSocket error:', { event });
    // The close handler will be called after this, which will handle reconnection
  }

  private attemptReconnect(): void {
    // Clear any existing reconnect timeout
    if (this.reconnectTimeout) {
      window.clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectInterval * Math.pow(1.5, this.reconnectAttempts - 1);

      if (debugState.isEnabled()) {
        logger.info(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      }

      this.reconnectTimeout = window.setTimeout(() => {
        this.connect().catch(error => {
          logger.error('Reconnect attempt failed:', createErrorMetadata(error));
        });
      }, delay);
    } else { // Added missing else block
      logger.error(`Maximum reconnect attempts (${this.maxReconnectAttempts}) reached. Giving up.`);
    }
  }

  public sendMessage(type: string, data?: any): void {
    if (!this.isConnected || !this.socket) {
      logger.warn('Cannot send message: WebSocket not connected');
      return;
    }

    try {
      const message: WebSocketMessage = { type, data };
      this.socket.send(JSON.stringify(message));

      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Sent message: ${type}`);
      }
    } catch (error) {
      logger.error('Error sending WebSocket message:', createErrorMetadata(error));
    }
  }

  public sendRawBinaryData(data: ArrayBuffer): void {
    if (!this.isConnected || !this.socket) {
      logger.warn('Cannot send binary data: WebSocket not connected');
      return;
    }

    try {
      this.socket.send(data);

      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Sent binary data: ${data.byteLength} bytes`);
      }
    } catch (error) {
      logger.error('Error sending binary data:', createErrorMetadata(error));
    }
  }

  /**
   * Send position updates for nodes that have been moved by user interaction
   * @param updates Array of node position updates
   */
  public sendNodePositionUpdates(updates: Array<{nodeId: number, position: {x: number, y: number, z: number}, velocity?: {x: number, y: number, z: number}}>): void {
    if (!this.isConnected || !this.socket) {
      logger.warn('Cannot send position updates: WebSocket not connected');
      return;
    }

    try {
      // Convert updates to binary format
      const binaryNodes = updates.map(update => ({
        nodeId: update.nodeId,
        position: update.position,
        velocity: update.velocity || {x: 0, y: 0, z: 0}
      }));
      
      const binaryData = createBinaryNodeData(binaryNodes);
      
      // Send as binary message
      this.socket.send(binaryData);
      
      if (debugState.isDataDebugEnabled()) {
        logger.debug(`Sent position updates for ${updates.length} nodes`);
      }
    } catch (error) {
      logger.error('Error sending position updates:', createErrorMetadata(error));
    }
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
    // Immediately notify of current status
    handler(this.isConnected);
    return () => {
      this.connectionStatusHandlers = this.connectionStatusHandlers.filter(h => h !== handler);
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

  public isReady(): boolean {
    return this.isConnected && this.isServerReady;
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
      // Clear reconnection timeout
      if (this.reconnectTimeout) {
        window.clearTimeout(this.reconnectTimeout);
        this.reconnectTimeout = null;
      }

      try {
        // Close the socket with a normal closure
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
        this.notifyConnectionStatusHandlers(false);
      }
    }
  }

  // Alias for compatibility
  public disconnect(): void {
    this.close();
  }
}

// Create and export singleton instance
export const webSocketService = WebSocketService.getInstance();

export default WebSocketService;
