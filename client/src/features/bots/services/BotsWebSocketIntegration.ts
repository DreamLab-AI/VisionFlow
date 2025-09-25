import { createLogger } from '../../../utils/loggerConfig';
import { agentTelemetry } from '../../../telemetry/AgentTelemetry';
import { webSocketService } from '../../../services/WebSocketService';
import { agentPollingService } from './AgentPollingService';
import type { BotsAgent, BotsEdge, BotsCommunication } from '../types/BotsTypes';

const logger = createLogger('BotsWebSocketIntegration');

/**
 * Integration service that handles WebSocket connections for real-time position updates
 *
 * ARCHITECTURE (post-fix):
 * - Handles ONLY WebSocket binary position updates (34-byte format) for real-time movement
 * - Does NOT handle REST polling - that's done by BotsDataContext via useAgentPolling hook
 * - Does NOT start any timers or polling loops - prevents duplicate data fetching
 *
 * Note: MCP connections are handled by the backend only - frontend uses REST API for metadata
 */
export class BotsWebSocketIntegration {
  private static instance: BotsWebSocketIntegration;
  private logseqConnected = false;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  // Removed: botsGraphInterval - no longer needed as polling is handled by REST
  private useRestPolling: boolean = true; // Use REST polling instead of WebSocket polling

  private constructor() {
    this.initializeConnections();
  }

  static getInstance(): BotsWebSocketIntegration {
    if (!BotsWebSocketIntegration.instance) {
      BotsWebSocketIntegration.instance = new BotsWebSocketIntegration();
    }
    return BotsWebSocketIntegration.instance;
  }

  private async initializeConnections() {
    logger.info('Initializing WebSocket connection for graph data');

    // Initialize main WebSocket for Logseq graph data and position updates
    this.initializeLogseqConnection();
  }

  private initializeLogseqConnection() {
    // Listen for main WebSocket connection events (Logseq graph)
    webSocketService.onConnectionStatusChange((connected) => {
      logger.info(`Logseq WebSocket connection status: ${connected}`);
      this.logseqConnected = connected;
      this.emit('logseq-connected', { connected });

      // Log connection status changes
      agentTelemetry.logWebSocketMessage('connection_status_change', 'incoming', { connected });
      
      // DO NOT start polling here - let BotsDataContext handle REST polling
      // This service only handles WebSocket binary position updates
      logger.info(`WebSocket connection ${connected ? 'established' : 'lost'} - position updates ${connected ? 'enabled' : 'disabled'}`);
    });

    // Listen for Logseq graph messages
    webSocketService.onMessage((message) => {
      // Log all incoming WebSocket messages
      agentTelemetry.logWebSocketMessage(message.type || 'unknown', 'incoming', message.data);

      if (message.type === 'graph-update') {
        logger.debug('Received Logseq graph update', message.data);
        this.emit('logseq-graph-update', message.data);
      } else if (message.type === 'botsGraphUpdate') {
        // NEW: Handle full graph data with nodes and edges
        const nodeCount = message.data?.nodes?.length || 0;
        const edgeCount = message.data?.edges?.length || 0;
        logger.debug('Received bots graph update with', nodeCount, 'nodes and', edgeCount, 'edges');

        // Enhanced telemetry logging
        agentTelemetry.logWebSocketMessage('botsGraphUpdate', 'incoming', {
          nodeCount,
          edgeCount,
          hasData: !!message.data
        });

        // Log node positions for debugging
        if (message.data?.nodes) {
          logger.debug(`[NODES] Received ${message.data.nodes.length} nodes with positions`);
        }

        // Add debug logging
        if (!message.data) {
          logger.warn('botsGraphUpdate message has no data field:', message);
        }
        this.emit('bots-graph-update', message.data);
      } else if (message.type === 'bots-full-update') {
        const agentCount = message.agents?.length || 0;
        logger.debug('Received bots full update with', agentCount, 'agents');

        // Log full update telemetry
        agentTelemetry.logWebSocketMessage('bots-full-update', 'incoming', {
          agentCount,
          hasMultiAgentMetrics: !!message.multiAgentMetrics
        });

        this.emit('bots-full-update', message);
        // Also emit individual updates for backward compatibility
        this.processBotsUpdate({
          agents: message.agents,
          multiAgentMetrics: message.multiAgentMetrics,
          timestamp: message.timestamp
        });
      }
    });

    // Listen for binary updates (node positions from Logseq)
    webSocketService.onBinaryMessage((data) => {
      logger.debug(`Received Logseq binary update: ${data.byteLength} bytes`);

      // Log binary message telemetry
      agentTelemetry.logWebSocketMessage('binary_position_update', 'incoming', undefined, data.byteLength);

      this.emit('logseq-binary-update', data);
    });

    // Listen for bots position updates (agent nodes from binary protocol)
    webSocketService.on('bots-position-update', (data: ArrayBuffer) => {
      logger.debug(`Received bots binary position update: ${data.byteLength} bytes`);

      // Log agent position update telemetry
      agentTelemetry.logWebSocketMessage('bots_binary_position_update', 'incoming', undefined, data.byteLength);

      // Emit for bots visualization to consume
      this.emit('bots-binary-position-update', data);
    });
  }

  private processBotsUpdate(data: any) {
    // Process different types of bots updates
    if (data.agents) {
      this.emit('bots-agents-update', data.agents);
    }

    if (data.edges) {
      this.emit('bots-edges-update', data.edges);
    }

    if (data.communications) {
      this.emit('bots-communications-update', data.communications);
    }

    if (data.tokenUsage) {
      this.emit('bots-token-usage', data.tokenUsage);
    }

    // Emit general bots update
    this.emit('bots-update', data);
  }

  /**
   * Configure polling mode (REST vs WebSocket)
   * @deprecated - Polling is now handled by BotsDataContext, this service only handles binary position updates
   */
  public setPollingMode(useRest: boolean): void {
    logger.warn('setPollingMode is deprecated. Polling is handled by BotsDataContext via REST API.');
    this.useRestPolling = useRest;
  }

  /**
   * Start polling for full bots graph data (legacy WebSocket polling)
   * @deprecated - Use REST polling via BotsDataContext instead
   */
  public startBotsGraphPolling(interval: number = 2000): void {
    logger.warn('startBotsGraphPolling is deprecated. Use REST polling via BotsDataContext instead.');
    // No longer start WebSocket polling to avoid duplicate data fetching
  }

  /**
   * Stop polling for bots graph data
   * @deprecated - No longer needed as polling is handled by BotsDataContext
   */
  public stopBotsGraphPolling(): void {
    logger.warn('stopBotsGraphPolling is deprecated - no WebSocket polling to stop');
    // No longer has botsGraphInterval timer to clear
  }

  // Public API methods

  /**
   * Request initial data for both visualizations
   * @deprecated - Initial data fetching is now handled by BotsDataContext via REST polling
   */
  async requestInitialData() {
    logger.info('requestInitialData is deprecated - initial data is now fetched via REST polling in BotsDataContext');
    // No longer fetch initial data here to avoid duplicate requests
    // BotsDataContext handles all REST API calls for agent metadata
  }

  /**
   * Send bots position update (deprecated - use REST API instead)
   */
  async sendBotsUpdate(data: {
    nodes: BotsAgent[],
    edges: BotsEdge[]
  }) {
    logger.warn('sendBotsUpdate is deprecated. Use REST API POST /api/bots/update instead');
    // Updates should be sent via REST API to the backend
    // The backend handles all communication with Claude Flow MCP
  }

  /**
   * Check connection status
   */
  getConnectionStatus() {
    return {
      mcp: false, // MCP connections are handled by backend only
      logseq: this.logseqConnected,
      overall: this.logseqConnected
    };
  }

  /**
   * Subscribe to events
   */
  on(event: string, callback: (data: any) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);

    return () => {
      this.listeners.get(event)?.delete(callback);
    };
  }

  private emit(event: string, data: any) {
    this.listeners.get(event)?.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        logger.error(`Error in event listener for ${event}:`, error);
      }
    });
  }

  /**
   * Clear all agents data
   */
  clearAgents() {
    logger.info('Clearing all agents data');
    // No longer has WebSocket polling to stop - handled by BotsDataContext
    // Emit cleared state via the proper event
    this.emit('bots-graph-update', {
      nodes: [],
      edges: [],
      metadata: {}
    });
  }

  /**
   * @deprecated Polling is now handled by BotsDataContext via REST API
   * This method is kept for backward compatibility but does nothing
   */
  restartPolling() {
    logger.warn('restartPolling is deprecated. Polling is handled by BotsDataContext via REST API.');
    // No-op - polling is handled by BotsDataContext
    // NOTE: Hive mind control is managed via docker exec with claude-flow CLI
    // from visionflow container to /ext/multi-agent-docker container
  }

  /**
   * Cleanup connections
   */
  disconnect() {
    logger.info('Disconnecting WebSocket services');
    this.clearAgents();
    
    // Stop only WebSocket polling - REST polling is managed by BotsDataContext
    this.stopBotsGraphPolling();
    
    webSocketService.close();
    this.logseqConnected = false;
  }
}

// Export singleton instance
export const botsWebSocketIntegration = BotsWebSocketIntegration.getInstance();