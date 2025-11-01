import { createLogger } from '../../../utils/loggerConfig';
import { agentTelemetry } from '../../../telemetry/AgentTelemetry';
import { webSocketService } from '../../../services/WebSocketService';
import { agentPollingService } from './AgentPollingService';
import type { BotsAgent, BotsEdge, BotsCommunication } from '../types/BotsTypes';

const logger = createLogger('BotsWebSocketIntegration');


export class BotsWebSocketIntegration {
  private static instance: BotsWebSocketIntegration;
  private logseqConnected = false;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  
  private useRestPolling: boolean = true; 

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

    
    this.initializeLogseqConnection();
  }

  private initializeLogseqConnection() {
    
    webSocketService.onConnectionStatusChange((connected) => {
      logger.info(`Logseq WebSocket connection status: ${connected}`);
      this.logseqConnected = connected;
      this.emit('logseq-connected', { connected });

      
      agentTelemetry.logWebSocketMessage('connection_status_change', 'incoming', { connected });
      
      
      
      logger.info(`WebSocket connection ${connected ? 'established' : 'lost'} - position updates ${connected ? 'enabled' : 'disabled'}`);
    });

    
    webSocketService.onMessage((message) => {
      
      agentTelemetry.logWebSocketMessage(message.type || 'unknown', 'incoming', message.data);

      if (message.type === 'graph-update') {
        logger.debug('Received Logseq graph update', message.data);
        this.emit('logseq-graph-update', message.data);
      } else if (message.type === 'botsGraphUpdate') {
        
        const nodeCount = message.data?.nodes?.length || 0;
        const edgeCount = message.data?.edges?.length || 0;
        logger.debug('Received bots graph update with', nodeCount, 'nodes and', edgeCount, 'edges');

        
        agentTelemetry.logWebSocketMessage('botsGraphUpdate', 'incoming', {
          nodeCount,
          edgeCount,
          hasData: !!message.data
        });

        
        if (message.data?.nodes) {
          logger.debug(`[NODES] Received ${message.data.nodes.length} nodes with positions`);
        }

        
        if (!message.data) {
          logger.warn('botsGraphUpdate message has no data field:', message);
        }
        this.emit('bots-graph-update', message.data);
      } else if (message.type === 'bots-full-update') {
        const agentCount = message.agents?.length || 0;
        logger.debug('Received bots full update with', agentCount, 'agents');

        
        agentTelemetry.logWebSocketMessage('bots-full-update', 'incoming', {
          agentCount,
          hasMultiAgentMetrics: !!message.multiAgentMetrics
        });

        this.emit('bots-full-update', message);
        
        this.processBotsUpdate({
          agents: message.agents,
          multiAgentMetrics: message.multiAgentMetrics,
          timestamp: message.timestamp
        });
      }
    });

    
    webSocketService.onBinaryMessage((data) => {
      logger.debug(`Received Logseq binary update: ${data.byteLength} bytes`);

      
      agentTelemetry.logWebSocketMessage('binary_position_update', 'incoming', undefined, data.byteLength);

      this.emit('logseq-binary-update', data);
    });

    
    webSocketService.on('bots-position-update', (data: ArrayBuffer) => {
      logger.debug(`Received bots binary position update: ${data.byteLength} bytes`);

      
      agentTelemetry.logWebSocketMessage('bots_binary_position_update', 'incoming', undefined, data.byteLength);

      
      this.emit('bots-binary-position-update', data);
    });
  }

  private processBotsUpdate(data: any) {
    
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

    
    this.emit('bots-update', data);
  }

  
  public setPollingMode(useRest: boolean): void {
    logger.warn('setPollingMode is deprecated. Polling is handled by BotsDataContext via REST API.');
    this.useRestPolling = useRest;
  }

  
  public startBotsGraphPolling(interval: number = 2000): void {
    logger.warn('startBotsGraphPolling is deprecated. Use REST polling via BotsDataContext instead.');
    
  }

  
  public stopBotsGraphPolling(): void {
    logger.warn('stopBotsGraphPolling is deprecated - no WebSocket polling to stop');
    
  }

  

  
  async requestInitialData() {
    logger.info('requestInitialData is deprecated - initial data is now fetched via REST polling in BotsDataContext');
    
    
  }

  
  async sendBotsUpdate(data: {
    nodes: BotsAgent[],
    edges: BotsEdge[]
  }) {
    logger.warn('sendBotsUpdate is deprecated. Use REST API POST /api/bots/update instead');
    
    
  }

  
  getConnectionStatus() {
    return {
      mcp: false, 
      logseq: this.logseqConnected,
      overall: this.logseqConnected
    };
  }

  
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

  
  clearAgents() {
    logger.info('Clearing all agents data');
    
    
    this.emit('bots-graph-update', {
      nodes: [],
      edges: [],
      metadata: {}
    });
  }

  
  restartPolling() {
    logger.warn('restartPolling is deprecated. Polling is handled by BotsDataContext via REST API.');
    
    
    
  }

  
  disconnect() {
    logger.info('Disconnecting WebSocket services');
    this.clearAgents();
    
    
    this.stopBotsGraphPolling();
    
    webSocketService.close();
    this.logseqConnected = false;
  }
}

// Export singleton instance
export const botsWebSocketIntegration = BotsWebSocketIntegration.getInstance();