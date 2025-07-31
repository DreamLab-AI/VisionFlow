// WebSocket Integration for Real-time Agent Data
// Connects to agentic-flow system for live updates

import { EventEmitter } from 'events';
import { 
  EnhancedAgent, 
  MessageFlow, 
  CoordinationInstance, 
  SystemMetrics,
  AgentEvent,
  MessageEvent,
  CoordinationEvent,
  SystemEvent,
  VisualizationState
} from './types';

interface WebSocketConfig {
  url: string;
  reconnectInterval: number;
  heartbeatInterval: number;
  maxReconnectAttempts: number;
  messageBufferSize: number;
}

class AgentWebSocketManager extends EventEmitter {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private messageBuffer: any[] = [];
  private isConnected = false;
  private subscriptions = new Set<string>();

  constructor(config: Partial<WebSocketConfig> = {}) {
    super();
    this.config = {
      url: config.url || 'ws://localhost:3001/agent-websocket',
      reconnectInterval: config.reconnectInterval || 5000,
      heartbeatInterval: config.heartbeatInterval || 30000,
      maxReconnectAttempts: config.maxReconnectAttempts || 10,
      messageBufferSize: config.messageBufferSize || 1000,
      ...config
    };
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.config.url);
        
        this.ws.onopen = () => {
          console.log('Agent WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.resubscribe();
          this.flushMessageBuffer();
          this.emit('connected');
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        this.ws.onclose = (event) => {
          console.log('Agent WebSocket disconnected:', event.code, event.reason);
          this.isConnected = false;
          this.stopHeartbeat();
          this.emit('disconnected', event);
          
          if (!event.wasClean && this.reconnectAttempts < this.config.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('Agent WebSocket error:', error);
          this.emit('error', error);
          reject(error);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
    }
    this.cleanup();
  }

  private cleanup(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopHeartbeat();
    this.ws = null;
    this.isConnected = false;
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts}`);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnect failed:', error);
      });
    }, this.config.reconnectInterval);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected) {
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data);
      
      switch (message.type) {
        case 'pong':
          // Heartbeat response - no action needed
          break;
          
        case 'agent_event':
          this.emit('agentEvent', message.data as AgentEvent);
          break;
          
        case 'message_event':
          this.emit('messageEvent', message.data as MessageEvent);
          break;
          
        case 'coordination_event':
          this.emit('coordinationEvent', message.data as CoordinationEvent);
          break;
          
        case 'system_event':
          this.emit('systemEvent', message.data as SystemEvent);
          break;
          
        case 'bulk_update':
          this.emit('bulkUpdate', message.data);
          break;
          
        default:
          console.warn('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  private send(message: any): void {
    if (this.isConnected && this.ws) {
      this.ws.send(JSON.stringify(message));
    } else {
      // Buffer messages when disconnected
      if (this.messageBuffer.length < this.config.messageBufferSize) {
        this.messageBuffer.push(message);
      }
    }
  }

  private flushMessageBuffer(): void {
    while (this.messageBuffer.length > 0) {
      const message = this.messageBuffer.shift();
      this.send(message);
    }
  }

  private resubscribe(): void {
    this.subscriptions.forEach(subscription => {
      this.send({ type: 'subscribe', subscription });
    });
  }

  // Public subscription methods
  subscribeToAgent(agentId: string): void {
    const subscription = `agent:${agentId}`;
    this.subscriptions.add(subscription);
    this.send({ type: 'subscribe', subscription });
  }

  subscribeToMessageFlow(sourceId: string, targetId: string): void {
    const subscription = `messageflow:${sourceId}:${targetId}`;
    this.subscriptions.add(subscription);
    this.send({ type: 'subscribe', subscription });
  }

  subscribeToCoordination(coordinationId: string): void {
    const subscription = `coordination:${coordinationId}`;
    this.subscriptions.add(subscription);
    this.send({ type: 'subscribe', subscription });
  }

  subscribeToSystemMetrics(): void {
    const subscription = 'system:metrics';
    this.subscriptions.add(subscription);
    this.send({ type: 'subscribe', subscription });
  }

  unsubscribe(subscription: string): void {
    this.subscriptions.delete(subscription);
    this.send({ type: 'unsubscribe', subscription });
  }

  // Request initial data
  requestInitialData(): void {
    this.send({ type: 'request_initial_data' });
  }

  // Agent actions
  sendAgentCommand(agentId: string, command: string, params: any = {}): void {
    this.send({
      type: 'agent_command',
      agentId,
      command,
      params,
      timestamp: Date.now()
    });
  }

  // Connection status
  getConnectionStatus(): { connected: boolean; reconnectAttempts: number } {
    return {
      connected: this.isConnected,
      reconnectAttempts: this.reconnectAttempts
    };
  }
}

// React Hook for Agent Data Store
import { useState, useEffect, useCallback } from 'react';

interface AgentDataStore {
  agents: Map<string, EnhancedAgent>;
  messageFlows: Map<string, MessageFlow>;
  coordinationInstances: Map<string, CoordinationInstance>;
  systemMetrics: SystemMetrics;
  connectionStatus: { connected: boolean; reconnectAttempts: number };
}

const defaultSystemMetrics: SystemMetrics = {
  activeAgents: 0,
  totalAgents: 0,
  averagePerformance: 0,
  messageRate: 0,
  errorRate: 0,
  networkHealth: 0,
  resourceUtilization: 0,
  coordinationEfficiency: 0
};

export const useAgentDataStore = (wsConfig?: Partial<WebSocketConfig>) => {
  const [dataStore, setDataStore] = useState<AgentDataStore>({
    agents: new Map(),
    messageFlows: new Map(),
    coordinationInstances: new Map(),
    systemMetrics: defaultSystemMetrics,
    connectionStatus: { connected: false, reconnectAttempts: 0 }
  });

  const [wsManager] = useState(() => new AgentWebSocketManager(wsConfig));

  // Update agent data
  const updateAgent = useCallback((agentId: string, agentData: Partial<EnhancedAgent>) => {
    setDataStore(prev => {
      const newAgents = new Map(prev.agents);
      const existingAgent = newAgents.get(agentId);
      
      if (existingAgent) {
        newAgents.set(agentId, { ...existingAgent, ...agentData });
      } else {
        // Create new agent with defaults
        const newAgent: EnhancedAgent = {
          id: { id: agentId },
          name: agentId,
          type: 'SPECIALIST' as any,
          state: 'IDLE' as any,
          capabilities: [],
          goals: [],
          performance: {
            tasksCompleted: 0,
            successRate: 0,
            averageResponseTime: 0,
            resourceUtilization: 0,
            communicationEfficiency: 0,
            uptime: 0,
            messageQueueSize: 0,
            connectionCount: 0,
            memoryUsage: 0
          },
          createdAt: new Date(),
          updatedAt: new Date(),
          version: '1.0.0',
          tags: [],
          childAgentIds: [],
          lastActivity: new Date(),
          processingLogs: [],
          ...agentData
        };
        newAgents.set(agentId, newAgent);
      }

      return { ...prev, agents: newAgents };
    });
  }, []);

  // Update message flow
  const updateMessageFlow = useCallback((flowId: string, flowData: Partial<MessageFlow>) => {
    setDataStore(prev => {
      const newFlows = new Map(prev.messageFlows);
      const existingFlow = newFlows.get(flowId);
      
      if (existingFlow) {
        newFlows.set(flowId, { ...existingFlow, ...flowData });
      } else {
        const newFlow: MessageFlow = {
          source: '',
          target: '',
          messages: [],
          averageLatency: 0,
          messageRate: 0,
          successRate: 0,
          bandwidth: 0,
          ...flowData
        };
        newFlows.set(flowId, newFlow);
      }

      return { ...prev, messageFlows: newFlows };
    });
  }, []);

  // Update coordination instance
  const updateCoordination = useCallback((coordId: string, coordData: Partial<CoordinationInstance>) => {
    setDataStore(prev => {
      const newCoords = new Map(prev.coordinationInstances);
      const existing = newCoords.get(coordId);
      
      if (existing) {
        newCoords.set(coordId, { ...existing, ...coordData });
      } else {
        const newCoord: CoordinationInstance = {
          id: coordId,
          pattern: 'MESH' as any,
          participants: [],
          status: 'forming',
          progress: 0,
          startedAt: new Date(),
          metadata: {},
          ...coordData
        };
        newCoords.set(coordId, newCoord);
      }

      return { ...prev, coordinationInstances: newCoords };
    });
  }, []);

  // Update system metrics
  const updateSystemMetrics = useCallback((metrics: Partial<SystemMetrics>) => {
    setDataStore(prev => ({
      ...prev,
      systemMetrics: { ...prev.systemMetrics, ...metrics }
    }));
  }, []);

  // WebSocket event handlers
  useEffect(() => {
    const handleAgentEvent = (event: AgentEvent) => {
      updateAgent(event.agentId.id, event.data);
    };

    const handleMessageEvent = (event: MessageEvent) => {
      const flowId = `${event.message.from.id}-${event.message.to}`;
      // Update message flow logic here
    };

    const handleCoordinationEvent = (event: CoordinationEvent) => {
      updateCoordination(event.coordination.id, event.coordination);
    };

    const handleSystemEvent = (event: SystemEvent) => {
      if (event.type === 'metrics_updated') {
        updateSystemMetrics(event.data);
      }
    };

    const handleConnectionStatus = () => {
      setDataStore(prev => ({
        ...prev,
        connectionStatus: wsManager.getConnectionStatus()
      }));
    };

    // Register event handlers
    wsManager.on('agentEvent', handleAgentEvent);
    wsManager.on('messageEvent', handleMessageEvent);
    wsManager.on('coordinationEvent', handleCoordinationEvent);
    wsManager.on('systemEvent', handleSystemEvent);
    wsManager.on('connected', handleConnectionStatus);
    wsManager.on('disconnected', handleConnectionStatus);

    // Connect to WebSocket
    wsManager.connect().catch(console.error);

    // Subscribe to all data
    wsManager.subscribeToSystemMetrics();
    wsManager.requestInitialData();

    return () => {
      wsManager.removeAllListeners();
      wsManager.disconnect();
    };
  }, [wsManager, updateAgent, updateMessageFlow, updateCoordination, updateSystemMetrics]);

  // Public API
  const actions = {
    subscribeToAgent: wsManager.subscribeToAgent.bind(wsManager),
    subscribeToMessageFlow: wsManager.subscribeToMessageFlow.bind(wsManager),
    subscribeToCoordination: wsManager.subscribeToCoordination.bind(wsManager),
    sendAgentCommand: wsManager.sendAgentCommand.bind(wsManager),
    unsubscribe: wsManager.unsubscribe.bind(wsManager)
  };

  return {
    ...dataStore,
    actions
  };
};

export { AgentWebSocketManager };
export type { WebSocketConfig, AgentDataStore };