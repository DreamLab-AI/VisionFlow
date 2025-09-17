export interface AgentLifecycleEvent {
  id: string;
  timestamp: Date;
  agentId: string;
  agentName: string;
  eventType: 'spawn' | 'activate' | 'deactivate' | 'error' | 'complete' | 'idle';
  details: Record<string, any>;
  metadata?: {
    position?: { x: number; y: number; z?: number };
    performance?: { cpu: number; memory: number; gpu?: number };
    communication?: { incoming: number; outgoing: number };
  };
}

export interface AgentPosition {
  agentId: string;
  position: { x: number; y: number; z?: number };
  velocity?: { x: number; y: number; z?: number };
  timestamp: Date;
  forceVector?: { x: number; y: number; magnitude: number };
}

export interface MCPBridgeStatus {
  isConnected: boolean;
  latency: number;
  messageCount: number;
  errorCount: number;
  lastHeartbeat: Date;
  bridgeId: string;
  status: 'active' | 'idle' | 'error' | 'disconnected';
}

export interface GPUMetrics {
  utilizationPercent: number;
  memoryUsed: number;
  memoryTotal: number;
  temperature: number;
  powerConsumption: number;
  timestamp: Date;
  computeTasks: Array<{
    taskId: string;
    agentId: string;
    intensity: number;
    duration: number;
  }>;
}

export interface PerformanceMetrics {
  timestamp: Date;
  overall: {
    cpu: number;
    memory: number;
    network: number;
    gpu?: number;
  };
  agents: Array<{
    agentId: string;
    cpu: number;
    memory: number;
    taskCount: number;
    health: number;
  }>;
}

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: 'debug' | 'info' | 'warn' | 'error' | 'critical';
  source: string;
  message: string;
  agentId?: string;
  metadata?: Record<string, any>;
  tags?: string[];
}

export interface AgentCommunication {
  id: string;
  fromAgentId: string;
  toAgentId: string;
  messageType: string;
  timestamp: Date;
  latency?: number;
  success: boolean;
  metadata?: Record<string, any>;
}

export interface NetworkGraphNode {
  id: string;
  label: string;
  agentType: string;
  status: 'active' | 'idle' | 'error';
  position?: { x: number; y: number };
  metrics: {
    messageCount: number;
    errorRate: number;
    performance: number;
  };
}

export interface NetworkGraphEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  latency: number;
  messageCount: number;
  lastActivity: Date;
}

export interface TelemetryFilters {
  dateRange?: {
    start: Date;
    end: Date;
  };
  logLevels?: LogEntry['level'][];
  agentIds?: string[];
  eventTypes?: AgentLifecycleEvent['eventType'][];
  sources?: string[];
  tags?: string[];
}