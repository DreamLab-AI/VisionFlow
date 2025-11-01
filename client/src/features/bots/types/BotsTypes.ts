// Bots visualization type definitions

// UPDATED: Enhanced agent types to match claude-flow hive-mind system (15+ types including Maestro specs-driven agents)
export interface BotsAgent {
  id: string;
  type: 'coordinator' | 'researcher' | 'coder' | 'analyst' | 'architect' | 'tester' | 'reviewer' | 'optimizer' | 'documenter' | 'monitor' | 'specialist' |
        'requirements_analyst' | 'design_architect' | 'task_planner' | 'implementation_coder' | 'quality_reviewer' | 'steering_documenter' | 'queen';
  status: 'idle' | 'busy' | 'active' | 'error' | 'initializing' | 'terminating' | 'offline';
  health: number; 
  cpuUsage: number; 
  memoryUsage: number; 
  workload?: number; 
  createdAt: string; 
  age: number; 
  name?: string;

  
  capabilities?: string[]; 
  currentTask?: string; 
  tasksActive?: number; 
  tasksCompleted?: number; 
  successRate?: number; 
  tokens?: number; 
  tokenRate?: number; 
  activity?: number; 

  
  position?: {
    x: number;
    y: number;
    z: number;
  };
  velocity?: {
    x: number;
    y: number;
    z: number;
  };

  
  ssspDistance?: number; 
  ssspParent?: number;   
  lastPositionUpdate?: number; 

  
  swarmId?: string; 
  agentMode?: 'centralized' | 'distributed' | 'strategic';
  parentQueenId?: string; 

  
  processingLogs?: string[]; 
}

export interface BotsCommunication {
  id: string;
  type: 'communication';
  timestamp: string;
  sender: string; 
  receivers: string[]; 
  metadata: {
    size: number; 
    type?: string; 
  };
}

export interface TokenUsage {
  total: number;
  byAgent: {
    [agentType: string]: number;
  };
}

export interface BotsEdge {
  id: string;
  source: string;
  target: string;
  dataVolume: number; 
  messageCount: number;
  lastMessageTime: number;
}

export interface BotsState {
  agents: Map<string, BotsAgent>;
  edges: Map<string, BotsEdge>;
  communications: BotsCommunication[];
  tokenUsage: TokenUsage;
  lastUpdate: number;
}

// MCP WebSocket message types
export interface MCPMessage {
  type: 'welcome' | 'mcp-update' | 'mcp-response' | 'ping' | 'pong';
  clientId?: string;
  data?: any;
  requestId?: string;
}

// Enhanced WebSocket message for full agent data updates
export interface BotsFullUpdateMessage {
  type: 'bots-full-update';
  agents: BotsAgent[];
  multiAgentMetrics: {
    totalAgents: number;
    activeAgents: number;
    totalTasks: number;
    completedTasks: number;
    avgSuccessRate: number;
    totalTokens: number;
  };
  timestamp: string; 
}

export interface MCPRequest {
  jsonrpc: '2.0';
  id: string;
  method: 'tools/call';
  params: {
    name: string;
    arguments: any;
  };
}

// UPDATED: Enhanced visual configuration for all claude-flow hive-mind agent types
export interface BotsVisualConfig {
  colors: {
    
    coordinator: string;
    researcher: string;
    coder: string;
    analyst: string;
    architect: string;
    tester: string;
    reviewer: string;
    optimizer: string;
    documenter: string;
    monitor: string;
    specialist: string;

    
    requirements_analyst: string;
    design_architect: string;
    task_planner: string;
    implementation_coder: string;
    quality_reviewer: string;
    steering_documenter: string;

    
    queen: string; 
  };

  
  physics: {
    springStrength: number;
    linkDistance: number;
    damping: number;
    nodeRepulsion: number;
    gravityStrength: number;
    maxVelocity: number;

    
    queenGravity: number; 
    multiAgentCohesion: number; 
    hierarchicalForce: number; 
  };

  
  sizes: {
    queen: number; 
    coordinator: number;
    architect: number;
    specialist: number;
    default: number; 
  };
}