export interface NeuralAgent {
  id: string;
  name: string;
  type: 'researcher' | 'coder' | 'analyst' | 'optimizer' | 'coordinator';
  status: 'active' | 'idle' | 'busy' | 'error';
  cognitivePattern: 'convergent' | 'divergent' | 'lateral' | 'systems' | 'critical' | 'adaptive';
  capabilities: string[];
  autonomyLevel: number;
  learningRate: number;
  memoryUsage: number;
  performance: {
    tasksCompleted: number;
    averageResponseTime: number;
    successRate: number;
    errorRate: number;
  };
  lastActivity: Date;
  coordinationNode?: string;
}

export interface SwarmTopology {
  type: 'mesh' | 'hierarchical' | 'ring' | 'star';
  nodes: SwarmNode[];
  connections: SwarmConnection[];
  consensus: 'proof-of-learning' | 'byzantine' | 'raft' | 'gossip';
}

export interface SwarmNode {
  id: string;
  agentId: string;
  position: { x: number; y: number; z?: number };
  role: 'worker' | 'coordinator' | 'validator' | 'optimizer';
  load: number;
  connections: string[];
  status: 'online' | 'offline' | 'syncing' | 'error';
}

export interface SwarmConnection {
  id: string;
  sourceId: string;
  targetId: string;
  weight: number;
  latency: number;
  bandwidth: number;
  status: 'active' | 'inactive' | 'congested';
}

export interface NeuralMemory {
  id: string;
  type: 'vector' | 'episodic' | 'semantic' | 'working';
  content: any;
  embedding?: number[];
  associations: string[];
  strength: number;
  created: Date;
  lastAccessed: Date;
  accessCount: number;
}

export interface ConsensusState {
  mechanism: 'proof-of-learning' | 'byzantine' | 'raft' | 'gossip';
  round: number;
  participants: string[];
  proposals: ConsensusProposal[];
  decisions: ConsensusDecision[];
  health: number;
  latency: number;
}

export interface ConsensusProposal {
  id: string;
  proposer: string;
  content: any;
  votes: { [agentId: string]: 'accept' | 'reject' | 'abstain' };
  timestamp: Date;
  status: 'pending' | 'accepted' | 'rejected' | 'expired';
}

export interface ConsensusDecision {
  id: string;
  proposalId: string;
  result: 'accepted' | 'rejected';
  finalVotes: { [agentId: string]: 'accept' | 'reject' | 'abstain' };
  timestamp: Date;
  confidence: number;
}

export interface ResourceMetrics {
  timestamp: Date;
  cpu: {
    usage: number;
    cores: number;
    frequency: number;
    temperature?: number;
  };
  gpu?: {
    usage: number;
    memory: number;
    temperature?: number;
    powerDraw?: number;
  };
  memory: {
    used: number;
    total: number;
    cached: number;
    buffers: number;
  };
  network: {
    bytesIn: number;
    bytesOut: number;
    packetsIn: number;
    packetsOut: number;
  };
  swarm: {
    activeAgents: number;
    totalTasks: number;
    completedTasks: number;
    errorRate: number;
  };
}

export interface NeuralWebSocketMessage {
  type: 'agent_update' | 'swarm_topology' | 'memory_sync' | 'consensus_update' | 'metrics_update' | 'task_result';
  payload: any;
  timestamp: Date;
  source: string;
}

export interface TaskResult {
  id: string;
  agentId: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: any;
  error?: string;
  startTime: Date;
  endTime?: Date;
  metrics: {
    executionTime: number;
    memoryUsed: number;
    cpuUsage: number;
  };
}

export interface CognitivePattern {
  type: 'convergent' | 'divergent' | 'lateral' | 'systems' | 'critical' | 'adaptive';
  description: string;
  strength: number;
  applications: string[];
  effectiveness: number;
  learningProgress: number;
}

export interface NeuralDashboardState {
  agents: NeuralAgent[];
  topology: SwarmTopology;
  memory: NeuralMemory[];
  consensus: ConsensusState;
  metrics: ResourceMetrics[];
  isConnected: boolean;
  lastUpdate: Date;
}