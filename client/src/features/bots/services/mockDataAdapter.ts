/**
 * Mock Data Adapter - Converts mock agent data to existing format structures
 *
 * This adapter ensures compatibility between the comprehensive mock data
 * and the existing AgentPollingService and BotsTypes interfaces.
 */

import { mockAgentData, mockSwarmMetadata, MockAgentStatus, generateMockAgentData } from './mockAgentData';
import type { AgentSwarmData } from './AgentPollingService';
import type { BotsAgent, BotsEdge } from '../types/BotsTypes';
import { createLogger } from '../../../utils/loggerConfig';

const logger = createLogger('MockDataAdapter');

/**
 * Convert MockAgentStatus to BotsAgent format
 */
export const convertMockAgentToBotsAgent = (mockAgent: MockAgentStatus): BotsAgent => {
  return {
    id: mockAgent.id,
    type: mockAgent.type as BotsAgent['type'],
    status: mockAgent.status as BotsAgent['status'],
    health: Math.round(mockAgent.health * 100), // Convert 0-1 to 0-100
    cpuUsage: Math.round(mockAgent.cpuUsage * 100), // Convert 0-1 to percentage
    memoryUsage: Math.round(mockAgent.memoryUsage * 100), // Convert 0-1 to percentage
    workload: mockAgent.workload,
    createdAt: mockAgent.createdAt,
    age: mockAgent.age,
    name: mockAgent.profile.name,

    // Enhanced properties
    capabilities: mockAgent.capabilities,
    currentTask: mockAgent.currentTask,
    tasksActive: mockAgent.tasksActive,
    tasksCompleted: mockAgent.tasksCompleted,
    successRate: mockAgent.successRate,
    tokens: mockAgent.tokens,
    tokenRate: mockAgent.tokenRate,
    activity: mockAgent.activity,

    // 3D positioning
    position: mockAgent.position,
    velocity: mockAgent.position ? {
      x: (Math.random() - 0.5) * 2,
      y: (Math.random() - 0.5) * 2,
      z: (Math.random() - 0.5) * 0.5
    } : undefined,

    // SSSP data (simulated)
    ssspDistance: Math.floor(Math.random() * 10) + 1,
    ssspParent: Math.floor(Math.random() * 12),
    lastPositionUpdate: Date.now(),

    // Multi-agent metadata
    swarmId: mockAgent.swarmId,
    agentMode: mockAgent.agentMode as BotsAgent['agentMode'],
    parentQueenId: mockAgent.parentQueenId,
    processingLogs: mockAgent.processingLogs
  };
};

/**
 * Convert MockAgentStatus to AgentSwarmData node format
 */
export const convertMockAgentToSwarmNode = (mockAgent: MockAgentStatus, index: number) => {
  return {
    id: index,
    metadata_id: mockAgent.id,
    label: mockAgent.profile.name,
    node_type: mockAgent.type,
    data: {
      position: mockAgent.position || { x: 0, y: 0, z: 0 },
      velocity: {
        x: (Math.random() - 0.5) * 2,
        y: (Math.random() - 0.5) * 2,
        z: (Math.random() - 0.5) * 0.5
      }
    },
    metadata: {
      agent_type: mockAgent.type,
      status: mockAgent.status,
      health: (mockAgent.health * 100).toFixed(1),
      cpu_usage: (mockAgent.cpuUsage * 100).toFixed(1),
      memory_usage: (mockAgent.memoryUsage * 100).toFixed(1),
      workload: mockAgent.workload?.toFixed(2) || '0.0',
      tokens: mockAgent.tokens.toString(),
      created_at: mockAgent.createdAt,
      age: mockAgent.age.toString(),
      swarm_id: mockAgent.swarmId,
      parent_queen_id: mockAgent.parentQueenId,
      capabilities: mockAgent.capabilities.join(', ')
    }
  };
};

/**
 * Generate mock edges connecting related agents
 */
export const generateMockEdges = (agents: MockAgentStatus[]): Array<{
  id: string;
  source: number;
  target: number;
  weight: number;
}> => {
  const edges = [];

  // Connect coordinators to other agents
  const coordinatorIndices = agents
    .map((agent, index) => ({ agent, index }))
    .filter(({ agent }) => agent.type === 'coordinator')
    .map(({ index }) => index);

  // Connect each coordinator to 2-4 other agents
  coordinatorIndices.forEach(coordIndex => {
    const connectionCount = 2 + Math.floor(Math.random() * 3);
    const targetIndices = new Set<number>();

    while (targetIndices.size < connectionCount) {
      const targetIndex = Math.floor(Math.random() * agents.length);
      if (targetIndex !== coordIndex) {
        targetIndices.add(targetIndex);
      }
    }

    targetIndices.forEach(targetIndex => {
      edges.push({
        id: `edge-${coordIndex}-${targetIndex}`,
        source: coordIndex,
        target: targetIndex,
        weight: Math.random() * 0.8 + 0.2 // 0.2 to 1.0
      });
    });
  });

  // Add some random connections between non-coordinator agents
  for (let i = 0; i < agents.length; i++) {
    if (agents[i].type !== 'coordinator' && Math.random() > 0.7) {
      let targetIndex;
      do {
        targetIndex = Math.floor(Math.random() * agents.length);
      } while (targetIndex === i);

      edges.push({
        id: `edge-${i}-${targetIndex}`,
        source: i,
        target: targetIndex,
        weight: Math.random() * 0.5 + 0.1 // 0.1 to 0.6 (weaker than coordinator connections)
      });
    }
  }

  return edges;
};

/**
 * Convert mock data to full AgentSwarmData format
 */
export const convertToAgentSwarmData = (agents: MockAgentStatus[] = mockAgentData): AgentSwarmData => {
  const nodes = agents.map(convertMockAgentToSwarmNode);
  const edges = generateMockEdges(agents);

  // Calculate metadata
  const activeAgents = agents.filter(a => a.status === 'active' || a.status === 'busy').length;
  const totalTasks = agents.reduce((sum, a) => sum + a.activeTasksCount, 0);
  const completedTasks = agents.reduce((sum, a) => sum + a.completedTasksCount, 0);
  const avgSuccessRate = agents.reduce((sum, a) => sum + a.successRate, 0) / agents.length;
  const totalTokens = agents.reduce((sum, a) => sum + a.tokens, 0);

  return {
    nodes,
    edges,
    metadata: {
      total_agents: agents.length,
      active_agents: activeAgents,
      total_tasks: totalTasks,
      completed_tasks: completedTasks,
      avg_success_rate: avgSuccessRate,
      total_tokens: totalTokens
    }
  };
};

/**
 * Convert mock data to BotsAgent array format
 */
export const convertToBotsAgents = (agents: MockAgentStatus[] = mockAgentData): BotsAgent[] => {
  return agents.map(convertMockAgentToBotsAgent);
};

/**
 * Generate mock edges in BotsEdge format
 */
export const generateMockBotsEdges = (agents: BotsAgent[]): BotsEdge[] => {
  const edges: BotsEdge[] = [];
  const now = Date.now();

  // Similar logic to generateMockEdges but for BotsEdge format
  const coordinators = agents.filter(a => a.type === 'coordinator');

  coordinators.forEach(coord => {
    // Connect each coordinator to 2-4 other agents
    const connectionCount = 2 + Math.floor(Math.random() * 3);
    const potentialTargets = agents.filter(a => a.id !== coord.id);

    for (let i = 0; i < Math.min(connectionCount, potentialTargets.length); i++) {
      const target = potentialTargets[Math.floor(Math.random() * potentialTargets.length)];

      edges.push({
        id: `${coord.id}-${target.id}`,
        source: coord.id,
        target: target.id,
        dataVolume: Math.floor(Math.random() * 10000) + 1000, // 1-10KB
        messageCount: Math.floor(Math.random() * 100) + 10,
        lastMessageTime: now - Math.floor(Math.random() * 60000) // Within last minute
      });
    }
  });

  return edges;
};

/**
 * Mock data service that can replace real API calls during development
 */
export class MockDataService {
  private agents: MockAgentStatus[];
  private updateCallbacks = new Set<(data: AgentSwarmData) => void>();
  private updateInterval: NodeJS.Timeout | null = null;
  private isRunning = false;

  constructor(initialAgentCount = 12) {
    this.agents = generateMockAgentData(initialAgentCount);
    logger.info(`MockDataService initialized with ${initialAgentCount} agents`);
  }

  /**
   * Start periodic updates to simulate real agent activity
   */
  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    logger.info('MockDataService started - simulating agent activity');

    // Update agent states every 3 seconds
    this.updateInterval = setInterval(() => {
      this.updateAgentStates();
      this.notifySubscribers();
    }, 3000);
  }

  /**
   * Stop periodic updates
   */
  stop(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    this.isRunning = false;
    logger.info('MockDataService stopped');
  }

  /**
   * Subscribe to data updates
   */
  subscribe(callback: (data: AgentSwarmData) => void): () => void {
    this.updateCallbacks.add(callback);

    // Send initial data
    callback(this.getAgentSwarmData());

    return () => {
      this.updateCallbacks.delete(callback);
    };
  }

  /**
   * Get current agent swarm data
   */
  getAgentSwarmData(): AgentSwarmData {
    return convertToAgentSwarmData(this.agents);
  }

  /**
   * Get current bots agent data
   */
  getBotsAgents(): BotsAgent[] {
    return convertToBotsAgents(this.agents);
  }

  /**
   * Simulate agent state changes
   */
  private updateAgentStates(): void {
    this.agents.forEach(agent => {
      // Randomly update some agent states
      if (Math.random() > 0.8) {
        // Update activity level
        agent.activity = Math.max(0.1, Math.min(1.0, agent.activity + (Math.random() - 0.5) * 0.2));

        // Update CPU/memory based on activity
        agent.cpuUsage = Math.max(0.05, Math.min(0.95, agent.activity * 0.8 + Math.random() * 0.2));
        agent.memoryUsage = Math.max(0.2, Math.min(0.9, agent.memoryUsage + (Math.random() - 0.5) * 0.1));

        // Occasionally change status
        if (Math.random() > 0.9) {
          const statuses = ['idle', 'busy', 'active'];
          agent.status = statuses[Math.floor(Math.random() * statuses.length)];
        }

        // Update position slightly
        if (agent.position) {
          agent.position.x += (Math.random() - 0.5) * 5;
          agent.position.y += (Math.random() - 0.5) * 5;
          agent.position.z += (Math.random() - 0.5) * 2;
        }

        // Update tokens
        agent.tokens += Math.floor(Math.random() * 100);

        // Update age
        agent.age = Date.now() - new Date(agent.createdAt).getTime();
      }
    });
  }

  /**
   * Notify all subscribers of data changes
   */
  private notifySubscribers(): void {
    const data = this.getAgentSwarmData();
    this.updateCallbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        logger.error('Error in mock data callback:', error);
      }
    });
  }

  /**
   * Add a new mock agent
   */
  addAgent(agentType: MockAgentStatus['type'] = 'generic'): void {
    const newAgent = generateMockAgentData(1)[0];
    newAgent.type = agentType;
    newAgent.profile.agentType = agentType as MockAgentStatus['profile']['agentType'];
    this.agents.push(newAgent);

    if (this.isRunning) {
      this.notifySubscribers();
    }

    logger.info(`Added new mock agent: ${newAgent.id} (${agentType})`);
  }

  /**
   * Remove a mock agent
   */
  removeAgent(agentId: string): void {
    const initialLength = this.agents.length;
    this.agents = this.agents.filter(agent => agent.id !== agentId);

    if (this.agents.length < initialLength) {
      if (this.isRunning) {
        this.notifySubscribers();
      }
      logger.info(`Removed mock agent: ${agentId}`);
    }
  }
}

/**
 * Export singleton instance for easy use
 */
export const mockDataService = new MockDataService();

/**
 * Development utility to enable/disable mock data
 */
export const useMockData = () => {
  return process.env.NODE_ENV === 'development' || process.env.REACT_APP_USE_MOCK_DATA === 'true';
};