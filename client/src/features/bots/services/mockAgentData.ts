/**
 * Mock Agent Telemetry Data - Comprehensive test data matching server AgentStatus structure
 *
 * This file provides realistic mock data for testing agent visualization components.
 * All fields match the server's AgentStatus structure with proper camelCase conversion.
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface AgentProfile {
  name: string;
  agentType: 'coordinator' | 'researcher' | 'coder' | 'analyst' | 'architect' | 'tester' | 'reviewer' | 'optimizer' | 'documenter' | 'generic';
  capabilities: string[];
  description?: string;
  version: string;
  tags: string[];
}

export interface PerformanceMetrics {
  tasksCompleted: number;
  successRate: number;
}

export interface TokenUsage {
  total: number;
  tokenRate: number;
}

export interface TaskReference {
  taskId: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

/**
 * Complete AgentStatus interface matching server structure with camelCase fields
 */
export interface MockAgentStatus {
  // Core identification (server sends as 'id', client receives as agentId)
  id: string;
  profile: AgentProfile;
  status: string;

  // Task information
  activeTasksCount: number;
  completedTasksCount: number;
  failedTasksCount: number;
  successRate: number;
  timestamp: string; // ISO string for client compatibility
  currentTask?: TaskReference;

  // Client compatibility fields (camelCase)
  type: string;
  currentTask: string | null;
  capabilities: string[];

  // Position as Vec3 structure
  position?: Vec3;

  // Performance metrics (0-1 normalized for client)
  cpuUsage: number;
  memoryUsage: number;
  health: number;
  activity: number;
  tasksActive: number;
  tasksCompleted: number;
  successRate: number; // Normalized 0-1
  tokens: number;
  tokenRate: number;

  // Additional fields
  performanceMetrics: PerformanceMetrics;
  tokenUsage: TokenUsage;
  swarmId?: string;
  agentMode?: string;
  parentQueenId?: string;
  processingLogs?: string[];
  createdAt: string;
  age: number; // milliseconds
  workload?: number;
}

/**
 * Generates realistic task descriptions for different agent types
 */
const generateTaskByType = (agentType: string): string | null => {
  const taskTemplates: Record<string, string[]> = {
    coordinator: [
      'Orchestrating team workflow optimization',
      'Managing resource allocation across agents',
      'Coordinating cross-team dependencies',
      'Monitoring overall system health'
    ],
    researcher: [
      'Analyzing market trends in AI development',
      'Researching best practices for microservices',
      'Investigating performance bottlenecks',
      'Studying user behavior patterns'
    ],
    coder: [
      'Implementing REST API endpoints',
      'Refactoring legacy authentication system',
      'Building responsive UI components',
      'Optimizing database queries'
    ],
    analyst: [
      'Generating performance metrics report',
      'Analyzing user engagement data',
      'Evaluating system resource utilization',
      'Creating data visualization dashboards'
    ],
    architect: [
      'Designing microservices architecture',
      'Planning database schema migration',
      'Architecting real-time communication system',
      'Designing scalable deployment pipeline'
    ],
    tester: [
      'Running integration test suite',
      'Performing load testing on API endpoints',
      'Validating user acceptance criteria',
      'Testing mobile responsiveness'
    ],
    reviewer: [
      'Reviewing code for security vulnerabilities',
      'Conducting architectural design review',
      'Evaluating performance impact of changes',
      'Assessing code quality and maintainability'
    ],
    optimizer: [
      'Optimizing React component render cycles',
      'Improving database query performance',
      'Reducing bundle size and load times',
      'Enhancing memory usage efficiency'
    ],
    documenter: [
      'Writing API documentation',
      'Updating technical specifications',
      'Creating user guides and tutorials',
      'Documenting deployment procedures'
    ]
  };

  const templates = taskTemplates[agentType] || taskTemplates.coordinator;
  return Math.random() > 0.3 ? templates[Math.floor(Math.random() * templates.length)] : null;
};

/**
 * Generates capabilities based on agent type
 */
const generateCapabilitiesByType = (agentType: string): string[] => {
  const capabilityMap: Record<string, string[]> = {
    coordinator: ['task_management', 'resource_allocation', 'team_coordination', 'workflow_optimization'],
    researcher: ['data_analysis', 'market_research', 'trend_analysis', 'documentation_review'],
    coder: ['code_generation', 'debugging', 'testing', 'refactoring', 'api_development'],
    analyst: ['data_visualization', 'metrics_analysis', 'reporting', 'performance_analysis'],
    architect: ['system_design', 'architecture_planning', 'scalability_design', 'integration_design'],
    tester: ['automated_testing', 'load_testing', 'integration_testing', 'quality_assurance'],
    reviewer: ['code_review', 'security_analysis', 'architecture_review', 'quality_assessment'],
    optimizer: ['performance_optimization', 'resource_optimization', 'efficiency_analysis'],
    documenter: ['technical_writing', 'api_documentation', 'user_guides', 'specification_writing'],
    generic: ['general_purpose', 'flexible_tasking']
  };

  return capabilityMap[agentType] || capabilityMap.generic;
};

/**
 * Generates processing logs based on agent activity
 */
const generateProcessingLogs = (agentType: string, activity: number): string[] => {
  const baseTime = Date.now();
  const logs = [];

  if (activity > 0.8) {
    logs.push(
      `[${new Date(baseTime - 30000).toISOString()}] Started high-priority task`,
      `[${new Date(baseTime - 20000).toISOString()}] Processing at 95% efficiency`,
      `[${new Date(baseTime - 10000).toISOString()}] Coordinating with 3 other agents`,
      `[${new Date(baseTime).toISOString()}] Completing final verification steps`
    );
  } else if (activity > 0.5) {
    logs.push(
      `[${new Date(baseTime - 45000).toISOString()}] Received new task assignment`,
      `[${new Date(baseTime - 15000).toISOString()}] Processing task at normal pace`,
      `[${new Date(baseTime).toISOString()}] Awaiting input from dependencies`
    );
  } else if (activity > 0.2) {
    logs.push(
      `[${new Date(baseTime - 60000).toISOString()}] Monitoring system metrics`,
      `[${new Date(baseTime).toISOString()}] Performing background optimization`
    );
  } else {
    logs.push(`[${new Date(baseTime).toISOString()}] Idle - ready for new tasks`);
  }

  return logs;
};

/**
 * Generates realistic 3D positions for different agent types
 */
const generatePositionByType = (agentType: string, index: number): Vec3 => {
  const baseRadius = 50;
  const angle = (index * Math.PI * 2) / 8; // Distribute around circle

  // Coordinators stay more central, specialists spread out
  const radiusMultiplier = agentType === 'coordinator' ? 0.5 :
                          agentType === 'architect' ? 0.7 :
                          agentType === 'generic' ? 1.2 : 1.0;

  return {
    x: Math.cos(angle) * baseRadius * radiusMultiplier + (Math.random() - 0.5) * 20,
    y: Math.sin(angle) * baseRadius * radiusMultiplier + (Math.random() - 0.5) * 20,
    z: (Math.random() - 0.5) * 30
  };
};

/**
 * Main function to generate comprehensive mock agent data
 */
export const generateMockAgentData = (count: number = 12): MockAgentStatus[] => {
  const agentTypes = ['coordinator', 'researcher', 'coder', 'analyst', 'architect', 'tester', 'reviewer', 'optimizer', 'documenter'];
  const statuses = ['idle', 'busy', 'active', 'initializing'];
  const agentModes = ['centralized', 'distributed', 'strategic'];
  const swarmIds = ['swarm-alpha', 'swarm-beta', 'swarm-gamma'];

  const agents: MockAgentStatus[] = [];
  const now = new Date();
  const baseTime = now.getTime();

  for (let i = 0; i < count; i++) {
    const agentType = agentTypes[i % agentTypes.length] as AgentProfile['agentType'];
    const agentId = `agent-${agentType}-${String(i + 1).padStart(3, '0')}`;
    const status = statuses[Math.floor(Math.random() * statuses.length)];
    const activity = Math.random();
    const health = 0.7 + Math.random() * 0.3; // 70-100%
    const cpuUsage = Math.random() * 0.8; // 0-80%
    const memoryUsage = 0.2 + Math.random() * 0.6; // 20-80%
    const workload = Math.random();
    const tasksCompleted = Math.floor(Math.random() * 50) + 10;
    const failedTasks = Math.floor(Math.random() * 5);
    const successRate = tasksCompleted / (tasksCompleted + failedTasks);
    const tokens = Math.floor(Math.random() * 10000) + 1000;
    const tokenRate = Math.random() * 2; // 0-2 tokens per second
    const age = Math.floor(Math.random() * 86400000) + 3600000; // 1-24 hours
    const createdAt = new Date(baseTime - age).toISOString();

    const currentTaskDesc = generateTaskByType(agentType);
    const capabilities = generateCapabilitiesByType(agentType);
    const position = generatePositionByType(agentType, i);
    const processingLogs = generateProcessingLogs(agentType, activity);

    const agent: MockAgentStatus = {
      // Core identification
      id: agentId,
      profile: {
        name: agentId,
        agentType: agentType,
        capabilities: capabilities,
        description: `${agentType.charAt(0).toUpperCase() + agentType.slice(1)} agent specialized in ${capabilities[0].replace('_', ' ')}`,
        version: '1.0.0',
        tags: ['general', 'production', agentType]
      },
      status,

      // Task information
      activeTasksCount: currentTaskDesc ? 1 : 0,
      completedTasksCount: tasksCompleted,
      failedTasksCount: failedTasks,
      successRate: successRate,
      timestamp: now.toISOString(),
      currentTask: currentTaskDesc ? {
        taskId: `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        description: currentTaskDesc,
        priority: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as TaskReference['priority']
      } : undefined,

      // Client compatibility fields
      type: agentType,
      currentTask: currentTaskDesc,
      capabilities,
      position,

      // Performance metrics (normalized 0-1)
      cpuUsage,
      memoryUsage,
      health,
      activity,
      tasksActive: currentTaskDesc ? 1 : 0,
      tasksCompleted: tasksCompleted,
      successRate: successRate,
      tokens,
      tokenRate,

      // Additional fields
      performanceMetrics: {
        tasksCompleted: tasksCompleted,
        successRate: successRate
      },
      tokenUsage: {
        total: tokens,
        tokenRate: tokenRate
      },
      swarmId: i < 4 ? swarmIds[0] : i < 8 ? swarmIds[1] : swarmIds[2],
      agentMode: agentModes[Math.floor(Math.random() * agentModes.length)],
      parentQueenId: i % 4 === 0 ? undefined : `agent-coordinator-${String(Math.floor(i / 4) + 1).padStart(3, '0')}`,
      processingLogs,
      createdAt,
      age,
      workload
    };

    agents.push(agent);
  }

  // Ensure we have at least one coordinator as queen
  if (agents.length > 0 && !agents.some(a => a.profile.agentType === 'coordinator')) {
    agents[0].profile.agentType = 'coordinator';
    agents[0].type = 'coordinator';
    agents[0].parentQueenId = undefined; // Coordinators are typically queens
  }

  return agents;
};

/**
 * Export pre-generated sample data for immediate use
 */
export const mockAgentData = generateMockAgentData(12);

/**
 * Generate mock swarm metadata
 */
export const generateMockSwarmMetadata = (agents: MockAgentStatus[]) => {
  const totalAgents = agents.length;
  const activeAgents = agents.filter(a => a.status === 'active' || a.status === 'busy').length;
  const totalTasks = agents.reduce((sum, a) => sum + a.activeTasksCount, 0);
  const completedTasks = agents.reduce((sum, a) => sum + a.completedTasksCount, 0);
  const avgSuccessRate = agents.reduce((sum, a) => sum + a.successRate, 0) / totalAgents;
  const totalTokens = agents.reduce((sum, a) => sum + a.tokens, 0);

  return {
    totalAgents,
    activeAgents,
    totalTasks,
    completedTasks,
    avgSuccessRate,
    totalTokens,
    timestamp: new Date().toISOString()
  };
};

/**
 * Export mock swarm metadata
 */
export const mockSwarmMetadata = generateMockSwarmMetadata(mockAgentData);

/**
 * Utility function to create specific agent types for testing
 */
export const createMockAgent = (overrides: Partial<MockAgentStatus> = {}): MockAgentStatus => {
  const baseAgent = generateMockAgentData(1)[0];
  return { ...baseAgent, ...overrides };
};

/**
 * Create a mock agent with high activity for testing active states
 */
export const createHighActivityAgent = (): MockAgentStatus => {
  return createMockAgent({
    status: 'busy',
    activity: 0.9,
    cpuUsage: 0.8,
    memoryUsage: 0.7,
    health: 0.95,
    currentTask: 'Processing high-priority computational task',
    tasksActive: 2,
    workload: 0.85
  });
};

/**
 * Create a mock agent with low activity for testing idle states
 */
export const createIdleAgent = (): MockAgentStatus => {
  return createMockAgent({
    status: 'idle',
    activity: 0.1,
    cpuUsage: 0.05,
    memoryUsage: 0.25,
    health: 0.98,
    currentTask: null,
    tasksActive: 0,
    workload: 0.1
  });
};

/**
 * Create a mock coordinator agent (potential queen)
 */
export const createCoordinatorAgent = (): MockAgentStatus => {
  return createMockAgent({
    profile: {
      name: 'coordinator-queen-001',
      agentType: 'coordinator',
      capabilities: ['task_management', 'resource_allocation', 'team_coordination', 'strategic_planning'],
      description: 'Primary coordinator agent managing swarm operations',
      version: '1.2.0',
      tags: ['queen', 'coordinator', 'primary']
    },
    type: 'coordinator',
    status: 'active',
    activity: 0.7,
    parentQueenId: undefined, // Queens have no parent
    swarmId: 'swarm-alpha',
    agentMode: 'strategic'
  });
};