/**
 * Neural Helpers - Utility functions for neural AI capabilities
 */

import { CognitivePattern, AgentType, WorkflowStep, TaskPriority } from '../types/neural';

// Type definitions for neural capabilities
export interface NeuralConfig {
  architecture: 'feedforward' | 'lstm' | 'gan' | 'autoencoder' | 'transformer';
  training: {
    epochs: number;
    batchSize: number;
    learningRate: number;
    optimizer: 'adam' | 'sgd' | 'rmsprop' | 'adagrad';
  };
  divergent?: {
    enabled: boolean;
    pattern: 'lateral' | 'quantum' | 'chaotic' | 'associative' | 'evolutionary';
    factor: number;
  };
}

export interface SwarmAgent {
  id: string;
  type: AgentType;
  status: 'active' | 'idle' | 'busy' | 'error';
  capabilities: string[];
  cognitivePattern?: CognitivePattern;
  performance: {
    tasksCompleted: number;
    successRate: number;
    averageTime: number;
  };
}

export interface WorkflowExecution {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime: Date;
  endTime?: Date;
  agents: string[];
  results?: any;
}

// Cognitive Pattern Management
export const cognitivePatterns: Record<CognitivePattern, {
  description: string;
  strengths: string[];
  bestFor: string[]
}> = {
  convergent: {
    description: 'Focused, logical thinking that narrows down to the best solution',
    strengths: ['Problem solving', 'Analysis', 'Optimization'],
    bestFor: ['Code debugging', 'Performance optimization', 'System design']
  },
  divergent: {
    description: 'Creative, exploratory thinking that generates multiple solutions',
    strengths: ['Innovation', 'Brainstorming', 'Alternative approaches'],
    bestFor: ['Feature ideation', 'Architecture exploration', 'Creative solutions']
  },
  lateral: {
    description: 'Non-linear thinking that makes unexpected connections',
    strengths: ['Pattern recognition', 'Cross-domain insights', 'Novel approaches'],
    bestFor: ['Complex problem solving', 'Integration challenges', 'Innovation']
  },
  systems: {
    description: 'Holistic thinking that considers interconnections and dependencies',
    strengths: ['Architecture design', 'Integration', 'Scalability'],
    bestFor: ['System architecture', 'Microservices design', 'Platform development']
  },
  critical: {
    description: 'Analytical thinking that evaluates and validates solutions',
    strengths: ['Quality assurance', 'Risk assessment', 'Validation'],
    bestFor: ['Code review', 'Security analysis', 'Testing strategies']
  },
  adaptive: {
    description: 'Flexible thinking that adjusts approach based on context',
    strengths: ['Learning', 'Adaptation', 'Context awareness'],
    bestFor: ['Dynamic optimization', 'User experience', 'Personalization']
  }
};

// Agent Type Management
export const agentTypes: Record<AgentType, {
  description: string;
  defaultCapabilities: string[];
  preferredPatterns: CognitivePattern[];
}> = {
  researcher: {
    description: 'Analyzes requirements, patterns, and best practices',
    defaultCapabilities: ['analysis', 'research', 'documentation'],
    preferredPatterns: ['divergent', 'lateral', 'systems']
  },
  coder: {
    description: 'Implements features and writes production code',
    defaultCapabilities: ['implementation', 'coding', 'debugging'],
    preferredPatterns: ['convergent', 'critical', 'adaptive']
  },
  analyst: {
    description: 'Evaluates performance, metrics, and system behavior',
    defaultCapabilities: ['analysis', 'metrics', 'optimization'],
    preferredPatterns: ['critical', 'systems', 'convergent']
  },
  optimizer: {
    description: 'Improves performance and efficiency of systems',
    defaultCapabilities: ['optimization', 'performance', 'scaling'],
    preferredPatterns: ['convergent', 'systems', 'adaptive']
  },
  coordinator: {
    description: 'Orchestrates workflows and manages agent interactions',
    defaultCapabilities: ['coordination', 'workflow', 'communication'],
    preferredPatterns: ['systems', 'adaptive', 'lateral']
  }
};

// Neural Configuration Helpers
export const createNeuralConfig = (
  architecture: NeuralConfig['architecture'],
  overrides?: Partial<NeuralConfig>
): NeuralConfig => {
  const baseConfigs: Record<NeuralConfig['architecture'], NeuralConfig> = {
    feedforward: {
      architecture: 'feedforward',
      training: { epochs: 100, batchSize: 32, learningRate: 0.001, optimizer: 'adam' }
    },
    lstm: {
      architecture: 'lstm',
      training: { epochs: 50, batchSize: 16, learningRate: 0.01, optimizer: 'rmsprop' }
    },
    transformer: {
      architecture: 'transformer',
      training: { epochs: 20, batchSize: 8, learningRate: 0.0001, optimizer: 'adam' },
      divergent: { enabled: true, pattern: 'associative', factor: 0.7 }
    },
    gan: {
      architecture: 'gan',
      training: { epochs: 200, batchSize: 64, learningRate: 0.0002, optimizer: 'adam' },
      divergent: { enabled: true, pattern: 'chaotic', factor: 0.8 }
    },
    autoencoder: {
      architecture: 'autoencoder',
      training: { epochs: 150, batchSize: 32, learningRate: 0.001, optimizer: 'adam' }
    }
  };

  return { ...baseConfigs[architecture], ...overrides };
};

// Agent Management Helpers
export const createAgent = (
  type: AgentType,
  customConfig?: Partial<SwarmAgent>
): Omit<SwarmAgent, 'id'> => {
  const agentConfig = agentTypes[type];

  return {
    type,
    status: 'idle',
    capabilities: agentConfig.defaultCapabilities,
    cognitivePattern: agentConfig.preferredPatterns[0],
    performance: {
      tasksCompleted: 0,
      successRate: 0,
      averageTime: 0
    },
    ...customConfig
  };
};

export const getOptimalCognitivePattern = (taskType: string): CognitivePattern => {
  const taskPatternMap: Record<string, CognitivePattern> = {
    'bug-fix': 'convergent',
    'feature-development': 'divergent',
    'system-design': 'systems',
    'code-review': 'critical',
    'optimization': 'convergent',
    'research': 'lateral',
    'integration': 'systems',
    'testing': 'critical',
    'innovation': 'divergent',
    'adaptation': 'adaptive'
  };

  return taskPatternMap[taskType] || 'adaptive';
};

// Workflow Management
export const createWorkflowStep = (
  name: string,
  agentType: AgentType,
  dependencies?: string[],
  priority: TaskPriority = 'medium'
): WorkflowStep => ({
  id: generateId(),
  name,
  agentType,
  dependencies: dependencies || [],
  priority,
  status: 'pending',
  estimatedDuration: 0
});

export const calculateWorkflowComplexity = (steps: WorkflowStep[]): number => {
  const baseComplexity = steps.length;
  const dependencyComplexity = steps.reduce((acc, step) => acc + step.dependencies.length, 0);
  const priorityComplexity = steps.filter(s => s.priority === 'high').length * 1.5;

  return baseComplexity + dependencyComplexity * 0.5 + priorityComplexity;
};

export const optimizeWorkflowExecution = (steps: WorkflowStep[]): WorkflowStep[][] => {
  const sortedSteps = [...steps].sort((a, b) => {
    const priorityWeight = { high: 3, medium: 2, low: 1 };
    return priorityWeight[b.priority] - priorityWeight[a.priority];
  });

  const phases: WorkflowStep[][] = [];
  const completed = new Set<string>();

  while (sortedSteps.some(step => !completed.has(step.id))) {
    const currentPhase: WorkflowStep[] = [];

    for (const step of sortedSteps) {
      if (completed.has(step.id)) continue;

      const dependenciesMet = step.dependencies.every(dep => completed.has(dep));
      if (dependenciesMet) {
        currentPhase.push(step);
        completed.add(step.id);
      }
    }

    if (currentPhase.length > 0) {
      phases.push(currentPhase);
    } else {
      // Break circular dependencies
      const remaining = sortedSteps.filter(step => !completed.has(step.id));
      if (remaining.length > 0) {
        phases.push([remaining[0]]);
        completed.add(remaining[0].id);
      }
    }
  }

  return phases;
};

// Performance Analytics
export const calculateAgentEfficiency = (agent: SwarmAgent): number => {
  const { tasksCompleted, successRate, averageTime } = agent.performance;

  if (tasksCompleted === 0) return 0;

  const completionScore = Math.min(tasksCompleted / 10, 1) * 30;
  const qualityScore = successRate * 40;
  const speedScore = averageTime > 0 ? Math.max(0, 30 - (averageTime / 1000 / 60)) : 0;

  return Math.round(completionScore + qualityScore + speedScore);
};

export const generateSwarmMetrics = (agents: SwarmAgent[]) => {
  const activeAgents = agents.filter(a => a.status === 'active').length;
  const totalTasks = agents.reduce((sum, a) => sum + a.performance.tasksCompleted, 0);
  const averageSuccess = agents.length > 0
    ? agents.reduce((sum, a) => sum + a.performance.successRate, 0) / agents.length
    : 0;
  const averageEfficiency = agents.length > 0
    ? agents.reduce((sum, a) => sum + calculateAgentEfficiency(a), 0) / agents.length
    : 0;

  return {
    totalAgents: agents.length,
    activeAgents,
    totalTasks,
    averageSuccess: Math.round(averageSuccess * 100),
    averageEfficiency: Math.round(averageEfficiency),
    cognitiveDistribution: getCognitiveDistribution(agents)
  };
};

// Utility Functions
export const generateId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

export const formatDuration = (milliseconds: number): string => {
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
};

export const formatTimestamp = (date: Date): string => {
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

export const getCognitiveDistribution = (agents: SwarmAgent[]) => {
  const distribution: Record<CognitivePattern, number> = {
    convergent: 0,
    divergent: 0,
    lateral: 0,
    systems: 0,
    critical: 0,
    adaptive: 0
  };

  agents.forEach(agent => {
    if (agent.cognitivePattern) {
      distribution[agent.cognitivePattern]++;
    }
  });

  return distribution;
};

// Validation Helpers
export const validateNeuralConfig = (config: NeuralConfig): string[] => {
  const errors: string[] = [];

  if (config.training.epochs <= 0) {
    errors.push('Epochs must be greater than 0');
  }

  if (config.training.batchSize <= 0) {
    errors.push('Batch size must be greater than 0');
  }

  if (config.training.learningRate <= 0 || config.training.learningRate > 1) {
    errors.push('Learning rate must be between 0 and 1');
  }

  if (config.divergent?.enabled && (!config.divergent.factor || config.divergent.factor <= 0)) {
    errors.push('Divergent factor must be greater than 0 when enabled');
  }

  return errors;
};

export const validateWorkflow = (steps: WorkflowStep[]): string[] => {
  const errors: string[] = [];
  const stepIds = new Set(steps.map(s => s.id));

  steps.forEach(step => {
    step.dependencies.forEach(dep => {
      if (!stepIds.has(dep)) {
        errors.push(`Step "${step.name}" has invalid dependency: ${dep}`);
      }
    });
  });

  return errors;
};

// API Helpers
export const apiRequest = async <T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> => {
  const response = await fetch(`/api/neural${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    ...options
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.statusText}`);
  }

  return response.json();
};

export const createNeuralModel = async (config: NeuralConfig) => {
  return apiRequest('/models', {
    method: 'POST',
    body: JSON.stringify(config)
  });
};

export const executeWorkflow = async (steps: WorkflowStep[]) => {
  return apiRequest('/workflows', {
    method: 'POST',
    body: JSON.stringify({ steps })
  });
};

export const getSwarmStatus = async () => {
  return apiRequest('/swarm/status');
};

export const spawnAgent = async (agentConfig: Partial<SwarmAgent>) => {
  return apiRequest('/agents', {
    method: 'POST',
    body: JSON.stringify(agentConfig)
  });
};

// Error Handling
export class NeuralError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: any
  ) {
    super(message);
    this.name = 'NeuralError';
  }
}

export const handleNeuralError = (error: unknown): NeuralError => {
  if (error instanceof NeuralError) {
    return error;
  }

  if (error instanceof Error) {
    return new NeuralError(error.message, 'UNKNOWN_ERROR', error);
  }

  return new NeuralError('An unknown error occurred', 'UNKNOWN_ERROR', error);
};

// Export all utilities
export const neuralHelpers = {
  cognitivePatterns,
  agentTypes,
  createNeuralConfig,
  createAgent,
  getOptimalCognitivePattern,
  createWorkflowStep,
  calculateWorkflowComplexity,
  optimizeWorkflowExecution,
  calculateAgentEfficiency,
  generateSwarmMetrics,
  generateId,
  formatDuration,
  formatTimestamp,
  getCognitiveDistribution,
  validateNeuralConfig,
  validateWorkflow,
  apiRequest,
  createNeuralModel,
  executeWorkflow,
  getSwarmStatus,
  spawnAgent,
  handleNeuralError
};