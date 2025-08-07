import { v4 as uuidv4 } from 'uuid';
import { createLogger } from './logger.js';

const logger = createLogger('AgentManager');

export class AgentManager {
  constructor() {
    this.agents = new Map();
    this.swarms = new Map();
    this.agentTypes = [
      'queen', 'coordinator', 'architect', 'specialist',
      'coder', 'researcher', 'tester', 'analyst', 
      'optimizer', 'monitor'
    ];
  }
  
  // Create a new agent
  createAgent({ name, type, capabilities = [], position = null }) {
    const id = uuidv4();
    
    const agent = {
      id,
      name: name || `${type}-${id.slice(0, 8)}`,
      type: type || 'coder',
      status: 'idle',
      capabilities: capabilities.length > 0 ? capabilities : this.getDefaultCapabilities(type),
      position: position || { x: 0, y: 0, z: 0 },
      velocity: { x: 0, y: 0, z: 0 },
      force: { x: 0, y: 0, z: 0 },
      performance: {
        tasksCompleted: 0,
        successRate: 100,
        avgResponseTime: 0,
        resourceUtilization: 0
      },
      connections: [],
      metadata: {
        createdAt: new Date().toISOString(),
        lastActivity: new Date().toISOString(),
        swarmId: null,
        teamRole: null
      }
    };
    
    this.agents.set(id, agent);
    logger.info(`Created agent: ${agent.name} (${agent.type})`);
    
    return agent;
  }
  
  // Update agent state
  updateAgent(agentId, updates) {
    const agent = this.agents.get(agentId);
    if (!agent) {
      throw new Error(`Agent not found: ${agentId}`);
    }
    
    // Update status
    if (updates.status) {
      agent.status = updates.status;
    }
    
    // Update performance metrics
    if (updates.performance) {
      Object.assign(agent.performance, updates.performance);
    }
    
    // Update position
    if (updates.position) {
      Object.assign(agent.position, updates.position);
    }
    
    // Update metadata
    agent.metadata.lastActivity = new Date().toISOString();
    
    logger.info(`Updated agent ${agentId}: status=${agent.status}`);
    
    return agent;
  }
  
  // Get agent by ID
  getAgent(agentId) {
    return this.agents.get(agentId);
  }
  
  // Get all agents
  getAllAgents() {
    return Array.from(this.agents.values());
  }
  
  // Get agents by type
  getAgentsByType(type) {
    return this.getAllAgents().filter(agent => agent.type === type);
  }
  
  // Get agents by status
  getAgentsByStatus(status) {
    return this.getAllAgents().filter(agent => agent.status === status);
  }
  
  // Remove agent
  removeAgent(agentId) {
    const agent = this.agents.get(agentId);
    if (agent) {
      this.agents.delete(agentId);
      logger.info(`Removed agent: ${agent.name}`);
      return true;
    }
    return false;
  }
  
  // Create swarm
  createSwarm({ topology = 'hierarchical', name = null }) {
    const id = uuidv4();
    
    const swarm = {
      id,
      name: name || `swarm-${id.slice(0, 8)}`,
      topology,
      agents: new Set(),
      createdAt: new Date().toISOString(),
      status: 'active',
      metrics: {
        totalAgents: 0,
        activeAgents: 0,
        messageRate: 0,
        coordinationEfficiency: 1.0
      }
    };
    
    this.swarms.set(id, swarm);
    logger.info(`Created swarm: ${swarm.name} (${swarm.topology})`);
    
    return swarm;
  }
  
  // Add agent to swarm
  addAgentToSwarm(agentId, swarmId) {
    const agent = this.agents.get(agentId);
    const swarm = this.swarms.get(swarmId);
    
    if (!agent || !swarm) {
      throw new Error('Agent or swarm not found');
    }
    
    swarm.agents.add(agentId);
    agent.metadata.swarmId = swarmId;
    
    // Update swarm metrics
    swarm.metrics.totalAgents = swarm.agents.size;
    swarm.metrics.activeAgents = Array.from(swarm.agents)
      .filter(id => this.agents.get(id)?.status === 'active').length;
    
    logger.info(`Added agent ${agent.name} to swarm ${swarm.name}`);
  }
  
  // Get swarm status
  getSwarmStatus(swarmId) {
    const swarm = this.swarms.get(swarmId);
    if (!swarm) {
      throw new Error(`Swarm not found: ${swarmId}`);
    }
    
    const agents = Array.from(swarm.agents)
      .map(id => this.agents.get(id))
      .filter(Boolean);
    
    return {
      swarmId: swarm.id,
      name: swarm.name,
      topology: swarm.topology,
      status: swarm.status,
      totalAgents: agents.length,
      activeAgents: agents.filter(a => a.status === 'active').length,
      healthScore: this.calculateSwarmHealth(agents),
      coordinationEfficiency: swarm.metrics.coordinationEfficiency
    };
  }
  
  // Calculate swarm health score
  calculateSwarmHealth(agents) {
    if (agents.length === 0) return 1.0;
    
    const avgSuccessRate = agents.reduce((sum, agent) => 
      sum + agent.performance.successRate, 0) / agents.length;
    
    const activeRatio = agents.filter(a => a.status === 'active').length / agents.length;
    
    return (avgSuccessRate / 100) * 0.7 + activeRatio * 0.3;
  }
  
  // Get default capabilities for agent type
  getDefaultCapabilities(type) {
    const capabilityMap = {
      queen: ['swarm_orchestration', 'strategic_planning', 'resource_allocation'],
      coordinator: ['task_distribution', 'team_coordination', 'status_monitoring'],
      architect: ['system_design', 'component_architecture', 'optimization_planning'],
      specialist: ['domain_expertise', 'problem_solving', 'analysis'],
      coder: ['implementation', 'debugging', 'code_review'],
      researcher: ['information_gathering', 'analysis', 'documentation'],
      tester: ['test_design', 'test_execution', 'bug_reporting'],
      analyst: ['data_analysis', 'metrics_collection', 'reporting'],
      optimizer: ['performance_tuning', 'resource_optimization', 'bottleneck_analysis'],
      monitor: ['system_monitoring', 'alert_management', 'health_checking']
    };
    
    return capabilityMap[type] || ['general_processing'];
  }
  
  // Update agent connections based on communication
  updateAgentConnections(fromId, toId, messageCount = 1) {
    const fromAgent = this.agents.get(fromId);
    const toAgent = this.agents.get(toId);
    
    if (!fromAgent || !toAgent) return;
    
    // Update from agent connections
    let fromConnection = fromAgent.connections.find(c => c.agentId === toId);
    if (!fromConnection) {
      fromConnection = { agentId: toId, strength: 0, messageCount: 0 };
      fromAgent.connections.push(fromConnection);
    }
    fromConnection.messageCount += messageCount;
    fromConnection.strength = Math.min(1, fromConnection.messageCount / 100);
    
    // Update to agent connections
    let toConnection = toAgent.connections.find(c => c.agentId === fromId);
    if (!toConnection) {
      toConnection = { agentId: fromId, strength: 0, messageCount: 0 };
      toAgent.connections.push(toConnection);
    }
    toConnection.messageCount += messageCount;
    toConnection.strength = Math.min(1, toConnection.messageCount / 100);
  }
  
  // Get agent metrics
  getAgentMetrics(agentIds = null) {
    const agents = agentIds 
      ? agentIds.map(id => this.agents.get(id)).filter(Boolean)
      : this.getAllAgents();
    
    return agents.map(agent => ({
      agentId: agent.id,
      name: agent.name,
      type: agent.type,
      performanceScore: agent.performance.successRate / 100,
      tasksCompleted: agent.performance.tasksCompleted,
      successRate: agent.performance.successRate / 100,
      resourceUtilization: agent.performance.resourceUtilization,
      connectionCount: agent.connections.length,
      avgConnectionStrength: agent.connections.reduce((sum, c) => sum + c.strength, 0) / 
        (agent.connections.length || 1)
    }));
  }
}