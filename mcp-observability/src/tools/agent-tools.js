import { createLogger } from '../logger.js';

const logger = createLogger('AgentTools');

export function agentTools(agentManager, physicsEngine) {
  return {
    'agent.create': {
      description: 'Create a new agent in the swarm with spring physics positioning',
      inputSchema: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Agent name' },
          type: { 
            type: 'string', 
            enum: ['queen', 'coordinator', 'architect', 'specialist', 'coder', 'researcher', 'tester', 'analyst', 'optimizer', 'monitor'],
            description: 'Agent type determines behavior and capabilities'
          },
          capabilities: { 
            type: 'array', 
            items: { type: 'string' },
            description: 'List of agent capabilities'
          },
          position: {
            type: 'object',
            properties: {
              x: { type: 'number' },
              y: { type: 'number' },
              z: { type: 'number' }
            },
            description: 'Initial 3D position (optional)'
          }
        },
        required: ['name', 'type']
      },
      handler: async (args) => {
        try {
          const agent = agentManager.createAgent(args);
          
          // Update physics engine with new agent
          physicsEngine.update(1/60, agentManager.getAllAgents());
          
          logger.info(`Created agent: ${agent.name} (${agent.type})`);
          
          return {
            success: true,
            agent: {
              id: agent.id,
              name: agent.name,
              type: agent.type,
              status: agent.status,
              capabilities: agent.capabilities,
              position: agent.position
            }
          };
        } catch (error) {
          logger.error('Failed to create agent:', error);
          throw error;
        }
      }
    },
    
    'agent.update': {
      description: 'Update agent state and trigger physics recalculation',
      inputSchema: {
        type: 'object',
        properties: {
          agentId: { type: 'string', description: 'Agent ID' },
          status: { 
            type: 'string',
            enum: ['active', 'busy', 'idle', 'error'],
            description: 'New agent status'
          },
          performance: {
            type: 'object',
            properties: {
              successRate: { type: 'number', minimum: 0, maximum: 100 },
              resourceUtilization: { type: 'number', minimum: 0, maximum: 1 }
            }
          }
        },
        required: ['agentId']
      },
      handler: async (args) => {
        try {
          const agent = agentManager.updateAgent(args.agentId, args);
          
          // Trigger physics update
          physicsEngine.update(1/60, agentManager.getAllAgents());
          
          return {
            success: true,
            agent: {
              id: agent.id,
              name: agent.name,
              status: agent.status,
              performance: agent.performance
            }
          };
        } catch (error) {
          logger.error('Failed to update agent:', error);
          throw error;
        }
      }
    },
    
    'agent.metrics': {
      description: 'Get detailed metrics for specific agents',
      inputSchema: {
        type: 'object',
        properties: {
          agentIds: { 
            type: 'array',
            items: { type: 'string' },
            description: 'List of agent IDs (empty for all agents)'
          },
          includeHistory: { 
            type: 'boolean',
            description: 'Include performance history'
          }
        }
      },
      handler: async (args) => {
        try {
          const metrics = agentManager.getAgentMetrics(args.agentIds);
          
          return {
            success: true,
            metrics,
            timestamp: new Date().toISOString()
          };
        } catch (error) {
          logger.error('Failed to get agent metrics:', error);
          throw error;
        }
      }
    },
    
    'agent.list': {
      description: 'List all agents in the system',
      inputSchema: {
        type: 'object',
        properties: {
          filter: {
            type: 'object',
            properties: {
              type: { type: 'string' },
              status: { type: 'string' }
            }
          }
        }
      },
      handler: async (args) => {
        try {
          let agents = agentManager.getAllAgents();
          
          // Apply filters
          if (args.filter) {
            if (args.filter.type) {
              agents = agents.filter(a => a.type === args.filter.type);
            }
            if (args.filter.status) {
              agents = agents.filter(a => a.status === args.filter.status);
            }
          }
          
          return {
            success: true,
            agents: agents.map(a => ({
              id: a.id,
              name: a.name,
              type: a.type,
              status: a.status,
              position: a.position,
              connectionCount: a.connections.length
            })),
            count: agents.length
          };
        } catch (error) {
          logger.error('Failed to list agents:', error);
          throw error;
        }
      }
    },
    
    'agent.remove': {
      description: 'Remove an agent from the swarm',
      inputSchema: {
        type: 'object',
        properties: {
          agentId: { type: 'string', description: 'Agent ID to remove' }
        },
        required: ['agentId']
      },
      handler: async (args) => {
        try {
          const success = agentManager.removeAgent(args.agentId);
          
          if (success) {
            // Update physics after removal
            physicsEngine.update(1/60, agentManager.getAllAgents());
          }
          
          return {
            success,
            message: success ? 'Agent removed successfully' : 'Agent not found'
          };
        } catch (error) {
          logger.error('Failed to remove agent:', error);
          throw error;
        }
      }
    },
    
    'agent.spawn': {
      description: 'Spawn multiple agents with optimal positioning',
      inputSchema: {
        type: 'object',
        properties: {
          count: { type: 'number', minimum: 1, maximum: 50 },
          type: { type: 'string' },
          namePrefix: { type: 'string' }
        },
        required: ['count', 'type']
      },
      handler: async (args) => {
        try {
          const agents = [];
          const angleStep = (2 * Math.PI) / args.count;
          const radius = 10 + args.count * 2;
          
          for (let i = 0; i < args.count; i++) {
            const angle = i * angleStep;
            const position = {
              x: Math.cos(angle) * radius,
              y: (Math.random() - 0.5) * 10,
              z: Math.sin(angle) * radius
            };
            
            const agent = agentManager.createAgent({
              name: `${args.namePrefix || args.type}-${i + 1}`,
              type: args.type,
              position
            });
            
            agents.push(agent);
          }
          
          // Update physics for all new agents
          physicsEngine.update(1/60, agentManager.getAllAgents());
          
          return {
            success: true,
            agents: agents.map(a => ({
              id: a.id,
              name: a.name,
              type: a.type,
              position: a.position
            })),
            count: agents.length
          };
        } catch (error) {
          logger.error('Failed to spawn agents:', error);
          throw error;
        }
      }
    }
  };
}