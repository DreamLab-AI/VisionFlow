import { createLogger } from '../logger.js';

const logger = createLogger('SwarmTools');

export function swarmTools(agentManager, physicsEngine) {
  return {
    'swarm.initialize': {
      description: 'Initialize a new swarm with topology and physics parameters',
      inputSchema: {
        type: 'object',
        properties: {
          topology: {
            type: 'string',
            enum: ['hierarchical', 'mesh', 'ring', 'star'],
            description: 'Swarm topology pattern'
          },
          physicsConfig: {
            type: 'object',
            properties: {
              springStrength: { type: 'number', minimum: 0, maximum: 1 },
              damping: { type: 'number', minimum: 0, maximum: 1 },
              linkDistance: { type: 'number', minimum: 1, maximum: 50 }
            },
            description: 'Physics engine configuration'
          },
          agentConfig: {
            type: 'object',
            properties: {
              coordinatorCount: { type: 'number', minimum: 1, maximum: 5 },
              workerTypes: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    type: { type: 'string' },
                    count: { type: 'number' }
                  }
                }
              }
            }
          }
        },
        required: ['topology']
      },
      handler: async (args) => {
        try {
          // Create swarm
          const swarm = agentManager.createSwarm({
            topology: args.topology,
            name: `${args.topology}-swarm`
          });
          
          // Update physics config if provided
          if (args.physicsConfig) {
            physicsEngine.updateConfig(args.physicsConfig);
          }
          
          // Create default agents based on topology
          const agents = [];
          
          // Always create a coordinator
          const coordinator = agentManager.createAgent({
            name: 'Swarm Coordinator',
            type: 'coordinator',
            position: { x: 0, y: 0, z: 0 }
          });
          agentManager.addAgentToSwarm(coordinator.id, swarm.id);
          agents.push(coordinator);
          
          // Create additional agents based on config
          if (args.agentConfig?.workerTypes) {
            for (const workerType of args.agentConfig.workerTypes) {
              for (let i = 0; i < workerType.count; i++) {
                const agent = agentManager.createAgent({
                  name: `${workerType.type}-${i + 1}`,
                  type: workerType.type
                });
                agentManager.addAgentToSwarm(agent.id, swarm.id);
                agents.push(agent);
              }
            }
          } else {
            // Default agent distribution
            const defaultTypes = ['coder', 'researcher', 'tester', 'analyst'];
            defaultTypes.forEach((type, i) => {
              const agent = agentManager.createAgent({
                name: `${type}-1`,
                type
              });
              agentManager.addAgentToSwarm(agent.id, swarm.id);
              agents.push(agent);
            });
          }
          
          // Apply topology-specific positioning
          applyTopologyPositioning(agents, args.topology);
          
          // Update physics
          physicsEngine.update(1/60, agentManager.getAllAgents());
          
          logger.info(`Initialized ${args.topology} swarm with ${agents.length} agents`);
          
          return {
            success: true,
            swarmId: swarm.id,
            topology: args.topology,
            agentCount: agents.length,
            agents: agents.map(a => ({
              id: a.id,
              name: a.name,
              type: a.type
            }))
          };
        } catch (error) {
          logger.error('Failed to initialize swarm:', error);
          throw error;
        }
      }
    },
    
    'swarm.status': {
      description: 'Get comprehensive swarm status including physics state',
      inputSchema: {
        type: 'object',
        properties: {
          swarmId: { type: 'string', description: 'Swarm ID (optional)' },
          includeAgents: { type: 'boolean', default: true },
          includeMetrics: { type: 'boolean', default: true },
          includePhysics: { type: 'boolean', default: true }
        }
      },
      handler: async (args) => {
        try {
          // Get all swarms if no specific ID
          const swarmIds = args.swarmId 
            ? [args.swarmId]
            : Array.from(agentManager.swarms.keys());
          
          const swarmStatuses = [];
          
          for (const swarmId of swarmIds) {
            const status = agentManager.getSwarmStatus(swarmId);
            
            const swarmData = {
              ...status,
              timestamp: new Date().toISOString()
            };
            
            if (args.includeAgents) {
              const swarmAgents = Array.from(agentManager.swarms.get(swarmId).agents)
                .map(id => agentManager.getAgent(id))
                .filter(Boolean);
              
              swarmData.agents = swarmAgents.map(a => ({
                id: a.id,
                name: a.name,
                type: a.type,
                status: a.status,
                position: a.position
              }));
            }
            
            if (args.includePhysics) {
              const agents = agentManager.getAllAgents();
              swarmData.physics = physicsEngine.getSnapshot(agents);
            }
            
            swarmStatuses.push(swarmData);
          }
          
          return {
            success: true,
            swarms: swarmStatuses,
            count: swarmStatuses.length
          };
        } catch (error) {
          logger.error('Failed to get swarm status:', error);
          throw error;
        }
      }
    },
    
    'swarm.monitor': {
      description: 'Get real-time swarm monitoring data',
      inputSchema: {
        type: 'object',
        properties: {
          swarmId: { type: 'string' },
          includeMessageFlow: { type: 'boolean', default: true },
          includePerformance: { type: 'boolean', default: true }
        }
      },
      handler: async (args) => {
        try {
          const swarmId = args.swarmId || Array.from(agentManager.swarms.keys())[0];
          if (!swarmId) {
            throw new Error('No swarm found');
          }
          
          const swarm = agentManager.swarms.get(swarmId);
          const agents = Array.from(swarm.agents)
            .map(id => agentManager.getAgent(id))
            .filter(Boolean);
          
          const monitorData = {
            swarmId,
            timestamp: new Date().toISOString(),
            agentStates: {}
          };
          
          // Collect agent states
          agents.forEach(agent => {
            monitorData.agentStates[agent.id] = {
              status: agent.status,
              performance: agent.performance,
              connections: agent.connections.length
            };
          });
          
          // Add swarm-level metrics
          monitorData.swarmMetrics = {
            totalAgents: agents.length,
            activeAgents: agents.filter(a => a.status === 'active').length,
            avgSuccessRate: agents.reduce((sum, a) => sum + a.performance.successRate, 0) / agents.length,
            totalConnections: agents.reduce((sum, a) => sum + a.connections.length, 0) / 2
          };
          
          return {
            success: true,
            monitoring: monitorData
          };
        } catch (error) {
          logger.error('Failed to monitor swarm:', error);
          throw error;
        }
      }
    },
    
    'swarm.reconfigure': {
      description: 'Reconfigure swarm topology and update physics',
      inputSchema: {
        type: 'object',
        properties: {
          swarmId: { type: 'string' },
          newTopology: {
            type: 'string',
            enum: ['hierarchical', 'mesh', 'ring', 'star']
          },
          physicsConfig: {
            type: 'object',
            properties: {
              springStrength: { type: 'number' },
              damping: { type: 'number' },
              linkDistance: { type: 'number' }
            }
          }
        },
        required: ['swarmId', 'newTopology']
      },
      handler: async (args) => {
        try {
          const swarm = agentManager.swarms.get(args.swarmId);
          if (!swarm) {
            throw new Error('Swarm not found');
          }
          
          // Update topology
          swarm.topology = args.newTopology;
          
          // Update physics if provided
          if (args.physicsConfig) {
            physicsEngine.updateConfig(args.physicsConfig);
          }
          
          // Get swarm agents and reposition
          const agents = Array.from(swarm.agents)
            .map(id => agentManager.getAgent(id))
            .filter(Boolean);
          
          applyTopologyPositioning(agents, args.newTopology);
          
          // Force physics update
          physicsEngine.update(1/60, agentManager.getAllAgents());
          
          return {
            success: true,
            swarmId: args.swarmId,
            newTopology: args.newTopology,
            agentCount: agents.length
          };
        } catch (error) {
          logger.error('Failed to reconfigure swarm:', error);
          throw error;
        }
      }
    }
  };
}

// Helper function to apply topology-specific positioning
function applyTopologyPositioning(agents, topology) {
  switch (topology) {
    case 'hierarchical':
      // Coordinator at top, others in layers below
      const coordinator = agents.find(a => a.type === 'coordinator');
      if (coordinator) {
        coordinator.position = { x: 0, y: 10, z: 0 };
      }
      
      const workers = agents.filter(a => a.type !== 'coordinator');
      const layerSize = Math.ceil(Math.sqrt(workers.length));
      workers.forEach((agent, i) => {
        const row = Math.floor(i / layerSize);
        const col = i % layerSize;
        agent.position = {
          x: (col - layerSize / 2) * 5,
          y: -row * 5,
          z: (row % 2) * 3
        };
      });
      break;
      
    case 'mesh':
      // All agents distributed evenly in 3D space
      const meshRadius = Math.max(10, agents.length * 2);
      agents.forEach((agent, i) => {
        const phi = Math.acos(1 - 2 * (i / agents.length));
        const theta = Math.sqrt(agents.length * Math.PI) * phi;
        agent.position = {
          x: meshRadius * Math.sin(phi) * Math.cos(theta),
          y: meshRadius * Math.sin(phi) * Math.sin(theta),
          z: meshRadius * Math.cos(phi)
        };
      });
      break;
      
    case 'ring':
      // Agents in a circle
      const ringRadius = Math.max(10, agents.length * 2);
      agents.forEach((agent, i) => {
        const angle = (i / agents.length) * 2 * Math.PI;
        agent.position = {
          x: ringRadius * Math.cos(angle),
          y: 0,
          z: ringRadius * Math.sin(angle)
        };
      });
      break;
      
    case 'star':
      // Coordinator in center, others radiating out
      const center = agents.find(a => a.type === 'coordinator');
      if (center) {
        center.position = { x: 0, y: 0, z: 0 };
      }
      
      const spokes = agents.filter(a => a.type !== 'coordinator');
      const starRadius = Math.max(10, spokes.length * 3);
      spokes.forEach((agent, i) => {
        const angle = (i / spokes.length) * 2 * Math.PI;
        agent.position = {
          x: starRadius * Math.cos(angle),
          y: (Math.random() - 0.5) * 5,
          z: starRadius * Math.sin(angle)
        };
      });
      break;
  }
}