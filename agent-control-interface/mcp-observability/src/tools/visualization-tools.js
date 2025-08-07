import { createLogger } from '../logger.js';

const logger = createLogger('VisualizationTools');

export function visualizationTools(agentManager, physicsEngine) {
  return {
    'visualization.snapshot': {
      description: 'Get current visualization state for rendering',
      inputSchema: {
        type: 'object',
        properties: {
          includePositions: { type: 'boolean', default: true },
          includeVelocities: { type: 'boolean', default: true },
          includeForces: { type: 'boolean', default: false },
          includeConnections: { type: 'boolean', default: true },
          agentFilter: {
            type: 'array',
            items: { type: 'string' },
            description: 'Filter specific agents'
          }
        }
      },
      handler: async (args) => {
        try {
          let agents = agentManager.getAllAgents();
          
          // Apply filter if provided
          if (args.agentFilter && args.agentFilter.length > 0) {
            agents = agents.filter(a => args.agentFilter.includes(a.id));
          }
          
          const snapshot = {
            timestamp: new Date().toISOString(),
            frameCount: physicsEngine.frameCount,
            agentCount: agents.length,
            agents: []
          };
          
          // Build agent data based on requested fields
          snapshot.agents = agents.map(agent => {
            const agentData = {
              id: agent.id,
              name: agent.name,
              type: agent.type,
              status: agent.status
            };
            
            if (args.includePositions) {
              agentData.position = { ...agent.position };
            }
            
            if (args.includeVelocities) {
              agentData.velocity = { ...agent.velocity };
            }
            
            if (args.includeForces) {
              agentData.force = { ...agent.force };
            }
            
            if (args.includeConnections) {
              agentData.connections = agent.connections.map(c => ({
                targetId: c.agentId,
                strength: c.strength,
                messageCount: c.messageCount
              }));
            }
            
            // Always include visual properties
            agentData.visual = {
              color: getAgentColor(agent.type, agent.status),
              size: getAgentSize(agent.type, agent.performance),
              opacity: agent.status === 'error' ? 0.5 : 1.0,
              glow: agent.status === 'active' ? true : false
            };
            
            return agentData;
          });
          
          // Add physics configuration
          snapshot.physics = {
            config: physicsEngine.config,
            gravity: { x: 0, y: 0, z: 0 }
          };
          
          return {
            success: true,
            snapshot
          };
        } catch (error) {
          logger.error('Failed to get visualization snapshot:', error);
          throw error;
        }
      }
    },
    
    'visualization.animate': {
      description: 'Generate animation sequence for state transitions',
      inputSchema: {
        type: 'object',
        properties: {
          fromState: { type: 'string', description: 'Starting state ID' },
          toState: { type: 'string', description: 'Target state ID' },
          duration: { type: 'number', description: 'Animation duration in seconds', default: 2 },
          fps: { type: 'number', description: 'Frames per second', default: 60 },
          easing: {
            type: 'string',
            enum: ['linear', 'easeIn', 'easeOut', 'easeInOut'],
            default: 'easeInOut'
          }
        },
        required: ['fromState', 'toState']
      },
      handler: async (args) => {
        try {
          const frames = [];
          const frameCount = Math.floor(args.duration * args.fps);
          
          // Get current agent positions as starting state
          const agents = agentManager.getAllAgents();
          const startPositions = new Map(agents.map(a => [a.id, { ...a.position }]));
          
          // Generate target positions based on state
          const targetPositions = generateTargetPositions(agents, args.toState);
          
          // Generate animation frames
          for (let i = 0; i <= frameCount; i++) {
            const t = i / frameCount;
            const easedT = applyEasing(t, args.easing);
            
            const frame = {
              frameNumber: i,
              time: (i / args.fps),
              agents: agents.map(agent => {
                const start = startPositions.get(agent.id);
                const target = targetPositions.get(agent.id);
                
                return {
                  id: agent.id,
                  position: {
                    x: lerp(start.x, target.x, easedT),
                    y: lerp(start.y, target.y, easedT),
                    z: lerp(start.z, target.z, easedT)
                  }
                };
              })
            };
            
            frames.push(frame);
          }
          
          return {
            success: true,
            animation: {
              duration: args.duration,
              fps: args.fps,
              frameCount: frames.length,
              frames: frames
            }
          };
        } catch (error) {
          logger.error('Failed to generate animation:', error);
          throw error;
        }
      }
    },
    
    'visualization.layout': {
      description: 'Apply predefined layout to agent positions',
      inputSchema: {
        type: 'object',
        properties: {
          layout: {
            type: 'string',
            enum: ['grid', 'circle', 'spiral', 'random', 'hierarchical', 'force'],
            description: 'Layout pattern to apply'
          },
          spacing: { type: 'number', default: 5 },
          center: {
            type: 'object',
            properties: {
              x: { type: 'number', default: 0 },
              y: { type: 'number', default: 0 },
              z: { type: 'number', default: 0 }
            }
          }
        },
        required: ['layout']
      },
      handler: async (args) => {
        try {
          const agents = agentManager.getAllAgents();
          
          switch (args.layout) {
            case 'grid':
              applyGridLayout(agents, args.spacing, args.center);
              break;
              
            case 'circle':
              applyCircleLayout(agents, args.spacing * agents.length, args.center);
              break;
              
            case 'spiral':
              applySpiralLayout(agents, args.spacing, args.center);
              break;
              
            case 'random':
              applyRandomLayout(agents, args.spacing * 10, args.center);
              break;
              
            case 'hierarchical':
              applyHierarchicalLayout(agents, args.spacing, args.center);
              break;
              
            case 'force':
              // Let physics engine handle it
              physicsEngine.update(1, agents);
              break;
          }
          
          return {
            success: true,
            layout: args.layout,
            agentCount: agents.length,
            bounds: calculateBounds(agents)
          };
        } catch (error) {
          logger.error('Failed to apply layout:', error);
          throw error;
        }
      }
    },
    
    'visualization.highlight': {
      description: 'Highlight specific agents or connections',
      inputSchema: {
        type: 'object',
        properties: {
          agents: {
            type: 'array',
            items: { type: 'string' },
            description: 'Agent IDs to highlight'
          },
          connections: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                from: { type: 'string' },
                to: { type: 'string' }
              }
            },
            description: 'Connections to highlight'
          },
          style: {
            type: 'object',
            properties: {
              color: { type: 'string', default: '#FFD700' },
              pulseSpeed: { type: 'number', default: 2 },
              glowIntensity: { type: 'number', default: 1.5 }
            }
          }
        }
      },
      handler: async (args) => {
        try {
          const highlights = {
            agents: {},
            connections: {}
          };
          
          // Process agent highlights
          if (args.agents) {
            args.agents.forEach(agentId => {
              highlights.agents[agentId] = {
                ...args.style,
                timestamp: new Date().toISOString()
              };
            });
          }
          
          // Process connection highlights
          if (args.connections) {
            args.connections.forEach(conn => {
              const key = `${conn.from}-${conn.to}`;
              highlights.connections[key] = {
                ...args.style,
                timestamp: new Date().toISOString()
              };
            });
          }
          
          return {
            success: true,
            highlights,
            count: {
              agents: Object.keys(highlights.agents).length,
              connections: Object.keys(highlights.connections).length
            }
          };
        } catch (error) {
          logger.error('Failed to create highlights:', error);
          throw error;
        }
      }
    },
    
    'visualization.camera': {
      description: 'Get camera position recommendations',
      inputSchema: {
        type: 'object',
        properties: {
          target: {
            type: 'string',
            enum: ['overview', 'active_agents', 'errors', 'specific_agent'],
            default: 'overview'
          },
          agentId: { type: 'string', description: 'For specific_agent target' },
          smooth: { type: 'boolean', default: true }
        }
      },
      handler: async (args) => {
        try {
          const agents = agentManager.getAllAgents();
          let cameraPosition, cameraTarget, cameraDistance;
          
          switch (args.target) {
            case 'overview':
              const bounds = calculateBounds(agents);
              cameraPosition = {
                x: bounds.center.x + bounds.size.x,
                y: bounds.center.y + bounds.size.y,
                z: bounds.center.z + bounds.size.z
              };
              cameraTarget = bounds.center;
              cameraDistance = Math.max(bounds.size.x, bounds.size.y, bounds.size.z) * 2;
              break;
              
            case 'active_agents':
              const activeAgents = agents.filter(a => a.status === 'active');
              if (activeAgents.length > 0) {
                const activeBounds = calculateBounds(activeAgents);
                cameraTarget = activeBounds.center;
                cameraDistance = Math.max(activeBounds.size.x, activeBounds.size.y, activeBounds.size.z) * 1.5;
                cameraPosition = {
                  x: cameraTarget.x + cameraDistance * 0.7,
                  y: cameraTarget.y + cameraDistance * 0.5,
                  z: cameraTarget.z + cameraDistance * 0.7
                };
              }
              break;
              
            case 'errors':
              const errorAgents = agents.filter(a => a.status === 'error');
              if (errorAgents.length > 0) {
                const errorBounds = calculateBounds(errorAgents);
                cameraTarget = errorBounds.center;
                cameraDistance = 20;
                cameraPosition = {
                  x: cameraTarget.x,
                  y: cameraTarget.y + cameraDistance,
                  z: cameraTarget.z + cameraDistance * 0.5
                };
              }
              break;
              
            case 'specific_agent':
              if (args.agentId) {
                const agent = agentManager.getAgent(args.agentId);
                if (agent) {
                  cameraTarget = agent.position;
                  cameraDistance = 15;
                  cameraPosition = {
                    x: agent.position.x + 10,
                    y: agent.position.y + 8,
                    z: agent.position.z + 10
                  };
                }
              }
              break;
          }
          
          return {
            success: true,
            camera: {
              position: cameraPosition || { x: 30, y: 30, z: 30 },
              target: cameraTarget || { x: 0, y: 0, z: 0 },
              distance: cameraDistance || 50,
              fov: 75,
              smooth: args.smooth
            }
          };
        } catch (error) {
          logger.error('Failed to calculate camera position:', error);
          throw error;
        }
      }
    }
  };
}

// Helper functions

function getAgentColor(type, status) {
  const typeColors = {
    queen: '#FFD700',
    coordinator: '#F1C40F',
    architect: '#E67E22',
    coder: '#2ECC71',
    researcher: '#3498DB',
    tester: '#E74C3C',
    analyst: '#9B59B6',
    optimizer: '#F39C12',
    monitor: '#1ABC9C'
  };
  
  const statusModifiers = {
    active: 1.2,
    busy: 1.0,
    idle: 0.6,
    error: 0.3
  };
  
  return typeColors[type] || '#95A5A6';
}

function getAgentSize(type, performance) {
  const baseSizes = {
    queen: 2.5,
    coordinator: 2.0,
    architect: 1.5,
    default: 1.0
  };
  
  const baseSize = baseSizes[type] || baseSizes.default;
  const performanceModifier = 0.5 + (performance.successRate / 200);
  
  return baseSize * performanceModifier;
}

function generateTargetPositions(agents, targetState) {
  const positions = new Map();
  
  // Simple example - in real implementation would be more sophisticated
  agents.forEach((agent, index) => {
    const angle = (index / agents.length) * Math.PI * 2;
    const radius = 20;
    
    positions.set(agent.id, {
      x: Math.cos(angle) * radius,
      y: Math.sin(index * 0.5) * 5,
      z: Math.sin(angle) * radius
    });
  });
  
  return positions;
}

function applyEasing(t, type) {
  switch (type) {
    case 'linear':
      return t;
    case 'easeIn':
      return t * t;
    case 'easeOut':
      return t * (2 - t);
    case 'easeInOut':
      return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    default:
      return t;
  }
}

function lerp(start, end, t) {
  return start + (end - start) * t;
}

function calculateBounds(agents) {
  if (agents.length === 0) {
    return {
      min: { x: 0, y: 0, z: 0 },
      max: { x: 0, y: 0, z: 0 },
      center: { x: 0, y: 0, z: 0 },
      size: { x: 0, y: 0, z: 0 }
    };
  }
  
  const positions = agents.map(a => a.position);
  
  const min = {
    x: Math.min(...positions.map(p => p.x)),
    y: Math.min(...positions.map(p => p.y)),
    z: Math.min(...positions.map(p => p.z))
  };
  
  const max = {
    x: Math.max(...positions.map(p => p.x)),
    y: Math.max(...positions.map(p => p.y)),
    z: Math.max(...positions.map(p => p.z))
  };
  
  const center = {
    x: (min.x + max.x) / 2,
    y: (min.y + max.y) / 2,
    z: (min.z + max.z) / 2
  };
  
  const size = {
    x: max.x - min.x,
    y: max.y - min.y,
    z: max.z - min.z
  };
  
  return { min, max, center, size };
}

// Layout functions

function applyGridLayout(agents, spacing, center) {
  const gridSize = Math.ceil(Math.sqrt(agents.length));
  
  agents.forEach((agent, index) => {
    const row = Math.floor(index / gridSize);
    const col = index % gridSize;
    
    agent.position.x = center.x + (col - gridSize / 2) * spacing;
    agent.position.y = center.y;
    agent.position.z = center.z + (row - gridSize / 2) * spacing;
  });
}

function applyCircleLayout(agents, radius, center) {
  agents.forEach((agent, index) => {
    const angle = (index / agents.length) * Math.PI * 2;
    
    agent.position.x = center.x + Math.cos(angle) * radius;
    agent.position.y = center.y;
    agent.position.z = center.z + Math.sin(angle) * radius;
  });
}

function applySpiralLayout(agents, spacing, center) {
  agents.forEach((agent, index) => {
    const angle = index * 0.5;
    const radius = spacing * Math.sqrt(index);
    
    agent.position.x = center.x + Math.cos(angle) * radius;
    agent.position.y = center.y + index * 0.5;
    agent.position.z = center.z + Math.sin(angle) * radius;
  });
}

function applyRandomLayout(agents, range, center) {
  agents.forEach(agent => {
    agent.position.x = center.x + (Math.random() - 0.5) * range;
    agent.position.y = center.y + (Math.random() - 0.5) * range;
    agent.position.z = center.z + (Math.random() - 0.5) * range;
  });
}

function applyHierarchicalLayout(agents, spacing, center) {
  // Group by type
  const groups = {};
  agents.forEach(agent => {
    if (!groups[agent.type]) groups[agent.type] = [];
    groups[agent.type].push(agent);
  });
  
  // Position groups in layers
  const types = Object.keys(groups);
  types.forEach((type, typeIndex) => {
    const layer = typeIndex * spacing * 2;
    const agentsInType = groups[type];
    
    agentsInType.forEach((agent, agentIndex) => {
      const angle = (agentIndex / agentsInType.length) * Math.PI * 2;
      const radius = spacing * agentsInType.length / 2;
      
      agent.position.x = center.x + Math.cos(angle) * radius;
      agent.position.y = center.y + layer;
      agent.position.z = center.z + Math.sin(angle) * radius;
    });
  });
}