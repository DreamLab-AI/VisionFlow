import { createLogger } from '../logger.js';

const logger = createLogger('MessageTools');

export function messageTools(messageFlowTracker, physicsEngine) {
  return {
    'message.send': {
      description: 'Send message between agents and update spring forces',
      inputSchema: {
        type: 'object',
        properties: {
          from: { type: 'string', description: 'Sender agent ID' },
          to: { 
            oneOf: [
              { type: 'string', description: 'Single recipient agent ID' },
              { type: 'array', items: { type: 'string' }, description: 'Multiple recipient agent IDs' }
            ]
          },
          type: {
            type: 'string',
            enum: ['coordination', 'task', 'status', 'data', 'error'],
            default: 'coordination'
          },
          priority: {
            type: 'number',
            minimum: 1,
            maximum: 5,
            default: 1
          },
          content: {
            type: 'object',
            description: 'Message payload'
          }
        },
        required: ['from', 'to']
      },
      handler: async (args) => {
        try {
          const message = messageFlowTracker.trackMessage(args);
          
          // Apply message force to physics
          const recipients = Array.isArray(args.to) ? args.to : [args.to];
          recipients.forEach(recipientId => {
            // Note: In a real implementation, we'd need to get agent references
            // from the agent manager and apply forces
            logger.debug(`Applying message force from ${args.from} to ${recipientId}`);
          });
          
          return {
            success: true,
            message: {
              id: message.id,
              timestamp: message.timestamp,
              latency: message.latency_ms,
              springForce: message.springForce
            }
          };
        } catch (error) {
          logger.error('Failed to send message:', error);
          throw error;
        }
      }
    },
    
    'message.flow': {
      description: 'Get message flow data for visualization',
      inputSchema: {
        type: 'object',
        properties: {
          timeWindow: {
            type: 'number',
            description: 'Time window in seconds',
            default: 300
          },
          agentFilter: {
            type: 'array',
            items: { type: 'string' },
            description: 'Filter by agent IDs'
          }
        }
      },
      handler: async (args) => {
        try {
          const flow = messageFlowTracker.getMessageFlow(
            args.timeWindow,
            args.agentFilter
          );
          
          // Get communication statistics
          const stats = messageFlowTracker.getCommunicationStats();
          
          return {
            success: true,
            messages: flow,
            count: flow.length,
            statistics: stats,
            patterns: messageFlowTracker.getCoordinationPatterns()
          };
        } catch (error) {
          logger.error('Failed to get message flow:', error);
          throw error;
        }
      }
    },
    
    'message.acknowledge': {
      description: 'Acknowledge message receipt and update latency',
      inputSchema: {
        type: 'object',
        properties: {
          messageId: { type: 'string', description: 'Message ID to acknowledge' },
          latency: { type: 'number', description: 'Actual latency in milliseconds' }
        },
        required: ['messageId']
      },
      handler: async (args) => {
        try {
          messageFlowTracker.acknowledgeMessage(args.messageId, args.latency);
          
          return {
            success: true,
            messageId: args.messageId,
            acknowledged: true
          };
        } catch (error) {
          logger.error('Failed to acknowledge message:', error);
          throw error;
        }
      }
    },
    
    'message.stats': {
      description: 'Get detailed communication statistics',
      inputSchema: {
        type: 'object',
        properties: {
          agentId: { type: 'string', description: 'Filter by specific agent' },
          includeBottlenecks: { type: 'boolean', default: true }
        }
      },
      handler: async (args) => {
        try {
          const stats = messageFlowTracker.getCommunicationStats(args.agentId);
          
          const result = {
            success: true,
            statistics: stats,
            timestamp: new Date().toISOString()
          };
          
          if (args.includeBottlenecks) {
            result.bottlenecks = messageFlowTracker.detectBottlenecks();
          }
          
          return result;
        } catch (error) {
          logger.error('Failed to get message stats:', error);
          throw error;
        }
      }
    },
    
    'message.broadcast': {
      description: 'Broadcast message to multiple agents',
      inputSchema: {
        type: 'object',
        properties: {
          from: { type: 'string', description: 'Sender agent ID' },
          type: {
            type: 'string',
            enum: ['announcement', 'alert', 'coordination', 'status'],
            default: 'announcement'
          },
          priority: {
            type: 'number',
            minimum: 1,
            maximum: 5,
            default: 3
          },
          content: { type: 'object' },
          targetTypes: {
            type: 'array',
            items: { type: 'string' },
            description: 'Broadcast to specific agent types'
          }
        },
        required: ['from', 'content']
      },
      handler: async (args) => {
        try {
          // In a real implementation, we'd get all agents of target types
          // For now, simulate broadcast
          const recipients = ['agent-1', 'agent-2', 'agent-3']; // Mock recipients
          
          const message = messageFlowTracker.trackMessage({
            from: args.from,
            to: recipients,
            type: args.type,
            priority: args.priority,
            content: args.content
          });
          
          return {
            success: true,
            message: {
              id: message.id,
              timestamp: message.timestamp,
              recipientCount: recipients.length,
              type: args.type
            }
          };
        } catch (error) {
          logger.error('Failed to broadcast message:', error);
          throw error;
        }
      }
    },
    
    'message.patterns': {
      description: 'Analyze and return communication patterns',
      inputSchema: {
        type: 'object',
        properties: {
          timeWindow: { type: 'number', default: 600 },
          minPatternSize: { type: 'number', default: 3 }
        }
      },
      handler: async (args) => {
        try {
          const patterns = messageFlowTracker.getCoordinationPatterns();
          const bottlenecks = messageFlowTracker.detectBottlenecks();
          
          // Analyze pattern efficiency
          const patternAnalysis = patterns.map(pattern => ({
            ...pattern,
            efficiency: calculatePatternEfficiency(pattern),
            recommendations: generatePatternRecommendations(pattern)
          }));
          
          return {
            success: true,
            patterns: patternAnalysis,
            bottlenecks,
            timestamp: new Date().toISOString()
          };
        } catch (error) {
          logger.error('Failed to analyze patterns:', error);
          throw error;
        }
      }
    }
  };
}

// Helper function to calculate pattern efficiency
function calculatePatternEfficiency(pattern) {
  switch (pattern.type) {
    case 'broadcast':
      // Broadcasts are efficient for announcements but not for coordination
      return pattern.count < 10 ? 0.8 : 0.5;
      
    case 'pipeline':
      // Pipelines are efficient for sequential processing
      const avgDuration = pattern.chains.reduce((sum, chain) => 
        sum + chain.duration, 0) / pattern.chains.length;
      return avgDuration < 1000 ? 0.9 : 0.6;
      
    case 'hub':
      // Hubs can become bottlenecks
      const maxConnections = Math.max(...pattern.hubs.map(h => h.connectionCount));
      return maxConnections < 20 ? 0.7 : 0.4;
      
    default:
      return 0.5;
  }
}

// Helper function to generate pattern recommendations
function generatePatternRecommendations(pattern) {
  const recommendations = [];
  
  switch (pattern.type) {
    case 'broadcast':
      if (pattern.count > 20) {
        recommendations.push('Consider using hierarchical broadcasting to reduce load');
      }
      break;
      
    case 'pipeline':
      pattern.chains.forEach(chain => {
        if (chain.duration > 2000) {
          recommendations.push('Pipeline processing is slow, consider parallel processing');
        }
      });
      break;
      
    case 'hub':
      pattern.hubs.forEach(hub => {
        if (hub.connectionCount > 30) {
          recommendations.push(`Agent ${hub.agentId} is overloaded, consider load balancing`);
        }
      });
      break;
  }
  
  return recommendations;
}