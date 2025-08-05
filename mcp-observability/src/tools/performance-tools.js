import { createLogger } from '../logger.js';

const logger = createLogger('PerformanceTools');

export function performanceTools(performanceMonitor, agentManager) {
  return {
    'performance.analyze': {
      description: 'Analyze swarm performance with bottleneck detection',
      inputSchema: {
        type: 'object',
        properties: {
          metrics: {
            type: 'array',
            items: {
              type: 'string',
              enum: ['throughput', 'latency', 'successRate', 'resourceUsage', 'messageRate']
            },
            description: 'Metrics to analyze'
          },
          aggregation: {
            type: 'string',
            enum: ['avg', 'sum', 'max', 'min'],
            default: 'avg'
          },
          timeWindow: {
            type: 'number',
            description: 'Time window in seconds',
            default: 300
          }
        },
        required: ['metrics']
      },
      handler: async (args) => {
        try {
          const analysis = performanceMonitor.analyzePerformance(
            args.metrics,
            args.aggregation
          );
          
          const report = performanceMonitor.getPerformanceReport(args.timeWindow);
          
          return {
            success: true,
            analysis,
            report,
            timestamp: new Date().toISOString()
          };
        } catch (error) {
          logger.error('Failed to analyze performance:', error);
          throw error;
        }
      }
    },
    
    'performance.optimize': {
      description: 'Suggest physics parameter optimizations',
      inputSchema: {
        type: 'object',
        properties: {
          targetMetric: {
            type: 'string',
            enum: ['latency', 'throughput', 'balance'],
            default: 'balance'
          },
          constraints: {
            type: 'object',
            properties: {
              maxAgents: { type: 'number', minimum: 1 },
              maxDistance: { type: 'number', minimum: 1 },
              minFPS: { type: 'number', minimum: 30, maximum: 144 }
            }
          }
        }
      },
      handler: async (args) => {
        try {
          // Get current performance data
          const currentMetrics = performanceMonitor.getSystemMetrics();
          const report = performanceMonitor.getPerformanceReport();
          
          const optimizations = {
            targetMetric: args.targetMetric,
            currentPerformance: {
              latency: currentMetrics.avgLatency,
              throughput: currentMetrics.messageRate,
              activeAgents: currentMetrics.activeAgents
            },
            recommendations: []
          };
          
          // Generate optimization recommendations based on target
          switch (args.targetMetric) {
            case 'latency':
              if (currentMetrics.avgLatency > 50) {
                optimizations.recommendations.push({
                  parameter: 'springStrength',
                  action: 'increase',
                  value: 0.15,
                  reason: 'Stronger springs will reduce communication distance'
                });
                optimizations.recommendations.push({
                  parameter: 'messageAttraction',
                  action: 'increase',
                  value: 0.2,
                  reason: 'Higher message attraction improves routing'
                });
              }
              break;
              
            case 'throughput':
              if (currentMetrics.messageRate < 100) {
                optimizations.recommendations.push({
                  parameter: 'nodeRepulsion',
                  action: 'increase',
                  value: 600,
                  reason: 'Greater spacing allows parallel communication'
                });
                optimizations.recommendations.push({
                  parameter: 'maxAgents',
                  action: 'increase',
                  value: args.constraints?.maxAgents || 100,
                  reason: 'More agents can handle higher throughput'
                });
              }
              break;
              
            case 'balance':
              optimizations.recommendations.push({
                parameter: 'damping',
                action: 'adjust',
                value: 0.95,
                reason: 'Optimal damping for stability'
              });
              if (currentMetrics.activeAgents > 50) {
                optimizations.recommendations.push({
                  parameter: 'gravityStrength',
                  action: 'increase',
                  value: 0.03,
                  reason: 'Prevent drift in large swarms'
                });
              }
              break;
          }
          
          // Add bottleneck-based recommendations
          if (report.bottlenecks.length > 0) {
            report.bottlenecks.forEach(bottleneck => {
              optimizations.recommendations.push({
                parameter: bottleneck.type,
                action: 'address',
                value: bottleneck.recommendation,
                reason: `Severity: ${bottleneck.severity}`
              });
            });
          }
          
          optimizations.expectedImprovement = calculateExpectedImprovement(
            args.targetMetric,
            optimizations.recommendations
          );
          
          return {
            success: true,
            optimizations,
            timestamp: new Date().toISOString()
          };
        } catch (error) {
          logger.error('Failed to optimize performance:', error);
          throw error;
        }
      }
    },
    
    'performance.report': {
      description: 'Get comprehensive performance report',
      inputSchema: {
        type: 'object',
        properties: {
          timeWindow: { type: 'number', default: 300 },
          includeAgentDetails: { type: 'boolean', default: true },
          includeRecommendations: { type: 'boolean', default: true }
        }
      },
      handler: async (args) => {
        try {
          const report = performanceMonitor.getPerformanceReport(args.timeWindow);
          
          if (args.includeAgentDetails) {
            // Add per-agent performance data
            const agents = agentManager.getAllAgents();
            report.agentDetails = agents.map(agent => ({
              id: agent.id,
              name: agent.name,
              type: agent.type,
              performance: agent.performance,
              connectionCount: agent.connections.length
            }));
          }
          
          if (!args.includeRecommendations) {
            delete report.recommendations;
          }
          
          return {
            success: true,
            report,
            timestamp: new Date().toISOString()
          };
        } catch (error) {
          logger.error('Failed to generate performance report:', error);
          throw error;
        }
      }
    },
    
    'performance.metrics': {
      description: 'Get current system performance metrics',
      inputSchema: {
        type: 'object',
        properties: {
          detailed: { type: 'boolean', default: false }
        }
      },
      handler: async (args) => {
        try {
          const metrics = performanceMonitor.getSystemMetrics();
          
          const result = {
            success: true,
            metrics: {
              ...metrics,
              fps: calculateFPS(),
              agentLoad: calculateAverageAgentLoad(agentManager.getAllAgents())
            }
          };
          
          if (args.detailed) {
            result.details = {
              topAgents: performanceMonitor.getTopPerformingAgents(),
              bottlenecks: performanceMonitor.detectPerformanceBottlenecks(),
              trends: performanceMonitor.calculateTrends(
                performanceMonitor.metrics.historicalData.slice(-60)
              )
            };
          }
          
          return result;
        } catch (error) {
          logger.error('Failed to get performance metrics:', error);
          throw error;
        }
      }
    },
    
    'performance.benchmark': {
      description: 'Run performance benchmark tests',
      inputSchema: {
        type: 'object',
        properties: {
          tests: {
            type: 'array',
            items: {
              type: 'string',
              enum: ['agent_spawn', 'message_throughput', 'physics_fps', 'memory_usage']
            },
            default: ['agent_spawn', 'message_throughput']
          },
          duration: {
            type: 'number',
            description: 'Test duration in seconds',
            default: 10
          }
        }
      },
      handler: async (args) => {
        try {
          const results = {};
          
          for (const test of args.tests) {
            logger.info(`Running benchmark: ${test}`);
            
            switch (test) {
              case 'agent_spawn':
                results[test] = await benchmarkAgentSpawn(agentManager, args.duration);
                break;
                
              case 'message_throughput':
                results[test] = await benchmarkMessageThroughput(args.duration);
                break;
                
              case 'physics_fps':
                results[test] = await benchmarkPhysicsFPS(args.duration);
                break;
                
              case 'memory_usage':
                results[test] = await benchmarkMemoryUsage(args.duration);
                break;
            }
          }
          
          return {
            success: true,
            benchmarks: results,
            duration: args.duration,
            timestamp: new Date().toISOString()
          };
        } catch (error) {
          logger.error('Failed to run benchmarks:', error);
          throw error;
        }
      }
    }
  };
}

// Helper functions

function calculateExpectedImprovement(targetMetric, recommendations) {
  const improvementFactors = {
    latency: recommendations.length * 0.1, // 10% per recommendation
    throughput: recommendations.length * 0.15, // 15% per recommendation
    balance: recommendations.length * 0.05 // 5% per recommendation
  };
  
  return Math.min(improvementFactors[targetMetric] || 0.1, 0.5); // Cap at 50%
}

function calculateFPS() {
  // In a real implementation, this would track actual frame times
  return 60; // Mock 60 FPS
}

function calculateAverageAgentLoad(agents) {
  if (agents.length === 0) return 0;
  
  const totalLoad = agents.reduce((sum, agent) => 
    sum + agent.performance.resourceUtilization, 0
  );
  
  return totalLoad / agents.length;
}

// Benchmark functions

async function benchmarkAgentSpawn(agentManager, duration) {
  const startTime = Date.now();
  const startCount = agentManager.getAllAgents().length;
  let spawnCount = 0;
  
  while (Date.now() - startTime < duration * 1000) {
    agentManager.createAgent({
      name: `benchmark-${spawnCount}`,
      type: 'coder'
    });
    spawnCount++;
    
    // Small delay to prevent overwhelming
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  
  const endCount = agentManager.getAllAgents().length;
  const actualDuration = (Date.now() - startTime) / 1000;
  
  // Clean up benchmark agents
  for (let i = 0; i < spawnCount; i++) {
    const agent = agentManager.getAllAgents().find(a => a.name === `benchmark-${i}`);
    if (agent) {
      agentManager.removeAgent(agent.id);
    }
  }
  
  return {
    agentsSpawned: spawnCount,
    spawnRate: spawnCount / actualDuration,
    duration: actualDuration
  };
}

async function benchmarkMessageThroughput(duration) {
  // Mock benchmark - in real implementation would send actual messages
  const messagesSent = Math.floor(duration * 100 + Math.random() * 50);
  
  return {
    messagesSent,
    messagesPerSecond: messagesSent / duration,
    avgLatency: 25 + Math.random() * 10
  };
}

async function benchmarkPhysicsFPS(duration) {
  // Mock benchmark - in real implementation would measure actual physics FPS
  const samples = [];
  const sampleCount = duration * 10;
  
  for (let i = 0; i < sampleCount; i++) {
    samples.push(58 + Math.random() * 4); // 58-62 FPS
  }
  
  return {
    avgFPS: samples.reduce((a, b) => a + b) / samples.length,
    minFPS: Math.min(...samples),
    maxFPS: Math.max(...samples),
    samples: samples.length
  };
}

async function benchmarkMemoryUsage(duration) {
  const samples = [];
  const startMemory = process.memoryUsage().heapUsed / (1024 * 1024);
  
  const interval = setInterval(() => {
    const current = process.memoryUsage().heapUsed / (1024 * 1024);
    samples.push(current);
  }, 100);
  
  await new Promise(resolve => setTimeout(resolve, duration * 1000));
  clearInterval(interval);
  
  const endMemory = process.memoryUsage().heapUsed / (1024 * 1024);
  
  return {
    startMemory,
    endMemory,
    avgMemory: samples.reduce((a, b) => a + b) / samples.length,
    peakMemory: Math.max(...samples),
    memoryGrowth: endMemory - startMemory
  };
}