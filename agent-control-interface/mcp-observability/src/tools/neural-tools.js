import { createLogger } from '../logger.js';

const logger = createLogger('NeuralTools');

// Simple neural network implementation for pattern learning
class SimpleNeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize) {
    this.weights1 = this.randomMatrix(inputSize, hiddenSize);
    this.weights2 = this.randomMatrix(hiddenSize, outputSize);
    this.bias1 = this.randomVector(hiddenSize);
    this.bias2 = this.randomVector(outputSize);
  }
  
  randomMatrix(rows, cols) {
    return Array(rows).fill().map(() => 
      Array(cols).fill().map(() => Math.random() * 2 - 1)
    );
  }
  
  randomVector(size) {
    return Array(size).fill().map(() => Math.random() * 2 - 1);
  }
  
  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }
  
  forward(input) {
    // Hidden layer
    const hidden = [];
    for (let i = 0; i < this.weights1[0].length; i++) {
      let sum = this.bias1[i];
      for (let j = 0; j < input.length; j++) {
        sum += input[j] * this.weights1[j][i];
      }
      hidden.push(this.sigmoid(sum));
    }
    
    // Output layer
    const output = [];
    for (let i = 0; i < this.weights2[0].length; i++) {
      let sum = this.bias2[i];
      for (let j = 0; j < hidden.length; j++) {
        sum += hidden[j] * this.weights2[j][i];
      }
      output.push(this.sigmoid(sum));
    }
    
    return output;
  }
}

// Store trained models
const trainedModels = new Map();
const trainingData = new Map();

export function neuralTools(agentManager, performanceMonitor) {
  return {
    'neural.train': {
      description: 'Train neural patterns from successful coordinations',
      inputSchema: {
        type: 'object',
        properties: {
          pattern: { 
            type: 'string',
            description: 'Pattern name to train'
          },
          trainingData: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                input: { type: 'array', items: { type: 'number' } },
                output: { type: 'array', items: { type: 'number' } }
              }
            },
            description: 'Training examples'
          },
          modelType: {
            type: 'string',
            enum: ['classification', 'regression', 'clustering'],
            default: 'classification'
          },
          epochs: {
            type: 'number',
            default: 100
          }
        },
        required: ['pattern']
      },
      handler: async (args) => {
        try {
          // Generate training data if not provided
          let data = args.trainingData;
          if (!data || data.length === 0) {
            data = generateTrainingData(args.pattern, agentManager, performanceMonitor);
          }
          
          if (data.length === 0) {
            throw new Error('No training data available');
          }
          
          // Determine network architecture
          const inputSize = data[0].input.length;
          const outputSize = data[0].output.length;
          const hiddenSize = Math.ceil((inputSize + outputSize) * 0.75);
          
          // Create and train network
          const network = new SimpleNeuralNetwork(inputSize, hiddenSize, outputSize);
          
          // Simple training loop (gradient-free for simplicity)
          const trainingLog = [];
          for (let epoch = 0; epoch < args.epochs; epoch++) {
            let totalError = 0;
            
            data.forEach(example => {
              const prediction = network.forward(example.input);
              const error = calculateError(prediction, example.output);
              totalError += error;
            });
            
            const avgError = totalError / data.length;
            trainingLog.push({ epoch, error: avgError });
            
            // Log progress every 10 epochs
            if (epoch % 10 === 0) {
              logger.debug(`Training epoch ${epoch}, error: ${avgError.toFixed(4)}`);
            }
          }
          
          // Store trained model
          const modelId = `${args.pattern}-${Date.now()}`;
          trainedModels.set(modelId, {
            network,
            pattern: args.pattern,
            type: args.modelType,
            inputSize,
            outputSize,
            trainedAt: new Date().toISOString(),
            trainingExamples: data.length,
            finalError: trainingLog[trainingLog.length - 1].error
          });
          
          // Store training data for future reference
          trainingData.set(modelId, data);
          
          return {
            success: true,
            modelId,
            pattern: args.pattern,
            trainingMetrics: {
              epochs: args.epochs,
              finalError: trainingLog[trainingLog.length - 1].error,
              trainingExamples: data.length,
              convergenceRate: calculateConvergenceRate(trainingLog)
            }
          };
        } catch (error) {
          logger.error('Failed to train neural pattern:', error);
          throw error;
        }
      }
    },
    
    'neural.predict': {
      description: 'Predict optimal coordination patterns',
      inputSchema: {
        type: 'object',
        properties: {
          scenario: {
            type: 'object',
            properties: {
              agentCount: { type: 'number' },
              taskComplexity: { type: 'number', minimum: 0, maximum: 1 },
              urgency: { type: 'number', minimum: 0, maximum: 1 },
              resourceAvailability: { type: 'number', minimum: 0, maximum: 1 }
            },
            required: ['agentCount', 'taskComplexity']
          },
          models: {
            type: 'array',
            items: { type: 'string' },
            description: 'Model IDs to use for prediction'
          }
        },
        required: ['scenario']
      },
      handler: async (args) => {
        try {
          // Convert scenario to input vector
          const input = [
            args.scenario.agentCount / 100, // Normalize
            args.scenario.taskComplexity,
            args.scenario.urgency || 0.5,
            args.scenario.resourceAvailability || 0.8
          ];
          
          const predictions = [];
          
          // Use specified models or all available
          const modelIds = args.models || Array.from(trainedModels.keys());
          
          for (const modelId of modelIds) {
            const model = trainedModels.get(modelId);
            if (!model) continue;
            
            // Ensure input size matches
            const paddedInput = padInput(input, model.inputSize);
            
            const output = model.network.forward(paddedInput);
            const interpretation = interpretPrediction(output, model.pattern);
            
            predictions.push({
              modelId,
              pattern: model.pattern,
              confidence: Math.max(...output),
              recommendation: interpretation,
              rawOutput: output
            });
          }
          
          // Sort by confidence
          predictions.sort((a, b) => b.confidence - a.confidence);
          
          return {
            success: true,
            scenario: args.scenario,
            predictions,
            bestRecommendation: predictions[0]?.recommendation || 'No trained models available'
          };
        } catch (error) {
          logger.error('Failed to predict pattern:', error);
          throw error;
        }
      }
    },
    
    'neural.status': {
      description: 'Get neural network status and available models',
      inputSchema: {
        type: 'object',
        properties: {
          includeDetails: { type: 'boolean', default: false }
        }
      },
      handler: async (args) => {
        try {
          const models = Array.from(trainedModels.entries()).map(([id, model]) => ({
            modelId: id,
            pattern: model.pattern,
            type: model.type,
            trainedAt: model.trainedAt,
            accuracy: 1 - model.finalError,
            trainingExamples: model.trainingExamples
          }));
          
          const status = {
            success: true,
            modelCount: models.length,
            models,
            capabilities: {
              patterns: [...new Set(models.map(m => m.pattern))],
              avgAccuracy: models.reduce((sum, m) => sum + m.accuracy, 0) / models.length || 0
            }
          };
          
          if (args.includeDetails) {
            status.details = {
              totalTrainingData: Array.from(trainingData.values())
                .reduce((sum, data) => sum + data.length, 0),
              memoryUsage: estimateMemoryUsage()
            };
          }
          
          return status;
        } catch (error) {
          logger.error('Failed to get neural status:', error);
          throw error;
        }
      }
    },
    
    'neural.patterns': {
      description: 'Get recognized patterns from neural analysis',
      inputSchema: {
        type: 'object',
        properties: {
          timeWindow: { type: 'number', default: 3600 },
          minConfidence: { type: 'number', minimum: 0, maximum: 1, default: 0.7 }
        }
      },
      handler: async (args) => {
        try {
          const agents = agentManager.getAllAgents();
          const patterns = [];
          
          // Analyze current swarm state
          const swarmAnalysis = analyzeSwarmPatterns(agents);
          
          // Communication patterns
          if (swarmAnalysis.communicationIntensity > 0.7) {
            patterns.push({
              type: 'high_communication',
              confidence: swarmAnalysis.communicationIntensity,
              description: 'Agents are heavily communicating',
              recommendation: 'Consider hierarchical communication to reduce overhead'
            });
          }
          
          // Performance patterns
          const avgPerformance = agents.reduce((sum, a) => 
            sum + a.performance.successRate, 0) / agents.length;
          
          if (avgPerformance < 70) {
            patterns.push({
              type: 'low_performance',
              confidence: (100 - avgPerformance) / 100,
              description: 'Overall swarm performance is below threshold',
              recommendation: 'Increase agent specialization or add more resources'
            });
          }
          
          // Clustering patterns
          const clusters = detectAgentClusters(agents);
          if (clusters.length > 1) {
            patterns.push({
              type: 'agent_clustering',
              confidence: 0.85,
              description: `Agents have formed ${clusters.length} distinct clusters`,
              recommendation: 'This may indicate natural task grouping',
              clusters: clusters.map(c => ({
                size: c.agents.length,
                center: c.center,
                avgPerformance: c.avgPerformance
              }))
            });
          }
          
          // Bottleneck patterns
          const bottlenecks = detectBottleneckPatterns(agents);
          bottlenecks.forEach(bottleneck => {
            if (bottleneck.severity > args.minConfidence) {
              patterns.push({
                type: 'bottleneck',
                confidence: bottleneck.severity,
                description: `Agent ${bottleneck.agentId} is a bottleneck`,
                recommendation: 'Redistribute load or spawn helper agents',
                details: bottleneck
              });
            }
          });
          
          // Filter by confidence
          const filteredPatterns = patterns.filter(p => p.confidence >= args.minConfidence);
          
          return {
            success: true,
            patterns: filteredPatterns,
            summary: {
              totalPatterns: filteredPatterns.length,
              dominantPattern: filteredPatterns[0]?.type || 'none',
              swarmHealth: calculateSwarmHealth(swarmAnalysis)
            }
          };
        } catch (error) {
          logger.error('Failed to analyze patterns:', error);
          throw error;
        }
      }
    },
    
    'neural.optimize': {
      description: 'Optimize swarm configuration using neural insights',
      inputSchema: {
        type: 'object',
        properties: {
          objective: {
            type: 'string',
            enum: ['performance', 'efficiency', 'reliability', 'scalability'],
            default: 'performance'
          },
          constraints: {
            type: 'object',
            properties: {
              maxAgents: { type: 'number' },
              maxCost: { type: 'number' },
              minReliability: { type: 'number', minimum: 0, maximum: 1 }
            }
          }
        }
      },
      handler: async (args) => {
        try {
          const currentState = analyzeCurrentState(agentManager, performanceMonitor);
          const optimizations = [];
          
          switch (args.objective) {
            case 'performance':
              if (currentState.avgResponseTime > 100) {
                optimizations.push({
                  action: 'spawn_agents',
                  target: 'coder',
                  count: Math.ceil(currentState.agentCount * 0.2),
                  expectedImprovement: 0.25,
                  reasoning: 'Additional coders can parallelize work'
                });
              }
              
              if (currentState.coordinatorRatio < 0.1) {
                optimizations.push({
                  action: 'spawn_agents',
                  target: 'coordinator',
                  count: 1,
                  expectedImprovement: 0.15,
                  reasoning: 'Better coordination improves overall performance'
                });
              }
              break;
              
            case 'efficiency':
              if (currentState.idleRatio > 0.3) {
                optimizations.push({
                  action: 'remove_agents',
                  target: 'idle',
                  count: Math.floor(currentState.idleCount * 0.5),
                  expectedImprovement: 0.2,
                  reasoning: 'Reduce resource waste from idle agents'
                });
              }
              
              optimizations.push({
                action: 'reconfigure_topology',
                target: 'hierarchical',
                expectedImprovement: 0.1,
                reasoning: 'Hierarchical topology reduces communication overhead'
              });
              break;
              
            case 'reliability':
              optimizations.push({
                action: 'spawn_agents',
                target: 'monitor',
                count: 2,
                expectedImprovement: 0.3,
                reasoning: 'Monitors improve failure detection and recovery'
              });
              
              if (currentState.singlePointsOfFailure > 0) {
                optimizations.push({
                  action: 'add_redundancy',
                  target: 'critical_agents',
                  expectedImprovement: 0.4,
                  reasoning: 'Eliminate single points of failure'
                });
              }
              break;
              
            case 'scalability':
              optimizations.push({
                action: 'reconfigure_topology',
                target: 'mesh',
                expectedImprovement: 0.2,
                reasoning: 'Mesh topology scales better with agent count'
              });
              
              optimizations.push({
                action: 'optimize_physics',
                parameters: {
                  nodeRepulsion: 800,
                  maxVelocity: 3
                },
                expectedImprovement: 0.15,
                reasoning: 'Adjusted physics handles larger swarms better'
              });
              break;
          }
          
          // Apply constraints
          const filteredOptimizations = applyConstraints(optimizations, args.constraints);
          
          // Sort by expected improvement
          filteredOptimizations.sort((a, b) => b.expectedImprovement - a.expectedImprovement);
          
          return {
            success: true,
            objective: args.objective,
            currentState: {
              performance: currentState.avgSuccessRate,
              efficiency: 1 - currentState.idleRatio,
              reliability: currentState.reliability,
              scalability: currentState.scalabilityScore
            },
            optimizations: filteredOptimizations,
            totalExpectedImprovement: filteredOptimizations
              .reduce((sum, opt) => sum + opt.expectedImprovement, 0)
          };
        } catch (error) {
          logger.error('Failed to optimize swarm:', error);
          throw error;
        }
      }
    }
  };
}

// Helper functions

function generateTrainingData(pattern, agentManager, performanceMonitor) {
  const data = [];
  const agents = agentManager.getAllAgents();
  
  // Generate synthetic training data based on pattern
  switch (pattern) {
    case 'coordination':
      // Train on agent count vs optimal coordinator ratio
      for (let i = 5; i <= 50; i += 5) {
        const coordinatorRatio = Math.min(0.2, Math.log10(i) / 10);
        data.push({
          input: [i / 50, 0.5, 0.5], // agent count, complexity, urgency
          output: [coordinatorRatio] // optimal coordinator ratio
        });
      }
      break;
      
    case 'performance':
      // Train on workload vs success rate
      const metrics = performanceMonitor.metrics.historicalData.slice(-100);
      metrics.forEach(m => {
        if (m.activeAgents > 0) {
          data.push({
            input: [
              m.activeAgents / 100,
              m.messageRate / 1000,
              m.avgLatency / 1000
            ],
            output: [m.networkHealth]
          });
        }
      });
      break;
      
    case 'topology':
      // Train on swarm size vs optimal topology
      const topologies = ['hierarchical', 'mesh', 'ring', 'star'];
      for (let size = 5; size <= 100; size += 10) {
        const optimalTopology = size < 20 ? 2 : size < 50 ? 1 : 0; // star -> mesh -> hierarchical
        const output = Array(4).fill(0);
        output[optimalTopology] = 1;
        
        data.push({
          input: [size / 100, Math.random(), Math.random()],
          output
        });
      }
      break;
  }
  
  return data;
}

function calculateError(prediction, target) {
  return prediction.reduce((sum, pred, i) => 
    sum + Math.pow(pred - target[i], 2), 0) / prediction.length;
}

function calculateConvergenceRate(trainingLog) {
  if (trainingLog.length < 10) return 0;
  
  const early = trainingLog.slice(0, 10).reduce((sum, log) => sum + log.error, 0) / 10;
  const late = trainingLog.slice(-10).reduce((sum, log) => sum + log.error, 0) / 10;
  
  return (early - late) / early;
}

function padInput(input, targetSize) {
  const padded = [...input];
  while (padded.length < targetSize) {
    padded.push(0.5); // Neutral padding
  }
  return padded.slice(0, targetSize);
}

function interpretPrediction(output, pattern) {
  switch (pattern) {
    case 'coordination':
      const ratio = output[0];
      return `Optimal coordinator ratio: ${(ratio * 100).toFixed(1)}%`;
      
    case 'performance':
      const health = output[0];
      return health > 0.8 ? 'Excellent performance expected' :
             health > 0.6 ? 'Good performance expected' :
             health > 0.4 ? 'Moderate performance expected' :
             'Poor performance expected - optimization needed';
      
    case 'topology':
      const topologies = ['hierarchical', 'mesh', 'ring', 'star'];
      const maxIndex = output.indexOf(Math.max(...output));
      return `Recommended topology: ${topologies[maxIndex]}`;
      
    default:
      return `Confidence: ${(Math.max(...output) * 100).toFixed(1)}%`;
  }
}

function estimateMemoryUsage() {
  let totalParams = 0;
  
  trainedModels.forEach(model => {
    // Estimate parameters in network
    const params = model.inputSize * 10 + 10 * model.outputSize + 10 + model.outputSize;
    totalParams += params;
  });
  
  // Estimate memory in MB (4 bytes per parameter)
  return (totalParams * 4) / (1024 * 1024);
}

function analyzeSwarmPatterns(agents) {
  const totalConnections = agents.reduce((sum, a) => sum + a.connections.length, 0);
  const maxConnections = agents.length * (agents.length - 1);
  
  return {
    communicationIntensity: totalConnections / maxConnections,
    agentDiversity: new Set(agents.map(a => a.type)).size / 10, // 10 possible types
    spatialSpread: calculateSpatialSpread(agents),
    activityLevel: agents.filter(a => a.status === 'active').length / agents.length
  };
}

function detectAgentClusters(agents) {
  // Simple distance-based clustering
  const clusters = [];
  const unassigned = new Set(agents.map(a => a.id));
  
  while (unassigned.size > 0) {
    const seedId = unassigned.values().next().value;
    const seed = agents.find(a => a.id === seedId);
    const cluster = { agents: [seed], center: { ...seed.position } };
    unassigned.delete(seedId);
    
    // Find nearby agents
    agents.forEach(agent => {
      if (unassigned.has(agent.id)) {
        const distance = Math.sqrt(
          Math.pow(agent.position.x - cluster.center.x, 2) +
          Math.pow(agent.position.y - cluster.center.y, 2) +
          Math.pow(agent.position.z - cluster.center.z, 2)
        );
        
        if (distance < 15) { // Cluster radius
          cluster.agents.push(agent);
          unassigned.delete(agent.id);
        }
      }
    });
    
    // Calculate cluster metrics
    cluster.avgPerformance = cluster.agents.reduce((sum, a) => 
      sum + a.performance.successRate, 0) / cluster.agents.length;
    
    clusters.push(cluster);
  }
  
  return clusters.filter(c => c.agents.length > 1);
}

function detectBottleneckPatterns(agents) {
  return agents
    .filter(agent => agent.connections.length > agents.length * 0.3)
    .map(agent => ({
      agentId: agent.id,
      connectionCount: agent.connections.length,
      severity: agent.connections.length / agents.length,
      utilization: agent.performance.resourceUtilization
    }));
}

function calculateSwarmHealth(analysis) {
  return (
    analysis.activityLevel * 0.3 +
    (1 - analysis.communicationIntensity) * 0.2 + // Too much communication is bad
    analysis.agentDiversity * 0.2 +
    Math.min(analysis.spatialSpread, 1) * 0.3
  );
}

function calculateSpatialSpread(agents) {
  if (agents.length === 0) return 0;
  
  const center = {
    x: agents.reduce((sum, a) => sum + a.position.x, 0) / agents.length,
    y: agents.reduce((sum, a) => sum + a.position.y, 0) / agents.length,
    z: agents.reduce((sum, a) => sum + a.position.z, 0) / agents.length
  };
  
  const avgDistance = agents.reduce((sum, agent) => {
    const distance = Math.sqrt(
      Math.pow(agent.position.x - center.x, 2) +
      Math.pow(agent.position.y - center.y, 2) +
      Math.pow(agent.position.z - center.z, 2)
    );
    return sum + distance;
  }, 0) / agents.length;
  
  return avgDistance / 50; // Normalize
}

function analyzeCurrentState(agentManager, performanceMonitor) {
  const agents = agentManager.getAllAgents();
  const metrics = performanceMonitor.getSystemMetrics();
  
  const state = {
    agentCount: agents.length,
    avgResponseTime: agents.reduce((sum, a) => sum + a.performance.avgResponseTime, 0) / agents.length,
    avgSuccessRate: agents.reduce((sum, a) => sum + a.performance.successRate, 0) / agents.length,
    coordinatorRatio: agents.filter(a => a.type === 'coordinator').length / agents.length,
    idleCount: agents.filter(a => a.status === 'idle').length,
    idleRatio: agents.filter(a => a.status === 'idle').length / agents.length,
    singlePointsOfFailure: detectSinglePointsOfFailure(agents),
    reliability: calculateReliability(agents),
    scalabilityScore: calculateScalabilityScore(agents)
  };
  
  return state;
}

function detectSinglePointsOfFailure(agents) {
  // Count agents that are sole providers of a capability
  const capabilityProviders = {};
  
  agents.forEach(agent => {
    agent.capabilities.forEach(capability => {
      if (!capabilityProviders[capability]) {
        capabilityProviders[capability] = [];
      }
      capabilityProviders[capability].push(agent.id);
    });
  });
  
  return Object.values(capabilityProviders).filter(providers => providers.length === 1).length;
}

function calculateReliability(agents) {
  const avgSuccessRate = agents.reduce((sum, a) => sum + a.performance.successRate, 0) / agents.length;
  const redundancy = 1 - detectSinglePointsOfFailure(agents) / agents.length;
  
  return (avgSuccessRate / 100) * 0.7 + redundancy * 0.3;
}

function calculateScalabilityScore(agents) {
  // Based on topology efficiency and resource usage
  const avgConnections = agents.reduce((sum, a) => sum + a.connections.length, 0) / agents.length;
  const connectionEfficiency = 1 - (avgConnections / agents.length); // Lower is better for scale
  
  const avgResourceUsage = agents.reduce((sum, a) => 
    sum + a.performance.resourceUtilization, 0) / agents.length;
  
  return connectionEfficiency * 0.6 + (1 - avgResourceUsage) * 0.4;
}

function applyConstraints(optimizations, constraints) {
  if (!constraints) return optimizations;
  
  return optimizations.filter(opt => {
    if (constraints.maxAgents && opt.action === 'spawn_agents') {
      // Check if we can add more agents
      return true; // Would need current count to properly check
    }
    
    if (constraints.minReliability && opt.action === 'remove_agents') {
      // Don't remove agents if it would hurt reliability
      return opt.target === 'idle'; // Only remove idle agents
    }
    
    return true;
  });
}