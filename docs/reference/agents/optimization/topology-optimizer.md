# Topology Optimizer Agent

*[Reference](../index.md) > [Agents](../../reference/agents/index.md) > [Optimization](../reference/agents/optimization/index.md)*

## Agent Profile
- **Name**: Topology Optimizer
- **Type**: Performance Optimisation Agent
- **Specialization**: Dynamic swarm topology reconfiguration and network optimisation
- **Performance Focus**: Communication pattern optimisation and adaptive network structures

## Core Capabilities

### 1. Dynamic Topology Reconfiguration
```javascript
// Advanced topology optimisation system
class TopologyOptimizer {
  constructor() {
    this.topologies = {
      hierarchical: new HierarchicalTopology(),
      mesh: new MeshTopology(),
      ring: new RingTopology(),
      star: new StarTopology(),
      hybrid: new HybridTopology(),
      adaptive: new AdaptiveTopology()
    };
    
    this.optimizer = new NetworkOptimizer();
    this.analyzer = new TopologyAnalyzer();
    this.predictor = new TopologyPredictor();
  }
  
  // Intelligent topology selection and optimisation
  async optimizeTopology(swarm, workloadProfile, constraints = {}) {
    // Analyze current topology performance
    const currentAnalysis = await this.analyzer.analyze(swarm.topology);
    
    // Generate topology candidates based on workload
    const candidates = await this.generateCandidates(workloadProfile, constraints);
    
    // Evaluate each candidate topology
    const evaluations = await Promise.all(
      candidates.map(candidate => this.evaluateTopology(candidate, workloadProfile))
    );
    
    // Select optimal topology using multi-objective optimisation
    const optimal = this.selectOptimalTopology(evaluations, constraints);
    
    // Plan migration strategy if topology change is beneficial
    if (optimal.improvement > constraints.minImprovement || 0.1) {
      const migrationPlan = await this.planMigration(swarm.topology, optimal.topology);
      return {
        recommended: optimal.topology,
        improvement: optimal.improvement,
        migrationPlan,
        estimatedDowntime: migrationPlan.estimatedDowntime,
        benefits: optimal.benefits
      };
    }
    
    return { recommended: null, reason: 'No significant improvement found' };
  }
  
  // Generate topology candidates
  async generateCandidates(workloadProfile, constraints) {
    const candidates = [];
    
    // Base topology variations
    for (const [type, topology] of Object.entries(this.topologies)) {
      if (this.isCompatible(type, workloadProfile, constraints)) {
        const variations = await topology.generateVariations(workloadProfile);
        candidates.push(...variations);
      }
    }
    
    // Hybrid topology generation
    const hybrids = await this.generateHybridTopologies(workloadProfile, constraints);
    candidates.push(...hybrids);
    
    // AI-generated novel topologies
    const aiGenerated = await this.generateAITopologies(workloadProfile);
    candidates.push(...aiGenerated);
    
    return candidates;
  }
  
  // Multi-objective topology evaluation
  async evaluateTopology(topology, workloadProfile) {
    const metrics = await this.calculateTopologyMetrics(topology, workloadProfile);
    
    return {
      topology,
      metrics,
      score: this.calculateOverallScore(metrics),
      strengths: this.identifyStrengths(metrics),
      weaknesses: this.identifyWeaknesses(metrics),
      suitability: this.calculateSuitability(metrics, workloadProfile)
    };
  }
}
```

### 2. Network Latency Optimisation
```javascript
// Advanced network latency optimisation
class NetworkLatencyOptimizer {
  constructor() {
    this.latencyAnalyzer = new LatencyAnalyzer();
    this.routingOptimizer = new RoutingOptimizer();
    this.bandwidthManager = new BandwidthManager();
  }
  
  // Comprehensive latency optimisation
  async optimizeLatency(network, communicationPatterns) {
    const optimisation = {
      // Physical network optimisation
      physical: await this.optimizePhysicalNetwork(network),
      
      // Logical routing optimisation
      routing: await this.optimizeRouting(network, communicationPatterns),
      
      // Protocol optimisation
      protocol: await this.optimizeProtocols(network),
      
      // Caching strategies
      caching: await this.optimizeCaching(communicationPatterns),
      
      // Compression optimisation
      compression: await this.optimizeCompression(communicationPatterns)
    };
    
    return optimisation;
  }
  
  // Physical network topology optimisation
  async optimizePhysicalNetwork(network) {
    // Calculate optimal agent placement
    const placement = await this.calculateOptimalPlacement(network.agents);
    
    // Minimise communication distance
    const distanceOptimization = this.optimizeCommunicationDistance(placement);
    
    // Bandwidth allocation optimisation
    const bandwidthOptimization = await this.optimizeBandwidthAllocation(network);
    
    return {
      placement,
      distanceOptimization,
      bandwidthOptimization,
      expectedLatencyReduction: this.calculateExpectedReduction(
        distanceOptimization, 
        bandwidthOptimization
      )
    };
  }
  
  // Intelligent routing optimisation
  async optimizeRouting(network, patterns) {
    // Analyze communication patterns
    const patternAnalysis = this.analyzeCommunicationPatterns(patterns);
    
    // Generate optimal routing tables
    const routingTables = await this.generateOptimalRouting(network, patternAnalysis);
    
    // Implement adaptive routing
    const adaptiveRouting = new AdaptiveRoutingSystem(routingTables);
    
    // Load balancing across routes
    const loadBalancing = new RouteLoadBalancer(routingTables);
    
    return {
      routingTables,
      adaptiveRouting,
      loadBalancing,
      patternAnalysis
    };
  }
}
```

### 3. Agent Placement Strategies
```javascript
// Sophisticated agent placement optimisation
class AgentPlacementOptimizer {
  constructor() {
    this.algorithms = {
      genetic: new GeneticPlacementAlgorithm(),
      simulated_annealing: new SimulatedAnnealingPlacement(),
      particle_swarm: new ParticleSwarmPlacement(),
      graph_partitioning: new GraphPartitioningPlacement(),
      machine_learning: new MLBasedPlacement()
    };
  }
  
  // Multi-algorithm agent placement optimisation
  async optimizePlacement(agents, constraints, objectives) {
    const results = new Map();
    
    // Run multiple algorithms in parallel
    const algorithmPromises = Object.entries(this.algorithms).map(
      async ([name, algorithm]) => {
        const result = await algorithm.optimise(agents, constraints, objectives);
        return [name, result];
      }
    );
    
    const algorithmResults = await Promise.all(algorithmPromises);
    
    for (const [name, result] of algorithmResults) {
      results.set(name, result);
    }
    
    // Ensemble optimisation - combine best results
    const ensembleResult = await this.ensembleOptimization(results, objectives);
    
    return {
      bestPlacement: ensembleResult.placement,
      algorithm: ensembleResult.algorithm,
      score: ensembleResult.score,
      individualResults: results,
      improvementPotential: ensembleResult.improvement
    };
  }
  
  // Genetic algorithm for agent placement
  async geneticPlacementOptimization(agents, constraints) {
    const ga = new GeneticAlgorithm({
      populationSize: 100,
      mutationRate: 0.1,
      crossoverRate: 0.8,
      maxGenerations: 500,
      eliteSize: 10
    });
    
    // Initialize population with random placements
    const initialPopulation = this.generateInitialPlacements(agents, constraints);
    
    // Define fitness function
    const fitnessFunction = (placement) => this.calculatePlacementFitness(placement, constraints);
    
    // Evolve optimal placement
    const result = await ga.evolve(initialPopulation, fitnessFunction);
    
    return {
      placement: result.bestIndividual,
      fitness: result.bestFitness,
      generations: result.generations,
      convergence: result.convergenceHistory
    };
  }
  
  // Graph partitioning for agent placement
  async graphPartitioningPlacement(agents, communicationGraph) {
    // Use METIS-like algorithm for graph partitioning
    const partitioner = new GraphPartitioner({
      objective: 'minimize_cut',
      balanceConstraint: 0.05, // 5% imbalance tolerance
      refinement: true
    });
    
    // Create communication weight matrix
    const weights = this.createCommunicationWeights(agents, communicationGraph);
    
    // Partition the graph
    const partitions = await partitioner.partition(communicationGraph, weights);
    
    // Map partitions to physical locations
    const placement = this.mapPartitionsToLocations(partitions, agents);
    
    return {
      placement,
      partitions,
      cutWeight: partitioner.getCutWeight(),
      balance: partitioner.getBalance()
    };
  }
}
```

### 4. Communication Pattern Optimisation
```javascript
// Advanced communication pattern optimisation
class CommunicationOptimizer {
  constructor() {
    this.patternAnalyzer = new PatternAnalyzer();
    this.protocolOptimizer = new ProtocolOptimizer();
    this.messageOptimizer = new MessageOptimizer();
    this.compressionEngine = new CompressionEngine();
  }
  
  // Comprehensive communication optimisation
  async optimizeCommunication(swarm, historicalData) {
    // Analyze communication patterns
    const patterns = await this.patternAnalyzer.analyze(historicalData);
    
    // Optimise based on pattern analysis
    const optimizations = {
      // Message batching optimisation
      batching: await this.optimizeMessageBatching(patterns),
      
      // Protocol selection optimisation
      protocols: await this.optimizeProtocols(patterns),
      
      // Compression optimisation
      compression: await this.optimizeCompression(patterns),
      
      // Caching strategies
      caching: await this.optimizeCaching(patterns),
      
      // Routing optimisation
      routing: await this.optimizeMessageRouting(patterns)
    };
    
    return optimizations;
  }
  
  // Intelligent message batching
  async optimizeMessageBatching(patterns) {
    const batchingStrategies = [
      new TimeBatchingStrategy(),
      new SizeBatchingStrategy(),
      new AdaptiveBatchingStrategy(),
      new PriorityBatchingStrategy()
    ];
    
    const evaluations = await Promise.all(
      batchingStrategies.map(strategy => 
        this.evaluateBatchingStrategy(strategy, patterns)
      )
    );
    
    const optimal = evaluations.reduce((best, current) => 
      current.score > best.score ? current : best
    );
    
    return {
      strategy: optimal.strategy,
      configuration: optimal.configuration,
      expectedImprovement: optimal.improvement,
      metrics: optimal.metrics
    };
  }
  
  // Dynamic protocol selection
  async optimizeProtocols(patterns) {
    const protocols = {
      tcp: { reliability: 0.99, latency: 'medium', overhead: 'high' },
      udp: { reliability: 0.95, latency: 'low', overhead: 'low' },
      websocket: { reliability: 0.98, latency: 'medium', overhead: 'medium' },
      grpc: { reliability: 0.99, latency: 'low', overhead: 'medium' },
      mqtt: { reliability: 0.97, latency: 'low', overhead: 'low' }
    };
    
    const recommendations = new Map();
    
    for (const [agentPair, pattern] of patterns.pairwisePatterns) {
      const optimal = this.selectOptimalProtocol(protocols, pattern);
      recommendations.set(agentPair, optimal);
    }
    
    return recommendations;
  }
}
```

## MCP Integration Hooks

### Topology Management Integration
```javascript
// Comprehensive MCP topology integration
const topologyIntegration = {
  // Real-time topology optimisation
  async optimizeSwarmTopology(swarmId, optimizationConfig = {}) {
    // Get current swarm status
    const swarmStatus = await mcp.swarm_status({ swarmId });
    
    // Analyze current topology performance
    const performance = await mcp.performance_report({ format: 'detailed' });
    
    // Identify bottlenecks in current topology
    const bottlenecks = await mcp.bottleneck_analyze({ component: 'topology' });
    
    // Generate optimisation recommendations
    const recommendations = await this.generateTopologyRecommendations(
      swarmStatus, 
      performance, 
      bottlenecks, 
      optimizationConfig
    );
    
    // Apply optimisation if beneficial
    if (recommendations.beneficial) {
      const result = await mcp.topology_optimize({ swarmId });
      
      // Monitor optimisation impact
      const impact = await this.monitorOptimizationImpact(swarmId, result);
      
      return {
        applied: true,
        recommendations,
        result,
        impact
      };
    }
    
    return {
      applied: false,
      recommendations,
      reason: 'No beneficial optimisation found'
    };
  },
  
  // Dynamic swarm scaling with topology consideration
  async scaleWithTopologyOptimization(swarmId, targetSize, workloadProfile) {
    // Current swarm state
    const currentState = await mcp.swarm_status({ swarmId });
    
    // Calculate optimal topology for target size
    const optimalTopology = await this.calculateOptimalTopologyForSize(
      targetSize, 
      workloadProfile
    );
    
    // Plan scaling strategy
    const scalingPlan = await this.planTopologyAwareScaling(
      currentState,
      targetSize,
      optimalTopology
    );
    
    // Execute scaling with topology optimisation
    const scalingResult = await mcp.swarm_scale({ 
      swarmId, 
      targetSize 
    });
    
    // Apply topology optimisation after scaling
    if (scalingResult.success) {
      await mcp.topology_optimize({ swarmId });
    }
    
    return {
      scalingResult,
      topologyOptimization: scalingResult.success,
      finalTopology: optimalTopology
    };
  },
  
  // Coordination optimisation
  async optimizeCoordination(swarmId) {
    // Analyze coordination patterns
    const coordinationMetrics = await mcp.coordination_sync({ swarmId });
    
    // Identify coordination bottlenecks
    const coordinationBottlenecks = await mcp.bottleneck_analyze({ 
      component: 'coordination' 
    });
    
    // Optimise coordination patterns
    const optimisation = await this.optimizeCoordinationPatterns(
      coordinationMetrics,
      coordinationBottlenecks
    );
    
    return optimisation;
  }
};
```

### Neural Network Integration
```javascript
// AI-powered topology optimisation
class NeuralTopologyOptimizer {
  constructor() {
    this.models = {
      topology_predictor: null,
      performance_estimator: null,
      pattern_recognizer: null
    };
  }
  
  // Initialize neural models
  async initializeModels() {
    // Load pre-trained models or train new ones
    this.models.topology_predictor = await mcp.model_load({ 
      modelPath: '/models/topology_optimizer.model' 
    });
    
    this.models.performance_estimator = await mcp.model_load({ 
      modelPath: '/models/performance_estimator.model' 
    });
    
    this.models.pattern_recognizer = await mcp.model_load({ 
      modelPath: '/models/pattern_recognizer.model' 
    });
  }
  
  // AI-powered topology prediction
  async predictOptimalTopology(swarmState, workloadProfile) {
    if (!this.models.topology_predictor) {
      await this.initializeModels();
    }
    
    // Prepare input features
    const features = this.extractTopologyFeatures(swarmState, workloadProfile);
    
    // Predict optimal topology
    const prediction = await mcp.neural_predict({
      modelId: this.models.topology_predictor.id,
      input: JSON.stringify(features)
    });
    
    return {
      predictedTopology: prediction.topology,
      confidence: prediction.confidence,
      expectedImprovement: prediction.improvement,
      reasoning: prediction.reasoning
    };
  }
  
  // Train topology optimisation model
  async trainTopologyModel(trainingData) {
    const trainingConfig = {
      pattern_type: 'optimisation',
      training_data: JSON.stringify(trainingData),
      epochs: 100
    };
    
    const trainingResult = await mcp.neural_train(trainingConfig);
    
    // Save trained model
    if (trainingResult.success) {
      await mcp.model_save({
        modelId: trainingResult.modelId,
        path: '/models/topology_optimizer.model'
      });
    }
    
    return trainingResult;
  }
}
```

## Advanced Optimisation Algorithms

### 1. Genetic Algorithm for Topology Evolution
```javascript
// Genetic algorithm implementation for topology optimisation
class GeneticTopologyOptimizer {
  constructor(config = {}) {
    this.populationSize = config.populationSize || 50;
    this.mutationRate = config.mutationRate || 0.1;
    this.crossoverRate = config.crossoverRate || 0.8;
    this.maxGenerations = config.maxGenerations || 100;
    this.eliteSize = config.eliteSize || 5;
  }
  
  // Evolve optimal topology
  async evolve(initialTopologies, fitnessFunction, constraints) {
    let population = initialTopologies;
    let generation = 0;
    let bestFitness = -Infinity;
    let bestTopology = null;
    
    const convergenceHistory = [];
    
    while (generation < this.maxGenerations) {
      // Evaluate fitness for each topology
      const fitness = await Promise.all(
        population.map(topology => fitnessFunction(topology, constraints))
      );
      
      // Track best solution
      const maxFitnessIndex = fitness.indexOf(Math.max(...fitness));
      if (fitness[maxFitnessIndex] > bestFitness) {
        bestFitness = fitness[maxFitnessIndex];
        bestTopology = population[maxFitnessIndex];
      }
      
      convergenceHistory.push({
        generation,
        bestFitness,
        averageFitness: fitness.reduce((a, b) => a + b) / fitness.length
      });
      
      // Selection
      const selected = this.selection(population, fitness);
      
      // Crossover
      const offspring = await this.crossover(selected);
      
      // Mutation
      const mutated = await this.mutation(offspring, constraints);
      
      // Next generation
      population = this.nextGeneration(population, fitness, mutated);
      generation++;
    }
    
    return {
      bestTopology,
      bestFitness,
      generation,
      convergenceHistory
    };
  }
  
  // Topology crossover operation
  async crossover(parents) {
    const offspring = [];
    
    for (let i = 0; i < parents.length - 1; i += 2) {
      if (Math.random() < this.crossoverRate) {
        const [child1, child2] = await this.crossoverTopologies(
          parents[i], 
          parents[i + 1]
        );
        offspring.push(child1, child2);
      } else {
        offspring.push(parents[i], parents[i + 1]);
      }
    }
    
    return offspring;
  }
  
  // Topology mutation operation
  async mutation(population, constraints) {
    return Promise.all(
      population.map(async topology => {
        if (Math.random() < this.mutationRate) {
          return await this.mutateTopology(topology, constraints);
        }
        return topology;
      })
    );
  }
}
```

### 2. Simulated Annealing for Topology Optimisation
```javascript
// Simulated annealing implementation
class SimulatedAnnealingOptimizer {
  constructor(config = {}) {
    this.initialTemperature = config.initialTemperature || 1000;
    this.coolingRate = config.coolingRate || 0.95;
    this.minTemperature = config.minTemperature || 1;
    this.maxIterations = config.maxIterations || 10000;
  }
  
  // Simulated annealing optimisation
  async optimise(initialTopology, objectiveFunction, constraints) {
    let currentTopology = initialTopology;
    let currentScore = await objectiveFunction(currentTopology, constraints);
    
    let bestTopology = currentTopology;
    let bestScore = currentScore;
    
    let temperature = this.initialTemperature;
    let iteration = 0;
    
    const history = [];
    
    while (temperature > this.minTemperature && iteration < this.maxIterations) {
      // Generate neighbour topology
      const neighborTopology = await this.generateNeighbor(currentTopology, constraints);
      const neighborScore = await objectiveFunction(neighborTopology, constraints);
      
      // Accept or reject the neighbour
      const deltaScore = neighborScore - currentScore;
      
      if (deltaScore > 0 || Math.random() < Math.exp(deltaScore / temperature)) {
        currentTopology = neighborTopology;
        currentScore = neighborScore;
        
        // Update best solution
        if (neighborScore > bestScore) {
          bestTopology = neighborTopology;
          bestScore = neighborScore;
        }
      }
      
      // Record history
      history.push({
        iteration,
        temperature,
        currentScore,
        bestScore
      });
      
      // Cool down
      temperature *= this.coolingRate;
      iteration++;
    }
    
    return {
      bestTopology,
      bestScore,
      iterations: iteration,
      history
    };
  }
  
  // Generate neighbour topology through local modifications
  async generateNeighbor(topology, constraints) {
    const modifications = [
      () => this.addConnection(topology, constraints),
      () => this.removeConnection(topology, constraints),
      () => this.modifyConnection(topology, constraints),
      () => this.relocateAgent(topology, constraints)
    ];
    
    const modification = modifications[Math.floor(Math.random() * modifications.length)];
    return await modification();
  }
}
```

## Operational Commands

### Topology Optimisation Commands
```bash
# Analyze current topology
npx claude-flow topology-analyze --swarm-id <id> --metrics performance

# Optimise topology automatically
npx claude-flow topology-optimise --swarm-id <id> --strategy adaptive

# Compare topology configurations
npx claude-flow topology-compare --topologies ["hierarchical", "mesh", "hybrid"]

# Generate topology recommendations
npx claude-flow topology-recommend --workload-profile <file> --constraints <file>

# Monitor topology performance
npx claude-flow topology-monitor --swarm-id <id> --interval 60
```

### Agent Placement Commands
```bash
# Optimise agent placement
npx claude-flow placement-optimise --algorithm genetic --agents <agent-list>

# Analyze placement efficiency
npx claude-flow placement-analyze --current-placement <config>

# Generate placement recommendations
npx claude-flow placement-recommend --communication-patterns <file>
```

## Integration Points

### With Other Optimisation Agents
- **Load Balancer**: Coordinates topology changes with load distribution
- **Performance Monitor**: Receives topology performance metrics
- **Resource Manager**: Considers resource constraints in topology decisions

### With Swarm Infrastructure
- **Task Orchestrator**: Adapts task distribution to topology changes
- **Agent Coordinator**: Manages agent connections during topology updates
- **Memory System**: Stores topology optimisation history and patterns

## Performance Metrics

### Topology Performance Indicators
```javascript
// Comprehensive topology metrics
const topologyMetrics = {
  // Communication efficiency
  communicationEfficiency: {
    latency: this.calculateAverageLatency(),
    throughput: this.calculateThroughput(),
    bandwidth_utilization: this.calculateBandwidthUtilization(),
    message_overhead: this.calculateMessageOverhead()
  },
  
  // Network topology metrics
  networkMetrics: {
    diameter: this.calculateNetworkDiameter(),
    clustering_coefficient: this.calculateClusteringCoefficient(),
    betweenness_centrality: this.calculateBetweennessCentrality(),
    degree_distribution: this.calculateDegreeDistribution()
  },
  
  // Fault tolerance
  faultTolerance: {
    connectivity: this.calculateConnectivity(),
    redundancy: this.calculateRedundancy(),
    single_point_failures: this.identifySinglePointFailures(),
    recovery_time: this.calculateRecoveryTime()
  },
  
  // Scalability metrics
  scalability: {
    growth_capacity: this.calculateGrowthCapacity(),
    scaling_efficiency: this.calculateScalingEfficiency(),
    bottleneck_points: this.identifyBottleneckPoints(),
    optimal_size: this.calculateOptimalSize()
  }
};
```

This Topology Optimizer agent provides sophisticated swarm topology optimisation with AI-powered decision making, advanced algorithms, and comprehensive performance monitoring for optimal swarm coordination.

## Related Topics

- [Agent Orchestration Architecture](../../../features/agent-orchestration.md)
- [Agent Type Conventions and Mapping](../../../AGENT_TYPE_CONVENTIONS.md)
- [Agent Visualisation Architecture](../../../agent-visualization-architecture.md)
- [Agentic Alliance](../../../archive/legacy/old_markdown/Agentic Alliance.md)
- [Agentic Metaverse for Global Creatives](../../../archive/legacy/old_markdown/Agentic Metaverse for Global Creatives.md)
- [Agentic Mycelia](../../../archive/legacy/old_markdown/Agentic Mycelia.md)
- [Agents](../../../archive/legacy/old_markdown/Agents.md)
- [Benchmark Suite Agent](../../../reference/agents/optimization/benchmark-suite.md)
- [Claude Code Agents Directory Structure](../../../reference/agents/README.md)
- [Claude Flow Commands to Agent System Migration Summary](../../../reference/agents/MIGRATION_SUMMARY.md)
- [Distributed Consensus Builder Agents](../../../reference/agents/consensus/README.md)
- [Financialised Agentic Memetics](../../../archive/legacy/old_markdown/Financialised Agentic Memetics.md)
- [Load Balancing Coordinator Agent](../../../reference/agents/optimization/load-balancer.md)
- [Multi Agent Orchestration](../../../server/agent-swarm.md)
- [Multi Agent RAG scrapbook](../../../archive/legacy/old_markdown/Multi Agent RAG scrapbook.md)
- [Multi-Agent Container Setup](../../../deployment/multi-agent-setup.md)
- [Multi-MCP Agent Visualisation System](../../../MCP_AGENT_VISUALIZATION.md)
- [Multi-MCP Agent Visualisation System](../../../multi-mcp-agent-visualization.md)
- [Performance Monitor Agent](../../../reference/agents/optimization/performance-monitor.md)
- [Performance Optimisation Agents](../../../reference/agents/optimization/README.md)
- [Resource Allocator Agent](../../../reference/agents/optimization/resource-allocator.md)
- [Swarm Coordination Agents](../../../reference/agents/swarm/README.md)
- [adaptive-coordinator](../../../reference/agents/swarm/adaptive-coordinator.md)
- [analyze-code-quality](../../../reference/agents/analysis/code-review/analyze-code-quality.md)
- [arch-system-design](../../../reference/agents/architecture/system-design/arch-system-design.md)
- [architecture](../../../reference/agents/sparc/architecture.md)
- [automation-smart-agent](../../../reference/agents/templates/automation-smart-agent.md)
- [base-template-generator](../../../reference/agents/base-template-generator.md)
- [byzantine-coordinator](../../../reference/agents/consensus/byzantine-coordinator.md)
- [code-analyzer](../../../reference/agents/analysis/code-analyzer.md)
- [code-review-swarm](../../../reference/agents/github/code-review-swarm.md)
- [coder](../../../reference/agents/core/coder.md)
- [coordinator-swarm-init](../../../reference/agents/templates/coordinator-swarm-init.md)
- [crdt-synchronizer](../../../reference/agents/consensus/crdt-synchronizer.md)
- [data-ml-model](../../../reference/agents/data/ml/data-ml-model.md)
- [dev-backend-api](../../../reference/agents/development/backend/dev-backend-api.md)
- [docs-api-openapi](../../../reference/agents/documentation/api-docs/docs-api-openapi.md)
- [github-modes](../../../reference/agents/github/github-modes.md)
- [github-pr-manager](../../../reference/agents/templates/github-pr-manager.md)
- [gossip-coordinator](../../../reference/agents/consensus/gossip-coordinator.md)
- [hierarchical-coordinator](../../../reference/agents/swarm/hierarchical-coordinator.md)
- [implementer-sparc-coder](../../../reference/agents/templates/implementer-sparc-coder.md)
- [issue-tracker](../../../reference/agents/github/issue-tracker.md)
- [memory-coordinator](../../../reference/agents/templates/memory-coordinator.md)
- [mesh-coordinator](../../../reference/agents/swarm/mesh-coordinator.md)
- [migration-plan](../../../reference/agents/templates/migration-plan.md)
- [multi-repo-swarm](../../../reference/agents/github/multi-repo-swarm.md)
- [ops-cicd-github](../../../reference/agents/devops/ci-cd/ops-cicd-github.md)
- [orchestrator-task](../../../reference/agents/templates/orchestrator-task.md)
- [performance-analyzer](../../../reference/agents/templates/performance-analyzer.md)
- [performance-benchmarker](../../../reference/agents/consensus/performance-benchmarker.md)
- [planner](../../../reference/agents/core/planner.md)
- [pr-manager](../../../reference/agents/github/pr-manager.md)
- [production-validator](../../../reference/agents/testing/validation/production-validator.md)
- [project-board-sync](../../../reference/agents/github/project-board-sync.md)
- [pseudocode](../../../reference/agents/sparc/pseudocode.md)
- [quorum-manager](../../../reference/agents/consensus/quorum-manager.md)
- [raft-manager](../../../reference/agents/consensus/raft-manager.md)
- [refinement](../../../reference/agents/sparc/refinement.md)
- [release-manager](../../../reference/agents/github/release-manager.md)
- [release-swarm](../../../reference/agents/github/release-swarm.md)
- [repo-architect](../../../reference/agents/github/repo-architect.md)
- [researcher](../../../reference/agents/core/researcher.md)
- [reviewer](../../../reference/agents/core/reviewer.md)
- [security-manager](../../../reference/agents/consensus/security-manager.md)
- [sparc-coordinator](../../../reference/agents/templates/sparc-coordinator.md)
- [spec-mobile-react-native](../../../reference/agents/specialized/mobile/spec-mobile-react-native.md)
- [specification](../../../reference/agents/sparc/specification.md)
- [swarm-issue](../../../reference/agents/github/swarm-issue.md)
- [swarm-pr](../../../reference/agents/github/swarm-pr.md)
- [sync-coordinator](../../../reference/agents/github/sync-coordinator.md)
- [tdd-london-swarm](../../../reference/agents/testing/unit/tdd-london-swarm.md)
- [tester](../../../reference/agents/core/tester.md)
- [workflow-automation](../../../reference/agents/github/workflow-automation.md)
