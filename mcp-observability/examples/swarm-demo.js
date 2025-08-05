#!/usr/bin/env node

/**
 * MCP Observability Swarm Demo
 * 
 * This example demonstrates how to:
 * 1. Initialize a hierarchical swarm
 * 2. Create multiple agents with different roles
 * 3. Send messages between agents
 * 4. Monitor performance
 * 5. Get visualization data
 */

import { spawn } from 'child_process';

class MCPClient {
  constructor() {
    this.process = null;
    this.responses = new Map();
    this.nextId = 1;
    this.ready = false;
  }
  
  async connect() {
    console.log('🔌 Connecting to MCP Observability Server...');
    
    this.process = spawn('node', ['../src/index.js'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: __dirname
    });
    
    this.process.stdout.on('data', (data) => {
      const lines = data.toString().split('\n').filter(l => l.trim());
      lines.forEach(line => {
        if (line.startsWith('{')) {
          try {
            const response = JSON.parse(line);
            if (response.id && this.responses.has(response.id)) {
              this.responses.get(response.id).resolve(response);
            }
          } catch (e) {
            // Ignore non-JSON output
          }
        }
      });
    });
    
    this.process.stderr.on('data', (data) => {
      console.error('MCP Error:', data.toString());
    });
    
    // Initialize
    await this.request('initialize', {
      protocolVersion: '0.1.0',
      clientInfo: {
        name: 'swarm-demo',
        version: '1.0.0'
      }
    });
    
    this.ready = true;
    console.log('✅ Connected to MCP server\n');
  }
  
  async request(method, params = {}) {
    const id = (this.nextId++).toString();
    const request = {
      jsonrpc: '2.0',
      id,
      method,
      params
    };
    
    return new Promise((resolve, reject) => {
      this.responses.set(id, { resolve, reject });
      this.process.stdin.write(JSON.stringify(request) + '\n');
      
      setTimeout(() => {
        if (this.responses.has(id)) {
          this.responses.delete(id);
          reject(new Error('Request timeout'));
        }
      }, 10000);
    });
  }
  
  async call(toolName, args) {
    const response = await this.request('tools/call', {
      name: toolName,
      arguments: args
    });
    
    if (response.error) {
      throw new Error(response.error.message);
    }
    
    return JSON.parse(response.result.content[0].text);
  }
  
  disconnect() {
    if (this.process) {
      this.process.kill();
    }
  }
}

// Demo functions

async function initializeSwarm(client) {
  console.log('🐝 Initializing Hierarchical Swarm...');
  
  const result = await client.call('swarm.initialize', {
    topology: 'hierarchical',
    physicsConfig: {
      springStrength: 0.12,
      damping: 0.95,
      linkDistance: 10.0,
      nodeRepulsion: 600.0
    },
    agentConfig: {
      coordinatorCount: 1,
      workerTypes: [
        { type: 'architect', count: 1 },
        { type: 'coder', count: 3 },
        { type: 'tester', count: 2 },
        { type: 'analyst', count: 1 },
        { type: 'monitor', count: 1 }
      ]
    }
  });
  
  console.log(`✅ Swarm initialized: ${result.swarmId}`);
  console.log(`   Topology: ${result.topology}`);
  console.log(`   Agents: ${result.agentCount}`);
  console.log('');
  
  return result;
}

async function simulateActivity(client, agents) {
  console.log('💬 Simulating Agent Communication...\n');
  
  // Find key agents
  const coordinator = agents.find(a => a.type === 'coordinator');
  const architect = agents.find(a => a.type === 'architect');
  const coders = agents.filter(a => a.type === 'coder');
  const testers = agents.filter(a => a.type === 'tester');
  
  // 1. Coordinator broadcasts task
  console.log('📢 Coordinator broadcasting task assignment...');
  await client.call('message.broadcast', {
    from: coordinator.id,
    type: 'coordination',
    priority: 5,
    content: {
      task: 'implement-user-authentication',
      deadline: '2024-01-25',
      requirements: ['JWT', 'OAuth2', 'MFA']
    }
  });
  
  // 2. Architect sends design to coders
  console.log('🏗️  Architect sending design specifications...');
  await client.call('message.send', {
    from: architect.id,
    to: coders.map(c => c.id),
    type: 'task',
    priority: 4,
    content: {
      design: 'microservice-architecture',
      endpoints: ['/auth/login', '/auth/refresh', '/auth/logout'],
      database: 'PostgreSQL'
    }
  });
  
  // 3. Coders communicate with each other
  console.log('💻 Coders coordinating implementation...');
  for (let i = 0; i < coders.length - 1; i++) {
    await client.call('message.send', {
      from: coders[i].id,
      to: coders[i + 1].id,
      type: 'coordination',
      priority: 3,
      content: {
        module: `auth-module-${i + 1}`,
        status: 'in-progress'
      }
    });
  }
  
  // 4. Coders send to testers
  console.log('🧪 Sending completed modules to testers...');
  for (let i = 0; i < Math.min(coders.length, testers.length); i++) {
    await client.call('message.send', {
      from: coders[i].id,
      to: testers[i % testers.length].id,
      type: 'data',
      priority: 3,
      content: {
        module: `auth-module-${i + 1}`,
        coverage: 85 + Math.random() * 15,
        status: 'ready-for-testing'
      }
    });
  }
  
  // 5. Update agent statuses
  console.log('📊 Updating agent statuses...');
  
  // Make coordinator busy
  await client.call('agent.update', {
    agentId: coordinator.id,
    status: 'busy',
    performance: {
      successRate: 98,
      resourceUtilization: 0.7
    }
  });
  
  // Make coders active
  for (const coder of coders) {
    await client.call('agent.update', {
      agentId: coder.id,
      status: 'active',
      performance: {
        successRate: 85 + Math.random() * 15,
        resourceUtilization: 0.6 + Math.random() * 0.3
      }
    });
  }
  
  console.log('');
}

async function monitorPerformance(client) {
  console.log('📈 Analyzing Performance...\n');
  
  // Get performance metrics
  const metrics = await client.call('performance.metrics', {
    detailed: true
  });
  
  console.log('System Metrics:');
  console.log(`  Active Agents: ${metrics.metrics.activeAgents}`);
  console.log(`  Message Rate: ${metrics.metrics.messageRate.toFixed(2)} msg/s`);
  console.log(`  Avg Latency: ${metrics.metrics.avgLatency.toFixed(2)} ms`);
  console.log(`  Network Health: ${(metrics.metrics.networkHealth * 100).toFixed(1)}%`);
  console.log(`  Memory Usage: ${metrics.metrics.memoryUsage.toFixed(2)} MB`);
  
  if (metrics.details.bottlenecks.length > 0) {
    console.log('\n⚠️  Detected Bottlenecks:');
    metrics.details.bottlenecks.forEach(b => {
      console.log(`  - ${b.type}: ${b.recommendation}`);
    });
  }
  
  console.log('');
}

async function getVisualizationData(client) {
  console.log('🎨 Getting Visualization Snapshot...\n');
  
  const snapshot = await client.call('visualization.snapshot', {
    includePositions: true,
    includeVelocities: true,
    includeForces: false,
    includeConnections: true
  });
  
  console.log(`Visualization Data:`);
  console.log(`  Frame: ${snapshot.snapshot.frameCount}`);
  console.log(`  Agents: ${snapshot.snapshot.agentCount}`);
  
  // Show agent positions
  console.log('\nAgent Positions:');
  snapshot.snapshot.agents.slice(0, 5).forEach(agent => {
    console.log(`  ${agent.name}: (${
      agent.position.x.toFixed(1)}, ${
      agent.position.y.toFixed(1)}, ${
      agent.position.z.toFixed(1)})`);
  });
  
  if (snapshot.snapshot.agents.length > 5) {
    console.log(`  ... and ${snapshot.snapshot.agents.length - 5} more agents`);
  }
  
  // Calculate message flow
  const messageFlow = await client.call('message.flow', {
    timeWindow: 60
  });
  
  console.log(`\nMessage Flow (last 60s):`);
  console.log(`  Total Messages: ${messageFlow.count}`);
  console.log(`  Avg Latency: ${messageFlow.statistics.avgLatency} ms`);
  
  if (messageFlow.patterns.length > 0) {
    console.log(`\nCommunication Patterns:`);
    messageFlow.patterns.forEach(pattern => {
      console.log(`  - ${pattern.type}: ${pattern.count} occurrences`);
    });
  }
  
  console.log('');
}

async function demonstrateNeuralLearning(client) {
  console.log('🧠 Training Neural Patterns...\n');
  
  // Train coordination pattern
  const training = await client.call('neural.train', {
    pattern: 'coordination',
    epochs: 50
  });
  
  console.log(`Training Complete:`);
  console.log(`  Model ID: ${training.modelId}`);
  console.log(`  Final Error: ${training.trainingMetrics.finalError.toFixed(4)}`);
  console.log(`  Convergence Rate: ${(training.trainingMetrics.convergenceRate * 100).toFixed(1)}%`);
  
  // Make prediction
  const prediction = await client.call('neural.predict', {
    scenario: {
      agentCount: 20,
      taskComplexity: 0.7,
      urgency: 0.8,
      resourceAvailability: 0.6
    }
  });
  
  console.log(`\nPrediction:`);
  console.log(`  ${prediction.bestRecommendation}`);
  
  console.log('');
}

async function demonstrateMemory(client) {
  console.log('💾 Demonstrating Memory System...\n');
  
  // Store swarm configuration
  await client.call('memory.store', {
    key: 'demo/swarm-config',
    value: {
      name: 'Authentication Team',
      topology: 'hierarchical',
      created: new Date().toISOString(),
      performance: {
        successRate: 92.5,
        avgResponseTime: 145
      }
    },
    section: 'swarm',
    ttl: 3600 // 1 hour
  });
  
  // Store pattern
  await client.call('memory.store', {
    key: 'demo/successful-pattern',
    value: {
      pattern: 'coordinator-architect-coder-tester',
      successRate: 0.95,
      avgCompletionTime: 3200
    },
    section: 'patterns'
  });
  
  // List stored keys
  const keys = await client.call('memory.list', {
    pattern: 'demo',
    section: 'all',
    includeMetadata: true
  });
  
  console.log('Stored Memory Keys:');
  keys.results.forEach(result => {
    if (result.keys.length > 0) {
      console.log(`  Section: ${result.section}`);
      result.keys.forEach(key => {
        console.log(`    - ${key.key || key} (${key.metadata?.size || 0} bytes)`);
      });
    }
  });
  
  console.log('');
}

// Main demo flow
async function runDemo() {
  const client = new MCPClient();
  
  try {
    // Connect to MCP server
    await client.connect();
    
    // 1. Initialize swarm
    const swarmResult = await initializeSwarm(client);
    
    // 2. Simulate agent activity
    await simulateActivity(client, swarmResult.agents);
    
    // Wait a bit for physics to settle
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // 3. Monitor performance
    await monitorPerformance(client);
    
    // 4. Get visualization data
    await getVisualizationData(client);
    
    // 5. Demonstrate neural learning
    await demonstrateNeuralLearning(client);
    
    // 6. Demonstrate memory system
    await demonstrateMemory(client);
    
    // Get final swarm status
    console.log('🏁 Final Swarm Status:\n');
    const status = await client.call('swarm.status', {
      includeAgents: true,
      includeMetrics: true,
      includePhysics: false
    });
    
    const swarm = status.swarms[0];
    console.log(`Swarm: ${swarm.name}`);
    console.log(`  Health Score: ${(swarm.healthScore * 100).toFixed(1)}%`);
    console.log(`  Active Agents: ${swarm.activeAgents}/${swarm.totalAgents}`);
    console.log(`  Coordination Efficiency: ${(swarm.coordinationEfficiency * 100).toFixed(1)}%`);
    
    console.log('\n✨ Demo completed successfully!');
    
  } catch (error) {
    console.error('❌ Demo failed:', error);
  } finally {
    // Cleanup
    client.disconnect();
    process.exit(0);
  }
}

// Run the demo
console.log('🚀 MCP Observability Swarm Demo\n');
console.log('This demo will:');
console.log('1. Initialize a hierarchical swarm');
console.log('2. Create agents with different roles');
console.log('3. Simulate communication patterns');
console.log('4. Monitor performance metrics');
console.log('5. Demonstrate neural learning');
console.log('6. Show memory persistence\n');

runDemo().catch(console.error);