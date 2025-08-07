#!/usr/bin/env node

import { spawn } from 'child_process';
import assert from 'assert';

const tests = [];

// Test helper
function test(name, fn) {
  tests.push({ name, fn });
}

// MCP client for testing
class TestMCPClient {
  constructor() {
    this.process = null;
    this.responses = new Map();
    this.nextId = 1;
  }
  
  async start() {
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
    
    // Wait for startup
    await new Promise(resolve => setTimeout(resolve, 1000));
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
      
      // Timeout after 5 seconds
      setTimeout(() => {
        if (this.responses.has(id)) {
          this.responses.delete(id);
          reject(new Error('Request timeout'));
        }
      }, 5000);
    });
  }
  
  async toolCall(name, args) {
    return this.request('tools/call', {
      name,
      arguments: args
    });
  }
  
  stop() {
    if (this.process) {
      this.process.kill();
    }
  }
}

// Tests

test('Initialize MCP connection', async () => {
  const client = new TestMCPClient();
  await client.start();
  
  const response = await client.request('initialize', {
    protocolVersion: '0.1.0',
    clientInfo: {
      name: 'test-client',
      version: '1.0.0'
    }
  });
  
  assert.strictEqual(response.result.protocolVersion, '0.1.0');
  assert.strictEqual(response.result.serverInfo.name, 'mcp-observability-server');
  
  client.stop();
});

test('List available tools', async () => {
  const client = new TestMCPClient();
  await client.start();
  
  await client.request('initialize');
  const response = await client.request('tools/list');
  
  assert(Array.isArray(response.result.tools));
  assert(response.result.tools.length > 40); // Should have 40+ tools
  
  // Check for key tools
  const toolNames = response.result.tools.map(t => t.name);
  assert(toolNames.includes('agent.create'));
  assert(toolNames.includes('swarm.initialize'));
  assert(toolNames.includes('message.send'));
  assert(toolNames.includes('visualization.snapshot'));
  
  client.stop();
});

test('Create an agent', async () => {
  const client = new TestMCPClient();
  await client.start();
  
  await client.request('initialize');
  
  const response = await client.toolCall('agent.create', {
    name: 'Test Agent',
    type: 'coder',
    capabilities: ['javascript', 'testing']
  });
  
  assert(response.result);
  const result = JSON.parse(response.result.content[0].text);
  assert(result.success);
  assert(result.agent.id);
  assert.strictEqual(result.agent.name, 'Test Agent');
  assert.strictEqual(result.agent.type, 'coder');
  
  client.stop();
});

test('Initialize a swarm', async () => {
  const client = new TestMCPClient();
  await client.start();
  
  await client.request('initialize');
  
  const response = await client.toolCall('swarm.initialize', {
    topology: 'mesh',
    agentConfig: {
      workerTypes: [
        { type: 'coder', count: 2 },
        { type: 'tester', count: 1 }
      ]
    }
  });
  
  const result = JSON.parse(response.result.content[0].text);
  assert(result.success);
  assert(result.swarmId);
  assert.strictEqual(result.topology, 'mesh');
  assert.strictEqual(result.agentCount, 4); // 1 coordinator + 2 coders + 1 tester
  
  client.stop();
});

test('Send a message', async () => {
  const client = new TestMCPClient();
  await client.start();
  
  await client.request('initialize');
  
  // Create two agents first
  const agent1 = await client.toolCall('agent.create', {
    name: 'Sender',
    type: 'coordinator'
  });
  const agent2 = await client.toolCall('agent.create', {
    name: 'Receiver',
    type: 'coder'
  });
  
  const sender = JSON.parse(agent1.result.content[0].text).agent;
  const receiver = JSON.parse(agent2.result.content[0].text).agent;
  
  // Send message
  const response = await client.toolCall('message.send', {
    from: sender.id,
    to: receiver.id,
    type: 'task',
    priority: 3,
    content: { task: 'test-task' }
  });
  
  const result = JSON.parse(response.result.content[0].text);
  assert(result.success);
  assert(result.message.id);
  assert(result.message.springForce > 0);
  
  client.stop();
});

test('Get visualization snapshot', async () => {
  const client = new TestMCPClient();
  await client.start();
  
  await client.request('initialize');
  
  // Create some agents
  await client.toolCall('agent.spawn', {
    count: 3,
    type: 'coder',
    namePrefix: 'test'
  });
  
  const response = await client.toolCall('visualization.snapshot', {
    includePositions: true,
    includeVelocities: true,
    includeConnections: true
  });
  
  const result = JSON.parse(response.result.content[0].text);
  assert(result.success);
  assert(result.snapshot.agents);
  assert(result.snapshot.agents.length >= 3);
  assert(result.snapshot.agents[0].position);
  assert(result.snapshot.physics);
  
  client.stop();
});

test('Store and retrieve memory', async () => {
  const client = new TestMCPClient();
  await client.start();
  
  await client.request('initialize');
  
  // Store data
  const storeResponse = await client.toolCall('memory.store', {
    key: 'test/data',
    value: { foo: 'bar', count: 42 },
    section: 'global'
  });
  
  const storeResult = JSON.parse(storeResponse.result.content[0].text);
  assert(storeResult.success);
  
  // Retrieve data
  const retrieveResponse = await client.toolCall('memory.retrieve', {
    key: 'test/data',
    section: 'global'
  });
  
  const retrieveResult = JSON.parse(retrieveResponse.result.content[0].text);
  assert(retrieveResult.success);
  assert.deepStrictEqual(retrieveResult.value, { foo: 'bar', count: 42 });
  
  client.stop();
});

test('Performance analysis', async () => {
  const client = new TestMCPClient();
  await client.start();
  
  await client.request('initialize');
  
  const response = await client.toolCall('performance.analyze', {
    metrics: ['throughput', 'latency'],
    aggregation: 'avg'
  });
  
  const result = JSON.parse(response.result.content[0].text);
  assert(result.success);
  assert(result.analysis);
  assert(result.report);
  
  client.stop();
});

// Run tests
async function runTests() {
  console.log('Running MCP Observability Tests...\n');
  
  let passed = 0;
  let failed = 0;
  
  for (const { name, fn } of tests) {
    try {
      await fn();
      console.log(`✅ ${name}`);
      passed++;
    } catch (error) {
      console.error(`❌ ${name}`);
      console.error(`   ${error.message}`);
      failed++;
    }
  }
  
  console.log(`\nTests: ${passed} passed, ${failed} failed, ${tests.length} total`);
  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(console.error);