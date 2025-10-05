#!/usr/bin/env node

/**
 * Test TCP MCP Endpoint
 * Tests the MCP server at 172.18.0.4:9500 to verify database isolation
 */

const net = require('net');
const readline = require('readline');

const HOST = process.env.MCP_HOST || '172.18.0.4';
const PORT = process.env.MCP_PORT || 9500;

class MCPClient {
  constructor(host, port) {
    this.host = host;
    this.port = port;
    this.socket = null;
    this.rl = null;
    this.messageId = 1;
  }

  connect() {
    return new Promise((resolve, reject) => {
      console.log(`🔌 Connecting to MCP server at ${this.host}:${this.port}...`);

      this.socket = net.createConnection(this.port, this.host, () => {
        console.log('✅ Connected to MCP server');

        this.rl = readline.createInterface({
          input: this.socket,
          crlfDelay: Infinity
        });

        this.rl.on('line', (line) => {
          if (line.trim()) {
            try {
              const msg = JSON.parse(line);
              console.log('📨 Received:', JSON.stringify(msg, null, 2));
            } catch (e) {
              console.log('📨 Raw:', line);
            }
          }
        });

        resolve();
      });

      this.socket.on('error', (err) => {
        console.error('❌ Connection error:', err.message);
        reject(err);
      });

      this.socket.on('close', () => {
        console.log('🔌 Connection closed');
      });
    });
  }

  sendMessage(method, params = {}) {
    const message = {
      jsonrpc: "2.0",
      id: this.messageId++,
      method,
      params
    };

    console.log('📤 Sending:', JSON.stringify(message, null, 2));
    this.socket.write(JSON.stringify(message) + '\n');
  }

  async test() {
    try {
      await this.connect();

      // Wait a bit for connection to stabilize
      await new Promise(resolve => setTimeout(resolve, 500));

      // 1. Initialize
      console.log('\n📋 Test 1: Initialize MCP connection');
      this.sendMessage('initialize', {
        protocolVersion: "2024-11-05",
        capabilities: {
          tools: { listChanged: true }
        },
        clientInfo: {
          name: "tcp-test-client",
          version: "1.0.0"
        }
      });

      await new Promise(resolve => setTimeout(resolve, 2000));

      // 2. List tools
      console.log('\n📋 Test 2: List available tools');
      this.sendMessage('tools/list');

      await new Promise(resolve => setTimeout(resolve, 2000));

      // 3. Call a tool (spawn agent)
      console.log('\n📋 Test 3: Spawn agent via claude-flow');
      this.sendMessage('tools/call', {
        name: 'spawn',
        arguments: {
          task: 'create hello world in python',
          agent_name: 'test-agent-' + Date.now()
        }
      });

      await new Promise(resolve => setTimeout(resolve, 5000));

      // 4. Check agent status
      console.log('\n📋 Test 4: List agents');
      this.sendMessage('tools/call', {
        name: 'list_agents',
        arguments: {}
      });

      await new Promise(resolve => setTimeout(resolve, 3000));

      console.log('\n✅ Test complete - disconnecting');
      this.socket.end();

    } catch (err) {
      console.error('❌ Test failed:', err);
      process.exit(1);
    }
  }

  close() {
    if (this.socket) {
      this.socket.end();
    }
  }
}

// Run test
const client = new MCPClient(HOST, PORT);
client.test().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n⚠️  Interrupted, closing connection...');
  client.close();
  process.exit(0);
});
