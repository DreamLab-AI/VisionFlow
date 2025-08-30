#!/usr/bin/env node

/**
 * Persistent MCP TCP Server
 * Maintains a single MCP instance and shares it across all TCP connections
 */

const { spawn } = require('child_process');
const net = require('net');
const readline = require('readline');

const TCP_PORT = process.env.MCP_TCP_PORT || 9500;
const LOG_LEVEL = process.env.MCP_LOG_LEVEL || 'info';

class PersistentMCPServer {
  constructor() {
    this.mcpProcess = null;
    this.mcpInterface = null;
    this.clients = new Map();
    this.requestQueue = [];
    this.isProcessing = false;
    this.initialized = false;
    this.initPromise = null;
  }

  log(level, message, ...args) {
    const levels = { debug: 0, info: 1, warn: 2, error: 3 };
    if (levels[level] >= levels[LOG_LEVEL]) {
      console.log(`[PMCP-${level.toUpperCase()}] ${new Date().toISOString()} ${message}`, ...args);
    }
  }

  async startMCPProcess() {
    if (this.mcpProcess) {
      this.log('warn', 'MCP process already running');
      return;
    }

    this.log('info', 'Starting persistent MCP process...');
    
    // Start a single MCP instance
    this.mcpProcess = spawn('npx', ['claude-flow@alpha', 'mcp', 'start', '--stdio'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: '/workspace',
      env: { ...process.env, CLAUDE_FLOW_DIRECT_MODE: 'true' }
    });

    // Create readline interface for parsing responses
    this.mcpInterface = readline.createInterface({
      input: this.mcpProcess.stdout,
      crlfDelay: Infinity
    });

    // Handle MCP output
    this.mcpInterface.on('line', (line) => {
      this.handleMCPOutput(line);
    });

    // Handle MCP errors
    this.mcpProcess.stderr.on('data', (data) => {
      this.log('error', `MCP stderr: ${data}`);
    });

    // Handle MCP exit
    this.mcpProcess.on('close', (code) => {
      this.log('error', `MCP process exited with code ${code}`);
      this.mcpProcess = null;
      this.mcpInterface = null;
      this.initialized = false;
      
      // Restart after a delay
      setTimeout(() => this.startMCPProcess(), 5000);
    });

    // Initialize MCP
    await this.initializeMCP();
  }

  async initializeMCP() {
    if (this.initialized) return;
    
    this.log('info', 'Initializing MCP protocol...');
    
    const initRequest = {
      jsonrpc: "2.0",
      id: "init-" + Date.now(),
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {
          tools: { listChanged: true },
          resources: { subscribe: true, listChanged: true }
        },
        clientInfo: {
          name: "persistent-tcp-wrapper",
          version: "1.0.0"
        }
      }
    };

    return new Promise((resolve) => {
      this.initPromise = { resolve, id: initRequest.id };
      this.mcpProcess.stdin.write(JSON.stringify(initRequest) + '\n');
    });
  }

  handleMCPOutput(line) {
    // Skip non-JSON lines
    if (!line.startsWith('{')) {
      this.log('debug', `Skipping non-JSON: ${line}`);
      return;
    }

    try {
      const message = JSON.parse(line);
      
      // Handle initialization response
      if (this.initPromise && message.id === this.initPromise.id) {
        this.initialized = true;
        this.log('info', 'MCP initialized successfully');
        this.initPromise.resolve();
        this.initPromise = null;
        return;
      }

      // Handle notifications (no id field)
      if (!message.id) {
        this.log('debug', `Notification: ${message.method}`);
        // Broadcast notifications to all clients
        this.broadcastToClients(line);
        return;
      }

      // Route response to appropriate client
      const clientId = this.findClientByRequestId(message.id);
      if (clientId) {
        const client = this.clients.get(clientId);
        if (client && client.socket) {
          client.socket.write(line + '\n');
          this.log('debug', `Routed response ${message.id} to client ${clientId}`);
        }
      } else {
        this.log('warn', `No client found for response ${message.id}`);
      }
    } catch (err) {
      this.log('error', `Failed to parse MCP output: ${err.message}`);
    }
  }

  findClientByRequestId(requestId) {
    for (const [clientId, client] of this.clients) {
      if (client.pendingRequests && client.pendingRequests.has(requestId)) {
        client.pendingRequests.delete(requestId);
        return clientId;
      }
    }
    return null;
  }

  broadcastToClients(message) {
    for (const [clientId, client] of this.clients) {
      if (client.socket && !client.socket.destroyed) {
        client.socket.write(message + '\n');
      }
    }
  }

  async handleClient(socket) {
    const clientId = `${socket.remoteAddress}:${socket.remotePort}-${Date.now()}`;
    this.log('info', `Client connected: ${clientId}`);

    // Wait for MCP to be ready
    if (!this.initialized) {
      socket.write('{"error":"MCP not initialized, please retry"}\n');
      socket.end();
      return;
    }

    // Store client
    this.clients.set(clientId, {
      socket,
      pendingRequests: new Set(),
      buffer: ''
    });

    // Handle client data
    socket.on('data', (data) => {
      const client = this.clients.get(clientId);
      if (!client) return;

      client.buffer += data.toString();
      
      // Process complete lines
      const lines = client.buffer.split('\n');
      client.buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim()) {
          this.handleClientRequest(clientId, line);
        }
      }
    });

    // Handle client disconnect
    socket.on('close', () => {
      this.log('info', `Client disconnected: ${clientId}`);
      this.clients.delete(clientId);
    });

    socket.on('error', (err) => {
      this.log('error', `Client error ${clientId}: ${err.message}`);
      this.clients.delete(clientId);
    });
  }

  handleClientRequest(clientId, requestStr) {
    try {
      const request = JSON.parse(requestStr);
      const client = this.clients.get(clientId);
      
      if (!client) return;

      // Skip initialization requests (we already initialized)
      if (request.method === 'initialize') {
        // Send cached initialization response
        const response = {
          jsonrpc: "2.0",
          id: request.id,
          result: {
            protocolVersion: "2024-11-05",
            serverInfo: {
              name: "claude-flow",
              version: "2.0.0-alpha.101"
            }
          }
        };
        client.socket.write(JSON.stringify(response) + '\n');
        return;
      }

      // Track request ID for response routing
      if (request.id) {
        client.pendingRequests.add(request.id);
      }

      // Forward request to MCP
      this.mcpProcess.stdin.write(requestStr + '\n');
      this.log('debug', `Forwarded request ${request.id} from ${clientId}`);
      
    } catch (err) {
      this.log('error', `Invalid request from ${clientId}: ${err.message}`);
    }
  }

  async start() {
    // Start MCP process
    await this.startMCPProcess();

    // Start TCP server
    const server = net.createServer((socket) => {
      this.handleClient(socket);
    });

    server.listen(TCP_PORT, '0.0.0.0', () => {
      this.log('info', `Persistent MCP TCP server listening on port ${TCP_PORT}`);
    });

    server.on('error', (err) => {
      this.log('error', `Server error: ${err.message}`);
      if (err.code === 'EADDRINUSE') {
        this.log('error', `Port ${TCP_PORT} is already in use`);
        process.exit(1);
      }
    });
  }
}

// Start server
const server = new PersistentMCPServer();
server.start().catch(err => {
  console.error('Failed to start server:', err);
  process.exit(1);
});

// Handle shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down...');
  if (server.mcpProcess) {
    server.mcpProcess.kill();
  }
  process.exit(0);
});