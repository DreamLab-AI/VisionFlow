#!/usr/bin/env node
/**
 * Unified MCP Gateway
 * Consolidates TCP, WebSocket, and shared context for MCP services
 */

const net = require('net');
const { WebSocketServer } = require('ws');
const http = require('http');

class SharedContext {
  constructor() {
    this.agents = new Map();
    this.sessions = new Map();
    this.tools = new Map();
  }

  registerAgent(id, metadata) {
    this.agents.set(id, { ...metadata, registeredAt: Date.now() });
    console.log(`[MCP Gateway] Agent registered: ${id}`);
  }

  getAgent(id) {
    return this.agents.get(id);
  }

  getAllAgents() {
    return Array.from(this.agents.values());
  }

  createSession(id, protocol, socket) {
    this.sessions.set(id, { id, protocol, socket, createdAt: Date.now() });
    console.log(`[MCP Gateway] Session created: ${id} (${protocol})`);
  }

  getSession(id) {
    return this.sessions.get(id);
  }

  closeSession(id) {
    this.sessions.delete(id);
    console.log(`[MCP Gateway] Session closed: ${id}`);
  }
}

class TCPServer {
  constructor(sharedContext, port = 9500) {
    this.context = sharedContext;
    this.port = port;
    this.server = null;
  }

  start() {
    this.server = net.createServer((socket) => {
      const sessionId = `tcp-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      this.context.createSession(sessionId, 'tcp', socket);

      socket.on('data', (data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleMessage(sessionId, message, socket);
        } catch (error) {
          console.error(`[TCP] Parse error:`, error);
          socket.write(JSON.stringify({
            jsonrpc: '2.0',
            error: { code: -32700, message: 'Parse error' },
            id: null
          }) + '\n');
        }
      });

      socket.on('error', (error) => {
        console.error(`[TCP] Socket error:`, error);
        this.context.closeSession(sessionId);
      });

      socket.on('close', () => {
        this.context.closeSession(sessionId);
      });
    });

    this.server.listen(this.port, () => {
      console.log(`[MCP Gateway] TCP server listening on port ${this.port}`);
    });
  }

  handleMessage(sessionId, message, socket) {
    const { id, method, params } = message;

    // Route to appropriate handler based on method
    let response;
    switch (method) {
      case 'tools/list':
        response = this.handleToolsList(id);
        break;
      case 'tools/call':
        response = this.handleToolCall(id, params);
        break;
      case 'agent/register':
        response = this.handleAgentRegister(id, params);
        break;
      case 'agent/list':
        response = this.handleAgentList(id);
        break;
      default:
        response = {
          jsonrpc: '2.0',
          error: { code: -32601, message: 'Method not found' },
          id
        };
    }

    socket.write(JSON.stringify(response) + '\n');
  }

  handleToolsList(id) {
    const tools = this.context.getAllAgents().map(agent => ({
      name: agent.id,
      description: agent.description || 'Agent tool',
      inputSchema: agent.inputSchema || {}
    }));

    return {
      jsonrpc: '2.0',
      result: { tools },
      id
    };
  }

  handleToolCall(id, params) {
    return {
      jsonrpc: '2.0',
      result: {
        content: [{ type: 'text', text: 'Tool executed via gateway' }]
      },
      id
    };
  }

  handleAgentRegister(id, params) {
    this.context.registerAgent(params.id, params);
    return {
      jsonrpc: '2.0',
      result: { success: true, agentId: params.id },
      id
    };
  }

  handleAgentList(id) {
    return {
      jsonrpc: '2.0',
      result: {
        agents: this.context.getAllAgents(),
        count: this.context.agents.size
      },
      id
    };
  }

  stop() {
    if (this.server) {
      this.server.close();
      console.log('[MCP Gateway] TCP server stopped');
    }
  }
}

class WebSocketServerWrapper {
  constructor(sharedContext, port = 3002) {
    this.context = sharedContext;
    this.port = port;
    this.httpServer = null;
    this.wss = null;
  }

  start() {
    this.httpServer = http.createServer((req, res) => {
      if (req.url === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          status: 'healthy',
          agents: this.context.agents.size,
          sessions: this.context.sessions.size
        }));
      } else {
        res.writeHead(404);
        res.end();
      }
    });

    this.wss = new WebSocketServer({ server: this.httpServer });

    this.wss.on('connection', (ws) => {
      const sessionId = `ws-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      this.context.createSession(sessionId, 'websocket', ws);

      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleMessage(sessionId, message, ws);
        } catch (error) {
          console.error(`[WS] Parse error:`, error);
          ws.send(JSON.stringify({
            jsonrpc: '2.0',
            error: { code: -32700, message: 'Parse error' },
            id: null
          }));
        }
      });

      ws.on('error', (error) => {
        console.error(`[WS] Socket error:`, error);
        this.context.closeSession(sessionId);
      });

      ws.on('close', () => {
        this.context.closeSession(sessionId);
      });
    });

    this.httpServer.listen(this.port, () => {
      console.log(`[MCP Gateway] WebSocket server listening on port ${this.port}`);
    });
  }

  handleMessage(sessionId, message, ws) {
    const { id, method, params } = message;

    // Reuse TCP server logic by creating a wrapper
    let response;
    switch (method) {
      case 'tools/list':
      case 'tools/call':
      case 'agent/register':
      case 'agent/list':
        // Delegate to TCP handler logic
        const tcpHandler = new TCPServer(this.context);
        response = tcpHandler.handleMessage(sessionId, message, { write: () => {} });
        break;
      default:
        response = {
          jsonrpc: '2.0',
          error: { code: -32601, message: 'Method not found' },
          id
        };
    }

    ws.send(JSON.stringify(response));
  }

  stop() {
    if (this.wss) {
      this.wss.close();
    }
    if (this.httpServer) {
      this.httpServer.close();
      console.log('[MCP Gateway] WebSocket server stopped');
    }
  }
}

class MCPGateway {
  constructor(config = {}) {
    this.config = {
      tcpPort: config.tcpPort || 9500,
      wsPort: config.wsPort || 3002,
      ...config
    };

    this.sharedContext = new SharedContext();
    this.tcpServer = new TCPServer(this.sharedContext, this.config.tcpPort);
    this.wsServer = new WebSocketServerWrapper(this.sharedContext, this.config.wsPort);
  }

  start() {
    console.log('[MCP Gateway] Starting unified MCP gateway...');
    this.tcpServer.start();
    this.wsServer.start();
    console.log('[MCP Gateway] All services started');
  }

  stop() {
    console.log('[MCP Gateway] Stopping all services...');
    this.tcpServer.stop();
    this.wsServer.stop();
    console.log('[MCP Gateway] All services stopped');
  }
}

// Main execution
if (require.main === module) {
  const config = {
    tcpPort: parseInt(process.env.MCP_TCP_PORT || '9500'),
    wsPort: parseInt(process.env.MCP_WS_PORT || '3002')
  };

  const gateway = new MCPGateway(config);

  // Handle graceful shutdown
  process.on('SIGTERM', () => {
    console.log('[MCP Gateway] SIGTERM received, shutting down...');
    gateway.stop();
    process.exit(0);
  });

  process.on('SIGINT', () => {
    console.log('[MCP Gateway] SIGINT received, shutting down...');
    gateway.stop();
    process.exit(0);
  });

  gateway.start();
}

module.exports = { MCPGateway, SharedContext, TCPServer, WebSocketServerWrapper };