#!/usr/bin/env node

/**
 * Agent Control Interface
 * 
 * TCP server that provides agent telemetry to VisionFlow visualization system.
 * Listens on port 9500 for JSON-RPC 2.0 requests over newline-delimited JSON.
 */

const net = require('net');
const path = require('path');
const fs = require('fs');
const dotenv = require('dotenv');

// Load environment variables from .env file if it exists
const envPath = path.join(__dirname, '..', '.env');
if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath });
}

const { Logger } = require('./logger');
const { JSONRPCHandler } = require('./json-rpc-handler');
const { TelemetryAggregator } = require('./telemetry-aggregator');
const { MCPBridge } = require('./mcp-bridge');

const PORT = process.env.AGENT_CONTROL_PORT || 9500;
const HOST = '0.0.0.0'; // Bind to all interfaces for Docker network access

class AgentControlServer {
    constructor() {
        this.logger = new Logger('AgentControlServer');
        this.sessions = new Map();
        this.telemetryAggregator = new TelemetryAggregator();
        this.mcpBridge = new MCPBridge();
        this.server = null;
    }

    async start() {
        try {
            // Initialize MCP connections
            await this.mcpBridge.initialize();
            
            // Start telemetry aggregation
            await this.telemetryAggregator.start(this.mcpBridge);

            // Create TCP server
            this.server = net.createServer((socket) => {
                this.handleConnection(socket);
            });

            this.server.listen(PORT, HOST, () => {
                this.logger.info(`Agent Control Interface listening on ${HOST}:${PORT}`);
                this.logger.info(`Ready to accept connections from VisionFlow`);
            });

            this.server.on('error', (err) => {
                this.logger.error('Server error:', err);
                if (err.code === 'EADDRINUSE') {
                    this.logger.error(`Port ${PORT} is already in use`);
                    process.exit(1);
                }
            });

            // Graceful shutdown
            process.on('SIGINT', () => this.shutdown());
            process.on('SIGTERM', () => this.shutdown());

        } catch (error) {
            this.logger.error('Failed to start server:', error);
            process.exit(1);
        }
    }

    handleConnection(socket) {
        const sessionId = `${socket.remoteAddress}:${socket.remotePort}`;
        this.logger.info(`New connection from ${sessionId}`);

        const handler = new JSONRPCHandler(socket, this.telemetryAggregator, this.logger);
        this.sessions.set(sessionId, { socket, handler });

        // Handle socket events
        socket.on('close', () => {
            this.logger.info(`Connection closed: ${sessionId}`);
            this.sessions.delete(sessionId);
        });

        socket.on('error', (err) => {
            this.logger.error(`Socket error for ${sessionId}:`, err);
            this.sessions.delete(sessionId);
        });

        // Start handling JSON-RPC requests
        handler.start();
    }

    async shutdown() {
        this.logger.info('Shutting down Agent Control Interface...');
        
        // Close all client connections
        for (const [sessionId, session] of this.sessions) {
            session.socket.end();
        }

        // Stop server
        if (this.server) {
            this.server.close();
        }

        // Cleanup MCP connections
        await this.mcpBridge.cleanup();
        
        // Stop telemetry aggregation
        this.telemetryAggregator.stop();

        this.logger.info('Shutdown complete');
        process.exit(0);
    }
}

// Start the server
if (require.main === module) {
    const server = new AgentControlServer();
    server.start().catch(err => {
        console.error('Failed to start:', err);
        process.exit(1);
    });
}