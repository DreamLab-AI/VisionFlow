/**
 * JSON-RPC 2.0 Protocol Handler
 * 
 * Handles JSON-RPC requests from VisionFlow client over TCP connection.
 * Implements the required methods for agent control and telemetry.
 */

class JSONRPCHandler {
    constructor(socket, telemetryAggregator, logger) {
        this.socket = socket;
        this.telemetryAggregator = telemetryAggregator;
        this.logger = logger;
        this.buffer = '';
        this.sessionInfo = null;
    }

    start() {
        this.socket.on('data', (data) => {
            this.buffer += data.toString();
            this.processBuffer();
        });
    }

    processBuffer() {
        let lines = this.buffer.split('\n');
        this.buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
            if (line.trim()) {
                this.handleRequest(line);
            }
        }
    }

    async handleRequest(line) {
        try {
            const request = JSON.parse(line);
            this.logger.debug('Received request:', request);

            if (!this.validateRequest(request)) {
                this.sendError(request.id || null, -32600, 'Invalid Request');
                return;
            }

            // Route to appropriate handler
            let result;
            switch (request.method) {
                case 'initialize':
                    result = await this.handleInitialize(request.params);
                    break;
                case 'agents/list':
                    result = await this.handleAgentsList(request.params);
                    break;
                case 'tools/call':
                    result = await this.handleToolsCall(request.params);
                    break;
                default:
                    this.sendError(request.id, -32601, 'Method not found');
                    return;
            }

            this.sendResult(request.id, result);

        } catch (error) {
            this.logger.error('Error handling request:', error);
            if (error instanceof SyntaxError) {
                this.sendError(null, -32700, 'Parse error');
            } else {
                this.sendError(null, -32603, 'Internal error', error.message);
            }
        }
    }

    validateRequest(request) {
        return request &&
               request.jsonrpc === '2.0' &&
               typeof request.method === 'string' &&
               (request.params === undefined || typeof request.params === 'object');
    }

    async handleInitialize(params) {
        this.logger.info('Initializing session', params);
        
        this.sessionInfo = {
            protocolVersion: params?.protocolVersion || '0.1.0',
            clientInfo: params?.clientInfo || { name: 'unknown', version: 'unknown' },
            connectedAt: new Date().toISOString()
        };

        return {
            serverInfo: {
                name: 'Agent Control System',
                version: '1.0.0',
                capabilities: ['swarm', 'telemetry', 'metrics']
            },
            protocolVersion: '0.1.0'
        };
    }

    async handleAgentsList(params) {
        this.logger.debug('Fetching agents list');
        const agents = await this.telemetryAggregator.getAgents(params);
        
        return {
            agents: agents.map(agent => this.formatAgent(agent))
        };
    }

    async handleToolsCall(params) {
        const { name, arguments: args } = params;
        this.logger.debug(`Tool call: ${name}`, args);

        switch (name) {
            case 'swarm.initialize':
                return await this.handleSwarmInitialize(args);
            case 'visualization.snapshot':
                return await this.handleVisualizationSnapshot(args);
            case 'metrics.get':
                return await this.handleMetricsGet(args);
            default:
                throw new Error(`Unknown tool: ${name}`);
        }
    }

    async handleSwarmInitialize(args) {
        const result = await this.telemetryAggregator.initializeSwarm(args);
        return {
            status: 'initialized',
            topology: args.topology || 'hierarchical',
            agentTypes: args.agentTypes || ['coordinator', 'coder'],
            swarmId: result.swarmId,
            timestamp: new Date().toISOString()
        };
    }

    async handleVisualizationSnapshot(args) {
        const telemetry = await this.telemetryAggregator.getSnapshot();
        
        // Format for VisionFlow - NO POSITIONS, just telemetry
        return {
            timestamp: new Date().toISOString(),
            agentCount: telemetry.agents.length,
            agents: telemetry.agents,
            connections: telemetry.connections,
            metrics: telemetry.metrics,
            // Positions will be calculated by Rust service
            positions: args.includePositions ? {} : undefined
        };
    }

    async handleMetricsGet(args) {
        const metrics = await this.telemetryAggregator.getSystemMetrics();
        
        return {
            timestamp: new Date().toISOString(),
            system: {
                uptime: process.uptime(),
                memoryUsage: process.memoryUsage(),
                cpuUsage: process.cpuUsage()
            },
            agents: metrics.agents,
            performance: metrics.performance
        };
    }

    formatAgent(agent) {
        return {
            id: agent.id,
            type: agent.type || 'worker',
            name: agent.name || `Agent-${agent.id}`,
            status: agent.status || 'active',
            health: agent.health || 100,
            capabilities: agent.capabilities || [],
            swarmId: agent.swarmId,
            createdAt: agent.createdAt || new Date().toISOString(),
            lastActivity: agent.lastActivity || new Date().toISOString(),
            metrics: {
                tasksCompleted: agent.tasksCompleted || 0,
                tasksActive: agent.tasksActive || 0,
                successRate: agent.successRate || 1.0,
                cpuUsage: agent.cpuUsage || 0.0,
                memoryUsage: agent.memoryUsage || 0.0
            }
        };
    }

    sendResult(id, result) {
        const response = {
            jsonrpc: '2.0',
            id: id,
            result: result
        };
        this.send(response);
    }

    sendError(id, code, message, data) {
        const response = {
            jsonrpc: '2.0',
            id: id,
            error: {
                code: code,
                message: message,
                data: data
            }
        };
        this.send(response);
    }

    send(response) {
        const line = JSON.stringify(response) + '\n';
        this.socket.write(line);
        this.logger.debug('Sent response:', response);
    }
}

module.exports = { JSONRPCHandler };