/**
 * MCP Bridge
 * 
 * Interfaces with MCP tools (claude-flow, ruv-swarm, mcp-observability)
 * to gather agent telemetry and execute swarm operations.
 */

const { spawn } = require('child_process');
const { EventEmitter } = require('events');
const { Logger } = require('./logger');

class MCPBridge extends EventEmitter {
    constructor() {
        super();
        this.logger = new Logger('MCPBridge');
        this.mcpProcesses = new Map();
        this.mcpClients = new Map();
        this.initialized = false;
    }

    async initialize() {
        this.logger.info('Initializing MCP Bridge...');
        
        try {
            // Start mcp-observability subprocess if available
            await this.startMCPObservability();
            
            // Initialize direct MCP tool access
            await this.initializeMCPTools();
            
            this.initialized = true;
            this.logger.info('MCP Bridge initialized successfully');
        } catch (error) {
            this.logger.error('Failed to initialize MCP Bridge:', error);
            // Continue anyway - we can work with limited functionality
            this.initialized = true;
        }
    }

    async startMCPObservability() {
        const mcpObservabilityPath = '/workspace/mcp-observability';
        
        try {
            this.logger.info('Starting mcp-observability subprocess...');
            
            const mcpProcess = spawn('node', ['src/index.js'], {
                cwd: mcpObservabilityPath,
                stdio: ['pipe', 'pipe', 'pipe'],
                env: { ...process.env, NODE_ENV: 'production' }
            });

            this.mcpProcesses.set('observability', mcpProcess);
            
            // Handle process output
            mcpProcess.stdout.on('data', (data) => {
                this.handleMCPResponse('observability', data.toString());
            });

            mcpProcess.stderr.on('data', (data) => {
                this.logger.warn('mcp-observability stderr:', data.toString());
            });

            mcpProcess.on('error', (error) => {
                this.logger.error('mcp-observability process error:', error);
            });

            mcpProcess.on('exit', (code) => {
                this.logger.warn(`mcp-observability exited with code ${code}`);
                this.mcpProcesses.delete('observability');
            });

            // Wait for initialization
            await this.waitForMCPReady('observability');
            
        } catch (error) {
            this.logger.warn('Could not start mcp-observability:', error.message);
            // Not critical - continue without it
        }
    }

    async initializeMCPTools() {
        // This would interface with claude-flow and ruv-swarm tools
        // For now, we'll simulate their availability
        
        this.mcpClients.set('claude-flow', {
            available: await this.checkToolAvailability('mcp__claude-flow__swarm_init'),
            prefix: 'mcp__claude-flow__'
        });

        this.mcpClients.set('ruv-swarm', {
            available: await this.checkToolAvailability('mcp__ruv-swarm__swarm_init'),
            prefix: 'mcp__ruv-swarm__'
        });

        this.logger.info('MCP Tools availability:', {
            'claude-flow': this.mcpClients.get('claude-flow').available,
            'ruv-swarm': this.mcpClients.get('ruv-swarm').available
        });
    }

    async checkToolAvailability(toolName) {
        // Check if MCP tool is available
        // This would normally check with the MCP server
        // For now, return true to simulate availability
        return true;
    }

    async waitForMCPReady(name, timeout = 5000) {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                reject(new Error(`Timeout waiting for ${name} to be ready`));
            }, timeout);

            const checkReady = () => {
                // In real implementation, would check for specific ready signal
                clearTimeout(timer);
                resolve();
            };

            // For now, just wait a bit
            setTimeout(checkReady, 1000);
        });
    }

    handleMCPResponse(source, data) {
        try {
            // Parse newline-delimited JSON from MCP
            const lines = data.split('\n').filter(line => line.trim());
            
            for (const line of lines) {
                try {
                    const message = JSON.parse(line);
                    this.emit('mcp-message', { source, message });
                } catch (e) {
                    // Not JSON - might be log output
                    this.logger.debug(`MCP ${source} output:`, line);
                }
            }
        } catch (error) {
            this.logger.error('Error handling MCP response:', error);
        }
    }

    async callMCPTool(tool, method, params = {}) {
        this.logger.debug(`Calling MCP tool: ${tool}.${method}`, params);
        
        // Route to appropriate MCP service
        if (tool === 'observability' && this.mcpProcesses.has('observability')) {
            return this.callMCPObservability(method, params);
        } else if (tool === 'claude-flow' && this.mcpClients.get('claude-flow').available) {
            return this.callClaudeFlow(method, params);
        } else if (tool === 'ruv-swarm' && this.mcpClients.get('ruv-swarm').available) {
            return this.callRuvSwarm(method, params);
        }
        
        // Fallback to mock data
        return this.getMockData(method, params);
    }

    async callMCPObservability(method, params) {
        const process = this.mcpProcesses.get('observability');
        if (!process) {
            throw new Error('mcp-observability not available');
        }

        // Send JSON-RPC request to mcp-observability
        const request = {
            jsonrpc: '2.0',
            id: Date.now().toString(),
            method: 'tools/call',
            params: {
                name: method,
                arguments: params
            }
        };

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('MCP call timeout'));
            }, 5000);

            const handler = (data) => {
                const { source, message } = data;
                if (source === 'observability' && message.id === request.id) {
                    clearTimeout(timeout);
                    this.removeListener('mcp-message', handler);
                    
                    if (message.error) {
                        reject(new Error(message.error.message));
                    } else {
                        resolve(message.result);
                    }
                }
            };

            this.on('mcp-message', handler);
            process.stdin.write(JSON.stringify(request) + '\n');
        });
    }

    async callClaudeFlow(method, params) {
        // Simulate claude-flow tool call
        // In real implementation, would execute the tool
        this.logger.debug(`Claude-flow call: ${method}`, params);
        return this.getMockData(method, params);
    }

    async callRuvSwarm(method, params) {
        // Simulate ruv-swarm tool call
        // In real implementation, would execute the tool
        this.logger.debug(`Ruv-swarm call: ${method}`, params);
        return this.getMockData(method, params);
    }

    getMockData(method, params) {
        // Provide mock telemetry data for testing
        switch (method) {
            case 'swarm.initialize':
                return {
                    swarmId: `swarm-${Date.now()}`,
                    status: 'initialized',
                    topology: params.topology || 'hierarchical'
                };
                
            case 'agent.list':
                return {
                    agents: [
                        {
                            id: 'agent-001',
                            type: 'coordinator',
                            name: 'Primary Coordinator',
                            status: 'active',
                            health: 95,
                            capabilities: ['planning', 'orchestration'],
                            cpuUsage: 0.45,
                            memoryUsage: 0.60
                        },
                        {
                            id: 'agent-002',
                            type: 'coder',
                            name: 'Code Agent Alpha',
                            status: 'busy',
                            health: 88,
                            capabilities: ['javascript', 'python'],
                            cpuUsage: 0.72,
                            memoryUsage: 0.55
                        },
                        {
                            id: 'agent-003',
                            type: 'analyst',
                            name: 'Data Analyst',
                            status: 'idle',
                            health: 100,
                            capabilities: ['analysis', 'reporting'],
                            cpuUsage: 0.15,
                            memoryUsage: 0.30
                        }
                    ]
                };
                
            case 'visualization.snapshot':
                return {
                    agents: this.getMockData('agent.list', {}).agents,
                    connections: [
                        {
                            id: 'conn-001',
                            from: 'agent-001',
                            to: 'agent-002',
                            messageCount: 42,
                            lastActivity: new Date().toISOString()
                        },
                        {
                            id: 'conn-002',
                            from: 'agent-001',
                            to: 'agent-003',
                            messageCount: 15,
                            lastActivity: new Date().toISOString()
                        }
                    ]
                };
                
            default:
                return {};
        }
    }

    async cleanup() {
        this.logger.info('Cleaning up MCP Bridge...');
        
        // Stop all MCP processes
        for (const [name, process] of this.mcpProcesses) {
            this.logger.info(`Stopping ${name} process...`);
            process.kill('SIGTERM');
        }
        
        this.mcpProcesses.clear();
        this.mcpClients.clear();
        this.removeAllListeners();
    }
}

module.exports = { MCPBridge };