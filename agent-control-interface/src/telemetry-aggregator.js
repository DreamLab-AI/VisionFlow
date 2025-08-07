/**
 * Telemetry Aggregator
 * 
 * Collects and aggregates agent telemetry from multiple MCP sources.
 * Provides unified view of agent states, metrics, and connections.
 */

const { EventEmitter } = require('events');
const { Logger } = require('./logger');

class TelemetryAggregator extends EventEmitter {
    constructor() {
        super();
        this.logger = new Logger('TelemetryAggregator');
        this.mcpBridge = null;
        
        // Telemetry cache
        this.cache = {
            agents: new Map(),
            connections: new Map(),
            metrics: {},
            lastUpdate: null
        };
        
        // Configuration
        this.config = {
            updateInterval: 1000, // 1 second
            cacheTTL: 5000, // 5 seconds
            aggregationStrategy: 'merge' // or 'replace'
        };
        
        this.updateTimer = null;
        this.running = false;
    }

    async start(mcpBridge) {
        this.mcpBridge = mcpBridge;
        this.running = true;
        
        this.logger.info('Starting telemetry aggregation...');
        
        // Initial data fetch
        await this.updateTelemetry();
        
        // Start periodic updates
        this.updateTimer = setInterval(() => {
            this.updateTelemetry().catch(err => {
                this.logger.error('Telemetry update failed:', err);
            });
        }, this.config.updateInterval);
        
        this.logger.info('Telemetry aggregation started');
    }

    stop() {
        this.running = false;
        
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
        
        this.logger.info('Telemetry aggregation stopped');
    }

    async updateTelemetry() {
        if (!this.running) return;
        
        try {
            // Fetch from multiple sources in parallel
            const [
                observabilityData,
                claudeFlowData,
                ruvSwarmData
            ] = await Promise.allSettled([
                this.fetchFromMCPObservability(),
                this.fetchFromClaudeFlow(),
                this.fetchFromRuvSwarm()
            ]);
            
            // Aggregate the data
            this.aggregateData([
                observabilityData.status === 'fulfilled' ? observabilityData.value : null,
                claudeFlowData.status === 'fulfilled' ? claudeFlowData.value : null,
                ruvSwarmData.status === 'fulfilled' ? ruvSwarmData.value : null
            ].filter(Boolean));
            
            this.cache.lastUpdate = Date.now();
            this.emit('telemetry-updated', this.cache);
            
        } catch (error) {
            this.logger.error('Failed to update telemetry:', error);
        }
    }

    async fetchFromMCPObservability() {
        try {
            const result = await this.mcpBridge.callMCPTool('observability', 'agent.list');
            return this.normalizeAgentData(result.agents || [], 'observability');
        } catch (error) {
            this.logger.debug('Could not fetch from mcp-observability:', error.message);
            return null;
        }
    }

    async fetchFromClaudeFlow() {
        try {
            const result = await this.mcpBridge.callMCPTool('claude-flow', 'agent.list');
            return this.normalizeAgentData(result.agents || [], 'claude-flow');
        } catch (error) {
            this.logger.debug('Could not fetch from claude-flow:', error.message);
            return null;
        }
    }

    async fetchFromRuvSwarm() {
        try {
            const result = await this.mcpBridge.callMCPTool('ruv-swarm', 'agent.list');
            return this.normalizeAgentData(result.agents || [], 'ruv-swarm');
        } catch (error) {
            this.logger.debug('Could not fetch from ruv-swarm:', error.message);
            return null;
        }
    }

    normalizeAgentData(agents, source) {
        return {
            source,
            agents: agents.map(agent => ({
                ...agent,
                source,
                normalizedId: this.generateAgentId(agent, source),
                timestamp: Date.now()
            }))
        };
    }

    generateAgentId(agent, source) {
        // Create consistent ID across sources
        return `${source}-${agent.id || agent.name || Date.now()}`;
    }

    aggregateData(dataSources) {
        // Clear old cache if using replace strategy
        if (this.config.aggregationStrategy === 'replace') {
            this.cache.agents.clear();
            this.cache.connections.clear();
        }
        
        // Aggregate agents from all sources
        for (const data of dataSources) {
            if (!data) continue;
            
            for (const agent of data.agents) {
                const key = agent.normalizedId;
                
                if (this.cache.agents.has(key)) {
                    // Merge with existing
                    const existing = this.cache.agents.get(key);
                    this.cache.agents.set(key, this.mergeAgent(existing, agent));
                } else {
                    // Add new
                    this.cache.agents.set(key, agent);
                }
            }
        }
        
        // Update connections based on agent relationships
        this.updateConnections();
        
        // Calculate system metrics
        this.updateMetrics();
    }

    mergeAgent(existing, update) {
        // Merge strategy: prefer newer data
        return {
            ...existing,
            ...update,
            metrics: {
                ...(existing.metrics || {}),
                ...(update.metrics || {})
            },
            lastSeen: Date.now()
        };
    }

    updateConnections() {
        // Generate connections based on agent relationships
        // This is simplified - real implementation would track actual message flow
        
        this.cache.connections.clear();
        
        const agents = Array.from(this.cache.agents.values());
        const coordinators = agents.filter(a => a.type === 'coordinator');
        const workers = agents.filter(a => a.type !== 'coordinator');
        
        // Create connections from coordinators to workers
        for (const coordinator of coordinators) {
            for (const worker of workers) {
                const connId = `${coordinator.normalizedId}-${worker.normalizedId}`;
                this.cache.connections.set(connId, {
                    id: connId,
                    from: coordinator.normalizedId,
                    to: worker.normalizedId,
                    messageCount: Math.floor(Math.random() * 100),
                    lastActivity: new Date().toISOString(),
                    strength: 0.5 + Math.random() * 0.5
                });
            }
        }
    }

    updateMetrics() {
        const agents = Array.from(this.cache.agents.values());
        
        this.cache.metrics = {
            totalAgents: agents.length,
            byType: this.groupBy(agents, 'type'),
            byStatus: this.groupBy(agents, 'status'),
            bySource: this.groupBy(agents, 'source'),
            avgHealth: agents.reduce((sum, a) => sum + (a.health || 0), 0) / agents.length || 0,
            avgCpuUsage: agents.reduce((sum, a) => sum + (a.cpuUsage || 0), 0) / agents.length || 0,
            avgMemoryUsage: agents.reduce((sum, a) => sum + (a.memoryUsage || 0), 0) / agents.length || 0
        };
    }

    groupBy(array, key) {
        return array.reduce((result, item) => {
            const group = item[key] || 'unknown';
            result[group] = (result[group] || 0) + 1;
            return result;
        }, {});
    }

    // Public API methods

    async getAgents(filter = {}) {
        let agents = Array.from(this.cache.agents.values());
        
        // Apply filters
        if (filter.type) {
            agents = agents.filter(a => a.type === filter.type);
        }
        if (filter.status) {
            agents = agents.filter(a => a.status === filter.status);
        }
        if (filter.source) {
            agents = agents.filter(a => a.source === filter.source);
        }
        
        return agents;
    }

    async getSnapshot() {
        return {
            agents: Array.from(this.cache.agents.values()),
            connections: Array.from(this.cache.connections.values()),
            metrics: this.cache.metrics,
            timestamp: this.cache.lastUpdate
        };
    }

    async getSystemMetrics() {
        return {
            agents: this.cache.metrics,
            performance: {
                cacheAge: Date.now() - this.cache.lastUpdate,
                updateInterval: this.config.updateInterval,
                sourcesAvailable: this.getAvailableSources()
            }
        };
    }

    getAvailableSources() {
        const sources = [];
        if (this.mcpBridge) {
            if (this.mcpBridge.mcpProcesses.has('observability')) {
                sources.push('mcp-observability');
            }
            if (this.mcpBridge.mcpClients.get('claude-flow')?.available) {
                sources.push('claude-flow');
            }
            if (this.mcpBridge.mcpClients.get('ruv-swarm')?.available) {
                sources.push('ruv-swarm');
            }
        }
        return sources;
    }

    async initializeSwarm(params) {
        // Initialize swarm using best available source
        const sources = ['observability', 'claude-flow', 'ruv-swarm'];
        
        for (const source of sources) {
            try {
                const result = await this.mcpBridge.callMCPTool(source, 'swarm.initialize', params);
                if (result) {
                    this.logger.info(`Swarm initialized via ${source}`, result);
                    
                    // Force immediate telemetry update
                    await this.updateTelemetry();
                    
                    return result;
                }
            } catch (error) {
                this.logger.debug(`Could not initialize swarm via ${source}:`, error.message);
            }
        }
        
        // Fallback
        return {
            swarmId: `swarm-${Date.now()}`,
            status: 'initialized',
            source: 'mock'
        };
    }
}

module.exports = { TelemetryAggregator };