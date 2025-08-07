#!/usr/bin/env node
import { createMCPServer } from './mcp-server.js';
import { AgentManager } from './agent-manager.js';
import { PhysicsEngine } from './physics-engine.js';
import { MessageFlowTracker } from './message-flow.js';
import { PerformanceMonitor } from './performance-monitor.js';
import { createLogger } from './logger.js';

const logger = createLogger('MCPObservability');

// Initialize core components
const agentManager = new AgentManager();
const physicsEngine = new PhysicsEngine();
const messageFlowTracker = new MessageFlowTracker();
const performanceMonitor = new PerformanceMonitor();

// Create and start MCP server
async function main() {
  try {
    logger.info('Starting MCP Observability Server...');
    
    const server = createMCPServer({
      agentManager,
      physicsEngine,
      messageFlowTracker,
      performanceMonitor
    });
    
    await server.start();
    
    logger.info('MCP Observability Server started successfully');
    logger.info('Available tools: agent.*, swarm.*, message.*, performance.*, visualization.*, neural.*, memory.*');
    
    // Start physics simulation loop (60 FPS)
    setInterval(() => {
      physicsEngine.update(1/60);
    }, 1000/60);
    
    // Start performance monitoring
    setInterval(() => {
      performanceMonitor.collect();
    }, 1000);
    
    // Handle graceful shutdown
    process.on('SIGINT', async () => {
      logger.info('Shutting down MCP Observability Server...');
      await server.stop();
      process.exit(0);
    });
    
  } catch (error) {
    logger.error('Failed to start MCP Observability Server:', error);
    process.exit(1);
  }
}

main();