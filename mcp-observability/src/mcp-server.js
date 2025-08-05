import { createLogger } from './logger.js';
import { agentTools } from './tools/agent-tools.js';
import { swarmTools } from './tools/swarm-tools.js';
import { messageTools } from './tools/message-tools.js';
import { performanceTools } from './tools/performance-tools.js';
import { visualizationTools } from './tools/visualization-tools.js';
import { neuralTools } from './tools/neural-tools.js';
import { memoryTools } from './tools/memory-tools.js';

const logger = createLogger('MCPServer');

export function createMCPServer(components) {
  const { agentManager, physicsEngine, messageFlowTracker, performanceMonitor } = components;
  
  // Combine all tools
  const tools = {
    ...agentTools(agentManager, physicsEngine),
    ...swarmTools(agentManager, physicsEngine),
    ...messageTools(messageFlowTracker, physicsEngine),
    ...performanceTools(performanceMonitor, agentManager),
    ...visualizationTools(agentManager, physicsEngine),
    ...neuralTools(agentManager, performanceMonitor),
    ...memoryTools()
  };
  
  // MCP message handler
  async function handleMessage(message) {
    try {
      const { jsonrpc, method, params, id } = message;
      
      if (jsonrpc !== '2.0') {
        return createErrorResponse(id, -32600, 'Invalid Request');
      }
      
      switch (method) {
        case 'initialize':
          return handleInitialize(id, params);
          
        case 'initialized':
          return handleInitialized(id);
          
        case 'tools/list':
          return handleToolsList(id);
          
        case 'tools/call':
          return handleToolCall(id, params);
          
        default:
          return createErrorResponse(id, -32601, 'Method not found');
      }
    } catch (error) {
      logger.error('Error handling message:', error);
      return createErrorResponse(message.id, -32603, 'Internal error');
    }
  }
  
  // Handle initialize request
  function handleInitialize(id, params) {
    logger.info('MCP client initializing with params:', params);
    
    return {
      jsonrpc: '2.0',
      id,
      result: {
        protocolVersion: '0.1.0',
        serverInfo: {
          name: 'mcp-observability-server',
          version: '1.0.0',
          vendor: 'Hive Mind Collective'
        },
        capabilities: {
          tools: true
        }
      }
    };
  }
  
  // Handle initialized notification
  function handleInitialized(id) {
    logger.info('MCP client initialized');
    return null; // No response for notifications
  }
  
  // Handle tools/list request
  function handleToolsList(id) {
    const toolList = Object.entries(tools).map(([name, tool]) => ({
      name,
      description: tool.description,
      inputSchema: tool.inputSchema
    }));
    
    logger.info(`Returning ${toolList.length} tools`);
    
    return {
      jsonrpc: '2.0',
      id,
      result: {
        tools: toolList
      }
    };
  }
  
  // Handle tools/call request
  async function handleToolCall(id, params) {
    const { name, arguments: args } = params;
    
    logger.info(`Calling tool: ${name}`);
    
    if (!tools[name]) {
      return createErrorResponse(id, -32602, `Tool not found: ${name}`);
    }
    
    try {
      const result = await tools[name].handler(args);
      
      return {
        jsonrpc: '2.0',
        id,
        result: {
          content: [
            {
              type: 'text',
              text: JSON.stringify(result, null, 2)
            }
          ]
        }
      };
    } catch (error) {
      logger.error(`Error executing tool ${name}:`, error);
      return createErrorResponse(id, -32603, `Tool execution failed: ${error.message}`);
    }
  }
  
  // Create error response
  function createErrorResponse(id, code, message) {
    return {
      jsonrpc: '2.0',
      id,
      error: {
        code,
        message
      }
    };
  }
  
  // Start the server
  async function start() {
    logger.info('Starting MCP server on stdio...');
    
    let buffer = '';
    
    // Read from stdin
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', async (chunk) => {
      buffer += chunk;
      
      // Process complete messages
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (line.trim()) {
          try {
            const message = JSON.parse(line);
            const response = await handleMessage(message);
            
            if (response) {
              process.stdout.write(JSON.stringify(response) + '\n');
            }
          } catch (error) {
            logger.error('Failed to parse message:', error);
          }
        }
      }
    });
    
    // Handle stdin close
    process.stdin.on('end', () => {
      logger.info('stdin closed, shutting down...');
      process.exit(0);
    });
  }
  
  // Stop the server
  async function stop() {
    logger.info('Stopping MCP server...');
    // Cleanup if needed
  }
  
  return {
    start,
    stop
  };
}