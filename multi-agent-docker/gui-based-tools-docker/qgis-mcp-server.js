#!/usr/bin/env node

const net = require('net');
const { exec } = require('child_process');

// Environment variables
const PORT = parseInt(process.env.QGIS_MCP_PORT || '9877', 10);
const HOST = process.env.QGIS_MCP_HOST || '0.0.0.0';

console.log(`Starting QGIS MCP Server on ${HOST}:${PORT}`);

// Create TCP server
const server = net.createServer((socket) => {
  console.log('Client connected to QGIS MCP');
  
  // Since QGIS MCP runs inside QGIS, we need to forward the connection
  // For now, return a simple response indicating QGIS is running
  socket.on('data', (data) => {
    const request = data.toString();
    console.log('Received request:', request);
    
    try {
      const parsed = JSON.parse(request.trim());
      
      if (parsed.method === 'initialize') {
        socket.write(JSON.stringify({
          jsonrpc: '2.0',
          id: parsed.id,
          result: {
            capabilities: {
              tools: {
                listChanged: true
              }
            }
          }
        }) + '\n');
      } else if (parsed.method === 'tools/list') {
        socket.write(JSON.stringify({
          jsonrpc: '2.0',
          id: parsed.id,
          result: {
            tools: [
              {
                name: 'qgis_load_layer',
                description: 'Load a layer in QGIS',
                inputSchema: {
                  type: 'object',
                  properties: {
                    path: { type: 'string', description: 'Path to the layer file' }
                  },
                  required: ['path']
                }
              },
              {
                name: 'qgis_export_map',
                description: 'Export current map view',
                inputSchema: {
                  type: 'object',
                  properties: {
                    output_path: { type: 'string', description: 'Output file path' },
                    format: { type: 'string', enum: ['png', 'pdf', 'svg'], description: 'Output format' }
                  },
                  required: ['output_path']
                }
              }
            ]
          }
        }) + '\n');
      } else {
        socket.write(JSON.stringify({
          jsonrpc: '2.0',
          id: parsed.id,
          error: {
            code: -32601,
            message: 'Method not found'
          }
        }) + '\n');
      }
    } catch (e) {
      console.error('Error processing request:', e);
      socket.write(JSON.stringify({
        jsonrpc: '2.0',
        id: null,
        error: {
          code: -32700,
          message: 'Parse error'
        }
      }) + '\n');
    }
  });

  socket.on('end', () => {
    console.log('Client disconnected from QGIS MCP');
  });

  socket.on('error', (err) => {
    console.error('Socket error:', err);
  });
});

server.listen(PORT, HOST, () => {
  console.log(`QGIS MCP Server listening on ${HOST}:${PORT}`);
  console.log(`Note: QGIS must be running with the MCP plugin loaded`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Received SIGTERM, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('Received SIGINT, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});