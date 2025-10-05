#!/usr/bin/env node

const net = require('net');
const { spawn } = require('child_process');
const path = require('path');

// Environment variables
const PORT = parseInt(process.env.PLAYWRIGHT_MCP_PORT || '9879', 10);
const HOST = process.env.PLAYWRIGHT_MCP_HOST || '0.0.0.0';
const DISPLAY = process.env.DISPLAY || ':1';

console.log(`Starting Playwright MCP Server on ${HOST}:${PORT}`);
console.log(`Using DISPLAY=${DISPLAY}`);

// Create TCP server to handle MCP requests
const server = net.createServer((socket) => {
  console.log('Client connected');
  
  // Spawn the actual Playwright MCP server process
  const playwrightMcp = spawn('npx', ['@executeautomation/playwright-mcp-server'], {
    env: {
      ...process.env,
      DISPLAY: DISPLAY,
      NODE_ENV: 'production',
      // Force browser to run in visible mode when in GUI container
      PLAYWRIGHT_HEADLESS: 'false',
      PLAYWRIGHT_BROWSERS_PATH: '/opt/playwright-browsers'
    },
    stdio: ['pipe', 'pipe', 'pipe']
  });

  // Connect socket to process stdio
  socket.pipe(playwrightMcp.stdin);
  playwrightMcp.stdout.pipe(socket);
  
  // Log errors
  playwrightMcp.stderr.on('data', (data) => {
    console.error(`Playwright MCP Error: ${data.toString()}`);
  });

  // Handle disconnection
  socket.on('end', () => {
    console.log('Client disconnected');
    playwrightMcp.kill();
  });

  socket.on('error', (err) => {
    console.error('Socket error:', err);
    playwrightMcp.kill();
  });

  playwrightMcp.on('exit', (code) => {
    console.log(`Playwright MCP process exited with code ${code}`);
    socket.end();
  });
});

// Health check endpoint
const healthServer = net.createServer((socket) => {
  socket.write('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nPlaywright MCP Server is running\n');
  socket.end();
});

healthServer.listen(PORT + 1, HOST, () => {
  console.log(`Health check listening on ${HOST}:${PORT + 1}`);
});

server.listen(PORT, HOST, () => {
  console.log(`Playwright MCP Server listening on ${HOST}:${PORT}`);
  console.log(`VNC available on port 5901 for visual browser interaction`);
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