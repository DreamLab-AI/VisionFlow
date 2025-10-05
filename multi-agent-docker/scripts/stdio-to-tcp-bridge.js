#!/usr/bin/env node

/**
 * STDIO to TCP Bridge for MCP
 * Connects Claude Code's stdio to TCP MCP server
 */

const net = require('net');

const TCP_HOST = process.env.MCP_HOST || '127.0.0.1';
const TCP_PORT = process.env.MCP_PORT || 9500;

// Connect to TCP MCP server
const client = net.connect(TCP_PORT, TCP_HOST, () => {
  console.error(`[BRIDGE] Connected to MCP at ${TCP_HOST}:${TCP_PORT}`);
});

// Pipe stdin to TCP socket
process.stdin.pipe(client);

// Pipe TCP socket to stdout
client.pipe(process.stdout);

// Handle errors
client.on('error', (err) => {
  console.error(`[BRIDGE] TCP error: ${err.message}`);
  process.exit(1);
});

client.on('close', () => {
  console.error('[BRIDGE] TCP connection closed');
  process.exit(0);
});

process.stdin.on('end', () => {
  console.error('[BRIDGE] stdin closed, closing TCP');
  client.end();
});
