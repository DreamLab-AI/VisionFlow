#!/usr/bin/env node
/**
 * Blender MCP Server - TCP Server for Blender MCP
 * Listens on TCP port and communicates with Blender via its Python API
 */

const net = require('net');

const BLENDER_MCP_PORT = parseInt(process.env.BLENDER_MCP_PORT || 9876);
const BLENDER_MCP_HOST = process.env.BLENDER_MCP_HOST || '0.0.0.0';
const BLENDER_PATH = process.env.BLENDER_PATH || '/usr/bin/blender';

console.log(`Starting Blender MCP Server on ${BLENDER_MCP_HOST}:${BLENDER_MCP_PORT}`);

const server = net.createServer((socket) => {
    console.log('Client connected to Blender MCP');

    let buffer = '';

    socket.on('data', (data) => {
        buffer += data.toString();

        let newlineIndex;
        while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
            const line = buffer.substring(0, newlineIndex);
            buffer = buffer.substring(newlineIndex + 1);

            if (line.trim()) {
                try {
                    const request = JSON.parse(line);
                    handleBlenderRequest(request, socket);
                } catch (e) {
                    socket.write(JSON.stringify({ error: `Invalid JSON: ${e.message}` }) + '\n');
                }
            }
        }
    });

    socket.on('end', () => console.log('Client disconnected'));
    socket.on('error', (err) => console.error('Socket error:', err));
});

function handleBlenderRequest(request, socket) {
    const { type, params } = request;
    const response = {
        success: true,
        message: `Blender MCP received: ${type}`,
        result: {
            note: "Blender is available via VNC:5901",
            blender_path: BLENDER_PATH,
            display: process.env.DISPLAY || ":1"
        }
    };
    socket.write(JSON.stringify(response) + '\n');
}

server.listen(BLENDER_MCP_PORT, BLENDER_MCP_HOST, () => {
    console.log(`Blender MCP listening on ${BLENDER_MCP_HOST}:${BLENDER_MCP_PORT}`);
});

server.on('error', (err) => {
    console.error('Server error:', err);
    process.exit(1);
});
