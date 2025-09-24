const net = require('net');
const stream = require('stream');

const HOST = process.env.CHROME_DEVTOOLS_PROXY_HOST || '127.0.0.1';
const PORT = process.env.CHROME_DEVTOOLS_PROXY_PORT || 9222;

const client = new net.Socket();

client.connect(PORT, HOST, () => {
    console.error('Connected to Chrome DevTools MCP');
});

client.on('data', (data) => {
    process.stdout.write(data);
});

client.on('close', () => {
    console.error('Connection to Chrome DevTools MCP closed');
});

client.on('error', (err) => {
    console.error('Chrome DevTools MCP connection error:', err);
});

process.stdin.pipe(client);