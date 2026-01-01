#!/usr/bin/env node
/**
 * HTTPS to HTTP Bridge Proxy
 * Bridges https://localhost:3001 -> http://<HOST_IP>:3001
 * Solves cross-origin security issues for local development
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

const HOST_IP = process.env.HOST_IP || '192.168.0.51';
const HTTPS_PORT = parseInt(process.env.HTTPS_PORT || '3001', 10);
const TARGET_PORT = parseInt(process.env.TARGET_PORT || '3001', 10);

const certDir = process.env.CERT_DIR || __dirname;
const keyPath = path.join(certDir, 'server.key');
const certPath = path.join(certDir, 'server.crt');

if (!fs.existsSync(keyPath) || !fs.existsSync(certPath)) {
  console.error('Certificate files not found. Run:');
  console.error(`  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\`);
  console.error(`    -keyout ${keyPath} -out ${certPath} -subj "/CN=localhost"`);
  process.exit(1);
}

const options = {
  key: fs.readFileSync(keyPath),
  cert: fs.readFileSync(certPath)
};

const server = https.createServer(options, (req, res) => {
  const proxyOptions = {
    hostname: HOST_IP,
    port: TARGET_PORT,
    path: req.url,
    method: req.method,
    headers: {
      ...req.headers,
      host: `${HOST_IP}:${TARGET_PORT}`
    }
  };

  const proxyReq = http.request(proxyOptions, (proxyRes) => {
    // Add CORS headers for browser compatibility
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });

  proxyReq.on('error', (err) => {
    console.error(`Proxy error: ${err.message}`);
    res.writeHead(502, { 'Content-Type': 'text/plain' });
    res.end(`Bad Gateway: ${err.message}`);
  });

  req.pipe(proxyReq);
});

server.listen(HTTPS_PORT, '0.0.0.0', () => {
  console.log(`HTTPS Bridge running: https://localhost:${HTTPS_PORT} -> http://${HOST_IP}:${TARGET_PORT}`);
});
