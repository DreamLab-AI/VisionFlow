import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ['@getalby/sdk']
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    host: '0.0.0.0',
    port: parseInt(process.env.VITE_DEV_SERVER_PORT || '5173'),
    strictPort: true,
    
    // HMR configuration for development
    hmr: {
      // Let Vite automatically use the browser's location host
      // This fixes WebSocket connections when accessing via IP address
      path: '/__vite_hmr', // Match nginx config
    },
    
    // File watching for Docker environments
    watch: {
      usePolling: true,
      interval: 1000,
    },
    
    // CORS headers for development
    cors: true,
    
    // Security headers to match nginx
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    
    // Proxy API requests to backend server
    // In Docker dev mode, Nginx handles proxying, but for local dev we need Vite proxy
    proxy: process.env.DOCKER_ENV ? {} : {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:4000',
        changeOrigin: true,
        secure: false,
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            console.log('Sending Request to the Target:', req.method, req.url);
          });
          proxy.on('proxyRes', (proxyRes, req, _res) => {
            console.log('Received Response from the Target:', proxyRes.statusCode, req.url);
          });
        }
      },
      '/ws': {
        target: process.env.VITE_WS_URL || 'ws://localhost:4000',
        ws: true,
        changeOrigin: true
      },
      '/wss': {
        target: process.env.VITE_WS_URL || 'ws://localhost:4000',
        ws: true,
        changeOrigin: true
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});