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

    // Allow Cloudflare Tunnel hostname
    allowedHosts: [
      'www.visionflow.info',
      'visionflow.info',
      'localhost',
      '192.168.0.51'
    ],

    // HMR configuration for development
    hmr: {
      // This is the crucial part for Docker environments.
      // It tells Vite's HMR client to connect to the Nginx proxy port (3001)
      // on the host machine, which then forwards the request to Vite's actual HMR port.
      // The path must match the location block in nginx.dev.conf.
      clientPort: 3001,
      path: '/vite-hmr',
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
    // Always use proxy for API requests - the backend is at port 4000
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://visionflow_container:4000',
        changeOrigin: true,
        secure: false,
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            // Proxy error handling
          });
          proxy.on('proxyReq', (_proxyReq, _req, _res) => {
            // Proxy request handling
          });
          proxy.on('proxyRes', (_proxyRes, _req, _res) => {
            // Proxy response handling
          });
        }
      },
      '/ws': {
        target: process.env.VITE_WS_URL || 'ws://visionflow_container:4000',
        ws: true,
        changeOrigin: true
      },
      '/wss': {
        target: process.env.VITE_WS_URL || 'ws://visionflow_container:4000',
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