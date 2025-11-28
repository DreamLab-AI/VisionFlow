import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  // Define global constants to replace process.env references
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
    'process.env.REACT_APP_API_URL': JSON.stringify(process.env.REACT_APP_API_URL || '/api'),
    'process.env.VISIONFLOW_TEST_MODE': JSON.stringify(process.env.VISIONFLOW_TEST_MODE || 'false'),
    'process.env.BYPASS_WEBGL': JSON.stringify(process.env.BYPASS_WEBGL || 'false'),
    'process.env': '({})', // Fallback for any other process.env access
  },
  optimizeDeps: {
    include: ['@getalby/sdk']
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // Enable minification and tree-shaking for production
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: process.env.NODE_ENV === 'production',
        drop_debugger: true,
      },
    },
    // Optimize chunking strategy
    rollupOptions: {
      output: {
        manualChunks: {
          // Separate heavy 3D libraries
          'babylon': ['@babylonjs/core', '@babylonjs/gui', '@babylonjs/loaders'],
          'three': ['three', '@react-three/fiber', '@react-three/drei'],
          // UI libraries
          'ui': ['react', 'react-dom', 'framer-motion'],
          // Icons - load separately
          'icons': ['lucide-react'],
          // State management
          'state': ['zustand', 'immer'],
        },
      },
    },
    // Target modern browsers for smaller bundles
    target: 'esnext',
    // Report compressed sizes
    reportCompressedSize: true,
    // Chunk size warnings
    chunkSizeWarningLimit: 1000,
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