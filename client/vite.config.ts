import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import path from 'path';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, path.resolve(__dirname, '.'), 'VITE_');
  return {
  plugins: [wasm(), topLevelAwait(), react()],
  // Define global constants to replace process.env references
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development'),
    'process.env.REACT_APP_API_URL': JSON.stringify(process.env.REACT_APP_API_URL || '/api'),
    'process.env.VISIONFLOW_TEST_MODE': JSON.stringify(process.env.VISIONFLOW_TEST_MODE || 'false'),
    'process.env.BYPASS_WEBGL': JSON.stringify(process.env.BYPASS_WEBGL || 'false'),
    'process.env': JSON.stringify({}), // Fallback for any other process.env access
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
          // Separate heavy 3D libraries (Three.js only - Babylon.js removed)
          'three': ['three', '@react-three/fiber', '@react-three/drei'],
          // UI libraries
          'ui': ['react', 'react-dom', 'framer-motion'],
          // Icons - load separately
          'icons': ['lucide-react'],
          // State management
          'state': ['immer'],
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

    // Allow hosts from environment (comma-separated) + local dev IP + docker hosts
    allowedHosts: [
      ...(process.env.VITE_ALLOWED_HOSTS?.split(',') || ['localhost']),
      process.env.VITE_LOCAL_DEV_IP,
      'host.docker.internal',  // For HTTPS bridge proxy
      '127.0.0.1',
    ].filter(Boolean) as string[],

    // HMR configuration for development (configurable via env)
    hmr: {
      // Client port for Docker/proxy environments (Nginx forwards to Vite)
      clientPort: parseInt(process.env.VITE_HMR_CLIENT_PORT || '3001'),
      path: process.env.VITE_HMR_PATH || '/vite-hmr',
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
        target: env.VITE_API_URL || 'http://127.0.0.1:4000',
        changeOrigin: true,
        secure: false,
        configure: (proxy, _options) => {
          proxy.on('error', (err, req, _res) => {
            console.error(`[SETTINGS-DIAG] Proxy error for ${req.method} ${req.url}:`, err.message);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            console.log(`[SETTINGS-DIAG] Proxy: ${req.method} ${req.url} → backend`);
          });
          proxy.on('proxyRes', (proxyRes, req, _res) => {
            console.log(`[SETTINGS-DIAG] Proxy response: ${req.method} ${req.url} → ${proxyRes.statusCode}`);
          });
        }
      },
      '/ws': {
        target: env.VITE_WS_URL || 'ws://127.0.0.1:4000',
        ws: true,
        changeOrigin: true
      },
      '/wss': {
        target: env.VITE_WS_URL || 'ws://127.0.0.1:4000',
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
};
});