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
    
    // No proxy needed - nginx handles all routing
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});