import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

// https://vite.dev/config/
export default defineConfig({
  base: '/',
  plugins: [
    react(),
    wasm(),
    topLevelAwait()
  ],
  resolve: {
    alias: {
      '@': '/src'
    }
  },
  optimizeDeps: {
    exclude: ['@dreamlab-ai/webvowl-wasm']
  },
  build: {
    target: 'esnext',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom', 'react-router-dom'],
          'fuse': ['fuse.js']
        }
      },
      // Tell Rollup to treat WASM imports as external during build
      // The WASM module is handled by vite-plugin-wasm
      external: (id) => {
        // Don't externalize the WASM package itself, let vite-plugin-wasm handle it
        return false;
      }
    }
  },
  // Ensure WASM files are copied to output
  assetsInclude: ['**/*.wasm']
})
