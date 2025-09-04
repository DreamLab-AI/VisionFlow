import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/tests/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/tests/setup.ts',
        '**/*.d.ts',
        '**/*.config.*',
        '**/types/**',
        '**/dist/**'
      ],
      thresholds: {
        global: {
          branches: 80,
          functions: 80,
          lines: 80,
          statements: 80
        }
      }
    },
    include: [
      'src/**/*.{test,spec}.{js,ts,tsx}',
      'src/tests/**/*.{test,spec}.{js,ts,tsx}'
    ],
    exclude: [
      'node_modules/**',
      'dist/**',
      'src/**/*.stories.*'
    ],
    testTimeout: 10000,
    hookTimeout: 10000,
    teardownTimeout: 10000,
    isolate: true,
    poolOptions: {
      threads: {
        singleThread: false,
        minThreads: 2,
        maxThreads: 4
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@tests': path.resolve(__dirname, './src/tests')
    },
  },
  define: {
    'import.meta.vitest': 'undefined',
  },
});