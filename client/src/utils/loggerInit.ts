/**
 * Logger System Initialization
 * Entry point for setting up the complete logging infrastructure
 */

import { initializeLoggerProvider, type LoggerProviderConfig } from './loggerProvider';
import { clientDebugState } from './clientDebugState';

/**
 * Initialize the complete logging system
 * Call this early in your application startup
 */
export function initializeLoggingSystem(config?: LoggerProviderConfig): void {
  // Set up default debug state if not already configured
  if (!clientDebugState.get('enabled') && !localStorage.getItem('debug.enabled')) {
    // Auto-enable debug in development
    const isDev = import.meta.env?.DEV || import.meta.env?.MODE === 'development';
    if (isDev) {
      clientDebugState.set('enabled', true);
      clientDebugState.set('consoleLogging', true);
      clientDebugState.set('logLevel', 'debug');
    }
  }

  // Initialize the logger provider
  initializeLoggerProvider(config);

  console.log('🔧 Logging system initialized');
}

/**
 * Quick setup for development environments
 */
export function setupDevelopmentLogging(): void {
  clientDebugState.set('enabled', true);
  clientDebugState.set('consoleLogging', true);
  clientDebugState.set('logLevel', 'debug');

  initializeLoggerProvider({
    enableAutoCleanup: false, // Keep all loggers in dev
    metricsReportingInterval: 30000, // More frequent reporting
    enableMetrics: true,
  });

  console.log('🚀 Development logging setup complete');
}

/**
 * Quick setup for production environments
 */
export function setupProductionLogging(): void {
  // Only enable if explicitly set by user
  if (!clientDebugState.get('enabled')) {
    clientDebugState.set('enabled', false);
    clientDebugState.set('consoleLogging', false);
    clientDebugState.set('logLevel', 'error');
  }

  initializeLoggerProvider({
    enableAutoCleanup: true,
    metricsReportingInterval: 300000, // 5 minutes
    enableMetrics: false, // Disable metrics in production
  });

  console.log('🏭 Production logging setup complete');
}

/**
 * Setup for testing environments
 */
export function setupTestLogging(): void {
  clientDebugState.set('enabled', true);
  clientDebugState.set('consoleLogging', false); // Reduce noise in tests
  clientDebugState.set('logLevel', 'warn');

  initializeLoggerProvider({
    enableAutoCleanup: true,
    enableMetrics: false,
  });

  console.log('🧪 Test logging setup complete');
}