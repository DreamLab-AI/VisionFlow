/**
 * Integration test for the runtime logger system
 * This can be run manually or imported to verify the integration works
 */

import { clientDebugState } from './clientDebugState';
import { createLogger, createAgentLogger, loggerManager, loggerFactory } from './dynamicLogger';
import { initializeLoggerProvider } from './loggerProvider';

/**
 * Test the runtime logger integration
 */
export function testLoggerIntegration(): void {
  console.log('🧪 Starting logger integration test...');

  // Initialize the provider
  initializeLoggerProvider({
    enableMetrics: true,
    metricsReportingInterval: 5000, // 5 seconds for testing
  });

  // Test 1: Create loggers
  console.log('📝 Test 1: Creating test loggers...');
  const uiLogger = loggerFactory.createUILogger('TestUI');
  const apiLogger = loggerFactory.createAPILogger('TestAPI');
  const wsLogger = loggerFactory.createWebSocketLogger('TestWS');

  // Test 2: Initial logging with current settings
  console.log('📝 Test 2: Testing initial logging...');
  uiLogger.info('UI logger initialized');
  apiLogger.info('API logger initialized');
  wsLogger.logWebSocketMessage('test', 'outgoing', { test: true });

  // Test 3: Change debug settings and verify updates
  console.log('📝 Test 3: Testing debug state changes...');

  // Save current state
  const originalEnabled = clientDebugState.get('enabled');
  const originalLevel = clientDebugState.get('logLevel');
  const originalConsole = clientDebugState.get('consoleLogging');

  // Test enabling debug
  console.log('  - Enabling debug mode...');
  clientDebugState.set('enabled', true);
  clientDebugState.set('consoleLogging', true);
  clientDebugState.set('logLevel', 'debug');

  // Wait a bit for async updates
  setTimeout(() => {
    uiLogger.debug('Debug logging should now be visible');
    apiLogger.debug('API debug message');
    wsLogger.debug('WebSocket debug message');

    // Test disabling console logging
    console.log('  - Disabling console logging...');
    clientDebugState.set('consoleLogging', false);

    setTimeout(() => {
      uiLogger.info('This should NOT appear in console');
      apiLogger.warn('This warning should NOT appear in console');

      // Test changing log level
      console.log('  - Setting log level to error only...');
      clientDebugState.set('consoleLogging', true);
      clientDebugState.set('logLevel', 'error');

      setTimeout(() => {
        uiLogger.info('This info should NOT appear');
        uiLogger.warn('This warning should NOT appear');
        uiLogger.error('This error SHOULD appear');

        // Test 4: Check logger registry
        console.log('📝 Test 4: Testing logger registry...');
        const registeredLoggers = loggerManager.getRegisteredLoggers();
        console.log('Registered loggers:', registeredLoggers);

        const settings = loggerManager.getCurrentSettings();
        console.log('Current settings:', settings);

        const metrics = loggerManager.getLoggerMetrics();
        console.log('Logger metrics:', metrics);

        // Test 5: Test logger categories
        console.log('📝 Test 5: Testing logger categories...');
        const uiLoggers = loggerManager.getLoggersByCategory('ui');
        const apiLoggers = loggerManager.getLoggersByCategory('api');
        console.log('UI loggers:', uiLoggers.length);
        console.log('API loggers:', apiLoggers.length);

        // Restore original state
        console.log('📝 Restoring original debug state...');
        clientDebugState.set('enabled', originalEnabled);
        clientDebugState.set('logLevel', originalLevel);
        clientDebugState.set('consoleLogging', originalConsole);

        console.log('✅ Logger integration test completed successfully!');
      }, 100);
    }, 100);
  }, 100);
}

/**
 * Test performance with many loggers
 */
export function testLoggerPerformance(): void {
  console.log('⚡ Starting logger performance test...');

  const startTime = performance.now();
  const loggers: ReturnType<typeof createLogger>[] = [];

  // Create 100 loggers
  for (let i = 0; i < 100; i++) {
    const logger = createLogger(`PerformanceTest${i}`, {
      category: 'performance',
      autoCleanup: true
    });
    loggers.push(logger);
  }

  const createTime = performance.now() - startTime;
  console.log(`Created 100 loggers in ${createTime.toFixed(2)}ms`);

  // Log messages with all loggers
  const logStartTime = performance.now();
  loggers.forEach((logger, i) => {
    logger.info(`Performance test message ${i}`);
  });
  const logTime = performance.now() - logStartTime;
  console.log(`Logged 100 messages in ${logTime.toFixed(2)}ms`);

  // Test settings update performance
  const updateStartTime = performance.now();
  clientDebugState.set('logLevel', 'debug');
  const updateTime = performance.now() - updateStartTime;
  console.log(`Updated all loggers in ${updateTime.toFixed(2)}ms`);

  console.log('✅ Performance test completed!');
}

/**
 * Manual test runner that can be called from browser console
 */
export function runManualTests(): void {
  console.log('🚀 Running manual logger integration tests...');

  testLoggerIntegration();

  setTimeout(() => {
    testLoggerPerformance();
  }, 2000);
}

// Export for browser console testing
if (typeof window !== 'undefined') {
  (window as any).testLoggers = runManualTests;
  console.log('💡 Run window.testLoggers() to test the logger integration');
}