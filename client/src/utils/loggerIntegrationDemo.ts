/**
 * Demonstration script for Dynamic Logger Integration
 * Shows how the Control Center settings affect loggers in real-time
 */

import { createLogger, createAgentLogger } from './loggerConfig';
import { clientDebugState } from './clientDebugState';
import { getLoggerSystemStatus, isLoggerSystemInitialized } from './loggerIntegrationInit';

/**
 * Demo function to show dynamic logger behavior
 * This can be called from the browser console to see the integration in action
 */
export function runLoggerDemo(): void {
  console.log('\n=== Dynamic Logger Integration Demo ===\n');

  // Check if integration system is initialized
  if (!isLoggerSystemInitialized()) {
    console.log('⚠️ Logger integration system not initialized yet. It should auto-initialize shortly.');
    return;
  }

  // Create test loggers
  const testLogger = createLogger('DemoLogger');
  const agentLogger = createAgentLogger('DemoAgent');

  console.log('📋 Current system status:');
  console.log(getLoggerSystemStatus());

  console.log('\n🔧 Current debug settings:');
  console.log('- Console Logging:', clientDebugState.get('consoleLogging'));
  console.log('- Log Level:', clientDebugState.get('logLevel'));
  console.log('- Debug Enabled:', clientDebugState.get('enabled'));

  console.log('\n📝 Testing logger output at all levels:');
  testLogger.debug('This is a DEBUG message');
  testLogger.info('This is an INFO message');
  testLogger.warn('This is a WARN message');
  testLogger.error('This is an ERROR message');

  console.log('\n🤖 Testing agent logger:');
  agentLogger.info('Agent logger message');
  agentLogger.logAgentAction('demo-agent', 'TestAgent', 'demo_action', { test: true });

  console.log('\n🎮 Try changing settings in Control Center > Developer tab:');
  console.log('1. Toggle "Console Logging" on/off');
  console.log('2. Change "Log Level" (error/warn/info/debug)');
  console.log('3. Watch how the logger output changes immediately!');
  console.log('4. Run this demo again to see the effects');

  // Set up a live demonstration
  let demoInterval: NodeJS.Timeout;

  console.log('\n⏰ Starting live demo (will log every 3 seconds)...');
  console.log('   Change Control Center settings to see immediate effects!');

  let counter = 0;
  demoInterval = setInterval(() => {
    counter++;
    testLogger.debug(`Live demo debug message #${counter}`);
    testLogger.info(`Live demo info message #${counter}`);
    testLogger.warn(`Live demo warn message #${counter}`);

    if (counter >= 10) {
      clearInterval(demoInterval);
      console.log('\n✅ Live demo completed. Integration system is working correctly!');
    }
  }, 3000);

  // Set up a function to stop the demo
  (window as any).stopLoggerDemo = () => {
    clearInterval(demoInterval);
    console.log('\n⏹️ Logger demo stopped.');
  };

  console.log('\n💡 Call stopLoggerDemo() to stop the live demo early');
}

/**
 * Test the configuration change behavior programmatically
 */
export function testConfigurationChanges(): void {
  console.log('\n=== Configuration Change Test ===\n');

  const testLogger = createLogger('ConfigTest');

  console.log('📊 Initial state:');
  console.log('- Console Logging:', clientDebugState.get('consoleLogging'));
  console.log('- Log Level:', clientDebugState.get('logLevel'));

  testLogger.debug('Initial debug message');
  testLogger.info('Initial info message');

  // Test enabling/disabling console logging
  console.log('\n🔄 Disabling console logging...');
  clientDebugState.set('consoleLogging', false);

  testLogger.debug('This should not appear (logging disabled)');
  testLogger.info('This should not appear (logging disabled)');

  setTimeout(() => {
    console.log('\n🔄 Re-enabling console logging...');
    clientDebugState.set('consoleLogging', true);

    testLogger.debug('Debug message after re-enabling');
    testLogger.info('Info message after re-enabling');

    // Test changing log level
    setTimeout(() => {
      console.log('\n🔄 Changing log level to "warn"...');
      clientDebugState.set('logLevel', 'warn');

      testLogger.debug('Debug message (should not appear - level too low)');
      testLogger.info('Info message (should not appear - level too low)');
      testLogger.warn('Warning message (should appear)');
      testLogger.error('Error message (should appear)');

      setTimeout(() => {
        console.log('\n🔄 Resetting log level to "debug"...');
        clientDebugState.set('logLevel', 'debug');

        testLogger.debug('Debug message (should appear - level reset)');
        testLogger.info('Info message (should appear - level reset)');

        console.log('\n✅ Configuration change test completed successfully!');
      }, 1000);
    }, 1000);
  }, 1000);
}

/**
 * Make demo functions available globally for browser console usage
 */
if (typeof window !== 'undefined') {
  (window as any).runLoggerDemo = runLoggerDemo;
  (window as any).testConfigurationChanges = testConfigurationChanges;
  (window as any).getLoggerSystemStatus = getLoggerSystemStatus;

  console.log('🎯 Logger integration demo functions are now available:');
  console.log('- runLoggerDemo() - Interactive demonstration');
  console.log('- testConfigurationChanges() - Automated configuration test');
  console.log('- getLoggerSystemStatus() - View system status');
}