/**
 * Example integration showing how to use the new runtime logger system
 * This demonstrates the complete integration with React components
 */

import React, { useEffect, useState } from 'react';
import { loggerFactory, loggerManager } from './dynamicLogger';
import { useLoggerProvider } from './loggerProvider';
import { clientDebugState } from './clientDebugState';

/**
 * Example React component that uses the logger system
 */
export const LoggerExampleComponent: React.FC = () => {
  const [logCount, setLogCount] = useState(0);
  const [loggerStatus, setLoggerStatus] = useState<any>(null);

  // Use the logger provider hook
  const {
    status,
    setDebugMode,
    setLogLevel,
    setConsoleLogging,
    forceUpdate
  } = useLoggerProvider();

  // Create component-specific loggers
  const logger = loggerFactory.createUILogger('LoggerExample', {
    autoCleanup: true // Will be cleaned up when component unmounts
  });

  const performanceLogger = loggerFactory.createPerformanceLogger('LoggerExamplePerf');

  useEffect(() => {
    logger.info('LoggerExampleComponent mounted');

    // Subscribe to metrics updates
    const handleMetrics = (event: CustomEvent) => {
      setLoggerStatus(event.detail);
    };

    window.addEventListener('logger-metrics', handleMetrics as EventListener);

    return () => {
      logger.info('LoggerExampleComponent unmounting');
      window.removeEventListener('logger-metrics', handleMetrics as EventListener);
    };
  }, []);

  const handleTestLogging = () => {
    const startTime = performance.now();

    logger.debug('Debug message test');
    logger.info('Info message test');
    logger.warn('Warning message test');
    logger.error('Error message test');

    const endTime = performance.now();
    performanceLogger.logPerformance('test-logging', endTime - startTime);

    setLogCount(prev => prev + 4);
  };

  const handleToggleDebug = () => {
    const currentEnabled = clientDebugState.isEnabled();
    setDebugMode(!currentEnabled);
    logger.info(`Debug mode ${!currentEnabled ? 'enabled' : 'disabled'}`);
  };

  const handleChangeLogLevel = () => {
    const levels = ['debug', 'info', 'warn', 'error'] as const;
    const currentLevel = clientDebugState.get('logLevel');
    const currentIndex = levels.indexOf(currentLevel as any);
    const nextLevel = levels[(currentIndex + 1) % levels.length];

    setLogLevel(nextLevel);
    logger.info(`Log level changed to: ${nextLevel}`);
  };

  const handleToggleConsole = () => {
    const currentConsole = clientDebugState.get('consoleLogging');
    setConsoleLogging(!currentConsole);
    logger.info(`Console logging ${!currentConsole ? 'enabled' : 'disabled'}`);
  };

  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', margin: '10px' }}>
      <h3>Logger System Integration Example</h3>

      <div style={{ marginBottom: '20px' }}>
        <h4>Current Status:</h4>
        <pre style={{ background: '#f5f5f5', padding: '10px', fontSize: '12px' }}>
          {JSON.stringify(status, null, 2)}
        </pre>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h4>Logger Controls:</h4>
        <button onClick={handleTestLogging} style={{ margin: '5px' }}>
          Test Logging (Count: {logCount})
        </button>
        <button onClick={handleToggleDebug} style={{ margin: '5px' }}>
          Toggle Debug Mode
        </button>
        <button onClick={handleChangeLogLevel} style={{ margin: '5px' }}>
          Cycle Log Level
        </button>
        <button onClick={handleToggleConsole} style={{ margin: '5px' }}>
          Toggle Console Logging
        </button>
        <button onClick={forceUpdate} style={{ margin: '5px' }}>
          Force Update Loggers
        </button>
      </div>

      {loggerStatus && (
        <div>
          <h4>Real-time Metrics:</h4>
          <pre style={{ background: '#f0f8ff', padding: '10px', fontSize: '12px' }}>
            {JSON.stringify(loggerStatus, null, 2)}
          </pre>
        </div>
      )}

      <div style={{ marginTop: '20px', fontSize: '12px', color: '#666' }}>
        <p><strong>Instructions:</strong></p>
        <ul>
          <li>Open browser console to see log output</li>
          <li>Use controls above to test different settings</li>
          <li>Watch how log visibility changes based on settings</li>
          <li>Metrics are updated every minute</li>
        </ul>
      </div>
    </div>
  );
};

/**
 * Higher-order component that provides logger context
 */
export function withLogger<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  loggerCategory: string = 'ui'
) {
  return function WithLoggerComponent(props: P) {
    const logger = loggerFactory.createUILogger(
      WrappedComponent.displayName || WrappedComponent.name || 'Component',
      { category: loggerCategory, autoCleanup: true }
    );

    useEffect(() => {
      logger.info(`${WrappedComponent.name} mounted with logger`);
      return () => {
        logger.info(`${WrappedComponent.name} unmounting`);
      };
    }, []);

    return <WrappedComponent {...props} logger={logger} />;
  };
}

/**
 * Custom hook for using loggers in functional components
 */
export function useLogger(namespace: string, category?: string) {
  const logger = React.useMemo(() => {
    return loggerFactory.createUILogger(namespace, {
      category: category || 'ui',
      autoCleanup: true
    });
  }, [namespace, category]);

  useEffect(() => {
    logger.info(`Logger created for ${namespace}`);
    return () => {
      logger.info(`Logger cleanup for ${namespace}`);
    };
  }, [logger, namespace]);

  return logger;
}

/**
 * Debug panel component for development
 */
export const LoggerDebugPanel: React.FC = () => {
  const [settings, setSettings] = useState(loggerManager.getCurrentSettings());
  const [registeredLoggers, setRegisteredLoggers] = useState(loggerManager.getRegisteredLoggers());

  useEffect(() => {
    const interval = setInterval(() => {
      setSettings(loggerManager.getCurrentSettings());
      setRegisteredLoggers(loggerManager.getRegisteredLoggers());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{
      position: 'fixed',
      top: '10px',
      right: '10px',
      background: 'white',
      border: '1px solid #ccc',
      padding: '10px',
      fontSize: '12px',
      maxWidth: '300px',
      zIndex: 9999
    }}>
      <h4>Logger Debug Panel</h4>
      <div><strong>Enabled:</strong> {settings.enabled ? 'Yes' : 'No'}</div>
      <div><strong>Console:</strong> {settings.consoleLogging ? 'Yes' : 'No'}</div>
      <div><strong>Level:</strong> {settings.logLevel}</div>
      <div><strong>Registered:</strong> {settings.registeredCount}</div>
      <div><strong>Categories:</strong></div>
      <ul style={{ margin: '5px 0', paddingLeft: '20px' }}>
        {Object.entries(settings.categoryCounts).map(([category, count]) => (
          <li key={category}>{category}: {count}</li>
        ))}
      </ul>
      <details>
        <summary>All Loggers ({registeredLoggers.length})</summary>
        <ul style={{ margin: '5px 0', paddingLeft: '20px', maxHeight: '200px', overflow: 'auto' }}>
          {registeredLoggers.map(name => (
            <li key={name} style={{ fontSize: '10px' }}>{name}</li>
          ))}
        </ul>
      </details>
    </div>
  );
};

export default LoggerExampleComponent;