import React, { useState } from 'react';
import { Button } from '../features/design-system/components';
import { 
  ErrorBoundary, 
  useErrorHandler, 
  ErrorNotification,
  ConnectionStatusNotification,
  SettingsSyncErrorNotification,
  GlobalSettingsRetryStatus,
  useWebSocketErrorHandler,
  reportClientError
} from '../components/error-handling';
import { webSocketService } from '../services/WebSocketService';

/**
 * Example demonstrating comprehensive error handling patterns
 */
export const ErrorHandlingExample: React.FC = () => {
  const { handleError, handleWebSocketError, handleSettingsError } = useErrorHandler();
  const [showExamples, setShowExamples] = useState(true);
  const [connectionState, setConnectionState] = useState(webSocketService.getConnectionState());
  
  // Set up WebSocket error handling
  useWebSocketErrorHandler();
  
  // Subscribe to connection state changes
  React.useEffect(() => {
    return webSocketService.onConnectionStateChange(setConnectionState);
  }, []);
  
  // Example error scenarios
  const triggerNetworkError = () => {
    handleError(new Error('Failed to fetch data from server'), {
      title: 'Network Error',
      category: 'network',
      retry: async () => {
        // Simulate retry
        await new Promise(resolve => setTimeout(resolve, 1000));
        console.log('Retrying network request...');
      },
      maxRetries: 3
    });
  };
  
  const triggerValidationError = () => {
    handleError(new Error('Invalid email format'), {
      title: 'Validation Error',
      category: 'validation',
      metadata: {
        field: 'email',
        value: 'not-an-email'
      }
    });
  };
  
  const triggerSettingsError = () => {
    handleSettingsError(
      new Error('Failed to update settings'),
      ['visualisation.glow.intensity', 'visualisation.glow.radius']
    );
  };
  
  const triggerWebSocketError = () => {
    handleWebSocketError(new Error('WebSocket connection lost'));
  };
  
  const triggerUnhandledError = () => {
    // This will be caught by ErrorBoundary
    throw new Error('Unhandled error in component');
  };
  
  const simulateServerError = () => {
    // Simulate receiving an error frame from server
    webSocketService.emit('error-frame', {
      code: 'RATE_LIMIT_EXCEEDED',
      message: 'Too many requests. Please slow down.',
      category: 'rate_limit',
      retryable: true,
      retryAfter: 5000,
      timestamp: Date.now()
    });
  };
  
  const reportBugToServer = () => {
    try {
      // Simulate a bug
      const bug = new Error('User reported bug: Button not working');
      reportClientError(bug, {
        component: 'ErrorHandlingExample',
        action: 'button_click',
        userAgent: navigator.userAgent
      });
      handleError(new Error('Bug report sent successfully'), {
        title: 'Success',
        category: 'general'
      });
    } catch (error) {
      handleError(error);
    }
  };
  
  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      <h1 className="text-3xl font-bold">Error Handling Examples</h1>
      
      {/* Connection Status */}
      <ConnectionStatusNotification
        isConnected={connectionState.status === 'connected'}
        isReconnecting={connectionState.status === 'reconnecting'}
        reconnectAttempts={connectionState.reconnectAttempts}
        onRetry={() => webSocketService.forceReconnect()}
      />
      
      {/* Global Settings Retry Status */}
      <GlobalSettingsRetryStatus />
      
      {/* Error Scenarios */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Test Error Scenarios</h2>
        
        <div className="grid grid-cols-2 gap-4">
          <Button onClick={triggerNetworkError}>
            Trigger Network Error (with retry)
          </Button>
          
          <Button onClick={triggerValidationError} variant="outline">
            Trigger Validation Error
          </Button>
          
          <Button onClick={triggerSettingsError} variant="outline">
            Trigger Settings Sync Error
          </Button>
          
          <Button onClick={triggerWebSocketError} variant="outline">
            Trigger WebSocket Error
          </Button>
          
          <Button onClick={simulateServerError} variant="destructive">
            Simulate Server Error Frame
          </Button>
          
          <Button onClick={reportBugToServer} variant="secondary">
            Report Bug to Server
          </Button>
          
          <Button onClick={triggerUnhandledError} variant="destructive">
            Trigger Unhandled Error (crashes component)
          </Button>
        </div>
      </div>
      
      {/* Example Notifications */}
      {showExamples && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Example Notifications</h2>
          
          <ErrorNotification
            type="error"
            title="Connection Failed"
            message="Unable to connect to the server. Please check your internet connection."
            retry={{
              onRetry: async () => {
                await new Promise(resolve => setTimeout(resolve, 1000));
                console.log('Retrying connection...');
              },
              maxRetries: 3
            }}
            onClose={() => console.log('Error notification closed')}
          />
          
          <ErrorNotification
            type="warning"
            title="Rate Limited"
            message="You're making requests too quickly. Please slow down."
            detail="Request limit: 60/minute. Current: 75/minute"
            autoClose={10000}
          />
          
          <ErrorNotification
            type="info"
            title="Settings Updated"
            message="Your settings have been successfully synchronized."
            autoClose={3000}
          />
          
          <SettingsSyncErrorNotification
            failedPaths={[
              'visualisation.glow.intensity',
              'visualisation.glow.radius',
              'visualisation.particles.count',
              'system.performance.targetFPS'
            ]}
            onRetry={async () => {
              console.log('Retrying settings sync...');
              await new Promise(resolve => setTimeout(resolve, 2000));
            }}
            onDismiss={() => setShowExamples(false)}
          />
        </div>
      )}
      
      {/* Error Boundary Demo */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Error Boundary Demo</h2>
        <ErrorBoundary
          onError={(error, errorInfo) => {
            console.error('ErrorBoundary caught:', error, errorInfo);
          }}
          fallback={(error, errorInfo, resetError) => (
            <div className="p-6 bg-destructive/10 rounded-lg">
              <h3 className="font-semibold text-destructive">
                Component Error: {error.message}
              </h3>
              <Button onClick={resetError} className="mt-4">
                Reset Component
              </Button>
            </div>
          )}
        >
          <ComponentThatMightError />
        </ErrorBoundary>
      </div>
      
      {/* Usage Guide */}
      <div className="prose prose-sm max-w-none">
        <h2>Usage Guide</h2>
        <h3>1. Basic Error Handling</h3>
        <pre className="bg-muted p-4 rounded-md overflow-x-auto">
{`const { handleError } = useErrorHandler();

try {
  await riskyOperation();
} catch (error) {
  handleError(error, {
    title: 'Operation Failed',
    category: 'network',
    retry: async () => await riskyOperation()
  });
}`}
        </pre>
        
        <h3>2. WebSocket Error Handling</h3>
        <pre className="bg-muted p-4 rounded-md overflow-x-auto">
{`// Set up in your app root
useWebSocketErrorHandler();

// Errors are automatically handled and displayed`}
        </pre>
        
        <h3>3. Settings Retry Management</h3>
        <pre className="bg-muted p-4 rounded-md overflow-x-auto">
{`// Add to your app root
<GlobalSettingsRetryStatus />

// Failed settings updates will automatically retry`}
        </pre>
        
        <h3>4. Error Boundaries</h3>
        <pre className="bg-muted p-4 rounded-md overflow-x-auto">
{`<ErrorBoundary
  fallback={(error, errorInfo, reset) => (
    <CustomErrorUI error={error} onReset={reset} />
  )}
>
  <YourComponent />
</ErrorBoundary>`}
        </pre>
      </div>
    </div>
  );
};

// Component that randomly throws errors for testing
const ComponentThatMightError: React.FC = () => {
  const [count, setCount] = useState(0);
  
  // Randomly throw error
  if (Math.random() > 0.7 && count > 0) {
    throw new Error(`Random error at count: ${count}`);
  }
  
  return (
    <div className="p-4 bg-muted rounded">
      <p>Click count: {count}</p>
      <Button onClick={() => setCount(c => c + 1)}>
        Increment (30% chance of error)
      </Button>
    </div>
  );
};