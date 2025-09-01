import React from 'react';
import { ErrorBoundary } from '../../../components/ErrorBoundary';
import GraphManager from './GraphManager';

/**
 * Optimized wrapper for GraphManager with error boundary and loading states
 */
const OptimizedGraphManager: React.FC = React.memo(() => {
  const [isLoaded, setIsLoaded] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  React.useEffect(() => {
    // Simulate loading completion after a brief delay
    const timer = setTimeout(() => setIsLoaded(true), 100);
    return () => clearTimeout(timer);
  }, []);

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-500">
          Graph rendering error: {error.message}
        </div>
      </div>
    );
  }

  if (!isLoaded) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
        <span className="ml-2 text-gray-500">Loading graph...</span>
      </div>
    );
  }

  return (
    <ErrorBoundary
      onError={(error, errorInfo) => {
        console.error('GraphManager error:', error, errorInfo);
        setError(error);
      }}
      fallback={({ error, reset }) => (
        <div className="flex flex-col items-center justify-center h-full p-4">
          <div className="text-red-500 mb-4">
            Graph rendering failed: {error.message}
          </div>
          <button
            onClick={reset}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      )}
    >
      <GraphManager />
    </ErrorBoundary>
  );
});

OptimizedGraphManager.displayName = 'OptimizedGraphManager';

export default OptimizedGraphManager;