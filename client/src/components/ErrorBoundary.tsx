import React, { Component, ReactNode, ErrorInfo } from 'react';
import { AlertTriangle, RefreshCw, Settings, Bug, Copy, ExternalLink } from 'lucide-react';

/**
 * Error boundary props
 */
interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  showDetails?: boolean;
  enableRecovery?: boolean;
  className?: string;
}

/**
 * Error boundary state
 */
interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
  errorId?: string;
  retryCount: number;
  showTechnicalDetails: boolean;
}

/**
 * Enhanced Error Boundary with settings integration
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    
    this.state = {
      hasError: false,
      retryCount: 0,
      showTechnicalDetails: false
    };
  }
  
  private getSettings() {
    try {
      return {
        debugMode: localStorage.getItem('settings.system.debug') === 'true',
        errorReporting: localStorage.getItem('settings.system.errorReporting') !== 'false',
        theme: localStorage.getItem('settings.ui.theme') || 'system',
        developerMode: localStorage.getItem('settings.system.developerMode') === 'true'
      };
    } catch {
      return {
        debugMode: false,
        errorReporting: true,
        theme: 'system',
        developerMode: false
      };
    }
  }
  
  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    // Generate unique error ID for tracking
    const errorId = `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    return {
      hasError: true,
      error,
      errorId
    };
  }
  
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });
    
    // Call custom error handler if provided
    this.props.onError?.(error, errorInfo);
    
    // Log error details
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    const settings = this.getSettings();
    
    // Report error if enabled
    if (settings.errorReporting) {
      this.reportError(error, errorInfo, settings);
    }
    
    // Log to performance monitoring if debug mode is enabled
    if (settings.debugMode) {
      this.logToPerformanceMonitor(error, errorInfo);
    }
  }
  
  private reportError(error: Error, errorInfo: ErrorInfo, settings: ReturnType<typeof this.getSettings>) {
    try {
      const errorReport = {
        errorId: this.state.errorId,
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href,
        userId: 'anonymous',
        buildVersion: process.env.REACT_APP_VERSION || 'development',
        settings: {
          debugMode: settings.debugMode,
          theme: settings.theme
        }
      };
      
      // Send to error reporting service (mock)
      console.info('Error report generated:', errorReport);
      
      // Save to localStorage for offline reporting
      const savedErrors = JSON.parse(localStorage.getItem('errorReports') || '[]');
      savedErrors.push(errorReport);
      
      // Keep only last 10 errors
      if (savedErrors.length > 10) {
        savedErrors.shift();
      }
      
      localStorage.setItem('errorReports', JSON.stringify(savedErrors));
    } catch (reportingError) {
      console.warn('Failed to report error:', reportingError);
    }
  }
  
  private logToPerformanceMonitor(error: Error, errorInfo: ErrorInfo) {
    try {
      // Mark performance with error information
      performance.mark(`error-${this.state.errorId}`);
      
      // Create custom performance entry
      if (window.performance && 'measure' in performance) {
        performance.measure(
          `error-boundary-${this.state.errorId}`,
          `error-${this.state.errorId}`
        );
      }
    } catch (perfError) {
      console.warn('Failed to log to performance monitor:', perfError);
    }
  }
  
  private handleRetry = () => {
    this.setState(prevState => ({
      hasError: false,
      error: undefined,
      errorInfo: undefined,
      errorId: undefined,
      retryCount: prevState.retryCount + 1,
      showTechnicalDetails: false
    }));
  };
  
  private handleReset = () => {
    // Clear any potentially corrupted state
    localStorage.removeItem('app-state');
    localStorage.removeItem('settings-cache');
    
    // Reset retry count and error state
    this.setState({
      hasError: false,
      error: undefined,
      errorInfo: undefined,
      errorId: undefined,
      retryCount: 0,
      showTechnicalDetails: false
    });
  };
  
  private handleCopyError = () => {
    if (!this.state.error || !this.state.errorInfo) return;
    
    const errorDetails = {
      errorId: this.state.errorId,
      message: this.state.error.message,
      stack: this.state.error.stack,
      componentStack: this.state.errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      retryCount: this.state.retryCount
    };
    
    navigator.clipboard.writeText(JSON.stringify(errorDetails, null, 2)).then(() => {
      console.info('Error details copied to clipboard');
    }).catch(err => {
      console.warn('Failed to copy error details:', err);
    });
  };
  
  private toggleTechnicalDetails = () => {
    this.setState(prevState => ({
      showTechnicalDetails: !prevState.showTechnicalDetails
    }));
  };
  
  private openSettingsDebug = () => {
    console.info('Opening settings debug page...');
  };
  
  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }
      
      const settings = this.getSettings();
      
      // Determine theme classes
      const isDark = settings.theme === 'dark' || 
                    (settings.theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
      
      const themeClasses = isDark
        ? 'bg-gray-900 text-white border-gray-700'
        : 'bg-white text-gray-900 border-gray-300';
      
      return (
        <div className={`min-h-screen flex items-center justify-center p-4 ${isDark ? 'bg-gray-950' : 'bg-gray-50'}`}>
          <div className={`max-w-2xl w-full ${themeClasses} rounded-lg shadow-2xl border p-8`}>
            {/* Header */}
            <div className="flex items-center gap-4 mb-6">
              <div className="p-3 bg-red-100 dark:bg-red-900/20 rounded-full">
                <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" />
              </div>
              
              <div>
                <h1 className="text-2xl font-bold text-red-600 dark:text-red-400">
                  Something went wrong
                </h1>
                <p className="text-gray-600 dark:text-gray-400">
                  An unexpected error has occurred in the application.
                </p>
              </div>
            </div>
            
            {/* Error Summary */}
            <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <h2 className="font-semibold mb-2 flex items-center gap-2">
                <Bug className="w-4 h-4" />
                Error Details
              </h2>
              
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                <strong>Error ID:</strong> {this.state.errorId}
              </p>
              
              <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                <strong>Message:</strong> {this.state.error?.message || 'Unknown error'}
              </p>
              
              {this.state.retryCount > 0 && (
                <p className="text-sm text-yellow-600 dark:text-yellow-400">
                  <strong>Retry attempts:</strong> {this.state.retryCount}
                </p>
              )}
            </div>
            
            {/* Technical Details (Collapsible) */}
            {(settings.debugMode || settings.developerMode) && (
              <div className="mb-6">
                <button
                  onClick={this.toggleTechnicalDetails}
                  className="flex items-center gap-2 text-sm text-blue-600 dark:text-blue-400 hover:underline"
                >
                  <Settings className="w-4 h-4" />
                  {this.state.showTechnicalDetails ? 'Hide' : 'Show'} Technical Details
                </button>
                
                {this.state.showTechnicalDetails && (
                  <div className="mt-3 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
                    <div className="space-y-3">
                      {/* Stack Trace */}
                      <div>
                        <h3 className="font-medium text-sm mb-1">Stack Trace:</h3>
                        <pre className="text-xs bg-gray-200 dark:bg-gray-900 p-2 rounded overflow-x-auto font-mono">
                          {this.state.error?.stack}
                        </pre>
                      </div>
                      
                      {/* Component Stack */}
                      {this.state.errorInfo?.componentStack && (
                        <div>
                          <h3 className="font-medium text-sm mb-1">Component Stack:</h3>
                          <pre className="text-xs bg-gray-200 dark:bg-gray-900 p-2 rounded overflow-x-auto font-mono">
                            {this.state.errorInfo.componentStack}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {/* Action Buttons */}
            <div className="flex flex-wrap gap-3">
              {/* Primary Actions */}
              {this.props.enableRecovery !== false && (
                <button
                  onClick={this.handleRetry}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  Try Again
                </button>
              )}
              
              <button
                onClick={this.handleReset}
                className="flex items-center gap-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Reset Application
              </button>
              
              {/* Developer Actions */}
              {settings.developerMode && (
                <>
                  <button
                    onClick={this.handleCopyError}
                    className="flex items-center gap-2 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800 px-4 py-2 rounded-lg font-medium transition-colors"
                  >
                    <Copy className="w-4 h-4" />
                    Copy Error Details
                  </button>
                  
                  <button
                    onClick={this.openSettingsDebug}
                    className="flex items-center gap-2 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800 px-4 py-2 rounded-lg font-medium transition-colors"
                  >
                    <Settings className="w-4 h-4" />
                    Debug Settings
                  </button>
                </>
              )}
              
              {/* Report Issue */}
              <button
                onClick={() => window.open('https://github.com/your-repo/issues/new', '_blank')}
                className="flex items-center gap-2 text-blue-600 dark:text-blue-400 hover:underline px-2 py-2 font-medium"
              >
                <ExternalLink className="w-4 h-4" />
                Report Issue
              </button>
            </div>
            
            {/* Error Prevention Tips */}
            {this.state.retryCount > 2 && (
              <div className="mt-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                <h3 className="font-medium text-yellow-800 dark:text-yellow-200 mb-2">
                  Persistent Error Detected
                </h3>
                <p className="text-sm text-yellow-700 dark:text-yellow-300">
                  This error has occurred {this.state.retryCount} times. Consider:
                </p>
                <ul className="text-sm text-yellow-700 dark:text-yellow-300 mt-2 ml-4 list-disc">
                  <li>Refreshing the page completely (Ctrl+F5)</li>
                  <li>Clearing browser cache and cookies</li>
                  <li>Checking your internet connection</li>
                  <li>Trying again in a few minutes</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      );
    }
    
    return this.props.children;
  }
}

/**
 * Settings-aware error boundary wrapper
 */
export function SettingsErrorBoundary({ children, ...props }: ErrorBoundaryProps) {
  return (
    <ErrorBoundary
      {...props}
      onError={(error, errorInfo) => {
        // Custom error handling that can access settings
        props.onError?.(error, errorInfo);
        
        // Log to settings-based error reporting
        console.group('🚨 Settings Error Boundary');
        console.error('Error:', error);
        console.error('Error Info:', errorInfo);
        console.groupEnd();
      }}
    >
      {children}
    </ErrorBoundary>
  );
}

/**
 * Development-only error boundary with enhanced debugging
 */
export function DevErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary
      showDetails={true}
      enableRecovery={true}
      onError={(error, errorInfo) => {
        // Enhanced development logging
        console.group('🔧 Development Error Boundary');
        console.error('Error:', error);
        console.error('Component Stack:', errorInfo.componentStack);
        console.error('Full Error Info:', errorInfo);
        console.groupEnd();
        
        // Break into debugger if DevTools are open
        if (process.env.NODE_ENV === 'development') {
          debugger;
        }
      }}
    >
      {children}
    </ErrorBoundary>
  );
}