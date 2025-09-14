import { useCallback, useRef, useState } from 'react';
import { useToast, ToastAction } from '../features/design-system/components';
import { createLogger } from '../utils/logger';
import { webSocketService } from '../services/WebSocketService';

const logger = createLogger('ErrorHandler');

export interface ErrorOptions {
  title?: string;
  showDetails?: boolean;
  actionLabel?: string;
  onAction?: () => void;
  retry?: () => Promise<void>;
  maxRetries?: number;
  category?: 'network' | 'validation' | 'settings' | 'websocket' | 'general';
  metadata?: Record<string, any>;
}

export interface ErrorContext {
  timestamp: number;
  category: string;
  message: string;
  count: number;
  lastOccurrence: number;
}

export function useErrorHandler() {
  const { toast } = useToast();
  const errorHistory = useRef<Map<string, ErrorContext>>(new Map());
  const [retryAttempts, setRetryAttempts] = useState<Map<string, number>>(new Map());

  const handleError = useCallback((error: unknown, options: ErrorOptions = {}) => {
    const {
      title = 'Something went wrong',
      showDetails = true,
      actionLabel,
      onAction,
      retry,
      maxRetries = 3,
      category = 'general',
      metadata
    } = options;

    // Extract error message
    let message = 'An unexpected error occurred';
    let details: string | undefined;
    let errorCode: string | undefined;

    if (error instanceof Error) {
      message = error.message;
      if (showDetails && error.stack) {
        details = error.stack;
      }
    } else if (typeof error === 'string') {
      message = error;
    } else if (error && typeof error === 'object' && 'message' in error) {
      message = String(error.message);
      // Check for error code
      if ('code' in error) {
        errorCode = String(error.code);
      }
    }

    // Track error frequency for smart handling
    const errorKey = `${category}:${message}`;
    const errorContext = errorHistory.current.get(errorKey) || {
      timestamp: Date.now(),
      category,
      message,
      count: 0,
      lastOccurrence: Date.now()
    };
    errorContext.count++;
    errorContext.lastOccurrence = Date.now();
    errorHistory.current.set(errorKey, errorContext);

    // Create user-friendly error messages with category-specific handling
    const userFriendlyMessages: Record<string, string> = {
      'Network request failed': 'Unable to connect to the server. Please check your internet connection.',
      'Failed to fetch': 'Could not load data. Please try again later.',
      'Unauthorized': 'You need to sign in to access this feature.',
      'Forbidden': 'You don\'t have permission to perform this action.',
      'Not found': 'The requested resource was not found.',
      'Internal server error': 'The server encountered an error. Please try again later.',
      'Timeout': 'The request took too long. Please try again.',
      'Invalid input': 'Please check your input and try again.',
      'Quota exceeded': 'You\'ve reached the usage limit. Please try again later.',
      '500 Internal Server Error': 'Server error. We\'re working on fixing this.',
      '502 Bad Gateway': 'Connection issue. Please try again in a moment.',
      '503 Service Unavailable': 'Service temporarily unavailable. Please try again later.',
      'WebSocket': 'Real-time connection issue. Attempting to reconnect...',
      'ECONNREFUSED': 'Cannot connect to server. Please check if the server is running.',
      'ETIMEDOUT': 'Connection timed out. Please check your network.',
      'batch update failed': 'Failed to save multiple settings. Trying individual updates...',
      'validation failed': 'Invalid data format. Please check your input.',
      'settings sync failed': 'Settings synchronization failed. Your changes may not be saved.'
    };

    // Category-specific error handling
    let friendlyMessage = message;
    let showRetry = false;
    
    switch (category) {
      case 'network':
        showRetry = true;
        if (message.includes('ECONNREFUSED')) {
          friendlyMessage = 'Cannot connect to server. Please ensure the server is running.';
        }
        break;
      
      case 'websocket':
        showRetry = true;
        friendlyMessage = 'Real-time updates temporarily unavailable.';
        break;
      
      case 'settings':
        if (message.includes('batch')) {
          friendlyMessage = 'Settings update partially failed. Some changes may not be saved.';
          showRetry = true;
        }
        break;
      
      case 'validation':
        friendlyMessage = 'Invalid data provided. Please check your input and try again.';
        break;
    }

    // Check for known error patterns
    for (const [pattern, friendly] of Object.entries(userFriendlyMessages)) {
      if (message.toLowerCase().includes(pattern.toLowerCase())) {
        friendlyMessage = friendly;
        break;
      }
    }

    // Auto-retry logic for certain errors
    const shouldAutoRetry = showRetry && retry && errorContext.count <= maxRetries;
    const currentRetries = retryAttempts.get(errorKey) || 0;

    if (shouldAutoRetry && currentRetries < maxRetries) {
      setRetryAttempts(prev => new Map(prev).set(errorKey, currentRetries + 1));
      
      // Exponential backoff
      const delay = Math.min(1000 * Math.pow(2, currentRetries), 10000);
      
      setTimeout(async () => {
        try {
          await retry();
          // Success - clear retry count
          setRetryAttempts(prev => {
            const newMap = new Map(prev);
            newMap.delete(errorKey);
            return newMap;
          });
          
          toast({
            title: 'Success',
            description: 'Operation completed successfully',
            variant: 'default',
          });
        } catch (retryError) {
          // Retry failed, will be handled by next error
          handleError(retryError, { ...options, category });
        }
      }, delay);
      
      friendlyMessage += ` Retrying in ${delay / 1000}s... (${currentRetries + 1}/${maxRetries})`;
    }

    // Show toast notification
    const toastId = toast({
      title,
      description: friendlyMessage,
      variant: 'destructive',
      action: actionLabel && onAction ? (
        <ToastAction altText={actionLabel} onClick={onAction}>
          {actionLabel}
        </ToastAction>
      ) : retry && !shouldAutoRetry ? (
        <ToastAction 
          altText="Retry" 
          onClick={async () => {
            try {
              await retry();
              toast({
                title: 'Success',
                description: 'Operation completed successfully',
                variant: 'default',
              });
            } catch (retryError) {
              handleError(retryError, options);
            }
          }}
        >
          Retry
        </ToastAction>
      ) : undefined,
    });

    // Log error with context
    logger.error('Error handled:', {
      error,
      message,
      details,
      friendlyMessage,
      category,
      errorCode,
      metadata,
      errorCount: errorContext.count,
      retryAttempts: currentRetries
    });

    // Send error telemetry for monitoring (non-blocking)
    if (category !== 'general' || errorContext.count > 3) {
      sendErrorTelemetry({
        message,
        category,
        errorCode,
        metadata,
        errorCount: errorContext.count,
        timestamp: Date.now()
      });
    }
  }, [toast, retryAttempts]);

  const handleAsyncError = useCallback((promise: Promise<any>, options?: ErrorOptions) => {
    return promise.catch(error => {
      handleError(error, options);
      throw error; // Re-throw to allow caller to handle if needed
    });
  }, [handleError]);

  // Handle WebSocket errors specifically
  const handleWebSocketError = useCallback((error: unknown) => {
    const connectionState = webSocketService.getConnectionState();
    
    handleError(error, {
      title: 'Connection Error',
      category: 'websocket',
      retry: async () => {
        await webSocketService.connect();
      },
      metadata: {
        connectionState,
        queuedMessages: webSocketService.getQueuedMessageCount()
      }
    });
  }, [handleError]);

  // Handle settings sync errors
  const handleSettingsError = useCallback((error: unknown, failedPaths?: string[]) => {
    handleError(error, {
      title: 'Settings Update Failed',
      category: 'settings',
      metadata: { failedPaths },
      retry: async () => {
        // Retry logic will be implemented by the settings store
        throw new Error('Settings retry not implemented');
      }
    });
  }, [handleError]);

  // Get error statistics
  const getErrorStats = useCallback(() => {
    const stats: Record<string, { count: number; lastOccurrence: number }> = {};
    errorHistory.current.forEach((context, key) => {
      stats[key] = {
        count: context.count,
        lastOccurrence: context.lastOccurrence
      };
    });
    return stats;
  }, []);

  // Clear error history
  const clearErrorHistory = useCallback(() => {
    errorHistory.current.clear();
    setRetryAttempts(new Map());
  }, []);

  return {
    handleError,
    handleAsyncError,
    handleWebSocketError,
    handleSettingsError,
    getErrorStats,
    clearErrorHistory
  };
}

// Helper function to send error telemetry
async function sendErrorTelemetry(errorData: Record<string, any>) {
  try {
    // Only send in production
    if (process.env.NODE_ENV !== 'production') return;
    
    await fetch('/api/telemetry/errors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(errorData)
    });
  } catch (error) {
    // Fail silently - don't want telemetry to affect user experience
    logger.debug('Failed to send error telemetry:', error);
  }
}

// Utility function for wrapping async functions with error handling
export function withErrorHandler<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  options?: ErrorOptions
): T {
  return (async (...args: Parameters<T>) => {
    try {
      return await fn(...args);
    } catch (error) {
      const { toast } = useToast();

      // Similar error handling logic as above
      let message = 'An unexpected error occurred';
      if (error instanceof Error) {
        message = error.message;
      } else if (typeof error === 'string') {
        message = error;
      }

      toast({
        title: options?.title || 'Error',
        description: message,
        variant: 'destructive',
      });

      throw error;
    }
  }) as T;
}