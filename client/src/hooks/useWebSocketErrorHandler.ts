import { useEffect } from 'react';
import { webSocketService, WebSocketErrorFrame } from '../services/WebSocketService';
import { useErrorHandler } from './useErrorHandler';
import { useSettingsStore } from '../store/settingsStore';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('WebSocketErrorHandler');

export function useWebSocketErrorHandler() {
  const { handleError, handleWebSocketError, handleSettingsError } = useErrorHandler();
  
  useEffect(() => {
    // Listen for structured error frames from WebSocket
    const unsubscribeErrorFrame = webSocketService.on('error-frame', (error: WebSocketErrorFrame) => {
      logger.warn('WebSocket error frame received:', error);
      
      // Handle based on category
      switch (error.category) {
        case 'validation':
          handleError(new Error(error.message), {
            title: 'Validation Error',
            category: 'validation',
            metadata: { 
              code: error.code,
              affectedPaths: error.affectedPaths,
              details: error.details
            }
          });
          break;
          
        case 'rate_limit':
          handleError(new Error(error.message), {
            title: 'Rate Limit Exceeded',
            category: 'network',
            retry: async () => {
              // Wait for rate limit to expire
              if (error.retryAfter) {
                await new Promise(resolve => setTimeout(resolve, error.retryAfter));
              }
              // Retry queued messages
              await webSocketService.processMessageQueue();
            },
            metadata: {
              retryAfter: error.retryAfter,
              code: error.code
            }
          });
          break;
          
        case 'auth':
          handleError(new Error(error.message), {
            title: 'Authentication Error',
            category: 'network',
            actionLabel: 'Sign In',
            onAction: () => {
              // Navigate to auth or trigger auth modal
              window.location.href = '/auth/login';
            },
            metadata: {
              code: error.code
            }
          });
          break;
          
        case 'server':
          if (error.retryable) {
            handleError(new Error(error.message), {
              title: 'Server Error',
              category: 'network',
              retry: async () => {
                await webSocketService.connect();
              },
              metadata: {
                code: error.code,
                retryAfter: error.retryAfter
              }
            });
          } else {
            handleError(new Error(error.message), {
              title: 'Server Error',
              category: 'network',
              metadata: {
                code: error.code
              }
            });
          }
          break;
          
        case 'protocol':
          handleWebSocketError(new Error(error.message));
          break;
      }
    });
    
    // Listen for validation errors
    const unsubscribeValidation = webSocketService.on('validation-error', (data: { paths: string[], message: string }) => {
      handleSettingsError(new Error(data.message), data.paths);
    });
    
    // Listen for rate limit events
    const unsubscribeRateLimit = webSocketService.on('rate-limit', (data: { retryAfter: number, message: string }) => {
      logger.info(`Rate limited. Will retry after ${data.retryAfter}ms`);
    });
    
    // Listen for auth errors
    const unsubscribeAuth = webSocketService.on('auth-error', (data: { code: string, message: string }) => {
      // Clear auth state
      useSettingsStore.getState().setAuthenticated(false);
      useSettingsStore.getState().setUser(null);
    });
    
    // Listen for connection state changes
    const unsubscribeConnectionState = webSocketService.onConnectionStateChange((state) => {
      if (state.status === 'failed') {
        handleWebSocketError(new Error(state.lastError || 'Connection failed'));
      }
    });
    
    return () => {
      unsubscribeErrorFrame();
      unsubscribeValidation();
      unsubscribeRateLimit();
      unsubscribeAuth();
      unsubscribeConnectionState();
    };
  }, [handleError, handleWebSocketError, handleSettingsError]);
}

// Helper to report client errors to server
export function reportClientError(error: Error, context?: Record<string, any>) {
  try {
    webSocketService.sendErrorFrame({
      code: 'CLIENT_ERROR',
      message: error.message,
      category: 'protocol',
      details: {
        stack: error.stack,
        context,
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString()
      },
      retryable: false
    });
  } catch (sendError) {
    // Fail silently - don't want error reporting to cause more errors
    logger.debug('Failed to report client error:', sendError);
  }
}