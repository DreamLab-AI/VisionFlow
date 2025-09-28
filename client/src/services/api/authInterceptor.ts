/**
 * Authentication Interceptor for UnifiedApiClient
 * Automatically includes Nostr authentication headers and request IDs in all API calls
 */

import { nostrAuth } from '../nostrAuthService';
import { createLogger } from '../../utils/loggerConfig';
import type { RequestConfig } from './UnifiedApiClient';
import { v4 as uuidv4 } from 'uuid';

const logger = createLogger('AuthInterceptor');

/**
 * Generate a unique request ID for tracing
 */
export function generateRequestId(): string {
  return uuidv4();
}

/**
 * Request interceptor that adds authentication headers
 */
export function authRequestInterceptor(config: RequestConfig, url: string): RequestConfig {
  const finalConfig = { ...config };

  // Initialize headers if not present
  if (!finalConfig.headers) {
    finalConfig.headers = {};
  }

  // Add request ID for tracing
  const requestId = generateRequestId();
  finalConfig.headers['X-Request-ID'] = requestId;

  // Add Nostr authentication headers if authenticated
  if (nostrAuth.isAuthenticated()) {
    const user = nostrAuth.getCurrentUser();
    const token = nostrAuth.getSessionToken();

    if (user?.pubkey) {
      finalConfig.headers['X-Nostr-Pubkey'] = user.pubkey;
      logger.debug(`[${requestId}] Added Nostr pubkey header for ${url}`, {
        pubkey: user.pubkey.slice(0, 8) + '...',
      });
    }

    if (token) {
      finalConfig.headers['X-Nostr-Token'] = token;
      logger.debug(`[${requestId}] Added Nostr token header for ${url}`);
    }
  } else {
    logger.debug(`[${requestId}] No Nostr auth headers added (not authenticated) for ${url}`);
  }

  return finalConfig;
}

/**
 * Initialize authentication interceptor for the UnifiedApiClient
 */
export function initializeAuthInterceptor(apiClient: any): void {
  // Set the interceptor
  apiClient.setInterceptors({
    onRequest: authRequestInterceptor,
  });

  logger.info('Authentication interceptor initialized for UnifiedApiClient');
}

/**
 * Update auth headers when authentication state changes
 */
export function setupAuthStateListener(): void {
  nostrAuth.onAuthStateChanged((state) => {
    if (state.authenticated) {
      logger.info('Authentication state changed: User authenticated', {
        pubkey: state.user?.pubkey?.slice(0, 8) + '...',
      });
    } else {
      logger.info('Authentication state changed: User logged out');
    }
  });
}