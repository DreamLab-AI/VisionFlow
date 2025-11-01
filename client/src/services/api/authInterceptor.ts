

import { nostrAuth } from '../nostrAuthService';
import { createLogger } from '../../utils/loggerConfig';
import type { RequestConfig } from './UnifiedApiClient';
import { v4 as uuidv4 } from 'uuid';

const logger = createLogger('AuthInterceptor');


export function generateRequestId(): string {
  return uuidv4();
}


export function authRequestInterceptor(config: RequestConfig, url: string): RequestConfig {
  const finalConfig = { ...config };

  
  if (!finalConfig.headers) {
    finalConfig.headers = {};
  }

  
  const requestId = generateRequestId();
  finalConfig.headers['X-Request-ID'] = requestId;

  
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


export function initializeAuthInterceptor(apiClient: any): void {
  
  apiClient.setInterceptors({
    onRequest: authRequestInterceptor,
  });

  logger.info('Authentication interceptor initialized for UnifiedApiClient');
}


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