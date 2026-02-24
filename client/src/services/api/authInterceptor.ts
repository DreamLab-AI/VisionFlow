import { nostrAuth } from '../nostrAuthService';
import { createLogger } from '../../utils/loggerConfig';
import type { RequestConfig } from './UnifiedApiClient';
import { v4 as uuidv4 } from 'uuid';

const logger = createLogger('AuthInterceptor');

export function generateRequestId(): string {
  return uuidv4();
}

export async function authRequestInterceptor(config: RequestConfig, url: string): Promise<RequestConfig> {
  const finalConfig = { ...config };

  if (!finalConfig.headers) {
    finalConfig.headers = {} as Record<string, string>;
  }

  const headers = finalConfig.headers as Record<string, string>;

  const requestId = generateRequestId();
  headers['X-Request-ID'] = requestId;

  if (nostrAuth.isAuthenticated()) {
    const user = nostrAuth.getCurrentUser();

    if (nostrAuth.isDevMode()) {
      // Dev mode: Bearer token
      headers['Authorization'] = 'Bearer dev-session-token';
      if (user?.pubkey) {
        headers['X-Nostr-Pubkey'] = user.pubkey;
      }
      logger.debug(`[${requestId}] Dev-mode auth headers for ${url}`);
    } else if (user?.pubkey) {
      // Always sign with NIP-98 ourselves. NIP-07 extensions like Podkey may
      // also intercept, but their retry-on-401 approach is unreliable for
      // PUT/POST mutations.
      try {
        const fullUrl = new URL(url, window.location.origin).href;
        const method = (finalConfig.method || 'GET').toUpperCase();
        const body = typeof finalConfig.body === 'string' ? finalConfig.body : undefined;
        const token = await nostrAuth.signRequest(fullUrl, method, body);
        headers['Authorization'] = `Nostr ${token}`;
        logger.debug(`[${requestId}] NIP-98 signed request for ${method} ${url}`);
      } catch (e) {
        logger.error(`[${requestId}] NIP-98 signing failed:`, e);
      }
    }
  } else {
    logger.debug(`[${requestId}] No auth headers (not authenticated) for ${url}`);
  }

  finalConfig.headers = headers;
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
