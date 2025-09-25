/**
 * API Services Index
 * Centralized exports for all API-related functionality
 */

// Export the unified API client as the primary interface
export {
  unifiedApiClient as default,
  unifiedApiClient,
  UnifiedApiClient,
  createApiClient,
  isApiError
} from './UnifiedApiClient';

// Re-export types for convenience
export type {
  ApiResponse,
  ApiError,
  RequestConfig,
  InterceptorConfig,
  RetryConfig
} from './UnifiedApiClient';