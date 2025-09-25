/**
 * Unified API Client
 * Production-ready HTTP client with centralized configuration, auth management,
 * request/response interceptors, retry logic, and comprehensive error handling
 */

import { createLogger } from '../../utils/loggerConfig';
import { createErrorMetadata } from '../../utils/loggerConfig';
import { debugState } from '../../utils/clientDebugState';

const logger = createLogger('UnifiedApiClient');

// Core interfaces
export interface ApiResponse<T = any> {
  data: T;
  status: number;
  statusText: string;
  headers: Record<string, string>;
}

export interface ApiError extends Error {
  status?: number;
  statusText?: string;
  data?: any;
  isApiError: true;
}

export interface RequestConfig extends RequestInit {
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  skipAuth?: boolean;
  skipInterceptors?: boolean;
}

export interface InterceptorConfig {
  onRequest?: (config: RequestConfig, url: string) => Promise<RequestConfig> | RequestConfig;
  onResponse?: <T>(response: ApiResponse<T>) => Promise<ApiResponse<T>> | ApiResponse<T>;
  onError?: (error: ApiError) => Promise<ApiError> | ApiError | never;
}

export interface RetryConfig {
  maxRetries: number;
  retryDelay: number;
  retryCondition: (error: ApiError, attempt: number) => boolean;
  onRetry?: (error: ApiError, attempt: number) => void;
}

// Default configurations
const DEFAULT_TIMEOUT = 30000; // 30 seconds
const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  retryDelay: 1000,
  retryCondition: (error: ApiError, attempt: number) => {
    // Retry on network errors or 5xx server errors, but not on auth errors
    return (
      attempt < 3 &&
      (!error.status || error.status >= 500 || error.status === 0) &&
      error.status !== 401 &&
      error.status !== 403
    );
  },
  onRetry: (error: ApiError, attempt: number) => {
    logger.warn(`Retrying request (attempt ${attempt + 1}/3):`, {
      status: error.status,
      message: error.message
    });
  }
};

/**
 * Unified API Client class with comprehensive features
 */
export class UnifiedApiClient {
  private baseURL: string;
  private defaultHeaders: Record<string, string>;
  private authToken: string | null = null;
  private interceptors: InterceptorConfig = {};
  private retryConfig: RetryConfig;
  private abortController: AbortController | null = null;

  constructor(baseURL: string = '/api', retryConfig: Partial<RetryConfig> = {}) {
    this.baseURL = baseURL.endsWith('/') ? baseURL.slice(0, -1) : baseURL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };
    this.retryConfig = { ...DEFAULT_RETRY_CONFIG, ...retryConfig };

    // Initialize abort controller for request cancellation
    this.abortController = new AbortController();
  }

  /**
   * Set authentication token
   */
  public setAuthToken(token: string): void {
    this.authToken = token;
    this.defaultHeaders['Authorization'] = `Bearer ${token}`;
    logger.info('Authentication token set');
  }

  /**
   * Remove authentication token
   */
  public removeAuthToken(): void {
    this.authToken = null;
    delete this.defaultHeaders['Authorization'];
    logger.info('Authentication token removed');
  }

  /**
   * Get current auth token
   */
  public getAuthToken(): string | null {
    return this.authToken;
  }

  /**
   * Set default header
   */
  public setDefaultHeader(key: string, value: string): void {
    this.defaultHeaders[key] = value;
  }

  /**
   * Remove default header
   */
  public removeDefaultHeader(key: string): void {
    delete this.defaultHeaders[key];
  }

  /**
   * Set request/response interceptors
   */
  public setInterceptors(interceptors: InterceptorConfig): void {
    this.interceptors = { ...this.interceptors, ...interceptors };
  }

  /**
   * Update retry configuration
   */
  public setRetryConfig(config: Partial<RetryConfig>): void {
    this.retryConfig = { ...this.retryConfig, ...config };
  }

  /**
   * Cancel all pending requests
   */
  public cancelRequests(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = new AbortController();
      logger.info('All pending requests cancelled');
    }
  }

  /**
   * Create API error from response or exception
   */
  private createApiError(
    message: string,
    status?: number,
    statusText?: string,
    data?: any
  ): ApiError {
    const error = new Error(message) as ApiError;
    error.name = 'ApiError';
    error.isApiError = true;
    error.status = status;
    error.statusText = statusText;
    error.data = data;
    return error;
  }

  /**
   * Execute request with retry logic
   */
  private async executeWithRetry<T>(
    url: string,
    config: RequestConfig,
    attempt: number = 0
  ): Promise<ApiResponse<T>> {
    try {
      return await this.executeRequest<T>(url, config);
    } catch (error) {
      const apiError = error as ApiError;

      if (
        attempt < this.retryConfig.maxRetries &&
        this.retryConfig.retryCondition(apiError, attempt)
      ) {
        if (this.retryConfig.onRetry) {
          this.retryConfig.onRetry(apiError, attempt);
        }

        // Exponential backoff
        const delay = this.retryConfig.retryDelay * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));

        return this.executeWithRetry<T>(url, config, attempt + 1);
      }

      throw error;
    }
  }

  /**
   * Core request execution method
   */
  private async executeRequest<T>(url: string, config: RequestConfig): Promise<ApiResponse<T>> {
    const fullUrl = url.startsWith('http') ? url : `${this.baseURL}${url}`;

    // Apply request interceptor
    let finalConfig = { ...config };
    if (this.interceptors.onRequest && !config.skipInterceptors) {
      finalConfig = await this.interceptors.onRequest(finalConfig, fullUrl);
    }

    // Merge headers
    const headers = {
      ...this.defaultHeaders,
      ...finalConfig.headers,
    };

    // Skip auth if requested
    if (finalConfig.skipAuth && headers.Authorization) {
      delete headers.Authorization;
    }

    // Set up abort signal with timeout
    const timeoutMs = finalConfig.timeout || DEFAULT_TIMEOUT;
    const timeoutId = setTimeout(() => {
      this.abortController?.abort();
    }, timeoutMs);

    const requestConfig: RequestInit = {
      ...finalConfig,
      headers,
      signal: this.abortController?.signal,
    };

    if (debugState.isEnabled()) {
      logger.debug(`Making ${finalConfig.method || 'GET'} request to ${fullUrl}`, {
        headers: Object.keys(headers),
        hasBody: !!finalConfig.body,
      });
    }

    let response: Response;
    try {
      response = await fetch(fullUrl, requestConfig);
    } catch (fetchError: any) {
      clearTimeout(timeoutId);

      if (fetchError.name === 'AbortError') {
        throw this.createApiError(
          'Request timeout or cancelled',
          0,
          'Timeout'
        );
      }

      throw this.createApiError(
        `Network error: ${fetchError.message}`,
        0,
        'Network Error',
        fetchError
      );
    } finally {
      clearTimeout(timeoutId);
    }

    // Parse response
    let responseData: any;
    const contentType = response.headers.get('content-type');

    try {
      if (contentType?.includes('application/json')) {
        responseData = await response.json();
      } else if (contentType?.includes('text/')) {
        responseData = await response.text();
      } else {
        responseData = await response.arrayBuffer();
      }
    } catch (parseError: any) {
      logger.warn('Failed to parse response body:', parseError);
      responseData = null;
    }

    // Create response object
    const apiResponse: ApiResponse<T> = {
      data: responseData,
      status: response.status,
      statusText: response.statusText,
      headers: Object.fromEntries(response.headers.entries()),
    };

    // Handle HTTP errors
    if (!response.ok) {
      const errorMessage = responseData?.error ||
                          responseData?.message ||
                          `HTTP ${response.status}: ${response.statusText}`;

      const apiError = this.createApiError(
        errorMessage,
        response.status,
        response.statusText,
        responseData
      );

      if (this.interceptors.onError && !config.skipInterceptors) {
        try {
          throw await this.interceptors.onError(apiError);
        } catch (interceptorError) {
          throw interceptorError;
        }
      }

      throw apiError;
    }

    // Apply response interceptor
    let finalResponse = apiResponse;
    if (this.interceptors.onResponse && !config.skipInterceptors) {
      finalResponse = await this.interceptors.onResponse(apiResponse);
    }

    if (debugState.isEnabled()) {
      logger.debug(`Request to ${fullUrl} succeeded`, {
        status: response.status,
        contentType,
      });
    }

    return finalResponse;
  }

  /**
   * Generic request method
   */
  public async request<T = any>(
    method: string,
    url: string,
    data?: any,
    config: RequestConfig = {}
  ): Promise<ApiResponse<T>> {
    const requestConfig: RequestConfig = {
      method: method.toUpperCase(),
      ...config,
    };

    // Add body for methods that support it
    if (data && ['POST', 'PUT', 'PATCH'].includes(requestConfig.method!)) {
      if (typeof data === 'string' || data instanceof ArrayBuffer || data instanceof FormData) {
        requestConfig.body = data;
      } else {
        requestConfig.body = JSON.stringify(data);
      }
    }

    try {
      return await this.executeWithRetry<T>(url, requestConfig);
    } catch (error) {
      const apiError = error as ApiError;
      logger.error(`${method.toUpperCase()} request to ${url} failed:`, createErrorMetadata(apiError));
      throw apiError;
    }
  }

  /**
   * GET request
   */
  public async get<T = any>(url: string, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('GET', url, undefined, config);
  }

  /**
   * POST request
   */
  public async post<T = any>(url: string, data?: any, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('POST', url, data, config);
  }

  /**
   * PUT request
   */
  public async put<T = any>(url: string, data?: any, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('PUT', url, data, config);
  }

  /**
   * PATCH request
   */
  public async patch<T = any>(url: string, data?: any, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('PATCH', url, data, config);
  }

  /**
   * DELETE request
   */
  public async delete<T = any>(url: string, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('DELETE', url, undefined, config);
  }

  /**
   * HEAD request
   */
  public async head<T = any>(url: string, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('HEAD', url, undefined, config);
  }

  /**
   * OPTIONS request
   */
  public async options<T = any>(url: string, config?: RequestConfig): Promise<ApiResponse<T>> {
    return this.request<T>('OPTIONS', url, undefined, config);
  }

  /**
   * Convenience methods that return data directly (similar to existing patterns)
   */
  public async getData<T = any>(url: string, config?: RequestConfig): Promise<T> {
    const response = await this.get<T>(url, config);
    return response.data;
  }

  public async postData<T = any>(url: string, data?: any, config?: RequestConfig): Promise<T> {
    const response = await this.post<T>(url, data, config);
    return response.data;
  }

  public async putData<T = any>(url: string, data?: any, config?: RequestConfig): Promise<T> {
    const response = await this.put<T>(url, data, config);
    return response.data;
  }

  public async patchData<T = any>(url: string, data?: any, config?: RequestConfig): Promise<T> {
    const response = await this.patch<T>(url, data, config);
    return response.data;
  }

  public async deleteData<T = any>(url: string, config?: RequestConfig): Promise<T> {
    const response = await this.delete<T>(url, config);
    return response.data;
  }

  /**
   * Upload file with progress tracking
   */
  public async uploadFile<T = any>(
    url: string,
    file: File | FormData,
    onProgress?: (progress: number) => void,
    config?: RequestConfig
  ): Promise<ApiResponse<T>> {
    const formData = file instanceof FormData ? file : new FormData();
    if (file instanceof File) {
      formData.append('file', file);
    }

    // Remove content-type to let browser set it with boundary
    const uploadConfig = { ...config };
    if (uploadConfig.headers) {
      delete uploadConfig.headers['Content-Type'];
    }

    // TODO: Implement progress tracking with XMLHttpRequest if needed
    return this.post<T>(url, formData, uploadConfig);
  }

  /**
   * Create a new instance with different base URL
   */
  public createInstance(baseURL: string, config: Partial<RetryConfig> = {}): UnifiedApiClient {
    const instance = new UnifiedApiClient(baseURL, config);
    instance.defaultHeaders = { ...this.defaultHeaders };
    instance.authToken = this.authToken;
    instance.interceptors = { ...this.interceptors };
    return instance;
  }

  /**
   * Health check method
   */
  public async healthCheck(): Promise<boolean> {
    try {
      await this.get('/health', { timeout: 5000, retries: 1 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get client configuration for debugging
   */
  public getConfig() {
    return {
      baseURL: this.baseURL,
      hasAuthToken: !!this.authToken,
      defaultHeaders: Object.keys(this.defaultHeaders),
      retryConfig: this.retryConfig,
      hasInterceptors: {
        onRequest: !!this.interceptors.onRequest,
        onResponse: !!this.interceptors.onResponse,
        onError: !!this.interceptors.onError,
      }
    };
  }
}

// Create and export singleton instance
export const unifiedApiClient = new UnifiedApiClient();

// Export utility functions
export const createApiClient = (baseURL: string, retryConfig?: Partial<RetryConfig>) => {
  return new UnifiedApiClient(baseURL, retryConfig);
};

// Type guards
export const isApiError = (error: any): error is ApiError => {
  return error && typeof error === 'object' && error.isApiError === true;
};

// Default export for easy importing
export default unifiedApiClient;