/**
 * Authentication initialization module
 * Handles authentication setup and configuration
 */

interface AuthConfig {
  apiUrl?: string;
  tokenKey?: string;
  refreshInterval?: number;
}

class AuthenticationManager {
  private initialized = false;
  private config: AuthConfig = {
    apiUrl: import.meta.env.VITE_API_URL || '/api',
    tokenKey: 'auth_token',
    refreshInterval: 30 * 60 * 1000 // 30 minutes
  };

  private token: string | null = null;
  private refreshTimer: NodeJS.Timeout | null = null;

  /**
   * Initialize authentication system
   */
  async initialize(config?: Partial<AuthConfig>) {
    if (this.initialized) {
      console.warn('Authentication already initialized');
      return;
    }

    // Merge config
    this.config = { ...this.config, ...config };

    // Load existing token from storage
    this.loadToken();

    // Setup token refresh if token exists
    if (this.token) {
      this.setupTokenRefresh();
    }

    this.initialized = true;
    console.info('Authentication system initialized');
  }

  /**
   * Load token from localStorage
   */
  private loadToken() {
    try {
      const stored = localStorage.getItem(this.config.tokenKey!);
      if (stored) {
        this.token = stored;
      }
    } catch (error) {
      console.error('Failed to load auth token:', error);
    }
  }

  /**
   * Save token to localStorage
   */
  private saveToken(token: string) {
    try {
      localStorage.setItem(this.config.tokenKey!, token);
      this.token = token;
    } catch (error) {
      console.error('Failed to save auth token:', error);
    }
  }

  /**
   * Clear token from storage
   */
  private clearToken() {
    try {
      localStorage.removeItem(this.config.tokenKey!);
      this.token = null;
    } catch (error) {
      console.error('Failed to clear auth token:', error);
    }
  }

  /**
   * Setup automatic token refresh
   */
  private setupTokenRefresh() {
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer);
    }

    this.refreshTimer = setInterval(() => {
      this.refreshToken();
    }, this.config.refreshInterval!);
  }

  /**
   * Refresh authentication token
   */
  private async refreshToken() {
    if (!this.token) return;

    try {
      // In a real implementation, this would call the refresh endpoint
      console.debug('Token refresh would happen here');
      // const response = await fetch(`${this.config.apiUrl}/auth/refresh`, {
      //   method: 'POST',
      //   headers: {
      //     'Authorization': `Bearer ${this.token}`
      //   }
      // });
      // const data = await response.json();
      // this.saveToken(data.token);
    } catch (error) {
      console.error('Failed to refresh token:', error);
      this.handleAuthError();
    }
  }

  /**
   * Handle authentication errors
   */
  private handleAuthError() {
    this.clearToken();
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer);
      this.refreshTimer = null;
    }

    // In a real app, redirect to login or show auth modal
    console.warn('Authentication failed, would redirect to login');
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!this.token;
  }

  /**
   * Get current auth token
   */
  getToken(): string | null {
    return this.token;
  }

  /**
   * Login with credentials
   */
  async login(username: string, password: string): Promise<boolean> {
    try {
      // In a real implementation, this would call the login endpoint
      console.info('Login attempt for user:', username);

      // Simulated successful login
      const mockToken = btoa(`${username}:${Date.now()}`);
      this.saveToken(mockToken);
      this.setupTokenRefresh();

      return true;
    } catch (error) {
      console.error('Login failed:', error);
      return false;
    }
  }

  /**
   * Logout current user
   */
  async logout() {
    try {
      // In a real implementation, this would call the logout endpoint
      this.clearToken();

      if (this.refreshTimer) {
        clearInterval(this.refreshTimer);
        this.refreshTimer = null;
      }

      console.info('User logged out successfully');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  }

  /**
   * Cleanup resources
   */
  destroy() {
    if (this.refreshTimer) {
      clearInterval(this.refreshTimer);
      this.refreshTimer = null;
    }
    this.initialized = false;
  }
}

// Create singleton instance
const authManager = new AuthenticationManager();

/**
 * Initialize authentication system
 * Called during app initialization
 */
export async function initializeAuth(config?: Partial<AuthConfig>) {
  await authManager.initialize(config);
  return authManager;
}

// Export manager for use in components
export { authManager };
export type { AuthConfig };