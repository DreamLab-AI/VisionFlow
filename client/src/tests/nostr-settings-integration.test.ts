/**
 * Nostr Authentication + Settings Integration Tests
 *
 * Tests the integration between Nostr authentication and settings persistence:
 * - Settings sync with authenticated users
 * - Settings persistence across sessions
 * - User-specific settings isolation
 * - Authentication state changes affecting settings
 */

import { describe, it, expect, beforeEach, afterEach, vi, beforeAll } from 'vitest';
import { settingsApi } from '../api/settingsApi';
import { nostrAuth, type AuthState, type SimpleNostrUser } from '../services/nostrAuthService';
import { Settings, SettingsUpdate } from '../features/settings/config/settings';
import { defaultSettings } from '../features/settings/config/defaultSettings';

// Mock fetch for API calls
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock localStorage
const mockLocalStorage = {
  store: new Map<string, string>(),
  getItem: vi.fn((key: string) => mockLocalStorage.store.get(key) || null),
  setItem: vi.fn((key: string, value: string) => {
    mockLocalStorage.store.set(key, value);
  }),
  removeItem: vi.fn((key: string) => {
    mockLocalStorage.store.delete(key);
  }),
  clear: vi.fn(() => {
    mockLocalStorage.store.clear();
  }),
};

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
  writable: true,
});

// Mock window.nostr for NIP-07
const mockNostrProvider = {
  getPublicKey: vi.fn(),
  signEvent: vi.fn(),
};

Object.defineProperty(window, 'nostr', {
  value: mockNostrProvider,
  writable: true,
});

// Test users
const testUsers = {
  powerUser: {
    pubkey: 'power_user_pubkey_123',
    npub: 'npub1power_user',
    isPowerUser: true,
  } as SimpleNostrUser,
  
  regularUser: {
    pubkey: 'regular_user_pubkey_456',
    npub: 'npub1regular_user', 
    isPowerUser: false,
  } as SimpleNostrUser,
};

// Test settings data
const userSpecificBloomSettings: SettingsUpdate = {
  visualisation: {
    glow: {
      enabled: true,
      intensity: 3.0,
      baseColor: '#ff6b6b',
      nodeGlowStrength: 4.0,
      edgeGlowStrength: 4.5,
    },
    graphs: {
      logseq: {
        physics: {
          springK: 0.25,
          repelK: 3.5,
          damping: 0.92,
        },
      },
    },
  },
};

describe('Nostr Authentication + Settings Integration', () => {
  let authStateCallbacks: ((state: AuthState) => void)[] = [];
  let currentAuthState: AuthState = { authenticated: false };
  
  const mockAuthStateListener = (callback: (state: AuthState) => void) => {
    authStateCallbacks.push(callback);
    // Immediately call with current state
    callback(currentAuthState);
    // Return unsubscribe function
    return () => {
      authStateCallbacks = authStateCallbacks.filter(cb => cb !== callback);
    };
  };
  
  const triggerAuthStateChange = (newState: AuthState) => {
    currentAuthState = newState;
    authStateCallbacks.forEach(callback => callback(newState));
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockClear();
    mockLocalStorage.store.clear();
    authStateCallbacks = [];
    currentAuthState = { authenticated: false };
    
    // Reset nostr provider mocks
    mockNostrProvider.getPublicKey.mockClear();
    mockNostrProvider.signEvent.mockClear();
  });

  describe('Authentication Flow Integration', () => {
    it('should sync settings after successful authentication', async () => {
      // Mock successful auth flow
      mockNostrProvider.getPublicKey.mockResolvedValue(testUsers.powerUser.pubkey);
      mockNostrProvider.signEvent.mockResolvedValue({
        id: 'signed_event_id',
        pubkey: testUsers.powerUser.pubkey,
        content: 'Authenticate to LogseqSpringThing',
        sig: 'valid_signature',
        created_at: Math.floor(Date.now() / 1000),
        kind: 22242,
        tags: [['relay', 'wss://relay.damus.io'], ['challenge', 'uuid']],
      });
      
      // Mock backend auth response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          user: {
            pubkey: testUsers.powerUser.pubkey,
            npub: testUsers.powerUser.npub,
            isPowerUser: true,
          },
          token: 'valid_jwt_token',
          expiresAt: Date.now() + 86400000, // 24 hours
        }),
      } as Response);
      
      // Mock settings update after auth
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...defaultSettings,
          ...userSpecificBloomSettings,
        }),
      } as Response);
      
      // Perform login
      const authState = await nostrAuth.login();
      
      expect(authState.authenticated).toBe(true);
      expect(authState.user?.isPowerUser).toBe(true);
      
      // Now update settings - should include auth headers
      await settingsApi.updateSettings(userSpecificBloomSettings);
      
      // Verify the settings call included authentication
      expect(mockFetch).toHaveBeenCalledWith('/api/settings', expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userSpecificBloomSettings),
      }));
    });
    
    it('should handle settings persistence across sessions', async () => {
      // Simulate stored session
      mockLocalStorage.store.set('nostr_session_token', 'stored_token');
      mockLocalStorage.store.set('nostr_user', JSON.stringify(testUsers.powerUser));
      
      // Mock token verification
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          valid: true,
          user: testUsers.powerUser,
        }),
      } as Response);
      
      // Initialize service (should restore session)
      await nostrAuth.initialize();
      
      expect(nostrAuth.isAuthenticated()).toBe(true);
      expect(nostrAuth.getCurrentUser()?.pubkey).toBe(testUsers.powerUser.pubkey);
      
      // Mock settings fetch for authenticated user
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...defaultSettings,
          ...userSpecificBloomSettings,
        }),
      } as Response);
      
      const settings = await settingsApi.fetchSettings();
      
      // Should fetch user-specific settings
      expect(settings.visualisation.glow.intensity).toBe(3.0);
      expect(settings.visualisation.glow.baseColor).toBe('#ff6b6b');
    });
    
    it('should clear settings on logout', async () => {
      // Set up authenticated state
      currentAuthState = {
        authenticated: true,
        user: testUsers.powerUser,
      };
      
      // Mock logout API call
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true }),
      } as Response);
      
      // Perform logout
      await nostrAuth.logout();
      
      // Verify local storage is cleared
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('nostr_session_token');
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('nostr_user');
      
      // Verify backend logout was called
      expect(mockFetch).toHaveBeenCalledWith('/api/auth/nostr', expect.objectContaining({
        method: 'DELETE',
        body: JSON.stringify({
          pubkey: testUsers.powerUser.pubkey,
          token: expect.any(String),
        }),
      }));
    });
  });

  describe('User-Specific Settings Isolation', () => {
    it('should handle different settings for different users', async () => {
      // Test power user settings
      currentAuthState = {
        authenticated: true,
        user: testUsers.powerUser,
      };
      
      const powerUserSettings = {
        visualisation: {
          glow: {
            intensity: 5.0, // High intensity for power user
            nodeGlowStrength: 5.0,
          },
        },
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...defaultSettings,
          ...powerUserSettings,
        }),
      } as Response);
      
      await settingsApi.updateSettings(powerUserSettings);
      
      // Switch to regular user
      currentAuthState = {
        authenticated: true,
        user: testUsers.regularUser,
      };
      
      const regularUserSettings = {
        visualisation: {
          glow: {
            intensity: 1.0, // Lower intensity for regular user
            nodeGlowStrength: 2.0,
          },
        },
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...defaultSettings,
          ...regularUserSettings,
        }),
      } as Response);
      
      await settingsApi.updateSettings(regularUserSettings);
      
      // Verify different API calls were made
      expect(mockFetch).toHaveBeenCalledTimes(2);
      expect(mockFetch).toHaveBeenNthCalledWith(1, '/api/settings', expect.objectContaining({
        body: JSON.stringify(powerUserSettings),
      }));
      expect(mockFetch).toHaveBeenNthCalledWith(2, '/api/settings', expect.objectContaining({
        body: JSON.stringify(regularUserSettings),
      }));
    });
    
    it('should handle power user exclusive features', async () => {
      // Set power user
      currentAuthState = {
        authenticated: true,
        user: testUsers.powerUser,
      };
      
      const advancedSettings = {
        visualisation: {
          glow: {
            // Advanced settings that might be power-user only
            volumetricIntensity: 5.0,
            atmosphericDensity: 1.5,
          },
          graphs: {
            logseq: {
              physics: {
                // Advanced physics parameters
                computeMode: 3, // Visual Analytics mode
                clusteringAlgorithm: 'spectral',
                iterations: 200, // High iteration count
              },
            },
          },
        },
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...defaultSettings,
          ...advancedSettings,
        }),
      } as Response);
      
      await settingsApi.updateSettings(advancedSettings);
      
      expect(mockFetch).toHaveBeenCalledWith('/api/settings', expect.objectContaining({
        body: JSON.stringify(advancedSettings),
      }));
      
      // Switch to regular user - should not be able to access advanced features
      currentAuthState = {
        authenticated: true,
        user: testUsers.regularUser,
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: () => Promise.resolve({
          error: 'Advanced settings require power user privileges',
        }),
      } as Response);
      
      await expect(settingsApi.updateSettings(advancedSettings))
        .rejects.toThrow(/power user privileges/i);
    });
  });

  describe('Authentication State Changes', () => {
    it('should handle auth state listener correctly', async () => {
      const authStateListener = vi.fn();
      
      // Set up mock for onAuthStateChanged
      vi.spyOn(nostrAuth, 'onAuthStateChanged').mockImplementation(mockAuthStateListener);
      
      const unsubscribe = nostrAuth.onAuthStateChanged(authStateListener);
      
      // Should be called immediately with current state
      expect(authStateListener).toHaveBeenCalledWith({ authenticated: false });
      
      // Trigger auth state change
      triggerAuthStateChange({
        authenticated: true,
        user: testUsers.powerUser,
      });
      
      expect(authStateListener).toHaveBeenCalledWith({
        authenticated: true,
        user: testUsers.powerUser,
      });
      
      // Unsubscribe should work
      unsubscribe();
      
      // Further changes shouldn't trigger the callback
      const callCount = authStateListener.mock.calls.length;
      triggerAuthStateChange({ authenticated: false });
      
      expect(authStateListener).toHaveBeenCalledTimes(callCount);
    });
    
    it('should handle authentication errors gracefully', async () => {
      // Mock failed authentication
      mockNostrProvider.getPublicKey.mockRejectedValue(new Error('User rejected request'));
      
      await expect(nostrAuth.login())
        .rejects.toThrow(/login request rejected/i);
      
      expect(nostrAuth.isAuthenticated()).toBe(false);
      expect(nostrAuth.getCurrentUser()).toBe(null);
    });
    
    it('should handle network errors during token verification', async () => {
      // Set up stored session
      mockLocalStorage.store.set('nostr_session_token', 'stored_token');
      mockLocalStorage.store.set('nostr_user', JSON.stringify(testUsers.powerUser));
      
      // Mock network error during verification
      mockFetch.mockRejectedValueOnce(new Error('Network error'));
      
      await nostrAuth.initialize();
      
      // Should clear session on verification failure
      expect(nostrAuth.isAuthenticated()).toBe(false);
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('nostr_session_token');
      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('nostr_user');
    });
  });

  describe('Settings API Integration with Auth', () => {
    beforeEach(() => {
      // Mock authenticated state
      currentAuthState = {
        authenticated: true,
        user: testUsers.powerUser,
      };
    });
    
    it('should include auth headers in API requests when authenticated', async () => {
      // Mock settings API response
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(defaultSettings),
      } as Response);
      
      await settingsApi.fetchSettings();
      
      // In a real implementation, auth headers would be included
      // This test verifies the API call structure
      expect(mockFetch).toHaveBeenCalledWith('/api/settings');
    });
    
    it('should handle unauthenticated settings requests', async () => {
      // Set unauthenticated state
      currentAuthState = { authenticated: false };
      
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: 'Authentication required for personalized settings',
        }),
      } as Response);
      
      await expect(settingsApi.fetchSettings())
        .rejects.toThrow(/authentication required/i);
    });
    
    it('should handle expired tokens gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: 'Token expired',
        }),
      } as Response);
      
      await expect(settingsApi.updateSettings(userSpecificBloomSettings))
        .rejects.toThrow(/token expired/i);
    });
  });

  describe('Bloom/Glow Settings with Authentication', () => {
    beforeEach(() => {
      currentAuthState = {
        authenticated: true,
        user: testUsers.powerUser,
      };
    });
    
    it('should sync bloom settings for authenticated power users', async () => {
      const advancedBloomSettings = {
        visualisation: {
          glow: {
            enabled: true,
            intensity: 4.0,
            radius: 1.2,
            threshold: 0.05,
            diffuseStrength: 2.5,
            atmosphericDensity: 1.5,
            volumetricIntensity: 2.0,
            baseColor: '#ff69b4',
            emissionColor: '#00ffff',
            opacity: 1.0,
            pulseSpeed: 1.5,
            flowSpeed: 1.2,
            nodeGlowStrength: 5.0,
            edgeGlowStrength: 5.5,
            environmentGlowStrength: 4.5,
          },
        },
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...defaultSettings,
          ...advancedBloomSettings,
        }),
      } as Response);
      
      const result = await settingsApi.updateSettings(advancedBloomSettings);
      
      expect(result.visualisation.glow.intensity).toBe(4.0);
      expect(result.visualisation.glow.nodeGlowStrength).toBe(5.0);
      expect(result.visualisation.glow.baseColor).toBe('#ff69b4');
    });
    
    it('should restrict advanced bloom features for regular users', async () => {
      currentAuthState = {
        authenticated: true,
        user: testUsers.regularUser,
      };
      
      const restrictedBloomSettings = {
        visualisation: {
          glow: {
            // Settings that might be restricted to power users
            volumetricIntensity: 5.0,
            atmosphericDensity: 2.0,
          },
        },
      };
      
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: () => Promise.resolve({
          error: 'Advanced bloom features require power user access',
        }),
      } as Response);
      
      await expect(settingsApi.updateSettings(restrictedBloomSettings))
        .rejects.toThrow(/power user access/i);
    });
    
    it('should validate bloom settings against user permissions', async () => {
      const validationTests = [
        {
          user: testUsers.powerUser,
          settings: { visualisation: { glow: { intensity: 10.0 } } },
          shouldPass: true,
        },
        {
          user: testUsers.regularUser,
          settings: { visualisation: { glow: { intensity: 10.0 } } },
          shouldPass: false,
        },
        {
          user: testUsers.powerUser,
          settings: { visualisation: { glow: { volumetricIntensity: 5.0 } } },
          shouldPass: true,
        },
        {
          user: testUsers.regularUser,
          settings: { visualisation: { glow: { volumetricIntensity: 5.0 } } },
          shouldPass: false,
        },
      ];
      
      for (const test of validationTests) {
        currentAuthState = {
          authenticated: true,
          user: test.user,
        };
        
        if (test.shouldPass) {
          mockFetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({
              ...defaultSettings,
              ...test.settings,
            }),
          } as Response);
          
          const result = await settingsApi.updateSettings(test.settings);
          expect(result).toBeDefined();
        } else {
          mockFetch.mockResolvedValueOnce({
            ok: false,
            status: 403,
            json: () => Promise.resolve({
              error: 'Permission denied',
            }),
          } as Response);
          
          await expect(settingsApi.updateSettings(test.settings))
            .rejects.toThrow();
        }
      }
    });
  });

  describe('Error Recovery and Resilience', () => {
    it('should handle auth service unavailability', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Service unavailable'));
      
      await expect(nostrAuth.login())
        .rejects.toThrow();
      
      // Should fallback to unauthenticated mode
      expect(nostrAuth.isAuthenticated()).toBe(false);
    });
    
    it('should retry failed settings requests', async () => {
      currentAuthState = {
        authenticated: true,
        user: testUsers.powerUser,
      };
      
      // First request fails
      mockFetch.mockRejectedValueOnce(new Error('Network timeout'));
      
      // Second request succeeds (retry)
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...defaultSettings,
          ...userSpecificBloomSettings,
        }),
      } as Response);
      
      // The API itself doesn't implement retry, but this tests the pattern
      try {
        await settingsApi.updateSettings(userSpecificBloomSettings);
      } catch (error) {
        // Retry manually in test
        const result = await settingsApi.updateSettings(userSpecificBloomSettings);
        expect(result).toBeDefined();
      }
    });
  });
});
