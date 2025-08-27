/**
 * Integration tests for settings sync functionality
 * 
 * Tests the complete client-server settings synchronization flow:
 * - REST API calls with bloom field validation
 * - Settings persistence and retrieval
 * - Bidirectional sync between client and server
 * - Nostr authentication integration
 * - Error handling and recovery scenarios
 */

import { describe, it, expect, beforeEach, afterEach, vi, beforeAll, afterAll } from 'vitest';
import { settingsApi } from '../api/settingsApi';
import { nostrAuth } from '../services/nostrAuthService';
import { Settings, SettingsUpdate } from '../features/settings/config/settings';
// import { defaultSettings } from '../features/settings/config/defaultSettings';

// Mock fetch for testing
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Test server base URL
const TEST_SERVER = process.env.TEST_SERVER_URL || 'http://localhost:8080';

// Test data for comprehensive bloom/glow settings
const bloomTestSettings: SettingsUpdate = {
  visualisation: {
    glow: {
      enabled: true,
      intensity: 2.5,
      radius: 0.9,
      threshold: 0.2,
      diffuseStrength: 1.8,
      atmosphericDensity: 0.9,
      volumetricIntensity: 1.5,
      baseColor: '#ff6b6b',
      emissionColor: '#4ecdc4',
      opacity: 0.95,
      pulseSpeed: 1.2,
      flowSpeed: 0.9,
      nodeGlowStrength: 3.5,
      edgeGlowStrength: 4.0,
      environmentGlowStrength: 3.2,
    },
    graphs: {
      logseq: {
        physics: {
          enabled: true,
          springK: 0.15,
          repelK: 2.5,
          attractionK: 0.02,
          gravity: 0.0002,
          damping: 0.9,
          maxVelocity: 8.0,
          dt: 0.02,
          temperature: 0.02,
          iterations: 75,
          boundsSize: 1200.0,
          separationRadius: 2.5,
          boundaryDamping: 0.95,
          massScale: 1.2,
          updateThreshold: 0.015,
        },
      },
      visionflow: {
        physics: {
          enabled: true,
          springK: 0.12,
          repelK: 2.2,
          attractionK: 0.015,
          gravity: 0.0001,
          damping: 0.88,
          maxVelocity: 7.0,
          dt: 0.018,
          temperature: 0.015,
          iterations: 60,
          boundsSize: 1000.0,
          separationRadius: 2.0,
          boundaryDamping: 0.92,
          massScale: 1.0,
          updateThreshold: 0.01,
        },
      },
    },
  },
};

// Invalid settings for validation testing
const invalidBloomSettings = [
  {
    name: 'negative intensity',
    settings: {
      visualisation: {
        glow: {
          intensity: -1.0,
        },
      },
    },
    expectedError: /intensity.*must be.*positive/i,
  },
  {
    name: 'invalid color format',
    settings: {
      visualisation: {
        glow: {
          baseColor: 'not-a-color',
        },
      },
    },
    expectedError: /invalid.*color/i,
  },
  {
    name: 'damping out of range',
    settings: {
      visualisation: {
        graphs: {
          logseq: {
            physics: {
              damping: 1.5, // > 1.0 is invalid
            },
          },
        },
      },
    },
    expectedError: /damping.*between.*0.*1/i,
  },
  {
    name: 'invalid iterations',
    settings: {
      visualisation: {
        graphs: {
          logseq: {
            physics: {
              iterations: 0, // Must be > 0
            },
          },
        },
      },
    },
    expectedError: /iterations.*must be.*positive/i,
  },
];

// Mock Nostr auth for testing
const mockNostrAuth = {
  isAuthenticated: vi.fn(() => false),
  getCurrentUser: vi.fn(() => null),
  login: vi.fn(),
  logout: vi.fn(),
  onAuthStateChanged: vi.fn(() => () => {}),
};

vi.mock('../services/nostrAuthService', () => ({
  nostrAuth: mockNostrAuth,
}));

describe('Settings Sync Integration Tests', () => {
  let originalFetch: typeof global.fetch;

  beforeAll(() => {
    originalFetch = global.fetch;
  });

  afterAll(() => {
    global.fetch = originalFetch;
  });

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockClear();
  });

  describe('REST API Endpoints', () => {
    it('should fetch settings with bloom fields', async () => {
      const mockSettings: Settings = {
        ...defaultSettings,
        visualisation: {
          ...defaultSettings.visualisation,
          glow: {
            enabled: true,
            intensity: 2.0,
            radius: 0.85,
            threshold: 0.15,
            diffuseStrength: 1.5,
            atmosphericDensity: 0.8,
            volumetricIntensity: 1.2,
            baseColor: '#00ffff',
            emissionColor: '#ffffff',
            opacity: 0.9,
            pulseSpeed: 1.0,
            flowSpeed: 0.8,
            nodeGlowStrength: 3.0,
            edgeGlowStrength: 3.5,
            environmentGlowStrength: 3.0,
          },
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSettings),
      } as Response);

      const settings = await settingsApi.fetchSettings();

      expect(mockFetch).toHaveBeenCalledWith('/api/settings');
      expect(settings).toBeDefined();
      expect(settings.visualisation.glow).toBeDefined();
      expect(settings.visualisation.glow.enabled).toBe(true);
      expect(settings.visualisation.glow.nodeGlowStrength).toBe(3.0);
      expect(settings.visualisation.glow.baseColor).toBe('#00ffff');
    });

    it('should update settings with bloom fields', async () => {
      const updatedSettings = {
        ...defaultSettings,
        ...bloomTestSettings,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(updatedSettings),
      } as Response);

      const result = await settingsApi.updateSettings(bloomTestSettings);

      expect(mockFetch).toHaveBeenCalledWith('/api/settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(bloomTestSettings),
      });

      expect(result.visualisation.glow.intensity).toBe(2.5);
      expect(result.visualisation.glow.baseColor).toBe('#ff6b6b');
    });

    it('should handle server validation errors', async () => {
      const invalidUpdate = invalidBloomSettings[0];

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid settings: intensity must be positive',
        }),
      } as Response);

      await expect(settingsApi.updateSettings(invalidUpdate.settings))
        .rejects.toThrow(invalidUpdate.expectedError);
    });

    it('should handle rate limiting', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () => Promise.resolve({
          error: 'rate_limit_exceeded',
          message: 'Too many settings update requests',
          retry_after: 60,
        }),
      } as Response);

      await expect(settingsApi.updateSettings(bloomTestSettings))
        .rejects.toThrow(/rate.*limit/i);
    });

    it('should reset settings to defaults', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(defaultSettings),
      } as Response);

      const result = await settingsApi.resetSettings();

      expect(mockFetch).toHaveBeenCalledWith('/api/settings/reset', {
        method: 'POST',
      });

      expect(result.visualisation.glow).toEqual(defaultSettings.visualisation.glow);
    });
  });

  describe('Physics Settings Propagation', () => {
    it('should update physics settings via dedicated endpoint', async () => {
      const physicsUpdate = {
        springK: 0.2,
        repelK: 3.0,
        damping: 0.95,
        maxVelocity: 10.0,
        iterations: 100,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: 'Physics settings updated successfully' }),
      } as Response);

      // await settingsApi.updatePhysics(physicsUpdate); // removed - use updateSettings instead

      expect(mockFetch).toHaveBeenCalledWith('/api/physics/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(physicsUpdate),
      });
    });

    it('should validate physics parameters', async () => {
      const invalidPhysics = {
        damping: 2.0, // Invalid: > 1.0
        iterations: -10, // Invalid: negative
      };

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid physics parameters: damping must be between 0.0 and 1.0',
        }),
      } as Response);

      // await expect(settingsApi.updatePhysics(invalidPhysics)) // removed - use updateSettings instead
      //   .rejects.toThrow(/physics parameters.*damping/i);
    });
  });

  describe('Bidirectional Sync', () => {
    it('should maintain consistency between client and server', async () => {
      // First, update settings
      const update = bloomTestSettings;
      const updatedSettings = { ...defaultSettings, ...update };

      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(updatedSettings),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(updatedSettings),
        } as Response);

      // Update and then fetch to verify
      await settingsApi.updateSettings(update);
      const fetchedSettings = await settingsApi.fetchSettings();

      expect(fetchedSettings.visualisation.glow.intensity).toBe(2.5);
      expect(fetchedSettings.visualisation.glow.baseColor).toBe('#ff6b6b');
      expect(fetchedSettings.visualisation.graphs.logseq.physics.springK).toBe(0.15);
    });

    it('should handle concurrent updates gracefully', async () => {
      const updates = [
        { visualisation: { glow: { intensity: 1.0 } } },
        { visualisation: { glow: { intensity: 2.0 } } },
        { visualisation: { glow: { intensity: 3.0 } } },
      ];

      // Mock successful responses
      updates.forEach(() => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(defaultSettings),
        } as Response);
      });

      const promises = updates.map(update => settingsApi.updateSettings(update));
      const results = await Promise.allSettled(promises);

      // All should resolve (no rejections)
      expect(results.every(r => r.status === 'fulfilled')).toBe(true);
    });
  });

  describe('Nostr Authentication Integration', () => {
    beforeEach(() => {
      mockNostrAuth.isAuthenticated.mockReturnValue(false);
      mockNostrAuth.getCurrentUser.mockReturnValue(null);
    });

    it('should persist settings after authentication', async () => {
      // Mock successful auth
      mockNostrAuth.isAuthenticated.mockReturnValue(true);
      mockNostrAuth.getCurrentUser.mockReturnValue({
        pubkey: 'test_pubkey',
        npub: 'npub_test',
        isPowerUser: true,
      });

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ ...defaultSettings, ...bloomTestSettings }),
      } as Response);

      const result = await settingsApi.updateSettings(bloomTestSettings);

      expect(result).toBeDefined();
      // In a real scenario, this would include authentication headers
    });

    it('should handle unauthenticated requests', async () => {
      mockNostrAuth.isAuthenticated.mockReturnValue(false);

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: 'Authentication required',
        }),
      } as Response);

      await expect(settingsApi.updateSettings(bloomTestSettings))
        .rejects.toThrow(/authentication/i);
    });
  });

  describe('Error Handling and Recovery', () => {
    it('should handle network errors gracefully', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(settingsApi.fetchSettings())
        .rejects.toThrow(/network error/i);
    });

    it('should handle malformed server responses', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.reject(new Error('Invalid JSON')),
      } as Response);

      await expect(settingsApi.fetchSettings())
        .rejects.toThrow();
    });

    it('should handle server errors gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: () => Promise.resolve({}),
      } as Response);

      await expect(settingsApi.fetchSettings())
        .rejects.toThrow(/internal server error/i);
    });

    it('should validate import/export functionality', () => {
      const exportedSettings = settingsApi.exportSettings(defaultSettings);
      expect(exportedSettings).toBeTruthy();
      expect(() => JSON.parse(exportedSettings)).not.toThrow();

      const importedSettings = settingsApi.importSettings(exportedSettings);
      expect(importedSettings).toEqual(defaultSettings);
    });

    it('should handle invalid imported settings', () => {
      const invalidJson = '{ invalid json }';
      expect(() => settingsApi.importSettings(invalidJson))
        .toThrow(/invalid settings file format/i);

      const incompleteSettings = JSON.stringify({ visualisation: {} });
      expect(() => settingsApi.importSettings(incompleteSettings))
        .toThrow(/invalid settings format/i);
    });
  });

  describe('Performance and Load Testing', () => {
    it('should handle large settings payloads', async () => {
      // Create a large but valid settings object
      const largeSettings = {
        ...bloomTestSettings,
        visualisation: {
          ...bloomTestSettings.visualisation!,
          // Add many custom properties to test payload size handling
          customData: Array.from({ length: 100 }, (_, i) => ({
            [`property${i}`]: `value${i}`,
          })).reduce((acc, obj) => ({ ...acc, ...obj }), {}),
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ ...defaultSettings, ...largeSettings }),
      } as Response);

      const startTime = Date.now();
      const result = await settingsApi.updateSettings(largeSettings);
      const endTime = Date.now();

      expect(result).toBeDefined();
      expect(endTime - startTime).toBeLessThan(5000); // Should complete within 5 seconds
    });

    it('should handle rapid successive requests', async () => {
      const requests = Array.from({ length: 10 }, (_, i) => ({
        visualisation: {
          glow: {
            intensity: i * 0.1,
          },
        },
      }));

      // Mock responses for all requests
      requests.forEach(() => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve(defaultSettings),
        } as Response);
      });

      const startTime = Date.now();
      const promises = requests.map(req => settingsApi.updateSettings(req));
      const results = await Promise.all(promises);
      const endTime = Date.now();

      expect(results).toHaveLength(10);
      expect(endTime - startTime).toBeLessThan(10000); // Should complete within 10 seconds
    });
  });

  describe('Validation Edge Cases', () => {
    it('should handle all invalid bloom settings types', async () => {
      for (const testCase of invalidBloomSettings) {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status: 400,
          json: () => Promise.resolve({
            error: `Invalid settings: ${testCase.name}`,
          }),
        } as Response);

        await expect(settingsApi.updateSettings(testCase.settings as SettingsUpdate))
          .rejects.toThrow(testCase.expectedError);
      }
    });

    it('should validate color format variations', async () => {
      const colorTests = [
        { color: '#fff', valid: true },
        { color: '#ffffff', valid: true },
        { color: '#ABCDEF', valid: true },
        { color: 'rgb(255,255,255)', valid: false },
        { color: 'blue', valid: false },
        { color: '#gggggg', valid: false },
      ];

      for (const test of colorTests) {
        const settings = {
          visualisation: {
            glow: {
              baseColor: test.color,
            },
          },
        };

        if (test.valid) {
          mockFetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ ...defaultSettings, ...settings }),
          } as Response);

          const result = await settingsApi.updateSettings(settings);
          expect(result).toBeDefined();
        } else {
          mockFetch.mockResolvedValueOnce({
            ok: false,
            status: 400,
            json: () => Promise.resolve({
              error: 'Invalid color format',
            }),
          } as Response);

          await expect(settingsApi.updateSettings(settings))
            .rejects.toThrow(/invalid.*color/i);
        }
      }
    });
  });
});
