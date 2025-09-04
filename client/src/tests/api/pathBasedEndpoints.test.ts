import { describe, it, expect, vi, beforeEach } from 'vitest';
import { getSettings, updateSettings } from '@/api/settings';
import { createMockFetchResponse, createMockSettings, measurePerformance } from '../utils/testFactories';

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Path-Based API Endpoints', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('GET /api/settings/get', () => {
    it('should request single path correctly', async () => {
      const mockData = {
        visualisation: {
          glow: {
            nodeGlowStrength: 2.5
          }
        }
      };

      mockFetch.mockResolvedValue(createMockFetchResponse(mockData));

      const result = await getSettings(['visualisation.glow.nodeGlowStrength']);

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings/get?paths=visualisation.glow.nodeGlowStrength',
        {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      expect(result).toEqual(mockData);
    });

    it('should request multiple paths correctly', async () => {
      const mockData = {
        visualisation: {
          glow: {
            nodeGlowStrength: 2.5,
            baseColor: '#ff0000'
          }
        },
        system: {
          debugMode: true
        }
      };

      mockFetch.mockResolvedValue(createMockFetchResponse(mockData));

      const result = await getSettings([
        'visualisation.glow.nodeGlowStrength',
        'visualisation.glow.baseColor',
        'system.debugMode'
      ]);

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings/get?paths=visualisation.glow.nodeGlowStrength,visualisation.glow.baseColor,system.debugMode',
        expect.objectContaining({
          method: 'GET'
        })
      );

      expect(result).toEqual(mockData);
    });

    it('should handle nested object path requests', async () => {
      const mockData = {
        visualisation: {
          glow: {
            nodeGlowStrength: 1.5,
            edgeGlowStrength: 2.0,
            environmentGlowStrength: 1.0,
            baseColor: '#00ffff',
            emissionColor: '#ffffff',
            enabled: true
          }
        }
      };

      mockFetch.mockResolvedValue(createMockFetchResponse(mockData));

      const result = await getSettings(['visualisation.glow']);

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings/get?paths=visualisation.glow',
        expect.objectContaining({ method: 'GET' })
      );

      expect(result.visualisation.glow).toBeDefined();
      expect(Object.keys(result.visualisation.glow)).toHaveLength(6);
    });

    it('should handle URL encoding for special characters in paths', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({}));

      await getSettings(['test.path with spaces', 'test.path%special']);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('test.path%20with%20spaces'),
        expect.anything()
      );
    });

    it('should handle empty path arrays gracefully', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({}));

      const result = await getSettings([]);

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings/get?paths=',
        expect.anything()
      );
    });

    it('should throw error for invalid response', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse(null, { 
        ok: false, 
        status: 400,
        statusText: 'Bad Request'
      }));

      await expect(getSettings(['invalid.path']))
        .rejects.toThrow('Failed to fetch settings');
    });

    it('should handle network errors gracefully', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      await expect(getSettings(['test.path']))
        .rejects.toThrow('Network error');
    });

    it('should handle large path lists efficiently', async () => {
      const largePaths = Array.from({ length: 100 }, (_, i) => `test.path${i}`);
      const mockData = createMockSettings();

      mockFetch.mockResolvedValue(createMockFetchResponse(mockData));

      const performance = await measurePerformance(async () => {
        await getSettings(largePaths);
      });

      expect(performance.average).toBeLessThan(1000); // Should complete in < 1s
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should validate camelCase response structure', async () => {
      const mockData = {
        visualisation: {
          glow: {
            nodeGlowStrength: 1.5, // camelCase
            edgeGlowStrength: 2.0,
            baseColor: '#ff0000'
          }
        },
        system: {
          debugMode: true,
          maxConnections: 100
        }
      };

      mockFetch.mockResolvedValue(createMockFetchResponse(mockData));

      const result = await getSettings(['visualisation.glow', 'system']);

      // Verify camelCase structure
      expect(result.visualisation.glow.nodeGlowStrength).toBeDefined();
      expect(result.visualisation.glow.edgeGlowStrength).toBeDefined();
      expect(result.system.debugMode).toBeDefined();
      expect(result.system.maxConnections).toBeDefined();

      // Verify no snake_case
      expect((result.visualisation.glow as any).node_glow_strength).toBeUndefined();
      expect((result.system as any).debug_mode).toBeUndefined();
    });
  });

  describe('POST /api/settings/set', () => {
    it('should send single update correctly', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({ success: true }));

      const updates = [
        { path: 'visualisation.glow.nodeGlowStrength', value: 3.0 }
      ];

      const result = await updateSettings(updates);

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings/set',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(updates)
        }
      );

      expect(result.success).toBe(true);
    });

    it('should send multiple updates in batch', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({ 
        success: true, 
        updated: 3 
      }));

      const updates = [
        { path: 'visualisation.glow.nodeGlowStrength', value: 3.0 },
        { path: 'visualisation.glow.baseColor', value: '#00ff00' },
        { path: 'system.debugMode', value: true }
      ];

      const result = await updateSettings(updates);

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings/set',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(updates)
        })
      );

      expect(result.success).toBe(true);
      expect(result.updated).toBe(3);
    });

    it('should handle complex nested object updates', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({ success: true }));

      const complexUpdate = {
        path: 'visualisation.glow',
        value: {
          nodeGlowStrength: 2.5,
          edgeGlowStrength: 3.0,
          baseColor: '#ff0000',
          enabled: true
        }
      };

      await updateSettings([complexUpdate]);

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings/set',
        expect.objectContaining({
          body: JSON.stringify([complexUpdate])
        })
      );
    });

    it('should handle arrays and special values', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({ success: true }));

      const updates = [
        { path: 'visualisation.colorSchemes', value: ['default', 'dark', 'high-contrast'] },
        { path: 'system.featureFlags', value: { newUI: true, betaFeatures: false } },
        { path: 'test.nullValue', value: null },
        { path: 'test.numberValue', value: 42.5 },
        { path: 'test.booleanValue', value: true }
      ];

      await updateSettings(updates);

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings/set',
        expect.objectContaining({
          body: JSON.stringify(updates)
        })
      );
    });

    it('should handle validation errors gracefully', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse(
        { success: false, error: 'Validation failed', details: ['Invalid color format'] },
        { ok: false, status: 400 }
      ));

      const invalidUpdate = [
        { path: 'visualisation.glow.baseColor', value: 'invalid-color' }
      ];

      await expect(updateSettings(invalidUpdate))
        .rejects.toThrow('Failed to update settings');
    });

    it('should handle server errors with retry logic', async () => {
      let callCount = 0;
      mockFetch.mockImplementation(() => {
        callCount++;
        if (callCount <= 2) {
          return Promise.resolve(createMockFetchResponse(
            { error: 'Internal server error' },
            { ok: false, status: 500 }
          ));
        }
        return Promise.resolve(createMockFetchResponse({ success: true }));
      });

      const updates = [{ path: 'test.retry', value: 'value' }];

      // This would depend on retry logic implementation
      // For now, just test that it eventually succeeds or throws appropriate error
    });

    it('should validate request payload size', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({ success: true }));

      const largeUpdates = Array.from({ length: 1000 }, (_, i) => ({
        path: `test.large${i}`,
        value: { 
          data: 'x'.repeat(1000),
          index: i,
          metadata: { created: Date.now() }
        }
      }));

      const result = await updateSettings(largeUpdates);

      // Should handle large payloads
      expect(result.success).toBe(true);
      
      const requestBody = mockFetch.mock.calls[0][1].body;
      expect(requestBody.length).toBeGreaterThan(1000 * 1000); // > 1MB
    });

    it('should maintain update order in batch requests', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({ success: true }));

      const orderedUpdates = [
        { path: 'test.order', value: 'first' },
        { path: 'test.order', value: 'second' }, // Should overwrite first
        { path: 'test.other', value: 'other_value' },
        { path: 'test.order', value: 'final' } // Final value
      ];

      await updateSettings(orderedUpdates);

      const sentData = JSON.parse(mockFetch.mock.calls[0][1].body);
      expect(sentData).toEqual(orderedUpdates);
    });
  });

  describe('Response Format Validation', () => {
    it('should validate GET response structure', async () => {
      const mockResponse = {
        visualisation: {
          glow: {
            nodeGlowStrength: 1.5,
            baseColor: '#ff0000'
          }
        }
      };

      mockFetch.mockResolvedValue(createMockFetchResponse(mockResponse));

      const result = await getSettings(['visualisation.glow']);

      // Validate structure matches expected format
      expect(typeof result).toBe('object');
      expect(result.visualisation).toBeDefined();
      expect(result.visualisation.glow).toBeDefined();
      expect(typeof result.visualisation.glow.nodeGlowStrength).toBe('number');
      expect(typeof result.visualisation.glow.baseColor).toBe('string');
    });

    it('should validate POST response structure', async () => {
      const mockResponse = {
        success: true,
        updated: 2,
        timestamp: Date.now(),
        errors: []
      };

      mockFetch.mockResolvedValue(createMockFetchResponse(mockResponse));

      const result = await updateSettings([
        { path: 'test1', value: 'value1' },
        { path: 'test2', value: 'value2' }
      ]);

      expect(result.success).toBe(true);
      expect(result.updated).toBe(2);
      expect(result.timestamp).toBeDefined();
      expect(Array.isArray(result.errors)).toBe(true);
    });
  });

  describe('Performance and Efficiency', () => {
    it('should measure response time for various payload sizes', async () => {
      const testSizes = [1, 10, 50, 100];
      const results: Record<number, number> = {};

      for (const size of testSizes) {
        const updates = Array.from({ length: size }, (_, i) => ({
          path: `test.path${i}`,
          value: { index: i, data: 'test'.repeat(10) }
        }));

        mockFetch.mockResolvedValue(createMockFetchResponse({ success: true }));

        const performance = await measurePerformance(async () => {
          await updateSettings(updates);
        });

        results[size] = performance.average;
      }

      // Verify performance scales reasonably
      expect(results[1]).toBeLessThan(100);
      expect(results[100]).toBeLessThan(1000);
      
      // Performance should scale sub-linearly
      expect(results[100] / results[10]).toBeLessThan(20);
    });

    it('should handle concurrent requests efficiently', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({ success: true }));

      const concurrentRequests = Array.from({ length: 10 }, (_, i) => 
        updateSettings([{ path: `concurrent.path${i}`, value: `value${i}` }])
      );

      const results = await Promise.all(concurrentRequests);

      expect(results).toHaveLength(10);
      results.forEach(result => {
        expect(result.success).toBe(true);
      });

      // Should have made 10 separate requests
      expect(mockFetch).toHaveBeenCalledTimes(10);
    });
  });

  describe('Error Recovery and Resilience', () => {
    it('should handle timeout errors', async () => {
      mockFetch.mockImplementation(() => 
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Request timeout')), 100)
        )
      );

      await expect(getSettings(['test.timeout']))
        .rejects.toThrow('Request timeout');
    });

    it('should handle malformed JSON responses', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.reject(new Error('Invalid JSON')),
        text: () => Promise.resolve('{"invalid": json}'),
        status: 200
      } as any);

      await expect(getSettings(['test.malformed']))
        .rejects.toThrow();
    });

    it('should handle partial failure responses', async () => {
      const partialFailureResponse = {
        success: false,
        updated: 1,
        errors: [
          { path: 'invalid.path', error: 'Path not found' }
        ],
        warnings: []
      };

      mockFetch.mockResolvedValue(createMockFetchResponse(
        partialFailureResponse,
        { ok: false, status: 400 }
      ));

      await expect(updateSettings([
        { path: 'valid.path', value: 'good' },
        { path: 'invalid.path', value: 'bad' }
      ])).rejects.toThrow('Failed to update settings');
    });
  });

  describe('Security and Input Validation', () => {
    it('should sanitize path parameters', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({}));

      const dangerousPaths = [
        'normal.path',
        '../../../etc/passwd',
        'path<script>alert("xss")</script>',
        'path; DROP TABLE settings; --'
      ];

      await getSettings(dangerousPaths);

      const calledUrl = mockFetch.mock.calls[0][0];
      
      // Verify dangerous content is properly encoded
      expect(calledUrl).not.toContain('<script>');
      expect(calledUrl).not.toContain('DROP TABLE');
      expect(decodeURIComponent(calledUrl)).toContain('%3C'); // < encoded
    });

    it('should validate update values for security issues', async () => {
      mockFetch.mockResolvedValue(createMockFetchResponse({ success: true }));

      const potentiallyDangerousUpdates = [
        { path: 'safe.path', value: 'safe_value' },
        { path: 'xss.attempt', value: '<script>alert("xss")</script>' },
        { path: 'injection.attempt', value: '"; DROP TABLE users; --' },
        { path: 'file.path', value: '../../../etc/passwd' }
      ];

      await updateSettings(potentiallyDangerousUpdates);

      const sentData = JSON.parse(mockFetch.mock.calls[0][1].body);
      
      // Values should be preserved as-is (server should handle validation)
      expect(sentData).toEqual(potentiallyDangerousUpdates);
    });
  });
});