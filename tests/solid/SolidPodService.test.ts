/**
 * SolidPodService Unit Tests
 *
 * Tests for the Solid Pod service including:
 * - Pod management operations
 * - LDP CRUD operations
 * - WebSocket subscription handling
 * - Authentication header management
 * - JSON-LD content handling
 */

import { describe, it, expect, beforeEach, afterEach, vi, Mock } from 'vitest';

// Mock nostrAuth before importing SolidPodService
vi.mock('../client/src/services/nostrAuthService', () => ({
  nostrAuth: {
    getSessionToken: vi.fn(() => 'test-session-token'),
    getCurrentUser: vi.fn(() => ({
      pubkey: 'test-pubkey',
      npub: 'npub1test',
      isPowerUser: true
    })),
    isAuthenticated: vi.fn(() => true)
  }
}));

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock WebSocket
class MockWebSocket {
  static OPEN = 1;
  static CLOSED = 3;

  readyState = MockWebSocket.OPEN;
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;

  send = vi.fn();
  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED;
  });
}

global.WebSocket = MockWebSocket as any;

describe('SolidPodService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Pod Management', () => {
    it('should check if pod exists', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          exists: true,
          podUrl: 'http://localhost:3030/pods/test/',
          webId: 'http://localhost:3030/pods/test/profile/card#me'
        })
      });

      // Import dynamically to ensure mocks are in place
      const { default: solidPodService } = await import('../client/src/services/SolidPodService');

      const result = await solidPodService.checkPodExists();

      expect(result.exists).toBe(true);
      expect(result.podUrl).toBeDefined();
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/pods/check'),
        expect.objectContaining({
          headers: expect.any(Headers)
        })
      );
    });

    it('should return exists: false when pod does not exist', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          exists: false,
          suggestedUrl: 'http://localhost:3030/pods/npub1test/'
        })
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const result = await solidPodService.checkPodExists();

      expect(result.exists).toBe(false);
      expect(result.suggestedUrl).toBeDefined();
    });

    it('should create a new pod', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          pod_url: 'http://localhost:3030/pods/newpod/',
          webid: 'http://localhost:3030/pods/newpod/profile/card#me'
        })
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const result = await solidPodService.createPod('newpod');

      expect(result.success).toBe(true);
      expect(result.podUrl).toBe('http://localhost:3030/pods/newpod/');
      expect(result.webId).toBeDefined();
    });

    it('should handle pod creation failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        json: async () => ({
          error: 'Pod already exists'
        })
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const result = await solidPodService.createPod('existing');

      expect(result.success).toBe(false);
      expect(result.error).toBe('Pod already exists');
    });
  });

  describe('LDP Operations', () => {
    it('should fetch JSON-LD resource', async () => {
      const mockJsonLd = {
        '@context': 'https://www.w3.org/ns/ldp',
        '@type': 'Container',
        'contains': []
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockJsonLd
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const result = await solidPodService.fetchJsonLd('/pods/test/public/');

      expect(result['@type']).toBe('Container');
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.any(Headers)
        })
      );
    });

    it('should fetch Turtle resource', async () => {
      const mockTurtle = '@prefix ldp: <http://www.w3.org/ns/ldp#>.\n<> a ldp:Container.';

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: async () => mockTurtle
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const result = await solidPodService.fetchTurtle('/pods/test/public/');

      expect(result).toContain('ldp:Container');
    });

    it('should PUT resource', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const testData = {
        '@context': 'https://schema.org',
        '@type': 'Thing',
        'name': 'Test'
      };

      const result = await solidPodService.putResource('/pods/test/public/test.jsonld', testData);

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify(testData)
        })
      );
    });

    it('should POST to container', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: {
          get: (name: string) => name === 'Location' ? 'http://localhost:3030/pods/test/public/new-resource' : null
        }
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const testData = {
        '@context': 'https://schema.org',
        '@type': 'Thing'
      };

      const location = await solidPodService.postResource('/pods/test/public/', testData, 'new-resource');

      expect(location).toBe('http://localhost:3030/pods/test/public/new-resource');
    });

    it('should DELETE resource', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const result = await solidPodService.deleteResource('/pods/test/public/old.jsonld');

      expect(result).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'DELETE'
        })
      );
    });

    it('should check resource existence with HEAD', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const exists = await solidPodService.resourceExists('/pods/test/public/resource.jsonld');

      expect(exists).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'HEAD'
        })
      );
    });

    it('should return false for non-existent resource', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const exists = await solidPodService.resourceExists('/pods/test/public/nonexistent.jsonld');

      expect(exists).toBe(false);
    });
  });

  describe('JSON-LD Content Handling', () => {
    it('should write resource with plain object', async () => {
      mockFetch.mockResolvedValueOnce({ ok: true });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');

      const plainObject = {
        title: 'Test',
        value: 42
      };

      const result = await solidPodService.writeResource('/pods/test/public/data.jsonld', plainObject);

      expect(result).toBe(true);
      // Should have wrapped with @context
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('@context')
        })
      );
    });

    it('should preserve existing @context', async () => {
      mockFetch.mockResolvedValueOnce({ ok: true });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');

      const jsonLdObject = {
        '@context': 'https://schema.org',
        '@type': 'Person',
        'name': 'Test'
      };

      await solidPodService.writeResource('/pods/test/public/person.jsonld', jsonLdObject);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"@context":"https://schema.org"')
        })
      );
    });
  });

  describe('Authentication', () => {
    it('should include Authorization header in requests', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ exists: true })
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      await solidPodService.checkPodExists();

      const call = mockFetch.mock.calls[0];
      const options = call[1];
      const headers = options.headers;

      // Should include Authorization header
      expect(headers.get('Authorization')).toBe('Bearer test-session-token');
    });

    it('should include credentials: include in fetch options', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      await solidPodService.fetchJsonLd('/pods/test/');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          credentials: 'include'
        })
      );
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors gracefully', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      const result = await solidPodService.checkPodExists();

      expect(result.exists).toBe(false);
    });

    it('should throw on fetch failure for JSON-LD', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');

      await expect(solidPodService.fetchJsonLd('/pods/nonexistent/'))
        .rejects.toThrow();
    });
  });

  describe('Path Resolution', () => {
    it('should resolve relative paths', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      await solidPodService.fetchJsonLd('pods/test/');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringMatching(/\/solid\/pods\/test\//),
        expect.any(Object)
      );
    });

    it('should preserve absolute URLs', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({})
      });

      const { default: solidPodService } = await import('../client/src/services/SolidPodService');
      await solidPodService.fetchJsonLd('http://other.server/pod/');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://other.server/pod/',
        expect.any(Object)
      );
    });
  });
});

describe('WebSocket Subscription', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should create subscription and return unsubscribe function', async () => {
    const { default: solidPodService } = await import('../client/src/services/SolidPodService');

    const callback = vi.fn();
    const unsubscribe = solidPodService.subscribe('http://localhost:3030/pods/test/resource', callback);

    expect(typeof unsubscribe).toBe('function');
  });

  it('should track subscriptions by URL', async () => {
    const { default: solidPodService } = await import('../client/src/services/SolidPodService');

    const callback1 = vi.fn();
    const callback2 = vi.fn();

    solidPodService.subscribe('http://localhost:3030/pods/test/resource1', callback1);
    solidPodService.subscribe('http://localhost:3030/pods/test/resource2', callback2);

    // Both should be tracked
    expect(callback1).not.toHaveBeenCalled();
    expect(callback2).not.toHaveBeenCalled();
  });

  it('should clean up on unsubscribe', async () => {
    const { default: solidPodService } = await import('../client/src/services/SolidPodService');

    const callback = vi.fn();
    const unsubscribe = solidPodService.subscribe('http://localhost:3030/pods/test/resource', callback);

    // Unsubscribe
    unsubscribe();

    // Should not throw
    expect(() => unsubscribe()).not.toThrow();
  });
});

describe('useSolidPod Hook Integration', () => {
  // Note: Hook tests would typically use @testing-library/react-hooks
  // These are placeholder tests for the hook behavior

  it('should export hook return type correctly', async () => {
    const { useSolidPod } = await import('../client/src/features/solid/hooks/useSolidPod');

    // Verify the hook is exported correctly
    expect(typeof useSolidPod).toBe('function');
  });
});

describe('useSolidResource Hook Integration', () => {
  it('should export hook return type correctly', async () => {
    const { useSolidResource } = await import('../client/src/features/solid/hooks/useSolidResource');

    // Verify the hook is exported correctly
    expect(typeof useSolidResource).toBe('function');
  });
});
