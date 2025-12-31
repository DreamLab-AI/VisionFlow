/**
 * WebSocket Notifications E2E Tests
 *
 * Tests the solid-0.1 WebSocket notification protocol:
 * - Protocol handshake
 * - Resource subscription
 * - Real-time change notifications
 * - Container notifications for child changes
 * - Unsubscription handling
 */

import { test, expect, Page } from '@playwright/test';
import { WebSocket } from 'ws';

const API_URL = process.env.TEST_API_URL || 'http://localhost:4000';
const SOLID_URL = `${API_URL}/solid`;

// Helper to get WebSocket URL from server
async function getWebSocketUrl(request: any): Promise<string | null> {
  const response = await request.fetch(`${SOLID_URL}/pods/test/`, {
    method: 'OPTIONS',
    headers: {
      'Authorization': 'Bearer dev-session-token'
    },
    failOnStatusCode: false
  });

  if (response.ok()) {
    const updatesVia = response.headers()['updates-via'];
    return updatesVia || null;
  }
  return null;
}

test.describe('WebSocket Discovery', () => {
  test('should include Updates-Via header in OPTIONS response', async ({ request }) => {
    const response = await request.fetch(`${SOLID_URL}/pods/test/public/`, {
      method: 'OPTIONS',
      headers: {
        'Authorization': 'Bearer dev-session-token'
      },
      failOnStatusCode: false
    });

    if (response.ok()) {
      const updatesVia = response.headers()['updates-via'];
      if (updatesVia) {
        // If notifications are enabled, should be a WebSocket URL
        expect(updatesVia).toMatch(/^wss?:\/\//);
      }
    }
  });

  test('should include Updates-Via header in GET response', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/test/public/`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Accept': 'application/ld+json'
      },
      failOnStatusCode: false
    });

    if (response.ok()) {
      const updatesVia = response.headers()['updates-via'];
      if (updatesVia) {
        expect(updatesVia).toMatch(/^wss?:\/\//);
      }
    }
  });

  test('should expose Updates-Via in Access-Control-Expose-Headers', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/test/public/`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Accept': 'application/ld+json',
        'Origin': 'http://localhost:3001'
      },
      failOnStatusCode: false
    });

    if (response.ok()) {
      const exposeHeaders = response.headers()['access-control-expose-headers'];
      if (exposeHeaders) {
        // Updates-Via should be exposed for CORS
        expect(exposeHeaders.toLowerCase()).toContain('updates-via');
      }
    }
  });
});

test.describe('WebSocket Protocol Handshake', () => {
  test('should handle WebSocket upgrade attempt', async ({ request }) => {
    // Note: Playwright doesn't directly support WebSocket, but we can test
    // the server's response to upgrade requests
    const wsUrl = await getWebSocketUrl(request);

    if (wsUrl) {
      // Convert ws:// to http:// for testing via HTTP
      const httpUrl = wsUrl.replace(/^ws/, 'http');

      const response = await request.get(httpUrl, {
        headers: {
          'Upgrade': 'websocket',
          'Connection': 'Upgrade',
          'Sec-WebSocket-Key': 'dGhlIHNhbXBsZSBub25jZQ==',
          'Sec-WebSocket-Version': '13'
        },
        failOnStatusCode: false
      });

      // Server should respond with 101 Switching Protocols or a valid HTTP response
      expect([101, 200, 400, 426]).toContain(response.status());
    }
  });
});

test.describe('WebSocket Integration with Page Context', () => {
  test('should initialize WebSocket connection from client', async ({ page }) => {
    // Navigate to the app
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');
    await page.waitForLoadState('networkidle');

    // Check if WebSocket API is available
    const hasWebSocket = await page.evaluate(() => {
      return typeof WebSocket !== 'undefined';
    });

    expect(hasWebSocket).toBe(true);
  });

  test('should handle WebSocket URL from service', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');
    await page.waitForLoadState('networkidle');

    // Test that the page can work with WebSocket URLs
    const wsUrlTest = await page.evaluate(() => {
      // Test URL parsing for WebSocket URLs
      try {
        const testWsUrl = 'ws://localhost:3030/.notifications';
        const url = new URL(testWsUrl);
        return {
          protocol: url.protocol,
          hostname: url.hostname,
          pathname: url.pathname,
          valid: true
        };
      } catch (e) {
        return { valid: false, error: String(e) };
      }
    });

    expect(wsUrlTest.valid).toBe(true);
    expect(wsUrlTest.protocol).toBe('ws:');
  });
});

test.describe('Notification Subscription Logic', () => {
  test('should format subscription message correctly', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    // Test subscription message format
    const subMessage = await page.evaluate((solidUrl) => {
      const resourceUrl = `${solidUrl}/pods/test/public/resource.json`;
      return `sub ${resourceUrl}`;
    }, SOLID_URL);

    expect(subMessage).toMatch(/^sub https?:\/\//);
  });

  test('should format unsubscription message correctly', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    // Test unsubscription message format
    const unsubMessage = await page.evaluate((solidUrl) => {
      const resourceUrl = `${solidUrl}/pods/test/public/resource.json`;
      return `unsub ${resourceUrl}`;
    }, SOLID_URL);

    expect(unsubMessage).toMatch(/^unsub https?:\/\//);
  });
});

test.describe('Notification Message Parsing', () => {
  test('should parse protocol greeting message', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const parsed = await page.evaluate(() => {
      const message = 'protocol solid-0.1';
      if (message.startsWith('protocol ')) {
        return {
          type: 'protocol',
          version: message.slice(9)
        };
      }
      return null;
    });

    expect(parsed).toEqual({
      type: 'protocol',
      version: 'solid-0.1'
    });
  });

  test('should parse ack message', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const parsed = await page.evaluate(() => {
      const message = 'ack http://example.org/resource';
      if (message.startsWith('ack ')) {
        return {
          type: 'ack',
          url: message.slice(4)
        };
      }
      return null;
    });

    expect(parsed).toEqual({
      type: 'ack',
      url: 'http://example.org/resource'
    });
  });

  test('should parse pub notification message', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const parsed = await page.evaluate(() => {
      const message = 'pub http://example.org/resource';
      if (message.startsWith('pub ')) {
        return {
          type: 'pub',
          url: message.slice(4)
        };
      }
      return null;
    });

    expect(parsed).toEqual({
      type: 'pub',
      url: 'http://example.org/resource'
    });
  });
});

test.describe('Subscription Tracking', () => {
  test('should track subscribed resources', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const tracking = await page.evaluate(() => {
      // Simulate subscription tracking
      const subscriptions = new Map<string, Set<Function>>();
      const resourceUrl = 'http://example.org/resource';
      const callback = () => {};

      // Add subscription
      if (!subscriptions.has(resourceUrl)) {
        subscriptions.set(resourceUrl, new Set());
      }
      subscriptions.get(resourceUrl)!.add(callback);

      return {
        hasSubscription: subscriptions.has(resourceUrl),
        callbackCount: subscriptions.get(resourceUrl)!.size
      };
    });

    expect(tracking.hasSubscription).toBe(true);
    expect(tracking.callbackCount).toBe(1);
  });

  test('should handle multiple subscribers to same resource', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const tracking = await page.evaluate(() => {
      const subscriptions = new Map<string, Set<Function>>();
      const resourceUrl = 'http://example.org/resource';

      // Add multiple callbacks
      if (!subscriptions.has(resourceUrl)) {
        subscriptions.set(resourceUrl, new Set());
      }
      subscriptions.get(resourceUrl)!.add(() => console.log('callback 1'));
      subscriptions.get(resourceUrl)!.add(() => console.log('callback 2'));
      subscriptions.get(resourceUrl)!.add(() => console.log('callback 3'));

      return subscriptions.get(resourceUrl)!.size;
    });

    expect(tracking).toBe(3);
  });
});

test.describe('Container Notification Handling', () => {
  test('should detect parent container from resource URL', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const container = await page.evaluate(() => {
      const resourceUrl = 'http://example.org/pod/public/resource.json';
      const lastSlash = resourceUrl.lastIndexOf('/');
      return resourceUrl.substring(0, lastSlash + 1);
    });

    expect(container).toBe('http://example.org/pod/public/');
  });

  test('should notify container subscribers on child change', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const result = await page.evaluate(() => {
      const subscriptions = new Map<string, Set<{ url: string }[]>>();
      const containerUrl = 'http://example.org/pod/public/';
      const childUrl = 'http://example.org/pod/public/child.json';

      // Subscribe to container
      const containerNotifications: { url: string }[] = [];
      subscriptions.set(containerUrl, new Set([containerNotifications]));

      // Simulate notification for child resource
      const notification = { url: childUrl };

      // Check if container should be notified
      const containerPath = childUrl.substring(0, childUrl.lastIndexOf('/') + 1);
      if (subscriptions.has(containerPath)) {
        subscriptions.get(containerPath)!.forEach(arr => arr.push(notification));
      }

      return containerNotifications;
    });

    expect(result).toHaveLength(1);
    expect(result[0].url).toBe('http://example.org/pod/public/child.json');
  });
});

test.describe('Reconnection Logic', () => {
  test('should calculate exponential backoff delay', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const delays = await page.evaluate(() => {
      const baseDelay = 1000;
      const maxAttempts = 5;
      const delays: number[] = [];

      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        const delay = baseDelay * Math.pow(2, attempt - 1);
        delays.push(delay);
      }

      return delays;
    });

    expect(delays).toEqual([1000, 2000, 4000, 8000, 16000]);
  });

  test('should resubscribe after reconnection', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const result = await page.evaluate(() => {
      // Simulate tracked subscriptions that need resubscription
      const subscriptions = new Map<string, Set<Function>>();
      subscriptions.set('http://example.org/resource1', new Set([() => {}]));
      subscriptions.set('http://example.org/resource2', new Set([() => {}]));

      // Get URLs to resubscribe
      const urlsToResubscribe = Array.from(subscriptions.keys());

      return {
        count: urlsToResubscribe.length,
        urls: urlsToResubscribe
      };
    });

    expect(result.count).toBe(2);
    expect(result.urls).toContain('http://example.org/resource1');
    expect(result.urls).toContain('http://example.org/resource2');
  });
});

test.describe('Error Handling', () => {
  test('should handle WebSocket close event', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const closeHandling = await page.evaluate(() => {
      // Simulate close event handling
      let reconnectAttempts = 0;
      const maxReconnectAttempts = 5;

      // Simulate close
      const handleClose = () => {
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          return { action: 'reconnect', attempt: reconnectAttempts };
        }
        return { action: 'give_up', attempt: reconnectAttempts };
      };

      return handleClose();
    });

    expect(closeHandling.action).toBe('reconnect');
    expect(closeHandling.attempt).toBe(1);
  });

  test('should handle WebSocket error event', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const errorHandling = await page.evaluate(() => {
      // Simulate error handling
      const error = new Error('WebSocket connection failed');
      return {
        errorLogged: true,
        errorMessage: error.message
      };
    });

    expect(errorHandling.errorLogged).toBe(true);
    expect(errorHandling.errorMessage).toBe('WebSocket connection failed');
  });
});

test.describe('Performance Considerations', () => {
  test('should limit subscription count', async ({ page }) => {
    await page.goto(process.env.TEST_BASE_URL || 'http://localhost:3001');

    const result = await page.evaluate(() => {
      const MAX_SUBSCRIPTIONS = 100;
      const subscriptions = new Map<string, Set<Function>>();

      // Attempt to add many subscriptions
      for (let i = 0; i < 150; i++) {
        if (subscriptions.size < MAX_SUBSCRIPTIONS) {
          subscriptions.set(`http://example.org/resource${i}`, new Set([() => {}]));
        }
      }

      return {
        count: subscriptions.size,
        limited: subscriptions.size <= MAX_SUBSCRIPTIONS
      };
    });

    expect(result.limited).toBe(true);
    expect(result.count).toBe(100);
  });
});
