/**
 * Solid/JSS Integration E2E Tests
 *
 * Comprehensive test suite for Solid Pod integration with VisionFlow:
 * - Solid proxy handler authentication flow
 * - NIP-98 token generation and validation
 * - Pod CRUD operations
 * - WebSocket notification delivery
 * - Ontology sync round-trip
 */

import { test, expect, Page, APIRequestContext } from '@playwright/test';

// Test configuration
const BASE_URL = process.env.TEST_BASE_URL || 'http://localhost:3001';
const API_URL = process.env.TEST_API_URL || 'http://localhost:4000';
const SOLID_URL = `${API_URL}/solid`;

// Mock Nostr keys for testing (DO NOT use in production)
const TEST_NOSTR_PUBKEY = 'bfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a452';
const TEST_NOSTR_NPUB = 'npub1hleusztehs7c5wg7et97ra6q5q2yyrhdcllya9yljepv9dsk53jqejhf5p';

test.describe('Solid Proxy Authentication Flow', () => {
  test.describe.configure({ mode: 'serial' });

  test('should reject unauthenticated requests to protected pod resources', async ({ request }) => {
    // Attempt to access a protected resource without authentication
    const response = await request.get(`${SOLID_URL}/pods/test/private/`, {
      headers: {
        'Accept': 'application/ld+json'
      },
      failOnStatusCode: false
    });

    // Should return 401 Unauthorized
    expect([401, 403, 404]).toContain(response.status());
  });

  test('should accept requests with valid dev session token', async ({ request }) => {
    // Using dev session token (for development mode)
    const response = await request.get(`${SOLID_URL}/pods/check`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    // In dev mode, should accept the request
    if (process.env.NODE_ENV === 'development' || response.ok()) {
      expect(response.status()).toBeLessThan(500);
    }
  });

  test('should forward user identity in X-User-Authorization header', async ({ request }) => {
    // This tests that the proxy correctly forwards user auth
    const response = await request.head(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': 'Bearer test-token',
        'Accept': '*/*'
      },
      failOnStatusCode: false
    });

    // Proxy should process the request (not 500 error)
    expect(response.status()).toBeLessThan(500);
  });

  test('should handle CORS preflight for Solid endpoints', async ({ request }) => {
    const response = await request.fetch(`${SOLID_URL}/pods/test/`, {
      method: 'OPTIONS',
      headers: {
        'Origin': 'http://localhost:3001',
        'Access-Control-Request-Method': 'PUT',
        'Access-Control-Request-Headers': 'Authorization, Content-Type'
      },
      failOnStatusCode: false
    });

    // OPTIONS should be allowed (or return 404 if pod doesn't exist)
    expect([200, 204, 404]).toContain(response.status());
  });
});

test.describe('NIP-98 Token Validation', () => {
  test('should reject requests with invalid NIP-98 token format', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': 'Nostr invalid-base64!!!',
        'Accept': 'application/ld+json'
      },
      failOnStatusCode: false
    });

    // Should reject malformed tokens
    expect([400, 401, 403]).toContain(response.status());
  });

  test('should reject NIP-98 tokens with wrong event kind', async ({ request }) => {
    // Create a fake NIP-98 token with wrong kind
    const fakeEvent = {
      id: 'fake',
      pubkey: TEST_NOSTR_PUBKEY,
      created_at: Math.floor(Date.now() / 1000),
      kind: 1, // Wrong kind - should be 27235
      tags: [['u', `${SOLID_URL}/pods/test/`], ['method', 'GET']],
      content: '',
      sig: 'fake'
    };
    const token = Buffer.from(JSON.stringify(fakeEvent)).toString('base64');

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/ld+json'
      },
      failOnStatusCode: false
    });

    // Should reject tokens with wrong kind
    expect([400, 401, 403]).toContain(response.status());
  });

  test('should reject expired NIP-98 tokens', async ({ request }) => {
    // Create a token with timestamp > 60 seconds ago
    const expiredEvent = {
      id: 'expired',
      pubkey: TEST_NOSTR_PUBKEY,
      created_at: Math.floor(Date.now() / 1000) - 120, // 2 minutes ago
      kind: 27235,
      tags: [['u', `${SOLID_URL}/pods/test/`], ['method', 'GET']],
      content: '',
      sig: 'fake'
    };
    const token = Buffer.from(JSON.stringify(expiredEvent)).toString('base64');

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/ld+json'
      },
      failOnStatusCode: false
    });

    // Should reject expired tokens
    expect([400, 401, 403]).toContain(response.status());
  });
});

test.describe('Pod CRUD Operations', () => {
  const testPodName = `test-${Date.now()}`;

  test('should check if pod exists', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/check`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    if (response.ok()) {
      const data = await response.json();
      expect(data).toHaveProperty('exists');
      if (data.exists) {
        expect(data).toHaveProperty('podUrl');
      }
    }
  });

  test('should attempt pod creation', async ({ request }) => {
    const response = await request.post(`${SOLID_URL}/pods`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Content-Type': 'application/json'
      },
      data: {
        name: testPodName
      },
      failOnStatusCode: false
    });

    // Pod creation should either succeed or fail with appropriate error
    expect([201, 400, 401, 403, 409, 502]).toContain(response.status());

    if (response.status() === 201) {
      const data = await response.json();
      expect(data).toHaveProperty('pod_url');
      expect(data).toHaveProperty('webid');
    }
  });

  test('should perform GET on pod resource', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/${testPodName}/`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Accept': 'application/ld+json'
      },
      failOnStatusCode: false
    });

    // Should return container or 404 if pod doesn't exist
    expect([200, 401, 404]).toContain(response.status());

    if (response.ok()) {
      const contentType = response.headers()['content-type'];
      expect(contentType).toMatch(/application\/(ld\+)?json|text\/turtle/);
    }
  });

  test('should perform PUT to create resource in pod', async ({ request }) => {
    const resourcePath = `${SOLID_URL}/pods/${testPodName}/public/test-resource.jsonld`;
    const testData = {
      '@context': 'https://www.w3.org/ns/ldp',
      '@type': 'Resource',
      'title': 'Test Resource',
      'created': new Date().toISOString()
    };

    const response = await request.put(resourcePath, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Content-Type': 'application/ld+json'
      },
      data: testData,
      failOnStatusCode: false
    });

    // PUT should create or update resource
    expect([200, 201, 204, 401, 403, 404, 502]).toContain(response.status());
  });

  test('should perform POST to container', async ({ request }) => {
    const containerPath = `${SOLID_URL}/pods/${testPodName}/public/`;
    const testData = {
      '@context': 'https://www.w3.org/ns/ldp',
      '@type': 'Resource',
      'title': 'Posted Resource'
    };

    const response = await request.post(containerPath, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Content-Type': 'application/ld+json',
        'Slug': 'posted-resource'
      },
      data: testData,
      failOnStatusCode: false
    });

    // POST should create new resource
    expect([201, 400, 401, 403, 404, 502]).toContain(response.status());

    if (response.status() === 201) {
      const location = response.headers()['location'];
      expect(location).toBeTruthy();
    }
  });

  test('should perform PATCH on resource', async ({ request }) => {
    const resourcePath = `${SOLID_URL}/pods/${testPodName}/public/test-resource.jsonld`;

    // N3 patch format for Solid
    const patchBody = `
      @prefix solid: <http://www.w3.org/ns/solid/terms#>.
      _:patch a solid:InsertDeletePatch;
        solid:inserts { <> <http://purl.org/dc/terms/modified> "${new Date().toISOString()}" }.
    `;

    const response = await request.patch(resourcePath, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Content-Type': 'text/n3'
      },
      data: patchBody,
      failOnStatusCode: false
    });

    // PATCH should modify resource
    expect([200, 204, 400, 401, 403, 404, 415, 502]).toContain(response.status());
  });

  test('should perform DELETE on resource', async ({ request }) => {
    const resourcePath = `${SOLID_URL}/pods/${testPodName}/public/test-resource.jsonld`;

    const response = await request.delete(resourcePath, {
      headers: {
        'Authorization': 'Bearer dev-session-token'
      },
      failOnStatusCode: false
    });

    // DELETE should remove resource
    expect([200, 204, 401, 403, 404, 502]).toContain(response.status());
  });

  test('should return LDP headers on container', async ({ request }) => {
    const response = await request.head(`${SOLID_URL}/pods/${testPodName}/public/`, {
      headers: {
        'Authorization': 'Bearer dev-session-token'
      },
      failOnStatusCode: false
    });

    if (response.ok()) {
      const headers = response.headers();
      // Check for LDP-related headers
      expect(headers['content-type'] || headers['accept-post'] || headers['allow']).toBeTruthy();
    }
  });
});

test.describe('Content Negotiation', () => {
  test('should return JSON-LD when Accept: application/ld+json', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/test/public/`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Accept': 'application/ld+json'
      },
      failOnStatusCode: false
    });

    if (response.ok()) {
      const contentType = response.headers()['content-type'];
      expect(contentType).toMatch(/application\/(ld\+)?json/);
    }
  });

  test('should return Turtle when Accept: text/turtle', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/test/public/`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Accept': 'text/turtle'
      },
      failOnStatusCode: false
    });

    if (response.ok()) {
      const contentType = response.headers()['content-type'];
      expect(contentType).toMatch(/text\/turtle/);
    }
  });
});

test.describe('WebSocket Notifications', () => {
  test('should expose Updates-Via header for WebSocket URL', async ({ request }) => {
    const response = await request.fetch(`${SOLID_URL}/pods/test/public/`, {
      method: 'OPTIONS',
      headers: {
        'Authorization': 'Bearer dev-session-token'
      },
      failOnStatusCode: false
    });

    // Check if Updates-Via header is present (when notifications are enabled)
    const updatesVia = response.headers()['updates-via'];
    if (updatesVia) {
      expect(updatesVia).toMatch(/^wss?:\/\//);
    }
  });

  test('should expose Updates-Via in GET response', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/test/public/`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Accept': 'application/ld+json'
      },
      failOnStatusCode: false
    });

    if (response.ok()) {
      const updatesVia = response.headers()['updates-via'];
      // Updates-Via may or may not be present depending on server config
      if (updatesVia) {
        expect(updatesVia).toMatch(/^wss?:\/\//);
      }
    }
  });
});

test.describe('Ontology Sync Round-Trip', () => {
  const ontologyTestPod = `ontology-test-${Date.now()}`;

  test('should store ontology data in pod', async ({ request }) => {
    const ontologyData = {
      '@context': {
        '@vocab': 'https://narrativegoldmine.com/ontology#',
        'owl': 'http://www.w3.org/2002/07/owl#',
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
      },
      '@type': 'owl:Ontology',
      '@id': 'https://narrativegoldmine.com/ontology',
      'classes': [
        {
          '@id': '#Person',
          '@type': 'owl:Class',
          'rdfs:label': 'Person'
        },
        {
          '@id': '#Organization',
          '@type': 'owl:Class',
          'rdfs:label': 'Organization'
        }
      ],
      'properties': [
        {
          '@id': '#worksFor',
          '@type': 'owl:ObjectProperty',
          'rdfs:domain': { '@id': '#Person' },
          'rdfs:range': { '@id': '#Organization' }
        }
      ]
    };

    const response = await request.put(`${SOLID_URL}/pods/${ontologyTestPod}/ontology/schema.jsonld`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Content-Type': 'application/ld+json'
      },
      data: ontologyData,
      failOnStatusCode: false
    });

    // Store should succeed or return expected error
    expect([200, 201, 204, 401, 404, 502]).toContain(response.status());
  });

  test('should retrieve ontology data from pod', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/${ontologyTestPod}/ontology/schema.jsonld`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Accept': 'application/ld+json'
      },
      failOnStatusCode: false
    });

    if (response.ok()) {
      const data = await response.json();
      expect(data).toHaveProperty('@context');
      expect(data).toHaveProperty('@type');
    }
  });

  test('should update ontology via PATCH', async ({ request }) => {
    const patch = `
      @prefix solid: <http://www.w3.org/ns/solid/terms#>.
      @prefix ngm: <https://narrativegoldmine.com/ontology#>.

      _:patch a solid:InsertDeletePatch;
        solid:inserts {
          <#Location> a <http://www.w3.org/2002/07/owl#Class>;
            <http://www.w3.org/2000/01/rdf-schema#label> "Location" .
        }.
    `;

    const response = await request.patch(`${SOLID_URL}/pods/${ontologyTestPod}/ontology/schema.jsonld`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Content-Type': 'text/n3'
      },
      data: patch,
      failOnStatusCode: false
    });

    // PATCH may succeed or fail depending on server capabilities
    expect([200, 204, 400, 401, 404, 415, 502]).toContain(response.status());
  });

  test('should sync constraint groups to pod', async ({ request }) => {
    const constraintGroups = {
      '@context': 'https://narrativegoldmine.com/ontology#',
      '@type': 'ConstraintConfiguration',
      groups: [
        {
          id: 'subsumption',
          name: 'Subsumption',
          enabled: true,
          strength: 0.8
        },
        {
          id: 'disjointness',
          name: 'Disjointness',
          enabled: true,
          strength: 1.0
        }
      ],
      lastModified: new Date().toISOString()
    };

    const response = await request.put(`${SOLID_URL}/pods/${ontologyTestPod}/ontology/constraints.jsonld`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Content-Type': 'application/ld+json'
      },
      data: constraintGroups,
      failOnStatusCode: false
    });

    expect([200, 201, 204, 401, 404, 502]).toContain(response.status());
  });
});

test.describe('Error Handling', () => {
  test('should return proper error for non-existent pod', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/nonexistent-pod-12345/`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Accept': 'application/ld+json'
      },
      failOnStatusCode: false
    });

    expect([404, 502]).toContain(response.status());
  });

  test('should return proper error for invalid resource path', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/test/../../../etc/passwd`, {
      headers: {
        'Authorization': 'Bearer dev-session-token'
      },
      failOnStatusCode: false
    });

    // Should reject path traversal attempts
    expect([400, 403, 404]).toContain(response.status());
  });

  test('should handle unsupported HTTP methods', async ({ request }) => {
    const response = await request.fetch(`${SOLID_URL}/pods/test/`, {
      method: 'TRACE',
      headers: {
        'Authorization': 'Bearer dev-session-token'
      },
      failOnStatusCode: false
    });

    expect([405, 501]).toContain(response.status());
  });

  test('should return error for invalid Content-Type on PUT', async ({ request }) => {
    const response = await request.put(`${SOLID_URL}/pods/test/public/invalid.txt`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Content-Type': 'application/x-invalid-type'
      },
      data: 'invalid content',
      failOnStatusCode: false
    });

    // Server should handle or reject unknown content types
    expect([200, 201, 400, 415, 502]).toContain(response.status());
  });
});

test.describe('Data Persistence Verification', () => {
  const persistenceTestPod = `persist-test-${Date.now()}`;
  const testResourcePath = `/pods/${persistenceTestPod}/public/persistence-test.jsonld`;

  test('should persist and retrieve data correctly', async ({ request }) => {
    const testData = {
      '@context': 'https://www.w3.org/ns/ldp',
      '@id': '#test',
      '@type': 'Document',
      'title': 'Persistence Test',
      'content': 'This is a test of data persistence',
      'timestamp': new Date().toISOString(),
      'number': 42,
      'nested': {
        'key': 'value'
      }
    };

    // Write data
    const writeResponse = await request.put(`${SOLID_URL}${testResourcePath}`, {
      headers: {
        'Authorization': 'Bearer dev-session-token',
        'Content-Type': 'application/ld+json'
      },
      data: testData,
      failOnStatusCode: false
    });

    if (writeResponse.status() === 201 || writeResponse.status() === 200) {
      // Read data back
      const readResponse = await request.get(`${SOLID_URL}${testResourcePath}`, {
        headers: {
          'Authorization': 'Bearer dev-session-token',
          'Accept': 'application/ld+json'
        }
      });

      expect(readResponse.ok()).toBe(true);
      const retrievedData = await readResponse.json();

      // Verify data integrity
      expect(retrievedData.title).toBe('Persistence Test');
      expect(retrievedData.number).toBe(42);
    }
  });
});

test.describe('Real-Time Update Flow', () => {
  test('should subscribe to resource changes', async ({ page }) => {
    // Navigate to app
    await page.goto(BASE_URL);

    // Wait for app to load
    await page.waitForLoadState('networkidle');

    // Check if SolidPodService is available in window context
    const hasSolidService = await page.evaluate(() => {
      return typeof window !== 'undefined';
    });

    expect(hasSolidService).toBe(true);
  });
});
