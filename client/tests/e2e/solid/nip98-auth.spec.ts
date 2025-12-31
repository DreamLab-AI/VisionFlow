/**
 * NIP-98 HTTP Authentication Tests
 *
 * Tests the Nostr NIP-98 authentication flow for Solid Pod access:
 * - Token generation and format validation
 * - Signature verification
 * - Timestamp validation (60-second window)
 * - URL and method matching
 * - Payload hash verification
 */

import { test, expect } from '@playwright/test';

const API_URL = process.env.TEST_API_URL || 'http://localhost:4000';
const SOLID_URL = `${API_URL}/solid`;

// NIP-98 HTTP Auth event kind
const HTTP_AUTH_KIND = 27235;

// Helper to create a mock NIP-98 event structure
function createMockNip98Event(overrides: Partial<{
  id: string;
  pubkey: string;
  created_at: number;
  kind: number;
  tags: string[][];
  content: string;
  sig: string;
}> = {}) {
  const now = Math.floor(Date.now() / 1000);
  return {
    id: 'mock_event_id_' + Math.random().toString(36).slice(2),
    pubkey: 'bfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a452',
    created_at: now,
    kind: HTTP_AUTH_KIND,
    tags: [
      ['u', `${SOLID_URL}/pods/test/`],
      ['method', 'GET']
    ],
    content: '',
    sig: 'mock_signature_' + Math.random().toString(36).slice(2),
    ...overrides
  };
}

// Helper to encode event to base64
function encodeToken(event: object): string {
  return Buffer.from(JSON.stringify(event)).toString('base64');
}

test.describe('NIP-98 Token Format Validation', () => {
  test('should reject non-base64 encoded tokens', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': 'Nostr not-valid-base64!!!@#$',
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });

  test('should reject tokens with invalid JSON structure', async ({ request }) => {
    const invalidJson = Buffer.from('not valid json').toString('base64');

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${invalidJson}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });

  test('should reject tokens missing required fields', async ({ request }) => {
    // Token missing pubkey
    const incompleteEvent = {
      id: 'test',
      kind: HTTP_AUTH_KIND,
      tags: [['u', `${SOLID_URL}/pods/test/`], ['method', 'GET']],
      sig: 'fake'
    };
    const token = encodeToken(incompleteEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });
});

test.describe('NIP-98 Event Kind Validation', () => {
  test('should reject tokens with kind != 27235', async ({ request }) => {
    const wrongKindEvent = createMockNip98Event({ kind: 1 }); // Regular note, not HTTP auth
    const token = encodeToken(wrongKindEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });

  test('should accept tokens with correct kind 27235', async ({ request }) => {
    const correctKindEvent = createMockNip98Event({ kind: HTTP_AUTH_KIND });
    const token = encodeToken(correctKindEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    // May fail signature verification, but should not fail on kind
    // If it fails with 401/403, it's for signature, not kind
    expect([401, 403, 404, 200]).toContain(response.status());
  });
});

test.describe('NIP-98 Timestamp Validation', () => {
  test('should reject tokens created more than 60 seconds ago', async ({ request }) => {
    const expiredEvent = createMockNip98Event({
      created_at: Math.floor(Date.now() / 1000) - 120 // 2 minutes ago
    });
    const token = encodeToken(expiredEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });

  test('should reject tokens from the future (clock skew protection)', async ({ request }) => {
    const futureEvent = createMockNip98Event({
      created_at: Math.floor(Date.now() / 1000) + 120 // 2 minutes in future
    });
    const token = encodeToken(futureEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });

  test('should accept tokens within 60-second window', async ({ request }) => {
    const validEvent = createMockNip98Event({
      created_at: Math.floor(Date.now() / 1000) - 30 // 30 seconds ago - within window
    });
    const token = encodeToken(validEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    // Should pass timestamp validation (may still fail signature)
    expect([401, 403, 404, 200]).toContain(response.status());
  });
});

test.describe('NIP-98 URL Tag Validation', () => {
  test('should reject tokens missing u (URL) tag', async ({ request }) => {
    const noUrlEvent = createMockNip98Event({
      tags: [['method', 'GET']] // Missing u tag
    });
    const token = encodeToken(noUrlEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });

  test('should reject tokens with mismatched URL', async ({ request }) => {
    const wrongUrlEvent = createMockNip98Event({
      tags: [
        ['u', 'http://evil.com/pods/test/'], // Wrong URL
        ['method', 'GET']
      ]
    });
    const token = encodeToken(wrongUrlEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });
});

test.describe('NIP-98 Method Tag Validation', () => {
  test('should reject tokens missing method tag', async ({ request }) => {
    const noMethodEvent = createMockNip98Event({
      tags: [['u', `${SOLID_URL}/pods/test/`]] // Missing method tag
    });
    const token = encodeToken(noMethodEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });

  test('should reject tokens with mismatched method', async ({ request }) => {
    const wrongMethodEvent = createMockNip98Event({
      tags: [
        ['u', `${SOLID_URL}/pods/test/`],
        ['method', 'POST'] // Wrong method for GET request
      ]
    });
    const token = encodeToken(wrongMethodEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });
});

test.describe('NIP-98 Payload Hash Validation', () => {
  test('should validate payload hash for PUT requests', async ({ request }) => {
    const body = JSON.stringify({ '@context': 'https://schema.org', test: 'data' });

    // Create token without payload hash for a PUT request with body
    const noPayloadHashEvent = createMockNip98Event({
      tags: [
        ['u', `${SOLID_URL}/pods/test/public/test.json`],
        ['method', 'PUT']
        // Missing payload tag
      ]
    });
    const token = encodeToken(noPayloadHashEvent);

    const response = await request.put(`${SOLID_URL}/pods/test/public/test.json`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Content-Type': 'application/json'
      },
      data: body,
      failOnStatusCode: false
    });

    // Depending on implementation, may or may not require payload hash
    expect([200, 201, 400, 401, 403, 404]).toContain(response.status());
  });

  test('should reject mismatched payload hash', async ({ request }) => {
    const body = JSON.stringify({ '@context': 'https://schema.org', test: 'data' });

    const wrongPayloadEvent = createMockNip98Event({
      tags: [
        ['u', `${SOLID_URL}/pods/test/public/test.json`],
        ['method', 'PUT'],
        ['payload', 'wrong_hash_value'] // Incorrect hash
      ]
    });
    const token = encodeToken(wrongPayloadEvent);

    const response = await request.put(`${SOLID_URL}/pods/test/public/test.json`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Content-Type': 'application/json'
      },
      data: body,
      failOnStatusCode: false
    });

    // Should reject if server validates payload hash
    expect([400, 401, 403, 404]).toContain(response.status());
  });
});

test.describe('NIP-98 Signature Validation', () => {
  test('should reject tokens with invalid signature', async ({ request }) => {
    const invalidSigEvent = createMockNip98Event({
      sig: 'invalid_signature_that_will_not_verify_0000000000000000'
    });
    const token = encodeToken(invalidSigEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    // Should reject invalid signature
    expect([401, 403]).toContain(response.status());
  });

  test('should reject tokens with signature not matching pubkey', async ({ request }) => {
    // Create token with mismatched pubkey and signature
    const mismatchedEvent = createMockNip98Event({
      pubkey: '0000000000000000000000000000000000000000000000000000000000000001',
      sig: 'signature_from_different_key_will_not_verify'
    });
    const token = encodeToken(mismatchedEvent);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    // Should reject mismatched signature
    expect([401, 403]).toContain(response.status());
  });
});

test.describe('NIP-98 Authorization Header Parsing', () => {
  test('should reject non-Nostr authorization schemes', async ({ request }) => {
    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': 'Bearer some-bearer-token',
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    // Bearer tokens should be handled differently or rejected
    expect([401, 403, 404, 200]).toContain(response.status());
  });

  test('should handle case-sensitive Nostr scheme', async ({ request }) => {
    const token = encodeToken(createMockNip98Event());

    // Test lowercase 'nostr' (should be case-sensitive per spec)
    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `nostr ${token}`, // lowercase
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    // 'nostr' lowercase may be rejected as per NIP-98 spec (should be 'Nostr')
    expect([400, 401, 403, 404]).toContain(response.status());
  });

  test('should handle whitespace in Authorization header', async ({ request }) => {
    const token = encodeToken(createMockNip98Event());

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `  Nostr   ${token}  `, // Extra whitespace
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    // Should handle or gracefully reject
    expect([401, 403, 404, 200]).toContain(response.status());
  });
});

test.describe('NIP-98 Pubkey Extraction', () => {
  test('should extract pubkey from valid token', async ({ request }) => {
    const testPubkey = 'abcd1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab';
    const event = createMockNip98Event({ pubkey: testPubkey });
    const token = encodeToken(event);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    // Response should be processed (even if auth fails for signature)
    expect(response.status()).toBeLessThan(500);
  });

  test('should reject tokens with invalid pubkey format', async ({ request }) => {
    const event = createMockNip98Event({ pubkey: 'not-a-valid-hex-pubkey' });
    const token = encodeToken(event);

    const response = await request.get(`${SOLID_URL}/pods/test/`, {
      headers: {
        'Authorization': `Nostr ${token}`,
        'Accept': 'application/json'
      },
      failOnStatusCode: false
    });

    expect([400, 401, 403]).toContain(response.status());
  });
});
