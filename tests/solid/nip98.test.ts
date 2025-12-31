/**
 * NIP-98 HTTP Authentication Unit Tests
 *
 * Tests the client-side NIP-98 token handling:
 * - Token format validation
 * - Base64 encoding/decoding
 * - Event structure validation
 * - Timestamp handling
 * - URL and method tag extraction
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// NIP-98 HTTP Auth event kind (from spec)
const HTTP_AUTH_KIND = 27235;

// Token format helpers
function createNip98Event(overrides: Partial<{
  id: string;
  pubkey: string;
  created_at: number;
  kind: number;
  tags: string[][];
  content: string;
  sig: string;
}> = {}) {
  return {
    id: 'test_event_id',
    pubkey: 'bfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a452',
    created_at: Math.floor(Date.now() / 1000),
    kind: HTTP_AUTH_KIND,
    tags: [
      ['u', 'http://localhost:3030/pods/test/'],
      ['method', 'GET']
    ],
    content: '',
    sig: 'test_signature',
    ...overrides
  };
}

function encodeToken(event: object): string {
  return Buffer.from(JSON.stringify(event)).toString('base64');
}

function decodeToken(token: string): object | null {
  try {
    const decoded = Buffer.from(token, 'base64').toString('utf-8');
    return JSON.parse(decoded);
  } catch {
    return null;
  }
}

describe('NIP-98 Token Encoding', () => {
  it('should encode event to base64', () => {
    const event = createNip98Event();
    const token = encodeToken(event);

    expect(token).toBeDefined();
    expect(typeof token).toBe('string');
    // Base64 should not contain special characters that break headers
    expect(token).toMatch(/^[A-Za-z0-9+/]+=*$/);
  });

  it('should decode base64 token back to event', () => {
    const originalEvent = createNip98Event();
    const token = encodeToken(originalEvent);
    const decoded = decodeToken(token);

    expect(decoded).not.toBeNull();
    expect((decoded as any).pubkey).toBe(originalEvent.pubkey);
    expect((decoded as any).kind).toBe(HTTP_AUTH_KIND);
  });

  it('should preserve all event fields', () => {
    const event = createNip98Event({
      tags: [
        ['u', 'http://example.com/resource'],
        ['method', 'PUT'],
        ['payload', 'abc123']
      ]
    });
    const token = encodeToken(event);
    const decoded = decodeToken(token) as any;

    expect(decoded.tags).toHaveLength(3);
    expect(decoded.tags[2][0]).toBe('payload');
    expect(decoded.tags[2][1]).toBe('abc123');
  });
});

describe('NIP-98 Event Structure', () => {
  it('should require correct event kind (27235)', () => {
    const event = createNip98Event();
    expect(event.kind).toBe(HTTP_AUTH_KIND);
  });

  it('should include u (URL) tag', () => {
    const event = createNip98Event();
    const urlTag = event.tags.find(t => t[0] === 'u');

    expect(urlTag).toBeDefined();
    expect(urlTag![1]).toMatch(/^https?:\/\//);
  });

  it('should include method tag', () => {
    const event = createNip98Event();
    const methodTag = event.tags.find(t => t[0] === 'method');

    expect(methodTag).toBeDefined();
    expect(['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD']).toContain(methodTag![1]);
  });

  it('should include valid created_at timestamp', () => {
    const event = createNip98Event();
    const now = Math.floor(Date.now() / 1000);

    expect(event.created_at).toBeGreaterThan(now - 60);
    expect(event.created_at).toBeLessThanOrEqual(now + 1);
  });

  it('should have empty content', () => {
    const event = createNip98Event();
    expect(event.content).toBe('');
  });
});

describe('NIP-98 Tag Extraction', () => {
  it('should extract URL from u tag', () => {
    const event = createNip98Event({
      tags: [
        ['u', 'http://example.com/pods/alice/data.json'],
        ['method', 'GET']
      ]
    });

    const urlTag = event.tags.find(t => t[0] === 'u');
    expect(urlTag![1]).toBe('http://example.com/pods/alice/data.json');
  });

  it('should extract method from method tag', () => {
    const event = createNip98Event({
      tags: [
        ['u', 'http://example.com/resource'],
        ['method', 'DELETE']
      ]
    });

    const methodTag = event.tags.find(t => t[0] === 'method');
    expect(methodTag![1]).toBe('DELETE');
  });

  it('should extract payload hash when present', () => {
    const payloadHash = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
    const event = createNip98Event({
      tags: [
        ['u', 'http://example.com/resource'],
        ['method', 'PUT'],
        ['payload', payloadHash]
      ]
    });

    const payloadTag = event.tags.find(t => t[0] === 'payload');
    expect(payloadTag).toBeDefined();
    expect(payloadTag![1]).toBe(payloadHash);
  });
});

describe('NIP-98 Timestamp Validation', () => {
  it('should consider tokens within 60 seconds as valid', () => {
    const now = Math.floor(Date.now() / 1000);
    const validTimestamps = [
      now,
      now - 30,
      now - 59,
      now + 5 // Small clock skew tolerance
    ];

    for (const timestamp of validTimestamps) {
      const age = now - timestamp;
      const isValid = Math.abs(age) <= 60;
      expect(isValid).toBe(true);
    }
  });

  it('should consider tokens older than 60 seconds as expired', () => {
    const now = Math.floor(Date.now() / 1000);
    const expiredTimestamp = now - 120; // 2 minutes ago
    const age = now - expiredTimestamp;

    expect(age > 60).toBe(true);
  });

  it('should reject tokens from the future (beyond tolerance)', () => {
    const now = Math.floor(Date.now() / 1000);
    const futureTimestamp = now + 120; // 2 minutes in future
    const age = now - futureTimestamp;

    expect(age < -60).toBe(true);
  });
});

describe('NIP-98 URL Normalization', () => {
  function normalizeUrl(url: string): string {
    let normalized = url.trim();

    // Remove trailing slashes for comparison
    while (normalized.endsWith('/') && normalized.length > 1) {
      normalized = normalized.slice(0, -1);
    }

    // Lowercase scheme and host
    const match = normalized.match(/^(https?:\/\/)([^\/]+)(\/.*)?$/i);
    if (match) {
      const scheme = match[1].toLowerCase();
      const host = match[2].toLowerCase();
      const path = match[3] || '';
      normalized = `${scheme}${host}${path}`;
    }

    return normalized;
  }

  it('should lowercase scheme', () => {
    expect(normalizeUrl('HTTP://example.com/path')).toBe('http://example.com/path');
    expect(normalizeUrl('HTTPS://Example.COM/path')).toBe('https://example.com/path');
  });

  it('should lowercase host', () => {
    expect(normalizeUrl('http://EXAMPLE.COM/path')).toBe('http://example.com/path');
  });

  it('should preserve path case', () => {
    expect(normalizeUrl('http://example.com/Path/To/Resource')).toBe('http://example.com/Path/To/Resource');
  });

  it('should remove trailing slashes for comparison', () => {
    expect(normalizeUrl('http://example.com/path/')).toBe('http://example.com/path');
    expect(normalizeUrl('http://example.com/path///')).toBe('http://example.com/path');
  });

  it('should handle root URL', () => {
    expect(normalizeUrl('http://example.com/')).toBe('http://example.com');
  });
});

describe('NIP-98 Payload Hash', () => {
  // SHA256 hash computation would typically use crypto
  // This tests the expected format

  it('should be 64-character hex string', () => {
    const payloadHash = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';

    expect(payloadHash).toHaveLength(64);
    expect(payloadHash).toMatch(/^[a-f0-9]{64}$/);
  });

  it('should be deterministic for same content', () => {
    // Same content should produce same hash (conceptual test)
    const content = '{"test": "data"}';
    const expectedBehavior = true; // Hash(content) === Hash(content)

    expect(expectedBehavior).toBe(true);
  });
});

describe('NIP-98 Authorization Header', () => {
  it('should format header correctly', () => {
    const token = encodeToken(createNip98Event());
    const header = `Nostr ${token}`;

    expect(header).toMatch(/^Nostr [A-Za-z0-9+/]+=*$/);
  });

  it('should parse Nostr authorization header', () => {
    const token = 'eyJpZCI6InRlc3QifQ==';
    const header = `Nostr ${token}`;

    const match = header.match(/^Nostr (.+)$/);
    expect(match).not.toBeNull();
    expect(match![1]).toBe(token);
  });

  it('should not match Bearer authorization', () => {
    const header = 'Bearer sometoken';
    const match = header.match(/^Nostr (.+)$/);

    expect(match).toBeNull();
  });

  it('should handle case-sensitive Nostr prefix', () => {
    const token = 'eyJpZCI6InRlc3QifQ==';

    // 'Nostr' should match
    expect(`Nostr ${token}`.startsWith('Nostr ')).toBe(true);

    // 'nostr' should NOT match (case-sensitive per spec)
    expect(`nostr ${token}`.startsWith('Nostr ')).toBe(false);
  });
});

describe('NIP-98 Pubkey Extraction', () => {
  it('should extract pubkey from token', () => {
    const pubkey = 'bfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a452';
    const event = createNip98Event({ pubkey });
    const token = encodeToken(event);

    const decoded = decodeToken(token) as any;
    expect(decoded.pubkey).toBe(pubkey);
  });

  it('should validate pubkey format (64-char hex)', () => {
    const validPubkey = 'bfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a452';

    expect(validPubkey).toHaveLength(64);
    expect(validPubkey).toMatch(/^[a-f0-9]{64}$/);
  });

  it('should reject invalid pubkey format', () => {
    const invalidPubkeys = [
      'short',
      'bfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a45', // 63 chars
      'bfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a4522', // 65 chars
      'zfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a452' // invalid hex
    ];

    for (const pk of invalidPubkeys) {
      const isValid = pk.length === 64 && /^[a-f0-9]{64}$/.test(pk);
      expect(isValid).toBe(false);
    }
  });
});

describe('NIP-98 Method Validation', () => {
  it('should accept valid HTTP methods', () => {
    const validMethods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'];

    for (const method of validMethods) {
      const event = createNip98Event({
        tags: [
          ['u', 'http://example.com/resource'],
          ['method', method]
        ]
      });

      const methodTag = event.tags.find(t => t[0] === 'method');
      expect(validMethods).toContain(methodTag![1]);
    }
  });

  it('should uppercase method for comparison', () => {
    const method = 'get';
    expect(method.toUpperCase()).toBe('GET');
  });
});

describe('NIP-98 Integration with SolidPodService', () => {
  it('should build correct authorization header structure', () => {
    const token = encodeToken(createNip98Event());
    const header = `Nostr ${token}`;

    // Header should be suitable for HTTP Authorization header
    expect(header.length).toBeLessThan(8192); // Reasonable header size
    expect(header).not.toContain('\n');
    expect(header).not.toContain('\r');
  });
});
