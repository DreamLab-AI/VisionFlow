/**
 * Pod Provisioning Tests
 *
 * Tests for automatic user pod creation when users authenticate via Nostr.
 * Each user gets a pod at /pods/{npub}/ with standard structure.
 *
 * @see docs/architecture/user-agent-pod-design.md
 */

import { vi, describe, it, expect, beforeEach } from 'vitest';

// Types matching pseudocode design
interface PodInfo {
  path: string;
  webid: string;
  created: boolean;
}

interface WebIdProfile {
  '@context': string[];
  '@id': string;
  '@type': string;
  'solid:oidcIssuer': string;
  'nostr:pubkey': string;
}

interface AclRule {
  '@type': string;
  'acl:agent': string;
  'acl:accessTo': string;
  'acl:mode': string[];
}

// Mock HTTP client for JSS
const mockHttpClient = {
  responses: new Map<string, { status: number; body?: unknown }>(),
  createdResources: new Set<string>(),

  head: vi.fn(async (url: string) => {
    if (mockHttpClient.createdResources.has(url)) {
      return { status: 200 };
    }
    return mockHttpClient.responses.get(url) || { status: 404 };
  }),

  put: vi.fn(async (url: string, body: unknown) => {
    mockHttpClient.createdResources.add(url);
    return { status: 201 };
  }),

  post: vi.fn(async (url: string, body: unknown) => {
    mockHttpClient.createdResources.add(url);
    return { status: 201 };
  }),

  reset: () => {
    mockHttpClient.responses.clear();
    mockHttpClient.createdResources.clear();
    mockHttpClient.head.mockClear();
    mockHttpClient.put.mockClear();
    mockHttpClient.post.mockClear();
  },
};

// Utility functions matching pseudocode
function npubToHex(npub: string): string {
  // Simplified mock - real implementation would decode bech32
  if (npub.startsWith('npub1')) {
    return npub.replace('npub1', '').padEnd(64, '0');
  }
  return npub;
}

function hexToNpub(hex: string): string {
  // Simplified mock - real implementation would encode bech32
  return 'npub1' + hex.substring(0, 59);
}

// Implementation under test (matching pseudocode)
async function podExists(jssBaseUrl: string, podPath: string): Promise<boolean> {
  const response = await mockHttpClient.head(`${jssBaseUrl}${podPath}`);
  return response.status === 200;
}

async function createContainer(jssBaseUrl: string, path: string): Promise<void> {
  await mockHttpClient.put(`${jssBaseUrl}${path}`, {
    '@type': 'ldp:BasicContainer',
  });
}

async function writeResource(
  jssBaseUrl: string,
  path: string,
  content: unknown
): Promise<void> {
  await mockHttpClient.put(`${jssBaseUrl}${path}`, content);
}

function generateWebIdProfile(podPath: string, userNpub: string): WebIdProfile {
  return {
    '@context': ['https://www.w3.org/ns/solid/terms'],
    '@id': `${podPath}profile/card#me`,
    '@type': 'foaf:Person',
    'solid:oidcIssuer': `did:nostr:${npubToHex(userNpub)}`,
    'nostr:pubkey': npubToHex(userNpub),
  };
}

function generateOwnerAcl(podPath: string, userNpub: string): AclRule[] {
  const userWebId = `${podPath}profile/card#me`;
  return [
    {
      '@type': 'acl:Authorization',
      'acl:agent': userWebId,
      'acl:accessTo': podPath,
      'acl:mode': ['acl:Read', 'acl:Write', 'acl:Control'],
    },
  ];
}

async function provisionUserPod(
  jssBaseUrl: string,
  userNpub: string
): Promise<PodInfo> {
  const podPath = `/pods/${userNpub}/`;

  // Check if pod exists
  const exists = await podExists(jssBaseUrl, podPath);

  if (!exists) {
    // Create pod container
    await createContainer(jssBaseUrl, podPath);

    // Create WebID profile linking Nostr identity
    const profile = generateWebIdProfile(podPath, userNpub);
    await writeResource(jssBaseUrl, `${podPath}profile/card`, profile);

    // Create agent-memory structure
    const memoryTypes = ['episodic', 'semantic', 'procedural', 'sessions'];
    for (const memoryType of memoryTypes) {
      await createContainer(jssBaseUrl, `${podPath}agent-memory/${memoryType}/`);
    }

    // Set default ACL (owner only)
    const acl = generateOwnerAcl(podPath, userNpub);
    await writeResource(jssBaseUrl, `${podPath}.acl`, acl);

    // Create delegations container
    await createContainer(jssBaseUrl, `${podPath}delegations/`);

    return {
      path: podPath,
      webid: `${podPath}profile/card#me`,
      created: true,
    };
  }

  return {
    path: podPath,
    webid: `${podPath}profile/card#me`,
    created: false,
  };
}

describe('Pod Provisioning', () => {
  const jssBaseUrl = 'http://jss:3030';
  const testNpub = 'npub1abc123def456789';

  beforeEach(() => {
    mockHttpClient.reset();
  });

  describe('npubToHex conversion', () => {
    it('should convert npub to hex pubkey', () => {
      const npub = 'npub1abc123';
      const hex = npubToHex(npub);

      expect(hex).not.toContain('npub1');
      expect(hex.length).toBe(64);
    });

    it('should handle already-hex input', () => {
      const hex = 'abc123'.padEnd(64, '0');
      const result = npubToHex(hex);

      expect(result).toBe(hex);
    });
  });

  describe('podExists', () => {
    it('should return false for non-existent pod', async () => {
      const exists = await podExists(jssBaseUrl, '/pods/nonexistent/');
      expect(exists).toBe(false);
    });

    it('should return true for existing pod', async () => {
      mockHttpClient.responses.set(`${jssBaseUrl}/pods/existing/`, { status: 200 });

      const exists = await podExists(jssBaseUrl, '/pods/existing/');
      expect(exists).toBe(true);
    });
  });

  describe('provisionUserPod', () => {
    it('should create new pod with correct structure', async () => {
      const result = await provisionUserPod(jssBaseUrl, testNpub);

      expect(result.created).toBe(true);
      expect(result.path).toBe(`/pods/${testNpub}/`);
      expect(result.webid).toBe(`/pods/${testNpub}/profile/card#me`);
    });

    it('should create pod container', async () => {
      await provisionUserPod(jssBaseUrl, testNpub);

      expect(mockHttpClient.put).toHaveBeenCalledWith(
        `${jssBaseUrl}/pods/${testNpub}/`,
        expect.any(Object)
      );
    });

    it('should create WebID profile with Nostr identity', async () => {
      await provisionUserPod(jssBaseUrl, testNpub);

      expect(mockHttpClient.put).toHaveBeenCalledWith(
        `${jssBaseUrl}/pods/${testNpub}/profile/card`,
        expect.objectContaining({
          '@type': 'foaf:Person',
          'nostr:pubkey': npubToHex(testNpub),
        })
      );
    });

    it('should create all memory type containers', async () => {
      await provisionUserPod(jssBaseUrl, testNpub);

      const memoryTypes = ['episodic', 'semantic', 'procedural', 'sessions'];
      for (const memoryType of memoryTypes) {
        expect(mockHttpClient.put).toHaveBeenCalledWith(
          `${jssBaseUrl}/pods/${testNpub}/agent-memory/${memoryType}/`,
          expect.any(Object)
        );
      }
    });

    it('should create ACL with owner-only access', async () => {
      await provisionUserPod(jssBaseUrl, testNpub);

      expect(mockHttpClient.put).toHaveBeenCalledWith(
        `${jssBaseUrl}/pods/${testNpub}/.acl`,
        expect.arrayContaining([
          expect.objectContaining({
            'acl:mode': expect.arrayContaining(['acl:Read', 'acl:Write', 'acl:Control']),
          }),
        ])
      );
    });

    it('should create delegations container', async () => {
      await provisionUserPod(jssBaseUrl, testNpub);

      expect(mockHttpClient.put).toHaveBeenCalledWith(
        `${jssBaseUrl}/pods/${testNpub}/delegations/`,
        expect.any(Object)
      );
    });

    it('should skip creation if pod already exists', async () => {
      // Pre-create the pod
      mockHttpClient.responses.set(`${jssBaseUrl}/pods/${testNpub}/`, { status: 200 });

      const result = await provisionUserPod(jssBaseUrl, testNpub);

      expect(result.created).toBe(false);
      expect(result.path).toBe(`/pods/${testNpub}/`);
      // Should only call HEAD, not PUT
      expect(mockHttpClient.head).toHaveBeenCalled();
      expect(mockHttpClient.put).not.toHaveBeenCalled();
    });
  });

  describe('WebID Profile', () => {
    it('should generate profile with correct structure', () => {
      const podPath = `/pods/${testNpub}/`;
      const profile = generateWebIdProfile(podPath, testNpub);

      expect(profile['@context']).toContain('https://www.w3.org/ns/solid/terms');
      expect(profile['@id']).toBe(`${podPath}profile/card#me`);
      expect(profile['@type']).toBe('foaf:Person');
    });

    it('should link Nostr identity via DID', () => {
      const podPath = `/pods/${testNpub}/`;
      const profile = generateWebIdProfile(podPath, testNpub);

      expect(profile['solid:oidcIssuer']).toContain('did:nostr:');
      expect(profile['nostr:pubkey']).toBe(npubToHex(testNpub));
    });
  });

  describe('ACL Generation', () => {
    it('should grant full control to pod owner', () => {
      const podPath = `/pods/${testNpub}/`;
      const acl = generateOwnerAcl(podPath, testNpub);

      expect(acl).toHaveLength(1);
      expect(acl[0]['acl:agent']).toBe(`${podPath}profile/card#me`);
      expect(acl[0]['acl:mode']).toContain('acl:Control');
    });

    it('should scope access to pod path', () => {
      const podPath = `/pods/${testNpub}/`;
      const acl = generateOwnerAcl(podPath, testNpub);

      expect(acl[0]['acl:accessTo']).toBe(podPath);
    });
  });

  describe('Pod Structure Verification', () => {
    it('should match expected directory structure', async () => {
      await provisionUserPod(jssBaseUrl, testNpub);

      const expectedPaths = [
        `/pods/${testNpub}/`,
        `/pods/${testNpub}/profile/card`,
        `/pods/${testNpub}/agent-memory/episodic/`,
        `/pods/${testNpub}/agent-memory/semantic/`,
        `/pods/${testNpub}/agent-memory/procedural/`,
        `/pods/${testNpub}/agent-memory/sessions/`,
        `/pods/${testNpub}/.acl`,
        `/pods/${testNpub}/delegations/`,
      ];

      for (const path of expectedPaths) {
        expect(mockHttpClient.createdResources.has(`${jssBaseUrl}${path}`)).toBe(true);
      }
    });
  });
});
