import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  deriveNostrKey,
  startRegistration,
  checkUsernameAvailable,
  startLogin,
  bytesToHex,
  downloadKeyBackup,
} from '../passkeyService';

// --- Mocks ---

// Mock fetch globally
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

// Mock crypto.subtle
const mockCrypto = {
  subtle: {
    importKey: vi.fn(),
    deriveBits: vi.fn(),
    digest: vi.fn(),
  },
};
vi.stubGlobal('crypto', mockCrypto);

describe('passkeyService', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // --- bytesToHex ---

  describe('bytesToHex', () => {
    it('should convert bytes to hex string', () => {
      const bytes = new Uint8Array([0, 1, 15, 16, 255]);
      expect(bytesToHex(bytes)).toBe('00010f10ff');
    });

    it('should return empty string for empty array', () => {
      expect(bytesToHex(new Uint8Array([]))).toBe('');
    });

    it('should pad single-digit hex values', () => {
      const bytes = new Uint8Array([0, 5, 10]);
      expect(bytesToHex(bytes)).toBe('00050a');
    });
  });

  // --- deriveNostrKey ---

  describe('deriveNostrKey', () => {
    it('should derive a 32-byte key from PRF output', async () => {
      const mockKeyMaterial = {};
      const mockDerived = new ArrayBuffer(32);
      new Uint8Array(mockDerived).fill(0xab);

      mockCrypto.subtle.importKey.mockResolvedValue(mockKeyMaterial);
      mockCrypto.subtle.deriveBits.mockResolvedValue(mockDerived);

      const result = await deriveNostrKey(new ArrayBuffer(32));

      expect(result).toBeInstanceOf(Uint8Array);
      expect(result.length).toBe(32);
      expect(mockCrypto.subtle.importKey).toHaveBeenCalledWith(
        'raw',
        expect.any(ArrayBuffer),
        'HKDF',
        false,
        ['deriveBits']
      );
      expect(mockCrypto.subtle.deriveBits).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'HKDF',
          hash: 'SHA-256',
        }),
        mockKeyMaterial,
        256
      );
    });
  });

  // --- startRegistration ---

  describe('startRegistration', () => {
    it('should return registration options on success', async () => {
      const mockOptions = {
        challenge: 'abc123',
        rp: { name: 'Test', id: 'localhost' },
        user: { id: 'uid', name: 'alice', displayName: 'Alice' },
        pubKeyCredParams: [{ type: 'public-key', alg: -7 }],
        challengeKey: 'ck1',
      };
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue(mockOptions),
      });

      const result = await startRegistration('alice');
      expect(result).toEqual(mockOptions);
      expect(mockFetch).toHaveBeenCalledWith('/idp/passkey/register-new/options', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: 'alice' }),
      });
    });

    it('should throw on 409 (username taken)', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 409 });
      await expect(startRegistration('taken')).rejects.toThrow('Username already taken');
    });

    it('should throw on 400 with server error message', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        json: vi.fn().mockResolvedValue({ error: 'Too short' }),
      });
      await expect(startRegistration('ab')).rejects.toThrow('Too short');
    });

    it('should throw on 405 (passkey not enabled)', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 405 });
      await expect(startRegistration('alice')).rejects.toThrow('not enabled');
    });

    it('should throw on other error status codes', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 500 });
      await expect(startRegistration('alice')).rejects.toThrow('Registration failed (500)');
    });
  });

  // --- checkUsernameAvailable ---

  describe('checkUsernameAvailable', () => {
    it('should return "available" on 200', async () => {
      mockFetch.mockResolvedValue({ ok: true, status: 200 });
      expect(await checkUsernameAvailable('newuser')).toBe('available');
    });

    it('should return "taken" on 409', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 409 });
      expect(await checkUsernameAvailable('alice')).toBe('taken');
    });

    it('should return "invalid" on 400', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 400 });
      expect(await checkUsernameAvailable('a')).toBe('invalid');
    });

    it('should return "error" on other status codes', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 500 });
      expect(await checkUsernameAvailable('x')).toBe('error');
    });

    it('should return "error" on network failure', async () => {
      mockFetch.mockRejectedValue(new Error('network'));
      expect(await checkUsernameAvailable('y')).toBe('error');
    });
  });

  // --- startLogin ---

  describe('startLogin', () => {
    it('should return authentication options on success', async () => {
      const mockOptions = {
        challenge: 'ch123',
        rpId: 'localhost',
        challengeKey: 'ck2',
      };
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue(mockOptions),
      });

      const result = await startLogin('alice');
      expect(result).toEqual(mockOptions);
      expect(mockFetch).toHaveBeenCalledWith('/idp/passkey/login/options', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: 'alice' }),
      });
    });

    it('should pass undefined username when not provided', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: vi.fn().mockResolvedValue({}),
      });

      await startLogin();
      expect(mockFetch).toHaveBeenCalledWith('/idp/passkey/login/options', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: undefined }),
      });
    });

    it('should throw on 405 (passkey login not enabled)', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 405 });
      await expect(startLogin()).rejects.toThrow('not enabled');
    });

    it('should throw on other error status', async () => {
      mockFetch.mockResolvedValue({ ok: false, status: 503 });
      await expect(startLogin()).rejects.toThrow('Failed to get login options (503)');
    });
  });

  // --- downloadKeyBackup ---

  describe('downloadKeyBackup', () => {
    it('should create and trigger a download link for PRF-derived key', () => {
      const mockClick = vi.fn();
      const mockAppendChild = vi.fn();
      const mockRemoveChild = vi.fn();
      const mockCreateElement = vi.fn(() => ({
        href: '',
        download: '',
        click: mockClick,
      }));
      const mockCreateObjectURL = vi.fn(() => 'blob:url');
      const mockRevokeObjectURL = vi.fn();

      vi.stubGlobal('document', {
        createElement: mockCreateElement,
        body: {
          appendChild: mockAppendChild,
          removeChild: mockRemoveChild,
        },
      });
      vi.stubGlobal('URL', {
        createObjectURL: mockCreateObjectURL,
        revokeObjectURL: mockRevokeObjectURL,
      });
      vi.stubGlobal('Blob', class {
        constructor(public content: string[], public options: any) {}
      });

      downloadKeyBackup('alice', 'pubhex', 'privhex', true);

      expect(mockCreateElement).toHaveBeenCalledWith('a');
      expect(mockClick).toHaveBeenCalled();
      expect(mockRevokeObjectURL).toHaveBeenCalledWith('blob:url');
    });

    it('should include warning about unique copy for non-PRF keys', () => {
      const capturedContent: string[] = [];
      vi.stubGlobal('Blob', class {
        constructor(public content: string[], public options: any) {
          capturedContent.push(...content);
        }
      });
      vi.stubGlobal('document', {
        createElement: vi.fn(() => ({ href: '', download: '', click: vi.fn() })),
        body: { appendChild: vi.fn(), removeChild: vi.fn() },
      });
      vi.stubGlobal('URL', {
        createObjectURL: vi.fn(() => 'blob:x'),
        revokeObjectURL: vi.fn(),
      });

      downloadKeyBackup('bob', 'pub', 'priv', false);

      const text = capturedContent.join('');
      expect(text).toContain('ONLY copy');
      expect(text).toContain('Random');
    });
  });
});
