import { createLogger } from '../utils/loggerConfig';
import { createErrorMetadata } from '../utils/loggerConfig';
import { nip19 } from 'nostr-tools';
import { getPublicKey, finalizeEvent } from 'nostr-tools/pure';
import type {} from '../types/nip07';

const logger = createLogger('NostrAuthService');

// --- Module-scoped key storage ---
// Private key held in memory only, never persisted to sessionStorage.
// This reduces the attack surface vs sessionStorage (which is queryable by
// any same-origin JS).  nostr-tools still needs raw bytes for signing, so
// this is the best we can do without a native secp256k1 WebCrypto curve.
let _localKeyHex: string | null = null;

/**
 * Store a hex-encoded Nostr private key in the module-scoped closure.
 * Call this instead of writing the key to sessionStorage.
 */
export function setLocalKey(hexKey: string): void {
  _localKeyHex = hexKey;
}

/**
 * Wipe the module-scoped private key and remove any legacy sessionStorage
 * entries that may still exist from older code paths.
 */
export function clearLocalKey(): void {
  _localKeyHex = null;
  try {
    sessionStorage.removeItem('nostr_passkey_key');
    sessionStorage.removeItem('nostr_privkey');
  } catch { /* sessionStorage may be unavailable */ }
}

// Clear key material when the tab / window is closed
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    _localKeyHex = null;
  });
}

// --- Interfaces ---

// User info stored locally and used in AuthState
export interface SimpleNostrUser {
  pubkey: string;
  npub?: string;
  isPowerUser: boolean;
}

// User info returned by backend (kept for backward compat; remove in Phase 2)
export interface BackendNostrUser {
  pubkey: string;
  npub?: string;
  isPowerUser: boolean;
}

// Legacy interfaces (kept for backward compat; remove in Phase 2)
export interface AuthResponse {
  user: BackendNostrUser;
  token: string;
  expiresAt: number;
  features?: string[];
}

export interface VerifyResponse {
  valid: boolean;
  user?: BackendNostrUser;
  features?: string[];
}

export interface AuthEventPayload {
  id: string;
  pubkey: string;
  content: string;
  sig: string;
  created_at: number;
  kind: number;
  tags: string[][];
}

// State exposed to the application
export interface AuthState {
  authenticated: boolean;
  user?: SimpleNostrUser;
  error?: string;
}

type AuthStateListener = (state: AuthState) => void;

// --- Service Implementation ---

class NostrAuthService {
  private static instance: NostrAuthService;
  private currentUser: SimpleNostrUser | null = null;
  private localPrivateKey: Uint8Array | null = null;
  private authStateListeners: AuthStateListener[] = [];
  private initialized = false;

  private constructor() {}

  public static getInstance(): NostrAuthService {
    if (!NostrAuthService.instance) {
      NostrAuthService.instance = new NostrAuthService();
    }
    return NostrAuthService.instance;
  }

  public hasNip07Provider(): boolean {
    return typeof window !== 'undefined' && window.nostr !== undefined;
  }

  /** Check if running in dev mode with auth bypass */
  public isDevMode(): boolean {
    return import.meta.env.DEV && import.meta.env.VITE_DEV_MODE_AUTH === 'true';
  }

  /**
   * Sign an HTTP request using NIP-98 (kind 27235).
   * Returns base64-encoded signed event for the Authorization header.
   * Prefers local passkey-derived key, falls back to NIP-07 extension.
   */
  public async signRequest(url: string, method: string, body?: string): Promise<string> {
    // Prefer local key (passkey-derived) over NIP-07 extension
    if (this.localPrivateKey) {
      return this.signWithLocalKey(url, method, body);
    }

    if (!this.hasNip07Provider()) {
      throw new Error('No signing method available (no passkey session or NIP-07 provider)');
    }

    const tags: string[][] = [
      ['u', url],
      ['method', method.toUpperCase()],
    ];

    if (body) {
      const encoder = new TextEncoder();
      const data = encoder.encode(body);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      tags.push(['payload', hashHex]);
    }

    const unsignedEvent = {
      created_at: Math.floor(Date.now() / 1000),
      kind: 27235,
      tags,
      content: '',
    };

    const signedEvent = await window.nostr!.signEvent(unsignedEvent);
    const eventJson = JSON.stringify(signedEvent);
    return btoa(eventJson);
  }

  /**
   * Sign a NIP-98 request using the local passkey-derived private key.
   */
  public async signWithLocalKey(url: string, method: string, body?: string): Promise<string> {
    if (!this.localPrivateKey) {
      throw new Error('No local private key available');
    }

    const tags: string[][] = [
      ['u', url],
      ['method', method.toUpperCase()],
    ];

    if (body) {
      const encoder = new TextEncoder();
      const data = encoder.encode(body);
      const hashBuffer = await crypto.subtle.digest('SHA-256', data);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
      tags.push(['payload', hashHex]);
    }

    const eventTemplate = {
      created_at: Math.floor(Date.now() / 1000),
      kind: 27235,
      tags,
      content: '',
    };

    const signedEvent = finalizeEvent(eventTemplate, this.localPrivateKey);
    const eventJson = JSON.stringify(signedEvent);
    return btoa(eventJson);
  }

  public async initialize(): Promise<void> {
    if (this.initialized) return;
    logger.debug('Initializing NostrAuthService...');

    // DEV MODE: Auto-login as power user
    if (this.isDevMode()) {
      logger.info('[DEV MODE] Auto-authenticating as power user');
      const devPowerUserPubkey = import.meta.env.VITE_DEV_POWER_USER_PUBKEY || 'bfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a452';
      this.currentUser = {
        pubkey: devPowerUserPubkey,
        npub: this.hexToNpub(devPowerUserPubkey),
        isPowerUser: true,
      };
      this.initialized = true;
      this.notifyListeners(this.getCurrentAuthState());
      logger.info(`[DEV MODE] Authenticated as power user: ${devPowerUserPubkey}`);
      return;
    }

    // Restore cached user from localStorage (no server verification — NIP-98 is per-request)
    const storedUserJson = localStorage.getItem('nostr_user');
    if (storedUserJson) {
      try {
        this.currentUser = JSON.parse(storedUserJson);
        logger.info(`Restored user from localStorage: ${this.currentUser?.pubkey}`);
      } catch (e) {
        logger.error('Failed to parse stored user data:', createErrorMetadata(e));
        localStorage.removeItem('nostr_user');
      }
    } else {
      logger.info('No stored session found.');
    }

    // Restore passkey session from sessionStorage (per-tab, survives page reload)
    this.restorePasskeySession();

    this.initialized = true;
    this.notifyListeners(this.getCurrentAuthState());
    logger.debug('NostrAuthService initialized.');
  }

  public async login(): Promise<AuthState> {
    logger.info('Attempting NIP-07 login...');
    if (!this.hasNip07Provider()) {
      const errorMsg = 'Nostr NIP-07 provider (e.g., Alby) not found. Please install a compatible extension.';
      logger.error(errorMsg);
      this.notifyListeners({ authenticated: false, error: errorMsg });
      throw new Error(errorMsg);
    }

    try {
      const pubkey = await window.nostr!.getPublicKey();
      if (!pubkey) {
        throw new Error('Could not get public key from NIP-07 provider.');
      }
      logger.info(`Got pubkey via NIP-07: ${pubkey}`);

      this.currentUser = {
        pubkey,
        npub: this.hexToNpub(pubkey),
        isPowerUser: false, // Server determines this per-request from power user list
      };

      this.storeCurrentUser();
      const newState = this.getCurrentAuthState();
      this.notifyListeners(newState);
      return newState;
    } catch (error: any) {
      let errorMessage = 'Login failed';
      if (error?.message?.includes('User rejected') || error?.message?.includes('extension rejected')) {
        errorMessage = 'Login request rejected in Nostr extension.';
      } else if (error?.message) {
        errorMessage = error.message;
      }
      const errorState: AuthState = { authenticated: false, error: errorMessage };
      this.notifyListeners(errorState);
      throw new Error(errorMessage);
    }
  }

  public async logout(): Promise<void> {
    logger.info('Logging out...');
    this.clearSession();
    this.notifyListeners({ authenticated: false });
  }

  /** @deprecated No session token in NIP-98 mode. Returns null. */
  public getSessionToken(): string | null {
    return null;
  }

  private storeCurrentUser(): void {
    if (this.currentUser) {
      localStorage.setItem('nostr_user', JSON.stringify(this.currentUser));
    } else {
      localStorage.removeItem('nostr_user');
    }
  }

  private clearSession(): void {
    this.currentUser = null;
    this.localPrivateKey = null;
    _localKeyHex = null;
    localStorage.removeItem('nostr_user');
    // Clean up legacy key if present
    localStorage.removeItem('nostr_session_token');
    // Clear passkey session (private key should already be absent, but belt-and-suspenders)
    try {
      sessionStorage.removeItem('nostr_passkey_key');
      sessionStorage.removeItem('nostr_privkey');
      sessionStorage.removeItem('nostr_passkey_pubkey');
      sessionStorage.removeItem('nostr_prf');
    } catch { /* sessionStorage may be unavailable */ }
  }

  public onAuthStateChanged(listener: AuthStateListener): () => void {
    this.authStateListeners.push(listener);
    if (this.initialized) {
      listener(this.getCurrentAuthState());
    }
    return () => {
      this.authStateListeners = this.authStateListeners.filter(l => l !== listener);
    };
  }

  private notifyListeners(state: AuthState): void {
    this.authStateListeners.forEach(listener => {
      try {
        listener(state);
      } catch (error) {
        logger.error('Error in auth state listener:', createErrorMetadata(error));
      }
    });
  }

  public getCurrentUser(): SimpleNostrUser | null {
    return this.currentUser;
  }

  public isAuthenticated(): boolean {
    return !!this.currentUser && (this.hasNip07Provider() || this.isDevMode() || this.localPrivateKey !== null);
  }

  public getCurrentAuthState(): AuthState {
    return {
      authenticated: this.isAuthenticated(),
      user: this.currentUser ? { ...this.currentUser } : undefined,
      error: undefined
    };
  }

  public hexToNpub(pubkey: string): string | undefined {
    if (!pubkey) return undefined;
    try {
      return nip19.npubEncode(pubkey);
    } catch (error) {
      logger.warn(`Failed to convert hex to npub: ${pubkey}`, createErrorMetadata(error));
      return undefined;
    }
  }

  public npubToHex(npub: string): string | undefined {
    if (!npub) return undefined;
    try {
      const decoded = nip19.decode(npub);
      if (decoded.type === 'npub') {
        return decoded.data;
      }
      throw new Error('Invalid npub format');
    } catch (error) {
      logger.warn(`Failed to convert npub to hex: ${npub}`, createErrorMetadata(error));
      return undefined;
    }
  }

  /**
   * Login using a passkey-derived private key.
   * Key is held in module-scoped memory only -- never persisted to sessionStorage.
   */
  public async loginWithPasskey(pubkey: string, privateKey: Uint8Array): Promise<AuthState> {
    logger.info('Passkey login...');
    this.localPrivateKey = privateKey;

    // Keep hex form in module closure for restorePasskeySession fallback
    const hexKey = Array.from(privateKey).map(b => b.toString(16).padStart(2, '0')).join('');
    _localKeyHex = hexKey;

    this.currentUser = {
      pubkey,
      npub: this.hexToNpub(pubkey),
      isPowerUser: false,
    };

    // Persist user in localStorage for cross-reload
    this.storeCurrentUser();
    // Only store pubkey and PRF flag -- NEVER store the private key
    try {
      sessionStorage.setItem('nostr_passkey_pubkey', pubkey);
    } catch { /* sessionStorage unavailable */ }

    const newState = this.getCurrentAuthState();
    this.notifyListeners(newState);
    logger.info(`Passkey login complete: ${pubkey}`);
    return newState;
  }

  /** Check if a passkey session exists (in-memory key or legacy sessionStorage) */
  public hasPasskeySession(): boolean {
    if (_localKeyHex) return true;
    if (this.localPrivateKey) return true;
    try {
      return !!sessionStorage.getItem('nostr_passkey_key');
    } catch {
      return false;
    }
  }

  /**
   * Restore passkey-derived key from the module-scoped variable or, as a
   * one-time migration, from legacy sessionStorage.  Legacy entries are
   * deleted immediately after migration so the private key is never left
   * in queryable browser storage.
   */
  public restorePasskeySession(): void {
    try {
      // Prefer the in-memory module-scoped key
      let hexKey = _localKeyHex;
      const pubkey = sessionStorage.getItem('nostr_passkey_pubkey');

      // Legacy migration: if sessionStorage still has the private key, ingest
      // it into memory and wipe the storage entry.
      if (!hexKey) {
        const legacyKey = sessionStorage.getItem('nostr_passkey_key');
        if (legacyKey) {
          hexKey = legacyKey;
          _localKeyHex = legacyKey;
          // Immediately remove legacy plaintext key from sessionStorage
          sessionStorage.removeItem('nostr_passkey_key');
          logger.info('Migrated legacy passkey key from sessionStorage to memory');
        }
      }
      // Also clean nostr_privkey if present (older legacy path)
      sessionStorage.removeItem('nostr_privkey');

      if (hexKey && pubkey) {
        this.localPrivateKey = new Uint8Array(
          hexKey.match(/.{1,2}/g)!.map(byte => parseInt(byte, 16))
        );
        // Verify the key matches by deriving pubkey
        const derivedPubkey = getPublicKey(this.localPrivateKey);
        if (derivedPubkey !== pubkey) {
          logger.warn('Passkey session pubkey mismatch, clearing');
          this.localPrivateKey = null;
          _localKeyHex = null;
          sessionStorage.removeItem('nostr_passkey_pubkey');
          return;
        }
        // Set current user if not already set from localStorage
        if (!this.currentUser) {
          this.currentUser = {
            pubkey,
            npub: this.hexToNpub(pubkey),
            isPowerUser: false,
          };
        }
        logger.info(`Restored passkey session: ${pubkey}`);
      }
    } catch (e) {
      logger.warn('Failed to restore passkey session:', createErrorMetadata(e));
    }
  }

  /**
   * Dev mode login - bypasses NIP-07 and logs in as power user
   * Only available in development mode on local network
   */
  public async devLogin(): Promise<AuthState> {
    if (!import.meta.env.DEV) {
      throw new Error('Dev login is only available in development mode');
    }

    const hostname = window.location.hostname;
    const isLocalNetwork =
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname.startsWith('192.168.') ||
      hostname.startsWith('10.') ||
      hostname.startsWith('172.16.') ||
      hostname.startsWith('172.17.') ||
      hostname.startsWith('172.18.') ||
      hostname.startsWith('172.19.') ||
      hostname.startsWith('172.2') ||
      hostname.startsWith('172.30.') ||
      hostname.startsWith('172.31.');

    if (!isLocalNetwork) {
      throw new Error('Dev login is only available on local network');
    }

    logger.info('[DEV MODE] Manual dev login triggered');
    const devPowerUserPubkey = import.meta.env.VITE_DEV_POWER_USER_PUBKEY ||
      'bfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a452';

    this.currentUser = {
      pubkey: devPowerUserPubkey,
      npub: this.hexToNpub(devPowerUserPubkey),
      isPowerUser: true,
    };

    this.storeCurrentUser();
    const newState = this.getCurrentAuthState();
    this.notifyListeners(newState);
    logger.info(`[DEV MODE] Logged in as power user: ${devPowerUserPubkey}`);
    return newState;
  }

  public isDevLoginAvailable(): boolean {
    if (!import.meta.env.DEV) return false;
    const hostname = window.location.hostname;
    return (
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname.startsWith('192.168.') ||
      hostname.startsWith('10.') ||
      hostname.startsWith('172.')
    );
  }
}

// Export a singleton instance
export const nostrAuth = NostrAuthService.getInstance();
