import { unifiedApiClient } from './api/UnifiedApiClient';
import { createLogger } from '../utils/loggerConfig';
import { createErrorMetadata } from '../utils/loggerConfig';
import { Event, UnsignedEvent, nip19 } from 'nostr-tools';
import { v4 as uuidv4 } from 'uuid'; 
import type {} from '../types/nip07'; 

const logger = createLogger('NostrAuthService');

// --- Interfaces ---

// User info stored locally and used in AuthState
export interface SimpleNostrUser {
  pubkey: string; 
  npub?: string; 
  isPowerUser: boolean; 
}

// User info returned by backend
export interface BackendNostrUser {
  pubkey: string;
  npub?: string;
  isPowerUser: boolean; 
  
}

// Response from POST /auth/nostr
export interface AuthResponse {
  user: BackendNostrUser;
  token: string;
  expiresAt: number; 
  features?: string[]; 
}

// Response from POST /auth/nostr/verify
export interface VerifyResponse {
  valid: boolean;
  user?: BackendNostrUser;
  features?: string[];
}

// Payload for POST /auth/nostr (signed NIP-42 event)
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
  private sessionToken: string | null = null;
  private currentUser: SimpleNostrUser | null = null;
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

  
  public async initialize(): Promise<void> {
    if (this.initialized) return;
    logger.debug('Initializing NostrAuthService...');

    // DEV MODE: Auto-login as power user when in development
    // SECURITY: Controlled via VITE_DEV_MODE_AUTH env var (default: false)
    const isDev = import.meta.env.DEV;
    const devModeAuthEnabled = import.meta.env.VITE_DEV_MODE_AUTH === 'true';
    if (isDev && devModeAuthEnabled) {
      logger.info('[DEV MODE] Auto-authenticating as power user');
      const devPowerUserPubkey = import.meta.env.VITE_DEV_POWER_USER_PUBKEY || 'bfcf20d472f0fb143b23cb5be3fa0a040d42176b71f73ca272f6912b1d62a452';
      this.sessionToken = 'dev-session-token';
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

    const storedToken = localStorage.getItem('nostr_session_token');
    const storedUserJson = localStorage.getItem('nostr_user');

    if (storedToken && storedUserJson) {
      let storedUser: SimpleNostrUser | null = null;
      try {
        storedUser = JSON.parse(storedUserJson);
      } catch (parseError) {
        logger.error('Failed to parse stored user data:', createErrorMetadata(parseError));
        this.clearSession();
      }

      if (storedUser) {
        logger.info(`Verifying stored session for pubkey: ${storedUser.pubkey}`);
        try {
          
          const verificationResponse = await unifiedApiClient.postData<VerifyResponse>('/auth/nostr/verify', {
            pubkey: storedUser.pubkey,
            token: storedToken
          });

          if (verificationResponse.valid) { 
            this.sessionToken = storedToken;
            if (verificationResponse.user) { 
              this.currentUser = {
                pubkey: verificationResponse.user.pubkey,
                npub: verificationResponse.user.npub || this.hexToNpub(verificationResponse.user.pubkey),
                isPowerUser: verificationResponse.user.isPowerUser,
              };
              logger.info('Token verified and user details updated from backend.');
            } else if (storedUser) {
              
              
              this.currentUser = storedUser; 
              logger.info('Token verified, using stored user details as backend did not provide them on verify.');
            } else {
              
              logger.error('Token verified but no user details available from backend or local storage. Clearing session.');
              this.clearSession();
              this.notifyListeners({ authenticated: false, error: 'User details missing after verification' });
              return; 
            }
            this.storeCurrentUser(); 
            this.notifyListeners(this.getCurrentAuthState());
            logger.info('Restored and verified session from local storage.');
          } else {
            
            logger.warn('Stored session token is invalid (verification failed), clearing session.');
            this.clearSession();
            this.notifyListeners({ authenticated: false });
          }
        } catch (error) {
          logger.error('Failed to verify stored session with backend:', createErrorMetadata(error));
          this.clearSession();
          this.notifyListeners({ authenticated: false, error: 'Session verification failed' });
        }
      }
    } else {
      logger.info('No stored session found.');
      this.notifyListeners({ authenticated: false });
    }
    this.initialized = true;
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

      
      const challenge = uuidv4(); 
      
      const relayUrl = 'wss://relay.damus.io';

      
      const unsignedNip07Event = {
        created_at: Math.floor(Date.now() / 1000),
        kind: 22242,
        tags: [
          ['relay', relayUrl],
          ['challenge', challenge]
        ],
        content: 'Authenticate to LogseqSpringThing' 
      };

      
      logger.debug('Requesting signature via NIP-07 for event:', unsignedNip07Event);
      const signedEvent: Event = await window.nostr!.signEvent(unsignedNip07Event);
      logger.debug('Event signed successfully via NIP-07.');

      
      const eventPayload: AuthEventPayload = {
        id: signedEvent.id,
        pubkey: signedEvent.pubkey, 
        content: signedEvent.content,
        sig: signedEvent.sig,
        created_at: signedEvent.created_at,
        kind: signedEvent.kind,
        tags: signedEvent.tags,
      };

      
      logger.info(`Sending auth event to backend for pubkey: ${pubkey}`);
      const response = await unifiedApiClient.postData<AuthResponse>('/auth/nostr', eventPayload);
      logger.info(`Backend auth successful for pubkey: ${response.user.pubkey}`);

      
      this.sessionToken = response.token;
      this.currentUser = {
        pubkey: response.user.pubkey,
        npub: response.user.npub || this.hexToNpub(response.user.pubkey),
        isPowerUser: response.user.isPowerUser,
      };

      this.storeSessionToken(response.token);
      this.storeCurrentUser(); 

      const newState = this.getCurrentAuthState();
      this.notifyListeners(newState);
      return newState;

    } catch (error: any) {
      const errorMeta = createErrorMetadata(error);
      logger.error(`NIP-07 login failed. Details: ${JSON.stringify(errorMeta, null, 2)}`);
      let errorMessage = 'Login failed';
      if (error?.response?.data?.error) { 
        errorMessage = error.response.data.error;
      } else if (error?.message) {
        errorMessage = error.message;
      } else if (typeof error === 'string') {
        errorMessage = error;
      }

      
      if (errorMessage.includes('User rejected') || errorMessage.includes('extension rejected')) {
        errorMessage = 'Login request rejected in Nostr extension.';
      } else if (errorMessage.includes('401') || errorMessage.includes('Invalid signature')) {
        errorMessage = 'Authentication failed: Invalid signature or credentials.';
      } else if (errorMessage.includes('Could not get public key')) {
        errorMessage = 'Failed to get public key from Nostr extension.';
      }

      const errorState: AuthState = { authenticated: false, error: errorMessage };
      this.notifyListeners(errorState);
      
      throw new Error(errorMessage);
    }
  }

  
  public async logout(): Promise<void> {
    logger.info('Attempting logout...');
    const token = this.sessionToken;
    const user = this.currentUser;

    
    const wasAuthenticated = this.isAuthenticated();
    this.clearSession();
    if (wasAuthenticated) {
        this.notifyListeners({ authenticated: false }); 
    }


    if (token && user) {
      try {
        logger.info(`Calling server logout for pubkey: ${user.pubkey}`);
        
        await unifiedApiClient.request<any>('DELETE', '/auth/nostr', {
          pubkey: user.pubkey,
          token: token
        });
        logger.info('Server logout successful.');
      } catch (error) {
        
        logger.error('Server logout call failed:', createErrorMetadata(error));
        
        
      }
    } else {
      logger.warn('Logout called but no active session found locally.');
    }
  }

  

  private storeSessionToken(token: string): void {
    localStorage.setItem('nostr_session_token', token);
  }

  private storeCurrentUser(): void {
    if (this.currentUser) {
      localStorage.setItem('nostr_user', JSON.stringify(this.currentUser));
    } else {
      localStorage.removeItem('nostr_user');
    }
  }

  private clearSession(): void {
    this.sessionToken = null;
    this.currentUser = null;
    localStorage.removeItem('nostr_session_token');
    localStorage.removeItem('nostr_user');
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

  public getSessionToken(): string | null {
    return this.sessionToken;
  }

  public isAuthenticated(): boolean {
    return !!this.sessionToken && !!this.currentUser;
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
   * Dev mode login - bypasses NIP-07 and logs in as power user
   * Only available in development mode on local network
   */
  public async devLogin(): Promise<AuthState> {
    // Security: Only allow in dev mode
    if (!import.meta.env.DEV) {
      throw new Error('Dev login is only available in development mode');
    }

    // Security: Only allow from local network IPs
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

    this.sessionToken = 'dev-session-token';
    this.currentUser = {
      pubkey: devPowerUserPubkey,
      npub: this.hexToNpub(devPowerUserPubkey),
      isPowerUser: true,
    };

    // Store in localStorage for persistence
    this.storeSessionToken(this.sessionToken);
    this.storeCurrentUser();

    const newState = this.getCurrentAuthState();
    this.notifyListeners(newState);
    logger.info(`[DEV MODE] Logged in as power user: ${devPowerUserPubkey}`);
    return newState;
  }

  /**
   * Check if dev login button should be shown
   */
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
