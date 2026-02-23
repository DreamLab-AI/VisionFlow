import { useState, useEffect } from 'react';
import { nostrAuth, AuthState } from '../services/nostrAuthService';
import { createLogger } from '../utils/loggerConfig';

const logger = createLogger('useNostrAuth');

export function useNostrAuth() {
  const [authState, setAuthState] = useState<AuthState>({
    authenticated: false,
    user: undefined,
    error: undefined
  });
  const [isLoading, setIsLoading] = useState(true);
  const [hasNip07, setHasNip07] = useState(() => nostrAuth.hasNip07Provider());

  useEffect(() => {
    let cancelled = false;

    // Initialize auth service
    const initAuth = async () => {
      try {
        await nostrAuth.initialize();
        const currentState = nostrAuth.getCurrentAuthState();
        if (!cancelled) setAuthState(currentState);
      } catch (error) {
        logger.error('Failed to initialize auth:', error);
        if (!cancelled) {
          setAuthState({
            authenticated: false,
            error: error instanceof Error ? error.message : 'Authentication initialization failed'
          });
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    initAuth();

    // Subscribe to auth state changes (fires when extension detected or stale session cleared)
    const unsubscribe = nostrAuth.onAuthStateChanged((newState) => {
      if (!cancelled) {
        setAuthState(newState);
        setIsLoading(false);
        // Re-check NIP-07 availability whenever auth state changes
        setHasNip07(nostrAuth.hasNip07Provider());
      }
    });

    // If NIP-07 is not yet available, wait for it reactively
    if (!nostrAuth.hasNip07Provider()) {
      nostrAuth.waitForNip07Provider(10000).then((detected) => {
        if (!cancelled && detected) {
          logger.info('NIP-07 extension detected — updating UI');
          setHasNip07(true);
        }
      });
    }

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const login = async () => {
    try {
      const newState = await nostrAuth.login();
      setAuthState(newState);
      return newState;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Login failed';
      setAuthState({
        authenticated: false,
        error: errorMessage
      });
      throw error;
    }
  };

  const logout = async () => {
    try {
      await nostrAuth.logout();
      setAuthState({
        authenticated: false
      });
    } catch (error) {
      logger.error('Logout error:', error);
    }
  };

  const devLogin = async () => {
    try {
      const newState = await nostrAuth.devLogin();
      setAuthState(newState);
      return newState;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Dev login failed';
      setAuthState({
        authenticated: false,
        error: errorMessage
      });
      throw error;
    }
  };

  const loginWithPasskey = async (pubkey: string, privateKey: Uint8Array) => {
    try {
      const newState = await nostrAuth.loginWithPasskey(pubkey, privateKey);
      setAuthState(newState);
      return newState;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Passkey login failed';
      setAuthState({ authenticated: false, error: errorMessage });
      throw error;
    }
  };

  return {
    ...authState,
    isLoading,
    login,
    logout,
    devLogin,
    loginWithPasskey,
    hasNip07,
    hasPasskeySession: nostrAuth.hasPasskeySession(),
    isDevLoginAvailable: nostrAuth.isDevLoginAvailable()
  };
}
