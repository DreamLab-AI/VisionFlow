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

  useEffect(() => {
    // Initialize auth service
    const initAuth = async () => {
      try {
        await nostrAuth.initialize();
        const currentState = nostrAuth.getCurrentAuthState();
        setAuthState(currentState);
      } catch (error) {
        logger.error('Failed to initialize auth:', error);
        setAuthState({
          authenticated: false,
          error: error instanceof Error ? error.message : 'Authentication initialization failed'
        });
      } finally {
        setIsLoading(false);
      }
    };

    initAuth();

    // Subscribe to auth state changes
    const unsubscribe = nostrAuth.onAuthStateChanged((newState) => {
      setAuthState(newState);
      setIsLoading(false);
    });

    return () => {
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
    hasNip07: nostrAuth.hasNip07Provider(),
    hasPasskeySession: nostrAuth.hasPasskeySession(),
    isDevLoginAvailable: nostrAuth.isDevLoginAvailable()
  };
}
