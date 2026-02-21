import React, { useState } from 'react';
import { useNostrAuth } from '../hooks/useNostrAuth';
import './NostrLoginScreen.css';

export const NostrLoginScreen: React.FC = () => {
  const { login, devLogin, hasNip07, error, isDevLoginAvailable } = useNostrAuth();
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [isDevLoggingIn, setIsDevLoggingIn] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  const handleLogin = async () => {
    setIsLoggingIn(true);
    setLoginError(null);

    try {
      await login();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Login failed';
      setLoginError(errorMessage);
    } finally {
      setIsLoggingIn(false);
    }
  };

  const handleDevLogin = async () => {
    setIsDevLoggingIn(true);
    setLoginError(null);

    try {
      await devLogin();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Dev login failed';
      setLoginError(errorMessage);
    } finally {
      setIsDevLoggingIn(false);
    }
  };

  const handlePasskeyLogin = () => {
    // Redirect to the IdP login page which supports passkey auth
    window.location.href = '/idp/register';
  };

  return (
    <div className="nostr-login-screen">
      <div className="nostr-login-container">
        <div className="nostr-login-header">
          <h1>VisionFlow</h1>
          <p className="nostr-login-subtitle">Sign in to continue</p>
        </div>

        <div className="nostr-login-content">
          <div className="nostr-login-form">
            <div className="nostr-login-icon">🔐</div>

            {(loginError || error) && (
              <div className="nostr-login-error-message">
                <strong>Error:</strong> {loginError || error}
              </div>
            )}

            {/* Primary: Passkey authentication */}
            <button
              className="nostr-login-button passkey-button"
              onClick={handlePasskeyLogin}
              disabled={isLoggingIn || isDevLoggingIn}
            >
              🔑 Sign in with Passkey
            </button>

            <p className="nostr-login-description">
              Use your device&apos;s passkey for passwordless authentication.
              Your Nostr identity is derived automatically.
            </p>

            {/* Secondary: PodKey extension (NIP-07) */}
            {hasNip07 && (
              <>
                <div className="auth-divider"><span>or</span></div>
                <button
                  className="nostr-login-button podkey-button"
                  onClick={handleLogin}
                  disabled={isLoggingIn || isDevLoggingIn}
                >
                  {isLoggingIn ? (
                    <>
                      <span className="spinner"></span>
                      Authenticating...
                    </>
                  ) : (
                    'Sign in with PodKey'
                  )}
                </button>
              </>
            )}

            {!hasNip07 && (
              <div className="podkey-hint">
                <p>
                  Have a PodKey extension?{' '}
                  <a href="#" onClick={(e) => { e.preventDefault(); window.location.reload(); }}>
                    Reload to detect it
                  </a>
                </p>
              </div>
            )}

            {isDevLoginAvailable && (
              <div className="dev-login-section">
                <div className="dev-login-divider">
                  <span>Development Mode</span>
                </div>
                <button
                  className="dev-login-button"
                  onClick={handleDevLogin}
                  disabled={isLoggingIn || isDevLoggingIn}
                >
                  {isDevLoggingIn ? (
                    <>
                      <span className="spinner"></span>
                      Logging in...
                    </>
                  ) : (
                    <>
                      <span className="dev-icon">🔧</span>
                      Dev Login (Bypass Auth)
                    </>
                  )}
                </button>
                <p className="dev-login-warning">
                  Development only - bypasses authentication
                </p>
              </div>
            )}

            <div className="nostr-login-info">
              <p>
                <strong>How it works</strong>
              </p>
              <p>
                VisionFlow uses passkeys and Nostr for decentralized identity.
                Your cryptographic keys are stored on your device — no passwords needed.
              </p>
            </div>
          </div>
        </div>

        <div className="nostr-login-footer">
          <p>
            New here?{' '}
            <a href="/idp/register">
              Create an account
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};
