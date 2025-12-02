import React, { useState } from 'react';
import { useNostrAuth } from '../hooks/useNostrAuth';
import './NostrLoginScreen.css';

export const NostrLoginScreen: React.FC = () => {
  const { login, hasNip07, error } = useNostrAuth();
  const [isLoggingIn, setIsLoggingIn] = useState(false);
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

  return (
    <div className="nostr-login-screen">
      <div className="nostr-login-container">
        <div className="nostr-login-header">
          <h1>VisionFlow</h1>
          <p className="nostr-login-subtitle">Nostr Authentication Required</p>
        </div>

        <div className="nostr-login-content">
          {!hasNip07 ? (
            <div className="nostr-login-error">
              <div className="error-icon">⚠️</div>
              <h2>Nostr Extension Required</h2>
              <p>
                VisionFlow requires a Nostr NIP-07 compatible browser extension to authenticate.
              </p>
              <div className="extension-recommendations">
                <h3>Recommended Extensions:</h3>
                <ul>
                  <li>
                    <a href="https://getalby.com" target="_blank" rel="noopener noreferrer">
                      Alby
                    </a>
                  </li>
                  <li>
                    <a href="https://chrome.google.com/webstore/detail/nos2x" target="_blank" rel="noopener noreferrer">
                      nos2x
                    </a>
                  </li>
                </ul>
              </div>
              <button
                className="nostr-login-retry-button"
                onClick={() => window.location.reload()}
              >
                Retry After Installing Extension
              </button>
            </div>
          ) : (
            <div className="nostr-login-form">
              <div className="nostr-login-icon">🔐</div>
              <p className="nostr-login-description">
                Click the button below to authenticate with your Nostr identity.
                Your browser extension will prompt you to sign a challenge.
              </p>

              {(loginError || error) && (
                <div className="nostr-login-error-message">
                  <strong>Error:</strong> {loginError || error}
                </div>
              )}

              <button
                className="nostr-login-button"
                onClick={handleLogin}
                disabled={isLoggingIn}
              >
                {isLoggingIn ? (
                  <>
                    <span className="spinner"></span>
                    Authenticating...
                  </>
                ) : (
                  'Login with Nostr'
                )}
              </button>

              <div className="nostr-login-info">
                <p>
                  <strong>What is Nostr?</strong>
                </p>
                <p>
                  Nostr is a decentralized protocol for social networking and identity.
                  Your identity is controlled by you via cryptographic keys.
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="nostr-login-footer">
          <p>
            Need help? Visit our{' '}
            <a href="https://docs.visionflow.io/auth" target="_blank" rel="noopener noreferrer">
              documentation
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};
