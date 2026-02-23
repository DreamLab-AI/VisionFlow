import React, { useReducer, useRef, useCallback, useEffect, useState } from 'react';
import { useNostrAuth } from '../hooks/useNostrAuth';
import { nostrAuth, setLocalKey } from '../services/nostrAuthService';
import {
  startRegistration,
  createPasskeyCredential,
  verifyRegistration,
  deriveNostrKey,
  startLogin,
  authenticatePasskey,
  verifyLogin,
  checkUsernameAvailable,
  bytesToHex,
  downloadKeyBackup,
  type UsernameCheckResult,
} from '../services/passkeyService';
import { getPublicKey, generateSecretKey } from 'nostr-tools/pure';
import { nip19 } from 'nostr-tools';
import './OnboardingWizard.css';

// --- State Machine ---

type WizardStep = 'welcome' | 'username' | 'create-passkey' | 'identity-ready'
  | 'sign-in-method' | 'sign-in' | 'sign-in-extension' | 'error';

interface WizardState {
  step: WizardStep;
  prevStep: WizardStep | null;
  username: string;
  pubkey: string;
  privateKey: Uint8Array | null;
  webId: string;
  error: string;
}

type WizardAction =
  | { type: 'START_REGISTER' }
  | { type: 'START_LOGIN' }
  | { type: 'START_LOGIN_PASSKEY' }
  | { type: 'START_LOGIN_EXTENSION' }
  | { type: 'SET_USERNAME'; username: string }
  | { type: 'PASSKEY_CREATED'; pubkey: string; privateKey: Uint8Array; webId: string }
  | { type: 'LOGIN_SUCCESS'; pubkey: string; privateKey: Uint8Array | null }
  | { type: 'ERROR'; message: string }
  | { type: 'BACK' };

const initialState: WizardState = {
  step: 'welcome',
  prevStep: null,
  username: '',
  pubkey: '',
  privateKey: null,
  webId: '',
  error: '',
};

function wizardReducer(state: WizardState, action: WizardAction): WizardState {
  switch (action.type) {
    case 'START_REGISTER':
      return { ...state, step: 'username', prevStep: 'welcome' };
    case 'START_LOGIN':
      return { ...state, step: 'sign-in-method', prevStep: 'welcome' };
    case 'START_LOGIN_PASSKEY':
      return { ...state, step: 'sign-in', prevStep: 'sign-in-method' };
    case 'START_LOGIN_EXTENSION':
      return { ...state, step: 'sign-in-extension', prevStep: 'sign-in-method' };
    case 'SET_USERNAME':
      return { ...state, step: 'create-passkey', prevStep: 'username', username: action.username };
    case 'PASSKEY_CREATED':
      return {
        ...state,
        step: 'identity-ready',
        prevStep: 'create-passkey',
        pubkey: action.pubkey,
        privateKey: action.privateKey,
        webId: action.webId,
      };
    case 'LOGIN_SUCCESS':
      return { ...state, pubkey: action.pubkey, privateKey: action.privateKey };
    case 'ERROR':
      return { ...state, step: 'error', error: action.message };
    case 'BACK':
      return state.prevStep
        ? { ...state, step: state.prevStep, prevStep: null, error: '' }
        : { ...state, step: 'welcome', prevStep: null, error: '' };
    default:
      return state;
  }
}

// --- Step Indicator ---

const REGISTER_STEPS = ['welcome', 'username', 'create-passkey', 'identity-ready'] as const;

function StepIndicator({ currentStep }: { currentStep: WizardStep }) {
  const idx = REGISTER_STEPS.indexOf(currentStep as typeof REGISTER_STEPS[number]);
  if (idx < 0) return null;
  return (
    <div className="wizard-steps">
      {REGISTER_STEPS.map((_, i) => (
        <span
          key={i}
          className={`wizard-step-dot${i <= idx ? ' active' : ''}${i === idx ? ' current' : ''}`}
        />
      ))}
    </div>
  );
}

// --- Wizard Component ---

interface OnboardingWizardProps {
  onComplete: () => void;
}

export const OnboardingWizard: React.FC<OnboardingWizardProps> = ({ onComplete }) => {
  const [state, dispatch] = useReducer(wizardReducer, initialState);
  const { isDevLoginAvailable, devLogin, hasNip07, login: nip07Login } = useNostrAuth();

  const finishAuth = useCallback(async (pubkey: string, privateKey: Uint8Array | null) => {
    if (privateKey) {
      await nostrAuth.loginWithPasskey(pubkey, privateKey);
    }
    onComplete();
  }, [onComplete]);

  return (
    <div className="wizard-screen">
      <div className="wizard-container">
        <StepIndicator currentStep={state.step} />
        <div className="wizard-content">
          {state.step === 'welcome' && (
            <WelcomeStep
              onRegister={() => dispatch({ type: 'START_REGISTER' })}
              onLogin={() => dispatch({ type: 'START_LOGIN' })}
              isDevLoginAvailable={isDevLoginAvailable}
              onDevLogin={devLogin}
            />
          )}
          {state.step === 'username' && (
            <UsernameStep
              onNext={(username) => dispatch({ type: 'SET_USERNAME', username })}
              onBack={() => dispatch({ type: 'BACK' })}
            />
          )}
          {state.step === 'create-passkey' && (
            <CreatePasskeyStep
              username={state.username}
              onCreated={(pubkey, privateKey, webId) =>
                dispatch({ type: 'PASSKEY_CREATED', pubkey, privateKey, webId })
              }
              onError={(msg) => dispatch({ type: 'ERROR', message: msg })}
              onBack={() => dispatch({ type: 'BACK' })}
            />
          )}
          {state.step === 'identity-ready' && (
            <IdentityReadyStep
              username={state.username}
              pubkey={state.pubkey}
              webId={state.webId}
              onEnter={() => finishAuth(state.pubkey, state.privateKey)}
            />
          )}
          {state.step === 'sign-in-method' && (
            <SignInMethodStep
              onPasskey={() => dispatch({ type: 'START_LOGIN_PASSKEY' })}
              onExtension={() => dispatch({ type: 'START_LOGIN_EXTENSION' })}
              hasExtension={hasNip07}
              onBack={() => dispatch({ type: 'BACK' })}
              onPrivKeyLogin={(pubkey, privateKey) => finishAuth(pubkey, privateKey)}
            />
          )}
          {state.step === 'sign-in' && (
            <SignInStep
              onSuccess={(pubkey, privateKey) => finishAuth(pubkey, privateKey)}
              onError={(msg) => dispatch({ type: 'ERROR', message: msg })}
              onBack={() => dispatch({ type: 'BACK' })}
            />
          )}
          {state.step === 'sign-in-extension' && (
            <ExtensionLoginStep
              nip07Login={nip07Login}
              onSuccess={(pubkey) => finishAuth(pubkey, null)}
              onError={(msg) => dispatch({ type: 'ERROR', message: msg })}
              onBack={() => dispatch({ type: 'BACK' })}
            />
          )}
          {state.step === 'error' && (
            <ErrorStep
              message={state.error}
              onRetry={() => dispatch({ type: 'BACK' })}
            />
          )}
        </div>
      </div>
    </div>
  );
};

// --- Step: Welcome ---

function WelcomeStep({
  onRegister,
  onLogin,
  isDevLoginAvailable,
  onDevLogin,
}: {
  onRegister: () => void;
  onLogin: () => void;
  isDevLoginAvailable: boolean;
  onDevLogin: () => Promise<unknown>;
}) {
  const [devLoading, setDevLoading] = useState(false);

  const handleDevLogin = async () => {
    setDevLoading(true);
    try { await onDevLogin(); } catch { /* handled by hook */ }
    setDevLoading(false);
  };

  return (
    <div className="wizard-step">
      <div className="wizard-header">
        <h1>VisionFlow</h1>
        <p className="wizard-subtitle">See your knowledge in three dimensions.<br />Own it with decentralized identity.</p>
      </div>
      <div className="wizard-actions">
        <button className="wizard-btn wizard-btn-primary" onClick={onRegister}>
          Create Account
        </button>
        <button className="wizard-btn wizard-btn-secondary" onClick={onLogin}>
          Sign In
        </button>
      </div>
      <div className="wizard-features">
        <div className="wizard-feature-card">
          <div className="wizard-feature-icon">~</div>
          <div className="wizard-feature-title">3D Knowledge Graph</div>
          <div className="wizard-feature-desc">
            Navigate your ideas, notes, and connections in an interactive three-dimensional space.
          </div>
        </div>
        <div className="wizard-feature-card">
          <div className="wizard-feature-icon">&#x25A1;</div>
          <div className="wizard-feature-title">Your Data, Your Pod</div>
          <div className="wizard-feature-desc">
            Store everything in a Solid pod you control. No vendor lock-in, full portability.
          </div>
        </div>
        <div className="wizard-feature-card">
          <div className="wizard-feature-icon">&#x2731;</div>
          <div className="wizard-feature-title">Decentralized Identity</div>
          <div className="wizard-feature-desc">
            Cryptographic keys replace passwords. Sign in with a passkey — no extensions required.
          </div>
        </div>
      </div>
      <div className="wizard-info">
        <p>
          Passwordless authentication. Your cryptographic keys are generated
          and stored on-device — never sent to a server.
        </p>
      </div>
      {isDevLoginAvailable && (
        <div className="wizard-dev-section">
          <div className="wizard-dev-label">Development Mode</div>
          <button
            className="wizard-btn wizard-btn-dev"
            onClick={handleDevLogin}
            disabled={devLoading}
          >
            {devLoading ? 'Logging in...' : 'Dev Login (Bypass Auth)'}
          </button>
        </div>
      )}
    </div>
  );
}

// --- Step: Username ---

function UsernameStep({
  onNext,
  onBack,
}: {
  onNext: (username: string) => void;
  onBack: () => void;
}) {
  const [username, setUsername] = useState('');
  const [checking, setChecking] = useState(false);
  const [checkResult, setCheckResult] = useState<UsernameCheckResult | null>(null);
  const [validationError, setValidationError] = useState('');
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  const validate = (value: string): string => {
    if (value.length === 0) return '';
    if (value.length < 3) return 'At least 3 characters';
    if (!/^[a-z0-9]+$/.test(value)) return 'Lowercase letters and numbers only';
    return '';
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value.toLowerCase().replace(/[^a-z0-9]/g, '');
    setUsername(val);
    setCheckResult(null);
    const err = validate(val);
    setValidationError(err);

    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (!err && val.length >= 3) {
      setChecking(true);
      debounceRef.current = setTimeout(async () => {
        const result = await checkUsernameAvailable(val);
        setCheckResult(result);
        setChecking(false);
      }, 400);
    }
  };

  useEffect(() => {
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, []);

  const canProceed = username.length >= 3 && !validationError && checkResult === 'available' && !checking;

  return (
    <div className="wizard-step">
      <button className="wizard-back" onClick={onBack}>Back</button>
      <h2>Choose a username</h2>
      <p className="wizard-hint">This will be your pod address and public identity.</p>
      <div className="wizard-input-group">
        <input
          type="text"
          className="wizard-input"
          placeholder="username"
          value={username}
          onChange={handleChange}
          autoFocus
          maxLength={32}
          onKeyDown={(e) => { if (e.key === 'Enter' && canProceed) onNext(username); }}
        />
        <div className="wizard-input-status">
          {checking && <span className="status-checking">Checking...</span>}
          {!checking && checkResult === 'available' && <span className="status-available">Available</span>}
          {!checking && checkResult === 'taken' && <span className="status-taken">Taken</span>}
          {!checking && checkResult === 'invalid' && <span className="status-error">Invalid username</span>}
          {!checking && checkResult === 'error' && <span className="status-error">Could not reach server</span>}
          {validationError && <span className="status-error">{validationError}</span>}
        </div>
      </div>
      <button
        className="wizard-btn wizard-btn-primary"
        disabled={!canProceed}
        onClick={() => onNext(username)}
      >
        Continue
      </button>
    </div>
  );
}

// --- Step: Create Passkey ---

function CreatePasskeyStep({
  username,
  onCreated,
  onError,
  onBack,
}: {
  username: string;
  onCreated: (pubkey: string, privateKey: Uint8Array, webId: string) => void;
  onError: (msg: string) => void;
  onBack: () => void;
}) {
  const [status, setStatus] = useState('Requesting registration options...');
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    (async () => {
      try {
        // 1. Get server options
        const options = await startRegistration(username);

        setStatus('Creating your passkey...');
        // 2. Create credential with PRF
        const { credential, prfOutput } = await createPasskeyCredential(options);

        setStatus('Generating identity...');
        // 3. Derive or generate Nostr key
        let secretKey: Uint8Array;
        let pubkey: string;
        let prfEnabled = false;

        if (prfOutput) {
          secretKey = await deriveNostrKey(prfOutput);
          pubkey = getPublicKey(secretKey);
          prfEnabled = true;
        } else {
          secretKey = generateSecretKey();
          pubkey = getPublicKey(secretKey);
          // Fallback: force download since key can't be re-derived
          downloadKeyBackup(username, pubkey, bytesToHex(secretKey), false);
        }

        setStatus('Setting up your pod...');
        // 4. Verify with server
        const result = await verifyRegistration({
          challengeKey: options.challengeKey,
          credential,
          pubkey,
          prfEnabled,
        });

        // 5. Store key in memory (not sessionStorage) for NIP-98 signing
        setLocalKey(bytesToHex(secretKey));
        try {
          // Only non-secret metadata goes to sessionStorage
          sessionStorage.setItem('nostr_passkey_pubkey', pubkey);
          sessionStorage.setItem('nostr_prf', prfEnabled ? '1' : '0');
        } catch { /* sessionStorage may be unavailable in some contexts */ }

        onCreated(pubkey, secretKey, result.webId);
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Passkey creation failed';
        onError(msg);
      }
    })();
  }, [username, onCreated, onError]);

  return (
    <div className="wizard-step wizard-step-centered">
      <button className="wizard-back" onClick={onBack}>Back</button>
      <div className="wizard-spinner-large" />
      <h2>{status}</h2>
      <p className="wizard-hint">
        Follow your browser's prompt to create a passkey.
        This stores your credential securely on your device.
      </p>
    </div>
  );
}

// --- Step: Identity Ready ---

function IdentityReadyStep({
  username,
  pubkey,
  webId,
  onEnter,
}: {
  username: string;
  pubkey: string;
  webId: string;
  onEnter: () => void;
}) {
  const shortPubkey = pubkey.slice(0, 8) + '...' + pubkey.slice(-8);

  return (
    <div className="wizard-step">
      <div className="wizard-success-icon">&#10003;</div>
      <h2>Your identity is ready</h2>
      <div className="wizard-identity-card">
        <div className="wizard-identity-row">
          <span className="wizard-identity-label">Username</span>
          <span className="wizard-identity-value">{username}</span>
        </div>
        <div className="wizard-identity-row">
          <span className="wizard-identity-label">Public Key</span>
          <span className="wizard-identity-value wizard-identity-mono">{shortPubkey}</span>
        </div>
        <div className="wizard-identity-row">
          <span className="wizard-identity-label">WebID</span>
          <span className="wizard-identity-value wizard-identity-mono wizard-identity-small">{webId}</span>
        </div>
      </div>
      <button className="wizard-btn wizard-btn-primary" onClick={onEnter}>
        Enter VisionFlow
      </button>
    </div>
  );
}

// --- Step: Sign In ---

function SignInStep({
  onSuccess,
  onError,
  onBack,
}: {
  onSuccess: (pubkey: string, privateKey: Uint8Array | null) => void;
  onError: (msg: string) => void;
  onBack: () => void;
}) {
  const [status, setStatus] = useState('Preparing sign-in...');
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    (async () => {
      try {
        // 1. Get authentication options (no username = discoverable credentials)
        const options = await startLogin();

        setStatus('Authenticate with your passkey...');
        // 2. Authenticate with PRF
        const { credential, prfOutput } = await authenticatePasskey(options);

        setStatus('Verifying...');
        // 3. Verify with server
        await verifyLogin({ challengeKey: options.challengeKey, credential });

        // 4. Derive Nostr key from PRF if available
        let secretKey: Uint8Array | null = null;
        let pubkey = '';

        if (prfOutput) {
          secretKey = await deriveNostrKey(prfOutput);
          pubkey = getPublicKey(secretKey);

          // Store key in memory only -- never in sessionStorage
          setLocalKey(bytesToHex(secretKey));
          try {
            // Only non-secret metadata goes to sessionStorage
            sessionStorage.setItem('nostr_passkey_pubkey', pubkey);
            sessionStorage.setItem('nostr_prf', '1');
          } catch { /* ignore */ }
        }

        onSuccess(pubkey, secretKey);
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Sign-in failed';
        onError(msg);
      }
    })();
  }, [onSuccess, onError]);

  return (
    <div className="wizard-step wizard-step-centered">
      <button className="wizard-back" onClick={onBack}>Back</button>
      <div className="wizard-spinner-large" />
      <h2>{status}</h2>
      <p className="wizard-hint">
        Select your passkey when prompted by your browser.
      </p>
    </div>
  );
}

// --- Step: Sign-In Method Picker ---

function SignInMethodStep({
  onPasskey,
  onExtension,
  hasExtension,
  onBack,
  onPrivKeyLogin,
}: {
  onPasskey: () => void;
  onExtension: () => void;
  hasExtension: boolean;
  onBack: () => void;
  onPrivKeyLogin: (pubkey: string, privateKey: Uint8Array) => void;
}) {
  const [showPrivKey, setShowPrivKey] = useState(false);
  const [privKeyInput, setPrivKeyInput] = useState('');
  const [privKeyError, setPrivKeyError] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [isValidPrivKey, setIsValidPrivKey] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  const validatePrivKey = useCallback((value: string): { valid: boolean; error: string } => {
    const trimmed = value.trim();
    if (!trimmed) return { valid: false, error: '' };

    if (trimmed.startsWith('nsec1')) {
      try {
        const decoded = nip19.decode(trimmed);
        if (decoded.type === 'nsec') return { valid: true, error: '' };
        return { valid: false, error: 'Invalid nsec key format' };
      } catch {
        return { valid: false, error: 'Invalid nsec key — check for typos' };
      }
    }

    if (/^[0-9a-f]{64}$/i.test(trimmed)) return { valid: true, error: '' };

    if (/^[0-9a-f]+$/i.test(trimmed) && trimmed.length < 64) {
      return { valid: false, error: `Hex key too short (${trimmed.length}/64 characters)` };
    }

    return { valid: false, error: 'Enter an nsec1... key or 64-character hex key' };
  }, []);

  const handlePrivKeyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setPrivKeyInput(val);
    setPrivKeyError('');
    setIsValidPrivKey(false);

    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      const { valid, error } = validatePrivKey(val);
      setIsValidPrivKey(valid);
      setPrivKeyError(error);
    }, 300);
  };

  useEffect(() => {
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, []);

  const handlePrivKeyLogin = () => {
    const trimmed = privKeyInput.trim();
    let secretKeyBytes: Uint8Array;
    let hexKey: string;

    try {
      if (trimmed.startsWith('nsec1')) {
        const decoded = nip19.decode(trimmed);
        if (decoded.type !== 'nsec') throw new Error('Invalid nsec');
        secretKeyBytes = decoded.data as Uint8Array;
        hexKey = bytesToHex(secretKeyBytes);
      } else {
        hexKey = trimmed.toLowerCase();
        secretKeyBytes = new Uint8Array(
          hexKey.match(/.{1,2}/g)!.map(byte => parseInt(byte, 16))
        );
      }

      const pubkey = getPublicKey(secretKeyBytes);
      setLocalKey(hexKey);

      try {
        sessionStorage.setItem('nostr_passkey_pubkey', pubkey);
      } catch { /* sessionStorage unavailable */ }

      // Clear input state before navigating away
      setPrivKeyInput('');
      setShowKey(false);

      onPrivKeyLogin(pubkey, secretKeyBytes);
    } catch (err) {
      setPrivKeyError(err instanceof Error ? err.message : 'Invalid key');
      setIsValidPrivKey(false);
    }
  };

  return (
    <div className="wizard-step">
      <button className="wizard-back" onClick={onBack}>Back</button>
      <h2>Sign in</h2>
      <p className="wizard-hint">Choose how to authenticate</p>
      <div className="wizard-actions">
        <button className="wizard-btn wizard-btn-primary" onClick={onPasskey}>
          Sign in with Passkey
        </button>
        {hasExtension && (
          <button className="wizard-btn wizard-btn-secondary" onClick={onExtension}>
            Sign in with Signing Extension
          </button>
        )}
      </div>
      {!hasExtension && (
        <p className="wizard-hint wizard-hint-small">
          Have a Nostr signing extension? It will be detected automatically.
        </p>
      )}
      <div className="wizard-divider" />
      <button
        className="wizard-advanced-link"
        onClick={() => setShowPrivKey(!showPrivKey)}
      >
        {showPrivKey ? '\u25BE' : '\u25B8'} Advanced: Import a private key
      </button>
      {showPrivKey && (
        <div className="wizard-privkey-section">
          <div className="wizard-privkey-input-wrap">
            <input
              type={showKey ? 'text' : 'password'}
              className="wizard-input wizard-input-mono"
              placeholder="nsec1... or 64-character hex key"
              value={privKeyInput}
              onChange={handlePrivKeyChange}
              autoComplete="off"
              spellCheck={false}
            />
            <button
              className="wizard-eye-toggle"
              onClick={() => setShowKey(!showKey)}
              type="button"
              aria-label={showKey ? 'Hide key' : 'Show key'}
            >
              {showKey ? '\u25C9' : '\u25CE'}
            </button>
          </div>
          {privKeyError && (
            <div className="status-error" style={{ marginTop: 6, fontSize: 13 }}>{privKeyError}</div>
          )}
          <div className="wizard-security-notice">
            <strong>Security Notice</strong>
            <ul>
              <li>Only paste keys on devices you fully trust</li>
              <li>Key held in browser memory only — never stored to disk</li>
              <li>Cleared when you close this tab</li>
              <li>Passkeys are the recommended sign-in method</li>
            </ul>
          </div>
          <button
            className="wizard-btn wizard-btn-outline"
            disabled={!isValidPrivKey}
            onClick={handlePrivKeyLogin}
          >
            Sign In with Key
          </button>
        </div>
      )}
    </div>
  );
}

// --- Step: Extension Login ---

function ExtensionLoginStep({
  nip07Login,
  onSuccess,
  onError,
  onBack,
}: {
  nip07Login: () => Promise<{ user?: { pubkey: string } }>;
  onSuccess: (pubkey: string) => void;
  onError: (msg: string) => void;
  onBack: () => void;
}) {
  const [status, setStatus] = useState('Requesting key from extension...');
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    (async () => {
      try {
        setStatus('Requesting key from extension...');
        const result = await nip07Login();
        if (result.user?.pubkey) {
          onSuccess(result.user.pubkey);
        } else {
          onError('Extension did not return a public key');
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Extension login failed';
        onError(msg);
      }
    })();
  }, [nip07Login, onSuccess, onError]);

  return (
    <div className="wizard-step wizard-step-centered">
      <button className="wizard-back" onClick={onBack}>Back</button>
      <div className="wizard-spinner-large" />
      <h2>{status}</h2>
      <p className="wizard-hint">
        Approve the request in your Nostr signing extension.
      </p>
    </div>
  );
}

// --- Step: Error ---

function ErrorStep({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="wizard-step wizard-step-centered">
      <div className="wizard-error-icon">!</div>
      <h2>Something went wrong</h2>
      <div className="wizard-error-message">{message}</div>
      <button className="wizard-btn wizard-btn-primary" onClick={onRetry}>
        Try Again
      </button>
    </div>
  );
}
