import React, { useReducer, useRef, useCallback, useEffect, useState } from 'react';
import { useNostrAuth } from '../hooks/useNostrAuth';
import { nostrAuth } from '../services/nostrAuthService';
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
import './OnboardingWizard.css';

// --- State Machine ---

type WizardStep = 'welcome' | 'username' | 'create-passkey' | 'identity-ready' | 'sign-in' | 'error';

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
      return { ...state, step: 'sign-in', prevStep: 'welcome' };
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
  const { isDevLoginAvailable, devLogin } = useNostrAuth();

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
          {state.step === 'sign-in' && (
            <SignInStep
              onSuccess={(pubkey, privateKey) => finishAuth(pubkey, privateKey)}
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
        <p className="wizard-subtitle">Decentralized identity, powered by passkeys</p>
      </div>
      <div className="wizard-actions">
        <button className="wizard-btn wizard-btn-primary" onClick={onRegister}>
          Create Account
        </button>
        <button className="wizard-btn wizard-btn-secondary" onClick={onLogin}>
          Sign In
        </button>
      </div>
      <div className="wizard-info">
        <p><strong>How it works</strong></p>
        <p>
          VisionFlow uses passkeys for passwordless authentication.
          Your cryptographic identity is derived on-device — no passwords, no extensions needed.
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

        // 5. Store in sessionStorage for NIP-98 signing
        try {
          sessionStorage.setItem('nostr_passkey_key', bytesToHex(secretKey));
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

          try {
            sessionStorage.setItem('nostr_passkey_key', bytesToHex(secretKey));
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
