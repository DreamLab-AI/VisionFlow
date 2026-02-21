/**
 * WebAuthn Passkey Service
 * Handles passkey registration, authentication, and PRF-based Nostr key derivation.
 * Ports the inline PRF/HKDF logic from src/idp/views.js into typed TypeScript.
 */

// --- Constants ---

/** Fixed RP-scoped PRF salt — must match server-side registration views */
const PRF_SALT = new Uint8Array(32);
new TextEncoder().encode('solid-nostr-prf-v1').forEach((b, i) => { PRF_SALT[i] = b; });

/** HKDF domain separation info for deriving secp256k1 keys */
const HKDF_INFO = new TextEncoder().encode('nostr-secp256k1-v1');

// --- Types ---

export interface RegistrationOptionsResponse {
  challenge: string;
  rp: { name: string; id: string };
  user: { id: string; name: string; displayName: string };
  pubKeyCredParams: Array<{ type: string; alg: number }>;
  authenticatorSelection?: Record<string, string>;
  excludeCredentials?: Array<{ id: string; type: string; transports?: string[] }>;
  challengeKey: string;
}

export interface AuthenticationOptionsResponse {
  challenge: string;
  rpId: string;
  allowCredentials?: Array<{ id: string; type: string; transports?: string[] }>;
  challengeKey: string;
  userVerification?: string;
}

export interface RegistrationVerifyResponse {
  success: boolean;
  accountId: string;
  webId: string;
  completionToken: string;
}

export interface AuthenticationVerifyResponse {
  success: boolean;
  accountId: string;
  webId: string;
}

export interface PasskeyCredentialResult {
  credential: PublicKeyCredential;
  prfOutput: ArrayBuffer | null;
}

// --- Key Derivation ---

/**
 * Derive a secp256k1 private key from WebAuthn PRF output via HKDF.
 * PRF output → HKDF(SHA-256, salt=zeros, info="nostr-secp256k1-v1") → 256-bit key
 */
export async function deriveNostrKey(prfOutput: ArrayBuffer): Promise<Uint8Array> {
  const keyMaterial = await crypto.subtle.importKey('raw', prfOutput, 'HKDF', false, ['deriveBits']);
  const derived = await crypto.subtle.deriveBits(
    {
      name: 'HKDF',
      hash: 'SHA-256',
      salt: new Uint8Array(32),
      info: HKDF_INFO,
    },
    keyMaterial,
    256
  );
  return new Uint8Array(derived);
}

// --- Registration Flow ---

/**
 * Request registration options for a new user from the server.
 * Also implicitly checks username availability (409 = taken).
 */
export async function startRegistration(username: string): Promise<RegistrationOptionsResponse> {
  const res = await fetch('/idp/passkey/register-new/options', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username }),
  });

  if (res.status === 409) {
    throw new Error('Username already taken');
  }
  if (res.status === 400) {
    const data = await res.json();
    throw new Error(data.error || 'Invalid username');
  }
  if (!res.ok) {
    throw new Error('Failed to get registration options');
  }

  return res.json();
}

/**
 * Check if a username is available without committing to registration.
 * Uses the same endpoint — a 409 means taken, 200 means available.
 */
export async function checkUsernameAvailable(username: string): Promise<boolean> {
  try {
    const res = await fetch('/idp/passkey/register-new/options', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username }),
    });
    if (res.status === 409) return false;
    if (res.status === 400) return false;
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Create a passkey credential via navigator.credentials.create() with PRF extension.
 * Returns the credential and optional PRF output.
 */
export async function createPasskeyCredential(
  options: RegistrationOptionsResponse
): Promise<PasskeyCredentialResult> {
  // Convert base64url fields to ArrayBuffer for the WebAuthn API
  const publicKeyOptions: PublicKeyCredentialCreationOptions = {
    challenge: base64urlToBuffer(options.challenge),
    rp: options.rp,
    user: {
      id: base64urlToBuffer(options.user.id),
      name: options.user.name,
      displayName: options.user.displayName,
    },
    pubKeyCredParams: options.pubKeyCredParams as PublicKeyCredentialParameters[],
    authenticatorSelection: options.authenticatorSelection as AuthenticatorSelectionCriteria,
    excludeCredentials: options.excludeCredentials?.map(c => ({
      ...c,
      id: base64urlToBuffer(c.id),
    })) as PublicKeyCredentialDescriptor[],
    extensions: {
      prf: { eval: { first: PRF_SALT } },
    } as AuthenticationExtensionsClientInputs,
  };

  const credential = (await navigator.credentials.create({
    publicKey: publicKeyOptions,
  })) as PublicKeyCredential | null;

  if (!credential) {
    throw new Error('Passkey creation was cancelled');
  }

  // Extract PRF result if available
  const extensions = credential.getClientExtensionResults() as Record<string, unknown>;
  const prfResult = extensions.prf as { results?: { first?: ArrayBuffer } } | undefined;
  const prfOutput = prfResult?.results?.first ?? null;

  return { credential, prfOutput };
}

/**
 * Send the registration credential to the server for verification.
 * Returns accountId, webId, and completionToken.
 */
export async function verifyRegistration(params: {
  challengeKey: string;
  credential: PublicKeyCredential;
  pubkey: string;
  prfEnabled: boolean;
}): Promise<RegistrationVerifyResponse> {
  const { challengeKey, credential, pubkey, prfEnabled } = params;
  const response = credential.response as AuthenticatorAttestationResponse;

  const res = await fetch('/idp/passkey/register-new/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      challengeKey,
      pubkey,
      prfEnabled,
      credential: {
        id: credential.id,
        rawId: bufferToBase64url(credential.rawId),
        type: credential.type,
        response: {
          clientDataJSON: bufferToBase64url(response.clientDataJSON),
          attestationObject: bufferToBase64url(response.attestationObject),
          transports: response.getTransports ? response.getTransports() : [],
        },
      },
      name: detectDeviceName(),
    }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: 'Registration verification failed' }));
    throw new Error(data.error || 'Registration verification failed');
  }

  return res.json();
}

// --- Authentication Flow ---

/**
 * Request authentication options from the server.
 * If username is provided, credentials are scoped to that user.
 */
export async function startLogin(username?: string): Promise<AuthenticationOptionsResponse> {
  const res = await fetch('/idp/passkey/login/options', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: username || undefined }),
  });

  if (!res.ok) {
    throw new Error('Failed to get login options');
  }

  return res.json();
}

/**
 * Authenticate with a passkey via navigator.credentials.get() with PRF extension.
 * Returns the assertion credential and optional PRF output.
 */
export async function authenticatePasskey(
  options: AuthenticationOptionsResponse
): Promise<PasskeyCredentialResult> {
  const publicKeyOptions: PublicKeyCredentialRequestOptions = {
    challenge: base64urlToBuffer(options.challenge),
    rpId: options.rpId,
    allowCredentials: options.allowCredentials?.map(c => ({
      ...c,
      id: base64urlToBuffer(c.id),
    })) as PublicKeyCredentialDescriptor[],
    userVerification: (options.userVerification || 'preferred') as UserVerificationRequirement,
    extensions: {
      prf: { eval: { first: PRF_SALT } },
    } as AuthenticationExtensionsClientInputs,
  };

  const credential = (await navigator.credentials.get({
    publicKey: publicKeyOptions,
  })) as PublicKeyCredential | null;

  if (!credential) {
    throw new Error('Passkey authentication was cancelled');
  }

  const extensions = credential.getClientExtensionResults() as Record<string, unknown>;
  const prfResult = extensions.prf as { results?: { first?: ArrayBuffer } } | undefined;
  const prfOutput = prfResult?.results?.first ?? null;

  return { credential, prfOutput };
}

/**
 * Send the authentication assertion to the server for verification.
 * Returns accountId and webId on success.
 */
export async function verifyLogin(params: {
  challengeKey: string;
  credential: PublicKeyCredential;
}): Promise<AuthenticationVerifyResponse> {
  const { challengeKey, credential } = params;
  const response = credential.response as AuthenticatorAssertionResponse;

  const res = await fetch('/idp/passkey/login/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      challengeKey,
      credential: {
        id: credential.id,
        rawId: bufferToBase64url(credential.rawId),
        type: credential.type,
        response: {
          clientDataJSON: bufferToBase64url(response.clientDataJSON),
          authenticatorData: bufferToBase64url(response.authenticatorData),
          signature: bufferToBase64url(response.signature),
          userHandle: response.userHandle
            ? bufferToBase64url(response.userHandle)
            : null,
        },
      },
    }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({ error: 'Authentication failed' }));
    throw new Error(data.error || 'Authentication failed');
  }

  return res.json();
}

// --- Helpers ---

function bufferToBase64url(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

function base64urlToBuffer(base64url: string): ArrayBuffer {
  const base64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
  const padLen = (4 - (base64.length % 4)) % 4;
  const padded = base64 + '='.repeat(padLen);
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

export function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
}

function detectDeviceName(): string {
  const ua = navigator.userAgent;
  if (/iPhone/.test(ua)) return 'iPhone';
  if (/iPad/.test(ua)) return 'iPad';
  if (/Mac/.test(ua)) return 'Mac';
  if (/Android/.test(ua)) return 'Android';
  if (/Windows/.test(ua)) return 'Windows';
  if (/Linux/.test(ua)) return 'Linux';
  return 'Security Key';
}

/**
 * Generate a random secp256k1 private key and trigger a file download.
 * Fallback for authenticators that don't support the PRF extension.
 */
export function downloadKeyBackup(username: string, pubkey: string, secretKeyHex: string, prfDerived: boolean): void {
  const source = prfDerived
    ? 'PRF-derived (re-derivable from this passkey)'
    : 'Random (this file is the ONLY copy)';
  const content = [
    'Nostr Private Key',
    '==================',
    `Username: ${username}`,
    `Public Key (hex): ${pubkey}`,
    `Private Key (hex): ${secretKeyHex}`,
    `Source: ${source}`,
    `Created: ${new Date().toISOString()}`,
    '',
    'IMPORTANT: Keep this file safe. Anyone with your private key can impersonate you.',
    prfDerived
      ? 'Since your key is PRF-derived, you can regenerate it by authenticating with the same passkey.'
      : 'This file is the ONLY backup of your private key. If you lose it, your identity is unrecoverable.',
  ].join('\n');

  const blob = new Blob([content], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `nostr-key-${username}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
