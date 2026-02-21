/**
 * Passkey (WebAuthn) authentication endpoints
 * Handles registration and authentication of passkey credentials
 */

import {
  generateRegistrationOptions,
  verifyRegistrationResponse,
  generateAuthenticationOptions,
  verifyAuthenticationResponse
} from '@simplewebauthn/server';
import crypto from 'crypto';
import * as accounts from './accounts.js';
import * as storageModule from '../storage/filesystem.js';
import { createPodStructure } from '../handlers/container.js';

// Temporary challenge storage (in-memory, cleared on restart)
// For production clusters, use Redis or session storage
const challenges = new Map();
const MAX_CHALLENGES = 10000; // Prevent unbounded growth

// Clean up expired challenges periodically
// Use unref() so this timer doesn't prevent process exit (important for tests)
const cleanupInterval = setInterval(() => {
  const now = Date.now();
  for (const [key, value] of challenges.entries()) {
    if (now > value.expires) {
      challenges.delete(key);
    }
  }
}, 60000);
cleanupInterval.unref();

/**
 * Store a challenge with size limit enforcement
 */
function storeChallenge(key, value) {
  // If at capacity, remove oldest expired entries first
  if (challenges.size >= MAX_CHALLENGES) {
    const now = Date.now();
    for (const [k, v] of challenges.entries()) {
      if (now > v.expires) {
        challenges.delete(k);
      }
      if (challenges.size < MAX_CHALLENGES) break;
    }
  }
  // If still at capacity, reject (DoS protection)
  if (challenges.size >= MAX_CHALLENGES) {
    return false;
  }
  challenges.set(key, value);
  return true;
}

/**
 * Get Relying Party configuration from request
 * Handles both IPv4 (with port) and IPv6 addresses correctly
 */
function getRP(request) {
  let hostname;
  try {
    // Use URL parsing to correctly extract hostname (handles IPv6)
    const url = new URL(`${request.protocol}://${request.hostname}`);
    hostname = url.hostname;
  } catch {
    // Fallback: strip port from hostname (IPv4 only)
    hostname = String(request.hostname || '').split(':')[0];
  }
  return {
    name: 'Solid Pod',
    id: hostname
  };
}

/**
 * Get origin from request
 */
function getOrigin(request) {
  return `${request.protocol}://${request.hostname}`;
}

/**
 * POST /idp/passkey/register/options
 * Generate registration options for a logged-in user
 */
export async function registrationOptions(request, reply) {
  const { accountId } = request.body || {};

  if (!accountId) {
    return reply.code(401).send({ error: 'Must provide accountId' });
  }

  const account = await accounts.findById(accountId);
  if (!account) {
    return reply.code(404).send({ error: 'Account not found' });
  }

  const rp = getRP(request);

  const options = await generateRegistrationOptions({
    rpName: rp.name,
    rpID: rp.id,
    userID: new TextEncoder().encode(account.id),
    userName: account.username,
    userDisplayName: account.username,
    attestationType: 'none', // Don't require attestation for privacy
    excludeCredentials: (account.passkeys || []).map(pk => ({
      id: Buffer.from(pk.credentialId, 'base64url'),
      type: 'public-key',
      transports: pk.transports
    })),
    authenticatorSelection: {
      residentKey: 'preferred',
      userVerification: 'preferred'
    }
  });

  // Store challenge for verification with unique key (prevents race conditions from multiple tabs)
  const challengeKey = crypto.randomUUID();
  const stored = storeChallenge(challengeKey, {
    challenge: options.challenge,
    type: 'registration',
    accountId: account.id,
    expires: Date.now() + 60000 // 1 minute
  });

  if (!stored) {
    return reply.code(503).send({ error: 'Server busy, try again later' });
  }

  return reply.send({ ...options, challengeKey });
}

/**
 * POST /idp/passkey/register/verify
 * Verify and store the registration response
 */
export async function registrationVerify(request, reply) {
  const { accountId, credential, name, challengeKey } = request.body || {};

  if (!accountId || !credential || !challengeKey) {
    return reply.code(400).send({ error: 'Missing required fields' });
  }

  const stored = challenges.get(challengeKey);
  if (!stored || stored.type !== 'registration' || Date.now() > stored.expires) {
    return reply.code(400).send({ error: 'Challenge expired or invalid' });
  }

  // Verify the accountId matches the challenge
  if (stored.accountId !== accountId) {
    return reply.code(403).send({ error: 'Account mismatch' });
  }

  const account = await accounts.findById(accountId);
  if (!account) {
    return reply.code(404).send({ error: 'Account not found' });
  }

  const rp = getRP(request);

  try {
    const verification = await verifyRegistrationResponse({
      response: credential,
      expectedChallenge: stored.challenge,
      expectedOrigin: getOrigin(request),
      expectedRPID: rp.id
    });

    if (!verification.verified || !verification.registrationInfo) {
      request.log.warn({ verified: verification.verified, hasInfo: !!verification.registrationInfo }, 'Passkey registration verification failed');
      return reply.code(400).send({ error: 'Verification failed' });
    }

    const { credential: regCredential } = verification.registrationInfo;

    await accounts.addPasskey(accountId, {
      credentialId: regCredential.id, // Already base64url string
      publicKey: Buffer.from(regCredential.publicKey).toString('base64url'),
      counter: regCredential.counter,
      transports: regCredential.transports || credential.response?.transports || [],
      name: name || 'Security Key'
    });

    challenges.delete(challengeKey);

    return reply.send({ success: true });
  } catch (err) {
    request.log.error({ err }, 'Passkey registration error');
    return reply.code(400).send({ error: 'Passkey registration failed' });
  }
}

/**
 * POST /idp/passkey/register-new/options
 * Generate registration options for a new user (passkey-only flow)
 */
export async function registrationOptionsForNewUser(request, reply) {
  const { username } = request.body || {};

  if (!username || !/^[a-z0-9]{3,}$/.test(username)) {
    return reply.code(400).send({ error: 'Username must be 3+ lowercase alphanumeric characters' });
  }

  const existing = await accounts.findByUsername(username);
  if (existing) {
    return reply.code(409).send({ error: 'Username already taken' });
  }

  const rp = getRP(request);

  const options = await generateRegistrationOptions({
    rpName: rp.name,
    rpID: rp.id,
    userID: new TextEncoder().encode(username),
    userName: username,
    userDisplayName: username,
    attestationType: 'none',
    authenticatorSelection: {
      residentKey: 'preferred',
      userVerification: 'preferred'
    }
  });

  const challengeKey = crypto.randomUUID();
  const stored = storeChallenge(challengeKey, {
    challenge: options.challenge,
    type: 'registration-new',
    username,
    expires: Date.now() + 60000
  });

  if (!stored) {
    return reply.code(503).send({ error: 'Server busy, try again later' });
  }

  return reply.send({ ...options, challengeKey });
}

/**
 * POST /idp/passkey/register-new/verify
 * Verify registration and create account + pod for new user
 */
export async function registrationVerifyNewUser(request, reply, issuer) {
  const { credential, challengeKey, pubkey, name, prfEnabled } = request.body || {};

  if (!credential || !challengeKey || !pubkey) {
    return reply.code(400).send({ error: 'Missing required fields' });
  }

  // Validate pubkey format (64-char hex)
  if (!/^[0-9a-f]{64}$/.test(pubkey)) {
    return reply.code(400).send({ error: 'Invalid public key (expected 64-char hex)' });
  }

  const stored = challenges.get(challengeKey);
  if (!stored || stored.type !== 'registration-new' || Date.now() > stored.expires) {
    return reply.code(400).send({ error: 'Challenge expired or invalid' });
  }

  const username = stored.username;
  const rp = getRP(request);

  try {
    const verification = await verifyRegistrationResponse({
      response: credential,
      expectedChallenge: stored.challenge,
      expectedOrigin: getOrigin(request),
      expectedRPID: rp.id
    });

    if (!verification.verified || !verification.registrationInfo) {
      return reply.code(400).send({ error: 'Verification failed' });
    }

    // Build pod URLs
    const baseUrl = issuer.endsWith('/') ? issuer.slice(0, -1) : issuer;
    const subdomainsEnabled = request.subdomainsEnabled;
    const baseDomain = request.baseDomain;

    let podUri, webId;
    if (subdomainsEnabled && baseDomain) {
      podUri = `${request.protocol}://${username}.${baseDomain}/`;
      webId = `${podUri}profile/card#me`;
    } else {
      podUri = `${baseUrl}/${username}/`;
      webId = `${podUri}profile/card#me`;
    }

    // Check if pod already exists
    const podExists = await storageModule.exists(`${username}/`);
    if (podExists) {
      return reply.code(409).send({ error: 'Username is already taken' });
    }

    // Create pod structure
    await createPodStructure(username, webId, podUri, issuer);

    // Create account (passkey-only, no password)
    const account = await accounts.createAccount({
      username,
      webId,
      podName: username,
      nostrPubkey: pubkey,
      prfEnabled: !!prfEnabled,
    });

    // Add passkey credential
    const { credential: regCredential } = verification.registrationInfo;
    await accounts.addPasskey(account.id, {
      credentialId: regCredential.id,
      publicKey: Buffer.from(regCredential.publicKey).toString('base64url'),
      counter: regCredential.counter,
      transports: regCredential.transports || credential.response?.transports || [],
      name: name || 'Security Key'
    });

    challenges.delete(challengeKey);

    // Generate a short-lived completion token to bind accountId to OIDC interaction
    // Prevents C-1: unauthenticated OIDC completion with arbitrary accountId
    const completionToken = crypto.randomUUID();
    storeChallenge(`reg-complete:${completionToken}`, {
      type: 'registration-complete',
      accountId: account.id,
      expires: Date.now() + 300000 // 5 minutes
    });

    request.log.info({ accountId: account.id, username, webId }, 'Passkey-only registration completed');

    return reply.send({
      success: true,
      accountId: account.id,
      webId: account.webId,
      completionToken,
    });
  } catch (err) {
    request.log.error({ err }, 'Passkey new user registration error');
    return reply.code(400).send({ error: 'Registration failed' });
  }
}

/**
 * POST /idp/passkey/login/options
 * Generate authentication options
 */
export async function authenticationOptions(request, reply) {
  const { username } = request.body || {};
  const rp = getRP(request);

  let allowCredentials = [];
  let accountId = null;

  // If username provided, limit to that user's credentials
  if (username) {
    const account = await accounts.findByUsername(username);
    if (account && account.passkeys?.length) {
      accountId = account.id;
      allowCredentials = account.passkeys.map(pk => ({
        id: Buffer.from(pk.credentialId, 'base64url'),
        type: 'public-key',
        transports: pk.transports
      }));
    }
  }

  const options = await generateAuthenticationOptions({
    rpID: rp.id,
    allowCredentials,
    userVerification: 'preferred'
  });

  // Always use a random challenge key to prevent client-controlled key overwrites
  const challengeKey = crypto.randomUUID();
  const stored = storeChallenge(challengeKey, {
    challenge: options.challenge,
    type: 'authentication',
    accountId,
    expires: Date.now() + 60000 // 1 minute
  });

  if (!stored) {
    return reply.code(503).send({ error: 'Server busy, try again later' });
  }

  return reply.send({ ...options, challengeKey });
}

/**
 * POST /idp/passkey/login/verify
 * Verify authentication and return account info
 */
export async function authenticationVerify(request, reply) {
  const { challengeKey, credential } = request.body || {};

  if (!challengeKey || !credential) {
    return reply.code(400).send({ error: 'Missing challengeKey or credential' });
  }

  const stored = challenges.get(challengeKey);
  if (!stored || stored.type !== 'authentication' || Date.now() > stored.expires) {
    return reply.code(400).send({ error: 'Challenge expired or invalid' });
  }

  // Find account by credential ID
  const credentialId = credential.id;
  const account = stored.accountId
    ? await accounts.findById(stored.accountId)
    : await accounts.findByCredentialId(credentialId);

  if (!account) {
    return reply.code(400).send({ error: 'Unknown credential' });
  }

  const passkey = account.passkeys?.find(pk => pk.credentialId === credentialId);
  if (!passkey) {
    return reply.code(400).send({ error: 'Credential not found' });
  }

  const rp = getRP(request);

  try {
    const verification = await verifyAuthenticationResponse({
      response: credential,
      expectedChallenge: stored.challenge,
      expectedOrigin: getOrigin(request),
      expectedRPID: rp.id,
      credential: {
        id: Buffer.from(passkey.credentialId, 'base64url'),
        publicKey: Buffer.from(passkey.publicKey, 'base64url'),
        counter: passkey.counter
      }
    });

    if (!verification.verified) {
      return reply.code(400).send({ error: 'Verification failed' });
    }

    // Update counter to prevent replay attacks
    await accounts.updatePasskeyCounter(
      account.id,
      credentialId,
      verification.authenticationInfo.newCounter
    );

    // Update last login
    await accounts.updateLastLogin(account.id);

    challenges.delete(challengeKey);

    // Return account info for session creation
    return reply.send({
      success: true,
      accountId: account.id,
      webId: account.webId
    });
  } catch (err) {
    request.log.error({ err }, 'Passkey authentication error');
    return reply.code(400).send({ error: 'Authentication failed' });
  }
}

/**
 * Validate a registration completion token
 * Used by handlePasskeyRegisterComplete to verify the caller completed registration
 * @param {string} token - Completion token from registrationVerifyNewUser
 * @param {string} accountId - Expected account ID
 * @returns {boolean} - True if valid
 */
export function validateCompletionToken(token, accountId) {
  const key = `reg-complete:${token}`;
  const stored = challenges.get(key);
  if (!stored || stored.type !== 'registration-complete' || Date.now() > stored.expires) {
    return false;
  }
  if (stored.accountId !== accountId) {
    return false;
  }
  challenges.delete(key);
  return true;
}
