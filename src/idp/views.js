/**
 * HTML templates for IdP login/consent pages
 * Minimal, functional design
 */

const styles = `
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f5f5;
    margin: 0;
    padding: 40px 20px;
    min-height: 100vh;
  }
  .container {
    max-width: 400px;
    margin: 0 auto;
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    padding: 40px;
  }
  h1 {
    margin: 0 0 8px 0;
    font-size: 24px;
    color: #333;
  }
  .subtitle {
    color: #666;
    margin: 0 0 30px 0;
    font-size: 14px;
  }
  .client-info {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
  }
  .client-name {
    font-weight: 600;
    color: #333;
  }
  .client-uri {
    font-size: 12px;
    color: #666;
    word-break: break-all;
  }
  label {
    display: block;
    font-size: 14px;
    font-weight: 500;
    color: #333;
    margin-bottom: 6px;
  }
  input[type="text"],
  input[type="email"],
  input[type="password"] {
    width: 100%;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 16px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
  }
  input:focus {
    outline: none;
    border-color: #0066cc;
  }
  .error {
    background: #fee;
    border: 1px solid #fcc;
    color: #c00;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 20px;
    font-size: 14px;
  }
  .btn {
    display: inline-block;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    border: none;
    text-decoration: none;
    text-align: center;
    transition: background-color 0.2s;
  }
  .btn-primary {
    background: #0066cc;
    color: white;
    width: 100%;
  }
  .btn-primary:hover {
    background: #0052a3;
  }
  .btn-secondary {
    background: #f0f0f0;
    color: #333;
    margin-top: 12px;
    width: 100%;
  }
  .btn-secondary:hover {
    background: #e0e0e0;
  }
  .btn-passkey {
    background: #1a73e8;
    color: white;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .btn-passkey:hover {
    background: #1557b0;
  }
  .btn-passkey svg {
    width: 20px;
    height: 20px;
  }
  .btn-schnorr {
    background: #7b1fa2;
    color: white;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 12px;
  }
  .btn-schnorr:hover {
    background: #6a1b9a;
  }
  .btn-schnorr svg {
    width: 20px;
    height: 20px;
  }
  .divider {
    display: flex;
    align-items: center;
    margin: 20px 0;
    color: #666;
    font-size: 14px;
  }
  .divider::before,
  .divider::after {
    content: '';
    flex: 1;
    border-bottom: 1px solid #ddd;
  }
  .divider span {
    padding: 0 12px;
  }
  .scopes {
    margin: 20px 0;
  }
  .scope {
    display: flex;
    align-items: center;
    padding: 12px;
    background: #f8f9fa;
    border-radius: 8px;
    margin-bottom: 8px;
  }
  .scope-icon {
    width: 24px;
    height: 24px;
    margin-right: 12px;
    opacity: 0.6;
  }
  .scope-name {
    font-weight: 500;
  }
  .scope-desc {
    font-size: 12px;
    color: #666;
  }
  .actions {
    margin-top: 24px;
  }
  .logo {
    text-align: center;
    margin-bottom: 24px;
  }
  .logo svg {
    width: 48px;
    height: 48px;
  }
`;

const solidLogo = `
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="45" fill="#7C4DFF" />
  <path d="M30 50 L45 65 L70 40" stroke="white" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
`;

const passkeyIcon = `
<svg viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
  <path d="M12.65 10C11.83 7.67 9.61 6 7 6c-3.31 0-6 2.69-6 6s2.69 6 6 6c2.61 0 4.83-1.67 5.65-4H17v4h4v-4h2v-4H12.65zM7 14c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2z"/>
</svg>
`;

const schnorrIcon = `
<svg viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
  <path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zm-6 9c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2zm3.1-9H8.9V6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2z"/>
</svg>
`;

const scopeDescriptions = {
  openid: 'Access your identity',
  webid: 'Access your WebID',
  profile: 'Access your name',
  email: 'Access your email address',
  offline_access: 'Stay logged in',
};

/**
 * Escape string for safe use in JavaScript
 */
function escapeJs(text) {
  if (!text) return '';
  return String(text)
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/"/g, '\\"')
    .replace(/</g, '\\x3c')
    .replace(/>/g, '\\x3e')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r');
}

/**
 * Login page HTML
 */
export function loginPage(uid, clientId, error = null, passkeyEnabled = true, schnorrEnabled = true) {
  const appName = clientId || 'An application';
  const safeUid = escapeJs(uid);

  const passkeySection = passkeyEnabled ? `
    <button type="button" class="btn btn-passkey" onclick="loginWithPasskey()">
      ${passkeyIcon}
      Sign in with Passkey
    </button>
  ` : '';

  const schnorrSection = schnorrEnabled ? `
    <button type="button" class="btn btn-schnorr" onclick="loginWithSchnorr()" id="schnorrBtn">
      ${schnorrIcon}
      Sign in with Schnorr
    </button>
  ` : '';

  const ssoSection = (passkeyEnabled || schnorrEnabled) ? `
    ${passkeySection}
    ${schnorrSection}
    <div class="divider"><span>or</span></div>
  ` : '';

  const passkeyScript = passkeyEnabled ? `
  <script>
    var INTERACTION_UID = '${safeUid}';

    // Fixed RP-scoped PRF salt (must match registration)
    var PRF_SALT = new Uint8Array(32);
    new TextEncoder().encode('solid-nostr-prf-v1').forEach(function(b, i) { PRF_SALT[i] = b; });

    // HKDF: PRF output -> secp256k1 private key
    async function deriveNostrKey(prfOutput) {
      var keyMaterial = await crypto.subtle.importKey('raw', prfOutput, 'HKDF', false, ['deriveBits']);
      var derived = await crypto.subtle.deriveBits({
        name: 'HKDF',
        hash: 'SHA-256',
        salt: new Uint8Array(32),
        info: new TextEncoder().encode('nostr-secp256k1-v1')
      }, keyMaterial, 256);
      return new Uint8Array(derived);
    }

    async function loginWithPasskey() {
      try {
        // Get authentication options
        const optionsRes = await fetch('/idp/passkey/login/options', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ visitorId: crypto.randomUUID() })
        });
        const options = await optionsRes.json();
        if (options.error) {
          alert('Error: ' + options.error);
          return;
        }

        var challengeKey = options.challengeKey;

        // Convert base64url to ArrayBuffer
        options.challenge = base64urlToBuffer(options.challenge);
        if (options.allowCredentials) {
          options.allowCredentials = options.allowCredentials.map(function(c) {
            return Object.assign({}, c, { id: base64urlToBuffer(c.id) });
          });
        }

        // Prompt user for passkey WITH PRF extension
        const credential = await navigator.credentials.get({
          publicKey: Object.assign({}, options, {
            extensions: {
              prf: { eval: { first: PRF_SALT } }
            }
          })
        });

        // Check PRF result — derive Nostr key if available
        var prfResult = credential.getClientExtensionResults().prf;
        if (prfResult && prfResult.results && prfResult.results.first) {
          try {
            var prfOutput = new Uint8Array(prfResult.results.first);
            var secretKey = await deriveNostrKey(prfOutput);
            var hexSecret = Array.from(secretKey).map(function(b) { return b.toString(16).padStart(2, '0'); }).join('');
            // Dynamic import for getPublicKey (login page is not type=module)
            var nostrTools = await import('https://esm.sh/nostr-tools@2.19.4/pure');
            var pubkey = nostrTools.getPublicKey(secretKey);
            // Store in sessionStorage for NIP-98 signing during this session
            sessionStorage.setItem('nostr_privkey', hexSecret);
            sessionStorage.setItem('nostr_pubkey', pubkey);
            sessionStorage.setItem('nostr_prf', '1');
            console.log('PRF key derived for NIP-98 signing');
          } catch (prfErr) {
            console.warn('PRF key derivation failed (non-fatal):', prfErr);
          }
        }

        // Send response to server
        const verifyRes = await fetch('/idp/passkey/login/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            challengeKey: challengeKey,
            credential: {
              id: credential.id,
              rawId: bufferToBase64url(credential.rawId),
              type: credential.type,
              response: {
                clientDataJSON: bufferToBase64url(credential.response.clientDataJSON),
                authenticatorData: bufferToBase64url(credential.response.authenticatorData),
                signature: bufferToBase64url(credential.response.signature),
                userHandle: credential.response.userHandle
                  ? bufferToBase64url(credential.response.userHandle)
                  : null
              }
            }
          })
        });

        const result = await verifyRes.json();
        if (result.success) {
          // Complete the OIDC interaction - build URL safely
          const redirectUrl = '/idp/interaction/' + encodeURIComponent(INTERACTION_UID) + '/passkey-complete?accountId=' + encodeURIComponent(result.accountId);
          window.location.href = redirectUrl;
        } else {
          alert('Passkey authentication failed: ' + (result.error || 'Unknown error'));
        }
      } catch (err) {
        if (err.name === 'NotAllowedError') {
          // User cancelled - do nothing
        } else {
          console.error('Passkey error:', err);
          alert('Passkey authentication failed: ' + err.message);
        }
      }
    }

    function base64urlToBuffer(base64url) {
      const base64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
      const padLen = (4 - base64.length % 4) % 4;
      const padded = base64 + '='.repeat(padLen);
      const binary = atob(padded);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    }

    function bufferToBase64url(buffer) {
      const bytes = new Uint8Array(buffer);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      return btoa(binary).replace(/[+]/g, '-').replace(/[/]/g, '_').replace(/=/g, '');
    }
  </script>
  ` : '';

  const schnorrScript = schnorrEnabled ? `
  <script>
    async function loginWithSchnorr() {
      const btn = document.getElementById('schnorrBtn');

      // Check for NIP-07 extension (window.nostr)
      if (typeof window.nostr === 'undefined') {
        alert('PodKey extension required for Nostr signing. Get it at: https://podkey.io');
        return;
      }

      btn.disabled = true;
      btn.textContent = 'Signing...';

      try {
        // Get the current URL for the auth event
        const authUrl = window.location.origin + '/idp/interaction/${safeUid}/schnorr-login';

        // Create NIP-98 event (kind 27235)
        const event = {
          kind: 27235,
          created_at: Math.floor(Date.now() / 1000),
          tags: [
            ['u', authUrl],
            ['method', 'POST']
          ],
          content: ''
        };

        // Sign with NIP-07 extension
        const signedEvent = await window.nostr.signEvent(event);

        // Send to server
        const response = await fetch(authUrl, {
          method: 'POST',
          headers: {
            'Authorization': 'Nostr ' + btoa(JSON.stringify(signedEvent))
          }
        });

        const result = await response.json();

        if (result.success && result.redirectUrl) {
          window.location.href = result.redirectUrl;
        } else if (result.error) {
          alert('Schnorr login failed: ' + result.error);
          btn.disabled = false;
          btn.textContent = 'Sign in with Schnorr';
        } else {
          alert('Schnorr login failed: Unknown error');
          btn.disabled = false;
          btn.textContent = 'Sign in with Schnorr';
        }
      } catch (err) {
        console.error('Schnorr login error:', err);
        if (err.message && err.message.includes('User rejected')) {
          // User cancelled signing - do nothing
        } else {
          alert('Schnorr login failed: ' + err.message);
        }
        btn.disabled = false;
        btn.textContent = 'Sign in with Schnorr';
      }
    }
  </script>
  ` : '';

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sign In - Solid IdP</title>
  <style>${styles}</style>
</head>
<body>
  <div class="container">
    <div class="logo">${solidLogo}</div>
    <h1>Sign In</h1>
    <p class="subtitle">Sign in to your Solid Pod</p>

    <div class="client-info">
      <div class="client-name">${escapeHtml(appName)}</div>
      <div class="client-uri">is requesting access to your pod</div>
    </div>

    ${error ? `<div class="error">${escapeHtml(error)}</div>` : ''}

    ${ssoSection}

    <form method="POST" action="/idp/interaction/${uid}/login">
      <label for="username">Username</label>
      <input type="text" id="username" name="username" required autofocus placeholder="Your username">

      <label for="password">Password</label>
      <input type="password" id="password" name="password" required placeholder="Your password">

      <button type="submit" class="btn btn-primary">Sign In</button>
    </form>

    <form method="POST" action="/idp/interaction/${uid}/abort">
      <button type="submit" class="btn btn-secondary">Cancel</button>
    </form>

    <p style="text-align: center; margin-top: 24px; color: #666; font-size: 14px;">
      Don't have an account? <a href="/idp/register?uid=${uid}" style="color: #0066cc;">Register</a>
    </p>
  </div>
  ${passkeyScript}
  ${schnorrScript}
</body>
</html>
  `;
}

/**
 * Consent page HTML
 */
export function consentPage(uid, client, params, account) {
  const scopes = (params.scope || 'openid').split(' ').filter(Boolean);
  const clientName = client?.clientName || client?.client_id || 'Unknown App';
  const clientUri = client?.clientUri || client?.redirect_uris?.[0] || '';

  const scopeItems = scopes.map(scope => `
    <div class="scope">
      <div class="scope-icon">✓</div>
      <div>
        <div class="scope-name">${escapeHtml(scope)}</div>
        <div class="scope-desc">${escapeHtml(scopeDescriptions[scope] || 'Access requested')}</div>
      </div>
    </div>
  `).join('');

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Authorize - Solid IdP</title>
  <style>${styles}</style>
</head>
<body>
  <div class="container">
    <div class="logo">${solidLogo}</div>
    <h1>Authorize Access</h1>
    <p class="subtitle">Allow this app to access your data?</p>

    <div class="client-info">
      <div class="client-name">${escapeHtml(clientName)}</div>
      ${clientUri ? `<div class="client-uri">${escapeHtml(clientUri)}</div>` : ''}
    </div>

    ${account ? `<p>Signed in as <strong>${escapeHtml(account.email)}</strong></p>` : ''}

    <div class="scopes">
      <label>This app is requesting access to:</label>
      ${scopeItems}
    </div>

    <div class="actions">
      <form method="POST" action="/idp/interaction/${uid}/confirm">
        <button type="submit" class="btn btn-primary">Allow Access</button>
      </form>

      <form method="POST" action="/idp/interaction/${uid}/abort">
        <button type="submit" class="btn btn-secondary">Deny</button>
      </form>
    </div>
  </div>
</body>
</html>
  `;
}

/**
 * Error page HTML
 */
export function errorPage(title, message) {
  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Error - Solid IdP</title>
  <style>${styles}</style>
</head>
<body>
  <div class="container">
    <div class="logo">${solidLogo}</div>
    <h1 style="color: #c00;">${escapeHtml(title)}</h1>
    <p>${escapeHtml(message)}</p>
    <a href="/" class="btn btn-secondary">Go Home</a>
  </div>
</body>
</html>
  `;
}

/**
 * Registration page HTML
 */
export function registerPage(uid = null, error = null, success = null, inviteOnly = false) {
  const safeUid = uid ? escapeJs(uid) : '';
  const inviteField = inviteOnly ? `
      <label for="invite">Invite Code</label>
      <input type="text" id="invite" name="invite" required
             placeholder="Enter your invite code" style="text-transform: uppercase;">
  ` : '';

  const passkeyRegisterSection = inviteOnly ? '' : `
    <div id="passkey-section">
      <label for="pk-username">Username</label>
      <input type="text" id="pk-username" placeholder="Choose a username" pattern="[a-z0-9]+"
             title="Lowercase letters and numbers only" autofocus>

      <button type="button" class="btn btn-passkey" onclick="registerWithPasskey()" id="pk-register-btn">
        ${passkeyIcon}
        Create Account with Passkey
      </button>

      <div id="pk-status" style="display:none; margin-top:12px; padding:12px; border-radius:8px; font-size:14px;"></div>

      <div class="divider"><span>or use a password</span></div>
    </div>
  `;

  const passkeyRegisterScript = inviteOnly ? '' : `
  <script type="module">
    import { generateSecretKey, getPublicKey } from 'https://esm.sh/nostr-tools@2.19.4/pure';

    // Fixed RP-scoped salt for PRF — deterministic, no server storage needed.
    // Different credentials still produce different PRF outputs (authenticator secret differs).
    const PRF_SALT = new Uint8Array(32);
    new TextEncoder().encode('solid-nostr-prf-v1').forEach(function(b, i) { PRF_SALT[i] = b; });

    // HKDF domain separation: PRF output -> secp256k1 private key
    async function deriveNostrKey(prfOutput) {
      var keyMaterial = await crypto.subtle.importKey('raw', prfOutput, 'HKDF', false, ['deriveBits']);
      var derived = await crypto.subtle.deriveBits({
        name: 'HKDF',
        hash: 'SHA-256',
        salt: new Uint8Array(32),
        info: new TextEncoder().encode('nostr-secp256k1-v1')
      }, keyMaterial, 256);
      return new Uint8Array(derived);
    }

    function bytesToHex(bytes) {
      return Array.from(bytes).map(function(b) { return b.toString(16).padStart(2, '0'); }).join('');
    }

    function base64urlToBuffer(base64url) {
      const base64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
      const padLen = (4 - base64.length % 4) % 4;
      const padded = base64 + '='.repeat(padLen);
      const binary = atob(padded);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    }

    function bufferToBase64url(buffer) {
      const bytes = new Uint8Array(buffer);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      return btoa(binary).replace(/[+]/g, '-').replace(/[/]/g, '_').replace(/=/g, '');
    }

    function detectDeviceName() {
      const ua = navigator.userAgent;
      if (/iPhone/.test(ua)) return 'iPhone';
      if (/iPad/.test(ua)) return 'iPad';
      if (/Mac/.test(ua)) return 'Mac';
      if (/Android/.test(ua)) return 'Android';
      if (/Windows/.test(ua)) return 'Windows';
      if (/Linux/.test(ua)) return 'Linux';
      return 'Security Key';
    }

    function showStatus(msg, isError) {
      const el = document.getElementById('pk-status');
      el.style.display = 'block';
      el.textContent = msg;
      el.style.background = isError ? '#fee' : '#efe';
      el.style.color = isError ? '#c00' : '#060';
      el.style.borderColor = isError ? '#fcc' : '#cfc';
      el.style.border = '1px solid';
    }

    function downloadKeyFile(username, pubkey, hexSecret, prfDerived) {
      var source = prfDerived ? 'PRF-derived (re-derivable from this passkey)' : 'Random (this file is the ONLY copy)';
      var keyContent = 'Nostr Private Key\\n==================\\n'
        + 'Username: ' + username + '\\n'
        + 'Public Key (hex): ' + pubkey + '\\n'
        + 'Private Key (hex): ' + hexSecret + '\\n'
        + 'DID: did:nostr:' + pubkey + '\\n'
        + 'Source: ' + source + '\\n\\n'
        + 'KEEP THIS FILE SAFE. Anyone with the private key can sign as your identity.\\n';
      var blob = new Blob([keyContent], { type: 'text/plain' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'nostr-keys-' + username + '.txt';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(a.href);
    }

    async function registerWithPasskey() {
      const btn = document.getElementById('pk-register-btn');
      const username = document.getElementById('pk-username').value.trim().toLowerCase();

      if (!username || !/^[a-z0-9]{3,}$/.test(username)) {
        showStatus('Username must be 3+ lowercase letters and numbers only.', true);
        return;
      }

      btn.disabled = true;
      btn.textContent = 'Setting up passkey...';

      try {
        // 1. Get WebAuthn registration options (username only, pubkey determined after PRF check)
        var optionsRes = await fetch('/idp/passkey/register-new/options', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: username })
        });
        var optionsData = await optionsRes.json();
        if (optionsData.error) {
          showStatus(optionsData.error, true);
          btn.disabled = false;
          btn.textContent = 'Create Account with Passkey';
          return;
        }

        var challengeKey = optionsData.challengeKey;

        // Convert base64url fields to ArrayBuffer
        optionsData.challenge = base64urlToBuffer(optionsData.challenge);
        optionsData.user.id = base64urlToBuffer(optionsData.user.id);
        if (optionsData.excludeCredentials) {
          optionsData.excludeCredentials = optionsData.excludeCredentials.map(function(c) {
            return Object.assign({}, c, { id: base64urlToBuffer(c.id) });
          });
        }

        // 2. Create passkey WITH PRF extension
        var credential = await navigator.credentials.create({
          publicKey: Object.assign({}, optionsData, {
            extensions: {
              prf: { eval: { first: PRF_SALT } }
            }
          })
        });

        btn.textContent = 'Generating identity...';

        // 3. Check PRF result — derive key or fall back
        var prfResult = credential.getClientExtensionResults().prf;
        var prfEnabled = false;
        var secretKey, pubkey;

        if (prfResult && prfResult.results && prfResult.results.first) {
          // PRF available: HKDF -> secp256k1 private key (deterministic, re-derivable)
          var prfOutput = new Uint8Array(prfResult.results.first);
          secretKey = await deriveNostrKey(prfOutput);
          pubkey = getPublicKey(secretKey);
          prfEnabled = true;
          console.log('PRF-derived Nostr key (re-derivable from this passkey)');
        } else {
          // PRF not available: random key (download is the only backup)
          secretKey = generateSecretKey();
          pubkey = getPublicKey(secretKey);
          console.log('Random Nostr key (PRF not supported by authenticator)');
        }

        // 4. Force download of private key backup
        downloadKeyFile(username, pubkey, bytesToHex(secretKey), prfEnabled);

        // Warn about cross-device PRF divergence
        if (prfEnabled && credential.authenticatorAttachment === 'cross-platform') {
          showStatus('Note: Cross-device passkeys may produce different keys. Keep your backup file safe.', true);
        }

        btn.textContent = 'Verifying...';

        // 5. Verify with server (pubkey sent here, after PRF derivation)
        var verifyRes = await fetch('/idp/passkey/register-new/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            challengeKey: challengeKey,
            pubkey: pubkey,
            prfEnabled: prfEnabled,
            credential: {
              id: credential.id,
              rawId: bufferToBase64url(credential.rawId),
              type: credential.type,
              response: {
                clientDataJSON: bufferToBase64url(credential.response.clientDataJSON),
                attestationObject: bufferToBase64url(credential.response.attestationObject),
                transports: credential.response.getTransports ? credential.response.getTransports() : []
              }
            },
            name: detectDeviceName()
          })
        });

        var result = await verifyRes.json();
        if (result.success) {
          // Store derived key in sessionStorage for NIP-98 signing
          try {
            sessionStorage.setItem('nostr_privkey', bytesToHex(secretKey));
            sessionStorage.setItem('nostr_pubkey', pubkey);
            sessionStorage.setItem('nostr_prf', prfEnabled ? '1' : '0');
          } catch (e) { /* sessionStorage may be unavailable */ }

          var uid = '${safeUid}';
          if (uid) {
            window.location.href = '/idp/interaction/' + encodeURIComponent(uid)
              + '/passkey-register-complete?accountId=' + encodeURIComponent(result.accountId)
              + '&token=' + encodeURIComponent(result.completionToken);
          } else {
            showStatus('Account created!' + (prfEnabled ? ' Your key is re-derivable from this passkey.' : ' Keep your downloaded key file safe.'), false);
            btn.textContent = 'Account Created';
          }
        } else {
          showStatus(result.error || 'Registration failed', true);
          btn.disabled = false;
          btn.textContent = 'Create Account with Passkey';
        }
      } catch (err) {
        if (err.name === 'NotAllowedError') {
          showStatus('Passkey creation was cancelled.', true);
        } else {
          console.error('Passkey registration error:', err);
          showStatus(err.message || 'Registration failed', true);
        }
        btn.disabled = false;
        btn.textContent = 'Create Account with Passkey';
      }
    }

    window.registerWithPasskey = registerWithPasskey;
  </script>
  `;

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Register - Solid IdP</title>
  <style>${styles}</style>
</head>
<body>
  <div class="container">
    <div class="logo">${solidLogo}</div>
    <h1>Create Account</h1>
    <p class="subtitle">Register for a new Solid Pod${inviteOnly ? ' (invite required)' : ''}</p>

    ${error ? `<div class="error">${escapeHtml(error)}</div>` : ''}
    ${success ? `<div class="error" style="background: #efe; border-color: #cfc; color: #060;">${escapeHtml(success)}</div>` : ''}

    ${passkeyRegisterSection}

    <form method="POST" action="/idp/register${uid ? `?uid=${uid}` : ''}">
      ${inviteField}

      <label for="username">Username</label>
      <input type="text" id="username" name="username" required ${inviteOnly ? '' : ''}
             placeholder="Choose a username" pattern="[a-z0-9]+"
             title="Lowercase letters and numbers only">

      <label for="password">Password</label>
      <input type="password" id="password" name="password" required
             placeholder="Choose a password">

      <label for="confirmPassword">Confirm Password</label>
      <input type="password" id="confirmPassword" name="confirmPassword" required
             placeholder="Confirm your password">

      <button type="submit" class="btn btn-primary">Create Account</button>
    </form>

    <p style="text-align: center; margin-top: 24px; color: #666; font-size: 14px;">
      Already have an account? <a href="${uid ? `/idp/interaction/${uid}` : '/idp/auth'}" style="color: #0066cc;">Sign In</a>
    </p>
  </div>
  ${passkeyRegisterScript}
</body>
</html>
  `;
}

/**
 * Passkey prompt page - shown after password login to encourage passkey setup
 */
export function passkeyPromptPage(uid, accountId) {
  const safeUid = escapeJs(uid);
  const safeAccountId = escapeJs(accountId);
  // Pre-escape the SVG for innerHTML assignment (no user data, just static SVG)
  const passkeyIconEscaped = passkeyIcon.replace(/'/g, "\\'").replace(/\n/g, '');

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Add a Passkey - Solid IdP</title>
  <style>${styles}</style>
</head>
<body>
  <div class="container">
    <div class="logo">${solidLogo}</div>
    <h1>Add a Passkey?</h1>
    <p class="subtitle">Sign in faster next time</p>

    <div class="client-info">
      <div class="client-name">Passkeys are more secure</div>
      <div class="client-uri">Use Touch ID, Face ID, or a security key instead of your password</div>
    </div>

    <button type="button" class="btn btn-passkey" onclick="registerPasskey()" id="addBtn">
      ${passkeyIcon}
      Add Passkey
    </button>

    <form method="GET" action="/idp/interaction/${escapeHtml(uid)}/passkey-skip">
      <button type="submit" class="btn btn-secondary">Skip for now</button>
    </form>
  </div>

  <script>
    var INTERACTION_UID = '${safeUid}';
    var ACCOUNT_ID = '${safeAccountId}';
    var PASSKEY_ICON = '${passkeyIconEscaped}';

    async function registerPasskey() {
      const btn = document.getElementById('addBtn');
      btn.disabled = true;
      btn.textContent = 'Setting up...';

      try {
        // Get registration options
        const optionsRes = await fetch('/idp/passkey/register/options', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ accountId: ACCOUNT_ID })
        });
        const options = await optionsRes.json();
        if (options.error) {
          alert('Error: ' + options.error);
          btn.disabled = false;
          btn.innerHTML = PASSKEY_ICON + ' Add Passkey';
          return;
        }

        // Save challengeKey for verification
        const challengeKey = options.challengeKey;

        // Convert base64url to ArrayBuffer
        options.challenge = base64urlToBuffer(options.challenge);
        options.user.id = base64urlToBuffer(options.user.id);
        if (options.excludeCredentials) {
          options.excludeCredentials = options.excludeCredentials.map(c => ({
            ...c,
            id: base64urlToBuffer(c.id)
          }));
        }

        // Prompt user to create passkey
        const credential = await navigator.credentials.create({ publicKey: options });

        // Send response to server
        const verifyRes = await fetch('/idp/passkey/register/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            accountId: ACCOUNT_ID,
            challengeKey: challengeKey,
            credential: {
              id: credential.id,
              rawId: bufferToBase64url(credential.rawId),
              type: credential.type,
              response: {
                clientDataJSON: bufferToBase64url(credential.response.clientDataJSON),
                attestationObject: bufferToBase64url(credential.response.attestationObject),
                transports: credential.response.getTransports ? credential.response.getTransports() : []
              }
            },
            name: detectDeviceName()
          })
        });

        const result = await verifyRes.json();
        if (result.success) {
          // Passkey added, continue to app - build URL safely
          const redirectUrl = '/idp/interaction/' + encodeURIComponent(INTERACTION_UID) + '/passkey-complete?accountId=' + encodeURIComponent(ACCOUNT_ID);
          window.location.href = redirectUrl;
        } else {
          alert('Failed to add passkey: ' + (result.error || 'Unknown error'));
          btn.disabled = false;
          btn.innerHTML = PASSKEY_ICON + ' Add Passkey';
        }
      } catch (err) {
        if (err.name === 'NotAllowedError') {
          // User cancelled
        } else {
          console.error('Passkey error:', err);
          alert('Failed to add passkey: ' + err.message);
        }
        btn.disabled = false;
        btn.innerHTML = PASSKEY_ICON + ' Add Passkey';
      }
    }

    function detectDeviceName() {
      const ua = navigator.userAgent;
      if (/iPhone/.test(ua)) return 'iPhone';
      if (/iPad/.test(ua)) return 'iPad';
      if (/Mac/.test(ua)) return 'Mac';
      if (/Android/.test(ua)) return 'Android';
      if (/Windows/.test(ua)) return 'Windows';
      if (/Linux/.test(ua)) return 'Linux';
      return 'Security Key';
    }

    function base64urlToBuffer(base64url) {
      const base64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
      const padLen = (4 - base64.length % 4) % 4;
      const padded = base64 + '='.repeat(padLen);
      const binary = atob(padded);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    }

    function bufferToBase64url(buffer) {
      const bytes = new Uint8Array(buffer);
      let binary = '';
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      return btoa(binary).replace(/[+]/g, '-').replace(/[/]/g, '_').replace(/=/g, '');
    }
  </script>
</body>
</html>
  `;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
