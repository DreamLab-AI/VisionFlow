# User Permissions System

## Overview

The VisionFlow settings system implements a permission-based access control model centered around the "power user" concept. This document explains the permission system, how it integrates with Nostr authentication, and how to grant/revoke permissions.

## Permission Model

### Power User Concept

**Power User** is a binary permission flag that grants users the ability to:
- Modify global default settings
- View other users' settings (read-only)
- Grant/revoke power user permissions (if they have it themselves)
- Access audit logs
- Override validation warnings (with explicit confirmation)

### Permission Hierarchy

```
┌─────────────────────────────────────┐
│         System Administrator        │
│  (Initial power user, via config)   │
└──────────────┬──────────────────────┘
               │ can grant
               ↓
┌─────────────────────────────────────┐
│           Power Users               │
│  • Modify global settings           │
│  • Grant power user to others       │
│  • View all user settings           │
│  • Access audit logs                │
└──────────────┬──────────────────────┘
               │ regular users
               ↓
┌─────────────────────────────────────┐
│          Regular Users              │
│  • Modify own settings only         │
│  • View own settings                │
│  • Cannot grant permissions         │
└─────────────────────────────────────┘
```

## Nostr Authentication Integration

### Authentication Flow

```
1. Client generates Nostr event with signature
   ↓
2. Server verifies signature using Nostr SDK
   ↓
3. Extract pubkey from event
   ↓
4. Query users table for permission status
   ↓
5. Attach user context to request
```

### Nostr Event Format

Authentication uses NIP-98 (HTTP Auth):

```json
{
  "kind": 27235,
  "created_at": 1634567890,
  "tags": [
    ["u", "https://visionflow.info/api/settings"],
    ["method", "PUT"]
  ],
  "content": "",
  "pubkey": "abc123...",
  "sig": "def456..."
}
```

### Verification Implementation

```rust
use nostr_sdk::prelude::*;

pub async fn verify_nostr_auth(event: Event) -> Result<String> {
    // Verify signature
    event.verify()?;

    // Check timestamp (within 5 minutes)
    let now = Utc::now().timestamp();
    let created = event.created_at.as_u64() as i64;
    if (now - created).abs() > 300 {
        return Err(Error::AuthExpired);
    }

    // Extract pubkey
    let pubkey = event.pubkey.to_hex();

    Ok(pubkey)
}
```

## Database Schema

### Users Table

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pubkey TEXT NOT NULL UNIQUE,          -- Nostr public key (hex)
    display_name TEXT,
    is_power_user BOOLEAN DEFAULT FALSE,  -- Permission flag
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    metadata TEXT                         -- JSON: {"email": "...", "avatar": "..."}
);

CREATE INDEX idx_users_pubkey ON users(pubkey);
CREATE INDEX idx_users_power ON users(is_power_user);
```

## Permission Checks

### Checking Power User Status

```rust
use crate::models::user::User;

pub fn check_power_user(user: &User) -> Result<()> {
    if !user.is_power_user {
        return Err(Error::PermissionDenied {
            message: "Power user permission required".to_string(),
            required_permission: "power_user".to_string(),
        });
    }
    Ok(())
}
```

### Middleware for Protected Routes

```rust
use actix_web::{dev::ServiceRequest, Error};
use actix_web_httpauth::extractors::bearer::BearerAuth;

pub async fn power_user_validator(
    req: ServiceRequest,
    credentials: BearerAuth,
) -> Result<ServiceRequest, Error> {
    let pubkey = verify_nostr_auth(credentials.token()).await?;

    let user = get_user_by_pubkey(&pubkey).await?;

    if !user.is_power_user {
        return Err(ErrorForbidden("Power user permission required"));
    }

    req.extensions_mut().insert(user);
    Ok(req)
}
```

### Route Protection

```rust
use actix_web::{web, HttpResponse};
use actix_web_httpauth::middleware::HttpAuthentication;

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    let power_user_auth = HttpAuthentication::bearer(power_user_validator);

    cfg.service(
        web::scope("/api/settings")
            .service(
                web::resource("")
                    .route(web::get().to(get_settings))      // Public
                    .route(web::put().to(update_settings)    // Protected
                        .wrap(power_user_auth.clone()))
            )
    );
}
```

## Granting and Revoking Permissions

### Grant Power User Permission

**API Endpoint:** `POST /api/admin/users/:pubkey/grant-power-user`

**Authentication:** Requires power user permission

**Request:**
```json
{
  "reason": "Trusted administrator for physics tuning"
}
```

**Implementation:**

```rust
use rusqlite::{Connection, params};

pub async fn grant_power_user(
    conn: &Connection,
    target_pubkey: &str,
    granted_by: &str,
    reason: Option<String>,
) -> Result<()> {
    // Update user record
    conn.execute(
        "UPDATE users SET is_power_user = TRUE WHERE pubkey = ?1",
        params![target_pubkey]
    )?;

    // Audit log
    conn.execute(
        "INSERT INTO settings_audit_log
         (user_id, action, details, timestamp)
         VALUES (?1, 'permission_granted', ?2, CURRENT_TIMESTAMP)",
        params![
            target_pubkey,
            serde_json::json!({
                "granted_by": granted_by,
                "reason": reason,
                "permission": "power_user"
            }).to_string()
        ]
    )?;

    info!("Power user granted: {} by {}", target_pubkey, granted_by);
    Ok(())
}
```

### Revoke Power User Permission

**API Endpoint:** `POST /api/admin/users/:pubkey/revoke-power-user`

**Authentication:** Requires power user permission

**Request:**
```json
{
  "reason": "No longer requires elevated permissions"
}
```

**Implementation:**

```rust
pub async fn revoke_power_user(
    conn: &Connection,
    target_pubkey: &str,
    revoked_by: &str,
    reason: Option<String>,
) -> Result<()> {
    // Cannot revoke yourself
    if target_pubkey == revoked_by {
        return Err(Error::CannotRevokeSelf);
    }

    // Update user record
    conn.execute(
        "UPDATE users SET is_power_user = FALSE WHERE pubkey = ?1",
        params![target_pubkey]
    )?;

    // Audit log
    conn.execute(
        "INSERT INTO settings_audit_log
         (user_id, action, details, timestamp)
         VALUES (?1, 'permission_revoked', ?2, CURRENT_TIMESTAMP)",
        params![
            target_pubkey,
            serde_json::json!({
                "revoked_by": revoked_by,
                "reason": reason,
                "permission": "power_user"
            }).to_string()
        ]
    )?;

    warn!("Power user revoked: {} by {}", target_pubkey, revoked_by);
    Ok(())
}
```

## Initial Power User Setup

### Bootstrap Administrator

On first startup, create an initial power user from environment variable:

```rust
use std::env;

pub async fn bootstrap_admin(conn: &Connection) -> Result<()> {
    let admin_pubkey = env::var("ADMIN_PUBKEY")
        .map_err(|_| Error::NoAdminConfigured)?;

    // Check if any power users exist
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM users WHERE is_power_user = TRUE",
        [],
        |row| row.get(0)
    )?;

    if count == 0 {
        info!("No power users found, creating bootstrap admin: {}", admin_pubkey);

        conn.execute(
            "INSERT INTO users (pubkey, display_name, is_power_user, created_at)
             VALUES (?1, 'Admin', TRUE, CURRENT_TIMESTAMP)
             ON CONFLICT(pubkey) DO UPDATE SET is_power_user = TRUE",
            params![admin_pubkey]
        )?;

        // Audit log
        conn.execute(
            "INSERT INTO settings_audit_log
             (user_id, action, details, timestamp)
             VALUES (?1, 'bootstrap_admin', 'Initial admin user created', CURRENT_TIMESTAMP)",
            params![admin_pubkey]
        )?;
    }

    Ok(())
}
```

### Environment Configuration

```bash
# .env file
ADMIN_PUBKEY=abc123...  # Your Nostr pubkey (hex format)
```

## Audit Logging

Every permission change is logged for security and compliance.

### Audit Log Schema

```sql
CREATE TABLE settings_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,            -- Subject of the action
    action TEXT NOT NULL,             -- 'permission_granted', 'permission_revoked'
    setting_key TEXT,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    details TEXT,                     -- JSON blob with additional context
    ip_address TEXT,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Query Audit Log

**Get Permission Changes:**

```sql
SELECT
    u.display_name,
    a.action,
    json_extract(a.details, '$.granted_by') AS granted_by,
    json_extract(a.details, '$.reason') AS reason,
    a.timestamp
FROM settings_audit_log a
JOIN users u ON a.user_id = u.pubkey
WHERE a.action IN ('permission_granted', 'permission_revoked')
ORDER BY a.timestamp DESC
LIMIT 50;
```

**API Endpoint:** `GET /api/admin/audit/permissions`

```bash
curl -X GET http://localhost:4000/api/admin/audit/permissions \
  -H "Authorization: Nostr <power_user_pubkey>"
```

**Response:**

```json
{
  "total": 23,
  "entries": [
    {
      "userId": "abc123...",
      "displayName": "Alice",
      "action": "permission_granted",
      "grantedBy": "admin123...",
      "reason": "Trusted administrator",
      "timestamp": "2025-10-17T10:30:00Z"
    },
    {
      "userId": "def456...",
      "displayName": "Bob",
      "action": "permission_revoked",
      "revokedBy": "admin123...",
      "reason": "No longer active",
      "timestamp": "2025-10-17T09:15:00Z"
    }
  ]
}
```

## Permission-Based UI Rendering

Frontend should adapt based on user permissions.

### Client-Side Permission Check

```typescript
interface User {
  pubkey: string;
  displayName: string;
  isPowerUser: boolean;
}

function canModifyGlobalSettings(user: User): boolean {
  return user.isPowerUser;
}

function canGrantPermissions(user: User): boolean {
  return user.isPowerUser;
}

// Usage in React component
function SettingsPanel({ user }: { user: User }) {
  return (
    <div>
      <h2>Settings</h2>

      {canModifyGlobalSettings(user) ? (
        <button onClick={updateGlobalSettings}>
          Update Global Settings
        </button>
      ) : (
        <p>You can only modify your personal settings.</p>
      )}

      {canGrantPermissions(user) && (
        <AdminPanel />
      )}
    </div>
  );
}
```

### Backend Permission Enforcement

Always enforce permissions on the backend, never trust client-side checks alone:

```rust
pub async fn update_global_settings(
    user: User,
    settings: AppFullSettings,
) -> Result<HttpResponse> {
    // Backend enforcement
    check_power_user(&user)?;

    // Validate settings
    ValidationService::validate_settings(&settings)?;

    // Persist
    save_global_settings(&settings).await?;

    Ok(HttpResponse::Ok().json(json!({
        "message": "Settings updated successfully"
    })))
}
```

## Security Considerations

### 1. Signature Verification

Always verify Nostr signatures:
- Check signature validity
- Verify timestamp (prevent replay attacks)
- Ensure pubkey matches signature

### 2. Prevent Privilege Escalation

- Users cannot grant themselves power user permission
- Power users cannot revoke their own permission
- All permission changes are audited

### 3. Rate Limiting

Apply stricter rate limits to permission-related endpoints:

```rust
use crate::utils::validation::rate_limit::RateLimiter;

// 10 permission grants per hour
let rate_limiter = RateLimiter::new(
    10,                           // max requests
    Duration::from_secs(3600),    // per hour
);
```

### 4. IP Whitelisting (Optional)

For high-security environments, restrict admin operations to specific IPs:

```rust
const ADMIN_IP_WHITELIST: &[&str] = &[
    "192.168.1.100",
    "10.0.0.50",
];

pub fn check_admin_ip(ip: &str) -> Result<()> {
    if !ADMIN_IP_WHITELIST.contains(&ip) {
        return Err(Error::IpNotWhitelisted);
    }
    Ok(())
}
```

## Multi-Factor Authentication (Future Enhancement)

Consider adding MFA for power user operations:

```rust
pub struct MfaRequirement {
    enabled: bool,
    method: MfaMethod,  // TOTP, WebAuthn, etc.
}

pub enum MfaMethod {
    TOTP { secret: String },
    WebAuthn { credential_id: Vec<u8> },
    Email { email: String },
}
```

## Permission API Reference

### List Power Users

**Endpoint:** `GET /api/admin/users/power-users`

**Authentication:** Power user required

**Response:**
```json
{
  "total": 3,
  "users": [
    {
      "pubkey": "abc123...",
      "displayName": "Alice",
      "grantedAt": "2025-10-01T12:00:00Z",
      "lastLogin": "2025-10-17T10:30:00Z"
    }
  ]
}
```

### Check Permission

**Endpoint:** `GET /api/admin/users/:pubkey/permissions`

**Authentication:** Self or power user

**Response:**
```json
{
  "pubkey": "abc123...",
  "displayName": "Alice",
  "permissions": {
    "isPowerUser": true,
    "canModifyGlobalSettings": true,
    "canGrantPermissions": true,
    "canAccessAuditLog": true
  }
}
```

### Grant Permission

**Endpoint:** `POST /api/admin/users/:pubkey/grant-power-user`

**Authentication:** Power user required

**Request:**
```json
{
  "reason": "Trusted administrator"
}
```

**Response:**
```json
{
  "message": "Power user permission granted",
  "userId": "abc123...",
  "grantedBy": "admin123...",
  "timestamp": "2025-10-17T10:30:00Z"
}
```

### Revoke Permission

**Endpoint:** `POST /api/admin/users/:pubkey/revoke-power-user`

**Authentication:** Power user required

**Request:**
```json
{
  "reason": "No longer requires access"
}
```

**Response:**
```json
{
  "message": "Power user permission revoked",
  "userId": "abc123...",
  "revokedBy": "admin123...",
  "timestamp": "2025-10-17T11:00:00Z"
}
```

## Best Practices

### For Administrators

1. **Grant sparingly:** Only give power user to trusted individuals
2. **Regular audits:** Review permission changes monthly
3. **Rotate permissions:** Revoke when no longer needed
4. **Use strong authentication:** Enable MFA when available
5. **Monitor audit logs:** Watch for suspicious activity

### For Developers

1. **Always check permissions server-side**
2. **Use middleware for route protection**
3. **Audit all permission changes**
4. **Fail securely:** Default to denying access
5. **Log permission failures for security monitoring**

### For Users

1. **Protect your Nostr private key**
2. **Use hardware wallets when possible**
3. **Log out when done**
4. **Report suspicious activity**

## Troubleshooting

### Permission Denied Errors

**Problem:** User gets 403 Forbidden when trying to modify settings.

**Solutions:**
1. Verify user has power user permission: `SELECT is_power_user FROM users WHERE pubkey = ?`
2. Check authentication token is valid
3. Ensure Nostr signature is not expired (5 minute window)
4. Review audit log for recent permission revocations

### Cannot Grant Permission

**Problem:** Power user cannot grant permission to another user.

**Solutions:**
1. Verify granting user has power user permission
2. Check target user exists in database
3. Ensure not trying to grant to self (use different endpoint)
4. Review rate limits on permission operations

### Bootstrap Admin Not Working

**Problem:** Initial admin user not created on startup.

**Solutions:**
1. Check `ADMIN_PUBKEY` environment variable is set
2. Verify pubkey is in hex format (not npub)
3. Check database is writable
4. Review startup logs for errors

## Related Documentation

- [Settings System Architecture](./settings-system.md)
- [Settings API Reference](./settings-api.md)
- [Database Schema](./settings-schema.md)
- [Migration Guide](./settings-migration-guide.md)
