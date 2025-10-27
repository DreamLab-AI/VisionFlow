# Security Architecture

## 🔒 Overview

VisionFlow v1.0.0 implements defense-in-depth security with multiple layers of protection across the hexagonal architecture.

---

## 🛡️ Security Principles

### 1. Defense in Depth
Multiple independent layers of security controls

### 2. Least Privilege
Components have minimum necessary permissions

### 3. Fail Securely
Errors don't expose sensitive information

### 4. Separation of Duties
Security boundaries between layers

### 5. Audit Everything
Complete logging of security-relevant events

---

## 🏗️ Security Layers

```
┌─────────────────────────────────────────────────────┐
│              Network Security                        │
│  (TLS, Firewall, Rate Limiting)                     │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│         Authentication & Authorization               │
│  (JWT, API Keys, RBAC)                              │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│            Input Validation                          │
│  (Type Safety, Sanitization, Validation)            │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│          Application Security                        │
│  (CQRS, Event Validation, Actor Isolation)          │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│            Data Security                             │
│  (Encryption at Rest, Parameterized Queries)        │
└─────────────────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│          Audit & Monitoring                          │
│  (Event Logging, Intrusion Detection)               │
└─────────────────────────────────────────────────────┘
```

---

## 🔐 Authentication

### JWT Token-Based Authentication

**Token Structure**:
```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct JWTClaims {
    pub sub: String,           // User ID
    pub exp: i64,              // Expiration (Unix timestamp)
    pub iat: i64,              // Issued at
    pub roles: Vec<String>,    // User roles
    pub permissions: Vec<String>, // Fine-grained permissions
}
```

**Token Generation**:
```rust
use jsonwebtoken::{encode, Header, EncodingKey};

pub fn generate_token(user_id: &str, roles: Vec<String>) -> Result<String, AuthError> {
    let claims = JWTClaims {
        sub: user_id.to_string(),
        exp: (Utc::now() + Duration::hours(24)).timestamp(),
        iat: Utc::now().timestamp(),
        roles,
        permissions: derive_permissions(&roles),
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(get_jwt_secret().as_bytes()),
    )
    .map_err(|e| AuthError::TokenGenerationFailed(e.to_string()))
}
```

**Token Validation**:
```rust
use jsonwebtoken::{decode, Validation, DecodingKey};

pub fn validate_token(token: &str) -> Result<JWTClaims, AuthError> {
    decode::<JWTClaims>(
        token,
        &DecodingKey::from_secret(get_jwt_secret().as_bytes()),
        &Validation::default(),
    )
    .map(|data| data.claims)
    .map_err(|e| AuthError::InvalidToken(e.to_string()))
}
```

### API Key Authentication

**API Key Format**:
```
vf_live_[32_random_alphanumeric_chars]
vf_test_[32_random_alphanumeric_chars]
```

**Storage** (hashed with bcrypt):
```rust
use bcrypt::{hash, verify, DEFAULT_COST};

pub fn hash_api_key(key: &str) -> Result<String, SecurityError> {
    hash(key, DEFAULT_COST)
        .map_err(|e| SecurityError::HashingFailed(e.to_string()))
}

pub fn verify_api_key(key: &str, hash: &str) -> Result<bool, SecurityError> {
    verify(key, hash)
        .map_err(|e| SecurityError::VerificationFailed(e.to_string()))
}
```

---

## 🔑 Authorization

### Role-Based Access Control (RBAC)

**Roles**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    Admin,           // Full system access
    Editor,          // Create/edit/delete graphs
    Viewer,          // Read-only access
    ApiUser,         // Programmatic API access
}
```

**Permissions**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Permission {
    // Graph permissions
    GraphCreate,
    GraphRead,
    GraphUpdate,
    GraphDelete,

    // Ontology permissions
    OntologyLoad,
    OntologyValidate,

    // Settings permissions
    SettingsRead,
    SettingsWrite,

    // Admin permissions
    UserManage,
    SystemConfig,
}
```

**Permission Checks**:
```rust
impl JWTClaims {
    pub fn has_permission(&self, permission: Permission) -> bool {
        self.permissions.contains(&permission.to_string())
    }

    pub fn has_role(&self, role: Role) -> bool {
        self.roles.contains(&role.to_string())
    }
}
```

**Middleware Enforcement**:
```rust
use actix_web::{dev::ServiceRequest, Error, HttpMessage};

pub async fn check_permission(
    req: ServiceRequest,
    required_permission: Permission,
) -> Result<ServiceRequest, Error> {
    let token = extract_token(&req)?;
    let claims = validate_token(&token)?;

    if !claims.has_permission(required_permission) {
        return Err(AuthError::InsufficientPermissions.into());
    }

    req.extensions_mut().insert(claims);
    Ok(req)
}
```

---

## 🛡️ Input Validation

### Type-Safe Validation

**Using `validator` crate**:
```rust
use validator::{Validate, ValidationError};

#[derive(Debug, Deserialize, Validate)]
pub struct CreateNodeRequest {
    #[validate(length(min = 1, max = 255))]
    pub label: String,

    #[validate(range(min = -1000.0, max = 1000.0))]
    pub x: f32,

    #[validate(range(min = -1000.0, max = 1000.0))]
    pub y: f32,

    #[validate(range(min = -1000.0, max = 1000.0))]
    pub z: f32,

    #[validate(custom = "validate_metadata")]
    pub metadata: serde_json::Value,
}

fn validate_metadata(metadata: &serde_json::Value) -> Result<(), ValidationError> {
    // Ensure metadata is an object
    if !metadata.is_object() {
        return Err(ValidationError::new("metadata_not_object"));
    }

    // Ensure metadata size is reasonable (<10KB)
    let json_str = serde_json::to_string(metadata).unwrap();
    if json_str.len() > 10_000 {
        return Err(ValidationError::new("metadata_too_large"));
    }

    Ok(())
}
```

**Validation in Handler**:
```rust
use actix_web::{web, HttpResponse};

pub async fn create_node(
    req: web::Json<CreateNodeRequest>,
) -> Result<HttpResponse, ApiError> {
    // Validate input
    req.validate()
        .map_err(|e| ApiError::ValidationFailed(e.to_string()))?;

    // Process valid request...
    Ok(HttpResponse::Created().json(response))
}
```

### SQL Injection Prevention

**Parameterized Queries** (REQUIRED):
```rust
// ✅ SAFE - Parameterized query
conn.execute(
    "INSERT INTO kg_nodes (id, label, x, y, z) VALUES (?, ?, ?, ?, ?)",
    params![node_id, label, x, y, z],
)?;

// ❌ UNSAFE - String concatenation (NEVER DO THIS)
conn.execute(&format!(
    "INSERT INTO kg_nodes (id, label) VALUES ('{}', '{}')",
    node_id, label
), [])?;
```

**Type System Enforcement**:
```rust
// Repository trait enforces type-safe queries
#[async_trait]
pub trait KnowledgeGraphRepository {
    async fn create_node(&self, node: &Node) -> Result<NodeId, RepositoryError>;
    // No raw SQL string parameters allowed!
}
```

---

## 🔒 Data Security

### Encryption at Rest

**SQLite Encryption Extension (Optional)**:
```toml
# Cargo.toml
[dependencies]
sqlcipher = { version = "0.4", optional = true }

[features]
encryption = ["sqlcipher"]
```

**Encrypted Database**:
```rust
#[cfg(feature = "encryption")]
pub fn create_encrypted_pool(db_path: &Path, key: &str) -> Result<Pool<SqliteConnectionManager>> {
    let manager = SqliteConnectionManager::file(db_path)
        .with_init(move |conn| {
            conn.execute_batch(&format!("PRAGMA key = '{}';", key))?;
            conn.pragma_update(None, "cipher_page_size", 4096)?;
            Ok(())
        });

    Pool::builder()
        .build(manager)
        .map_err(|e| SecurityError::EncryptionFailed(e.to_string()))
}
```

### Sensitive Data Handling

**Scrubbing Errors**:
```rust
impl From<rusqlite::Error> for ApiError {
    fn from(err: rusqlite::Error) -> Self {
        // Don't expose internal database errors
        error!("Database error: {}", err);
        ApiError::InternalServerError("Database operation failed".to_string())
    }
}
```

**Logging Sanitization**:
```rust
pub fn sanitize_log_message(message: &str) -> String {
    // Remove potential PII, API keys, passwords
    let patterns = [
        (r"password=\S+", "password=***"),
        (r"api_key=\S+", "api_key=***"),
        (r"token=\S+", "token=***"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "***@***.***"),
    ];

    let mut sanitized = message.to_string();
    for (pattern, replacement) in patterns {
        let re = Regex::new(pattern).unwrap();
        sanitized = re.replace_all(&sanitized, replacement).to_string();
    }
    sanitized
}
```

---

## 🚨 Threat Model

### Threat Categories

#### 1. Network Attacks
| Threat | Mitigation |
|--------|-----------|
| **Man-in-the-Middle** | TLS 1.3, Certificate pinning |
| **DDoS** | Rate limiting, CDN, Load balancer |
| **Replay Attacks** | JWT expiration, Nonce validation |

#### 2. Authentication Attacks
| Threat | Mitigation |
|--------|-----------|
| **Brute Force** | Rate limiting, Account lockout |
| **Token Theft** | Short token expiration, Secure cookies |
| **Session Hijacking** | CSRF tokens, SameSite cookies |

#### 3. Injection Attacks
| Threat | Mitigation |
|--------|-----------|
| **SQL Injection** | Parameterized queries (enforced by type system) |
| **Command Injection** | No shell execution, Validated inputs |
| **XSS** | Content Security Policy, Output encoding |

#### 4. Data Security
| Threat | Mitigation |
|--------|-----------|
| **Data Breach** | Encryption at rest, Access controls |
| **Data Tampering** | Audit logs, Integrity checks |
| **Data Loss** | Backups, Redundancy |

---

## 📊 Security Monitoring

### Audit Logging

**Audit Events**:
```rust
#[derive(Debug, Serialize)]
pub struct AuditEvent {
    pub event_type: AuditEventType,
    pub user_id: String,
    pub timestamp: DateTime<Utc>,
    pub resource: String,
    pub action: String,
    pub result: AuditResult,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    ConfigurationChange,
    SecurityViolation,
}

#[derive(Debug, Serialize)]
pub enum AuditResult {
    Success,
    Failure { reason: String },
    Blocked { reason: String },
}
```

**Logging Implementation**:
```rust
impl AuditLogger for SqliteAuditLogger {
    async fn log_event(&self, event: AuditEvent) -> Result<(), AuditError> {
        let conn = self.pool.get()?;

        conn.execute(
            "INSERT INTO audit_log (event_type, user_id, timestamp, resource, action, result, metadata)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                event.event_type.to_string(),
                event.user_id,
                event.timestamp.to_rfc3339(),
                event.resource,
                event.action,
                serde_json::to_string(&event.result)?,
                serde_json::to_string(&event.metadata)?,
            ],
        )?;

        Ok(())
    }
}
```

### Intrusion Detection

**Suspicious Activity Detection**:
```rust
pub async fn detect_suspicious_activity(event: &AuditEvent) -> Option<SecurityAlert> {
    // Failed login attempts
    if event.event_type == AuditEventType::Authentication {
        let failed_attempts = count_failed_logins(&event.user_id, Duration::minutes(15));
        if failed_attempts >= 5 {
            return Some(SecurityAlert::BruteForceAttempt {
                user_id: event.user_id.clone(),
                attempts: failed_attempts,
            });
        }
    }

    // Unusual data access patterns
    if event.event_type == AuditEventType::DataAccess {
        let access_rate = calculate_access_rate(&event.user_id, Duration::minutes(1));
        if access_rate > 100 {
            return Some(SecurityAlert::UnusualAccessPattern {
                user_id: event.user_id.clone(),
                rate: access_rate,
            });
        }
    }

    None
}
```

---

## 🔧 Security Best Practices

### Development

1. **Never hardcode secrets** (use environment variables)
2. **Use type-safe queries** (no string concatenation)
3. **Validate all inputs** (use `validator` crate)
4. **Sanitize outputs** (prevent XSS)
5. **Log security events** (for audit trail)

### Deployment

1. **Enable TLS** (use Let's Encrypt)
2. **Firewall configuration** (whitelist only necessary ports)
3. **Regular updates** (security patches)
4. **Backup encryption keys** (secure key management)
5. **Monitor audit logs** (detect intrusions)

### Code Review Checklist

- [ ] No hardcoded secrets or API keys
- [ ] All database queries parameterized
- [ ] Input validation on all API endpoints
- [ ] Authorization checks before sensitive operations
- [ ] Error messages don't leak sensitive data
- [ ] Audit logging for security-relevant events
- [ ] Dependencies up to date (cargo audit)

---

## 📞 Vulnerability Reporting

### Reporting Process

1. **Email**: security@visionflow.io
2. **Subject**: "VisionFlow Security Vulnerability"
3. **Include**:
   - Vulnerability description
   - Steps to reproduce
   - Potential impact
   - Suggested fix (optional)

### Response Timeline

- **24 hours**: Initial acknowledgment
- **72 hours**: Initial assessment
- **30 days**: Fix and disclosure (if applicable)

### Responsible Disclosure

We follow responsible disclosure practices:
- Security researchers acknowledged in release notes
- No legal action against good-faith security research
- Coordinated public disclosure after fix

---

**VisionFlow Security Architecture**
Version 1.0.0 | Last Updated: 2025-10-27
