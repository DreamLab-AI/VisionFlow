# Security Architecture

This document outlines the comprehensive, enterprise-grade security features integrated into the multi-agent Docker environment.

### Guiding Principles
- **Defense in Depth**: Multiple layers of security controls are implemented.
- **Secure by Default**: The default configuration enables key security features.
- **Configurability**: Security settings can be tailored for development or hardened for production.

### Feature Breakdown

#### 🛡️ 1. Authentication & Authorization
- **Token-Based Auth**: Both WebSocket and TCP endpoints are protected by secret tokens.
  - **WebSocket**: Uses a standard `Authorization: Bearer <token>` header.
  - **TCP**: Uses a custom JSON-based authentication handshake.
- **JWT Support**: Includes support for JSON Web Tokens for managing sessions and API access.
- **Configuration**:
  - `WS_AUTH_ENABLED`, `WS_AUTH_TOKEN`
  - `TCP_AUTH_TOKEN`
  - `JWT_SECRET`

#### 🚦 2. Rate Limiting & DDoS Protection
- **Per-Client Rate Limiting**: Prevents abuse by limiting the number of requests a single client can make in a given time window.
- **Automatic IP Blocking**: Malicious IPs that repeatedly fail authentication or exceed rate limits are automatically blocked for a configurable duration.
- **Connection Limiting**: Restricts the maximum number of concurrent WebSocket and TCP connections to prevent resource exhaustion.
- **Configuration**:
  - `RATE_LIMIT_ENABLED`, `RATE_LIMIT_WINDOW_MS`, `RATE_LIMIT_MAX_REQUESTS`
  - `AUTO_BLOCK_ENABLED`, `BLOCK_DURATION`, `MAX_FAILED_ATTEMPTS`
  - `WS_MAX_CONNECTIONS`, `TCP_MAX_CONNECTIONS`

#### 🔍 3. Input Validation & Sanitization
- **Size Limits**: Enforces maximum sizes for incoming requests and messages to prevent buffer overflow and memory exhaustion attacks.
- **Protocol Validation**: Ensures all incoming messages conform to the JSON-RPC 2.0 specification.
- **Content Sanitization**: Actively filters requests to prevent common injection attacks like XSS and prototype pollution.
- **Configuration**:
  - `MAX_REQUEST_SIZE`, `MAX_MESSAGE_SIZE`

#### 🌐 4. Network Security
- **CORS Protection**: Implements Cross-Origin Resource Sharing policies to control which web domains can access the WebSocket endpoint.
- **SSL/TLS Encryption**: Full support for enabling SSL/TLS to encrypt all network traffic. This is essential for production deployments.
- **Security Headers**: The WebSocket server can be configured to send security-related HTTP headers like `Content-Security-Policy` and `X-Frame-Options`.
- **Configuration**:
  - `CORS_ENABLED`, `CORS_ALLOWED_ORIGINS`
  - `SSL_ENABLED`, `SSL_CERT_PATH`, `SSL_KEY_PATH`
  - `SECURITY_HEADERS_ENABLED`

#### 📊 5. Monitoring & Auditing
- **Security Audit Logging**: All significant security events (e.g., failed logins, rate limit violations, blocked IPs) are logged for auditing and threat analysis.
- **Health Check Endpoints**: Secure, real-time endpoints for monitoring the health and status of services.
- **Circuit Breaker Pattern**: Automatically detects failing services and temporarily stops sending requests to them, preventing cascading failures and improving system resilience.
- **Configuration**:
  - `SECURITY_AUDIT_LOG`
  - `HEALTH_CHECK_ENABLED`
  - `CIRCUITBREAKER_ENABLED`

### 🚨 Security Best Practices for Deployment

1.  **Change All Defaults**: Immediately change `WS_AUTH_TOKEN`, `TCP_AUTH_TOKEN`, and `JWT_SECRET` in your `.env` file.
2.  **Enable SSL/TLS**: For any non-local deployment, set `SSL_ENABLED=true` and provide valid SSL certificates.
3.  **Restrict CORS**: Set `CORS_ALLOWED_ORIGINS` to only the specific domains that need access.
4.  **Review Rate Limits**: Adjust rate limits to match your expected traffic patterns.
5.  **Monitor Logs**: Integrate the security audit logs with a monitoring and alerting system.
6.  **Use Docker Secrets**: For production, use Docker secrets to manage sensitive values instead of environment variables.

### Security Monitoring Commands

Monitor security status using these commands from within the container:

```bash
# View security audit logs
mcp-security-audit

# Check current connections
mcp-connections

# Run health check
mcp-health

# View blocked IPs (in security logs)
grep "BLOCKED" /app/mcp-logs/security/*.log
```

### Incident Response

In case of a security incident:

1. **Immediate Actions**:
   - Review security logs: `/app/mcp-logs/security/`
   - Check active connections: `ss -tulnp | grep -E ":(3002|9500)"`
   - Identify blocked IPs in logs

2. **Containment**:
   - Update authentication tokens immediately
   - Restart affected services: `mcp-restart`
   - Block malicious IPs at the firewall level if needed

3. **Recovery**:
   - Review and update security configuration
   - Ensure all patches are applied
   - Consider enabling additional security features

### Compliance Considerations

The security architecture supports common compliance requirements:

- **Audit Trails**: Comprehensive logging of security events
- **Access Control**: Strong authentication and authorization mechanisms
- **Data Protection**: Support for encryption in transit (SSL/TLS)
- **Session Management**: Configurable session timeouts and limits
- **Monitoring**: Real-time health checks and performance metrics