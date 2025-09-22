# Reference: Environment Variables

This document provides a comprehensive reference for all environment variables used to configure the Multi-Agent Docker Environment. These should be set in the `.env` file in the project root.

### General & Resource Configuration

| Variable | Description | Default |
| :--- | :--- | :--- |
| `EXTERNAL_DIR` | Host path to mount into `/workspace/ext` for user projects. | `./.agent-mount/ext` |
| `HOST_UID` | Host user ID to map to the container's `dev` user for file permissions. | `1000` |
| `HOST_GID` | Host group ID to map to the container's `dev` group. | `1000` |
| `DOCKER_CPUS` | Number of CPU cores to allocate. Empty for auto-detection. | (auto) |
| `DOCKER_MEMORY` | RAM to allocate (e.g., `8g`). Empty for auto-detection. | (auto) |
| `NODE_ENV` | Sets the Node.js environment. Use `production` for deployments. | `development` |
| `DEBUG_MODE` | Enables (`true`) or disables (`false`) extra debugging output. | `true` |
| `VERBOSE_LOGGING` | Enables (`true`) or disables (`false`) verbose logging. | `true` |

### MCP & Application Ports

| Variable | Description | Default |
| :--- | :--- | :--- |
| `BLENDER_PORT` | TCP port for the Blender MCP server. | `9876` |
| `QGIS_PORT` | TCP port for the QGIS MCP server. | `9877` |
| `MCP_LOG_LEVEL` | Logging level for MCP services (`debug`, `info`, `warn`, `error`). | `info` |
| `MCP_TCP_PORT` | Main TCP port for MCP communication. | `9500` |
| `MCP_BRIDGE_PORT` | WebSocket bridge port for MCP communication. | `3002` |
| `MCP_WS_URL` | The URL the WebSocket client should connect to. | `ws://localhost:3002` |
| `MCP_TCP_HOST` | The host for the TCP server. | `localhost` |
| `MCP_HEALTH_PORT` | Port for MCP health check endpoint. | `9501` |

### Security: Authentication

| Variable | Description | Default |
| :--- | :--- | :--- |
| `WS_AUTH_ENABLED` | Enable (`true`) or disable (`false`) WebSocket authentication. | `true` |
| `WS_AUTH_TOKEN` | **CHANGE THIS!** Secret bearer token for WebSocket connections. | `your-secure-websocket-token-change-me` |
| `TCP_AUTH_TOKEN` | **CHANGE THIS!** Secret token for TCP connections. | `your-secure-tcp-token-change-me` |
| `JWT_SECRET` | **CHANGE THIS!** Secret key (min 32 chars) for signing JSON Web Tokens. | `your-super-secret-jwt-key-minimum-32-chars` |
| `API_KEY` | A generic API key for authenticating with external services. | `your-api-key-for-external-services` |
| `API_KEY_VALIDATION` | Enable (`true`) or disable (`false`) API key validation. | `true` |

### Security: Connection & Rate Limiting

| Variable | Description | Default |
| :--- | :--- | :--- |
| `WS_MAX_CONNECTIONS` | Maximum concurrent WebSocket connections. | `100` |
| `TCP_MAX_CONNECTIONS` | Maximum concurrent TCP connections. | `50` |
| `WS_CONNECTION_TIMEOUT` | Idle connection timeout in milliseconds. | `300000` (5 min) |
| `RATE_LIMIT_ENABLED` | Enable (`true`) or disable (`false`) rate limiting. | `true` |
| `RATE_LIMIT_WINDOW_MS` | The time window for rate limiting in milliseconds. | `60000` (1 min) |
| `RATE_LIMIT_MAX_REQUESTS` | Max requests per window per client. | `100` |
| `RATE_LIMIT_BURST_REQUESTS` | Max burst requests allowed. | `20` |
| `MAX_REQUEST_SIZE` | Maximum request body size in bytes. | `10485760` (10MB) |
| `MAX_MESSAGE_SIZE` | Maximum WebSocket message size in bytes. | `1048576` (1MB) |
| `MAX_BUFFER_SIZE` | Maximum internal buffer size in bytes. | `16777216` (16MB) |

### Security: Network & Encryption

| Variable | Description | Default |
| :--- | :--- | :--- |
| `CORS_ENABLED` | Enable (`true`) or disable (`false`) CORS protection. | `true` |
| `CORS_ALLOWED_ORIGINS` | Comma-separated list of allowed origins. | `http://localhost:3000,https://localhost:3000` |
| `CORS_ALLOWED_METHODS` | Comma-separated list of allowed HTTP methods. | `GET,POST,PUT,DELETE,OPTIONS` |
| `CORS_ALLOWED_HEADERS` | Comma-separated list of allowed headers. | `Content-Type,Authorization` |
| `SSL_ENABLED` | Enable (`true`) or disable (`false`) SSL/TLS encryption. | `false` |
| `SSL_CERT_PATH` | Container path to the SSL certificate file. | `/app/certs/server.crt` |
| `SSL_KEY_PATH` | Container path to the SSL private key file. | `/app/certs/server.key` |
| `SSL_CA_PATH` | Container path to the SSL CA certificate. | `/app/certs/ca.crt` |
| `ENCRYPTION_ENABLED` | Enable (`true`) or disable (`false`) end-to-end encryption. | `false` |
| `ENCRYPTION_ALGORITHM` | Encryption algorithm to use. | `aes-256-gcm` |
| `ENCRYPTION_KEY` | **CHANGE THIS!** Secret key for data encryption. | (none) |

### Security: Session & IP Management

| Variable | Description | Default |
| :--- | :--- | :--- |
| `SESSION_TIMEOUT` | Session inactivity timeout in milliseconds. | `1800000` (30 min) |
| `SESSION_CLEANUP_INTERVAL` | Session cleanup interval in milliseconds. | `300000` (5 min) |
| `MAX_CONCURRENT_SESSIONS` | Max concurrent sessions per client. | `10` |
| `AUTO_BLOCK_ENABLED` | Enable (`true`) or disable (`false`) automatic IP blocking. | `true` |
| `BLOCK_DURATION` | Duration to block malicious IPs in milliseconds. | `3600000` (1 hour) |
| `MAX_FAILED_ATTEMPTS` | Number of failed auth attempts before blocking an IP. | `3` |

### Monitoring & Health Checks

| Variable | Description | Default |
| :--- | :--- | :--- |
| `SECURITY_AUDIT_LOG` | Enable (`true`) or disable (`false`) security audit logging. | `true` |
| `PERFORMANCE_MONITORING` | Enable (`true`) or disable (`false`) performance metrics. | `true` |
| `HEALTH_CHECK_ENABLED` | Enable (`true`) or disable (`false`) health check endpoints. | `true` |
| `HEALTH_CHECK_INTERVAL` | Health check interval in milliseconds. | `30000` (30 sec) |

### Circuit Breaker

| Variable | Description | Default |
| :--- | :--- | :--- |
| `CIRCUITBREAKER_ENABLED` | Enable (`true`) or disable (`false`) circuit breaker. | `true` |
| `CIRCUITBREAKER_FAILURE_THRESHOLD` | Number of failures before opening circuit. | `5` |
| `CIRCUITBREAKER_TIMEOUT` | Circuit breaker timeout in milliseconds. | `30000` (30 sec) |
| `CIRCUITBREAKER_RESET_TIMEOUT` | Time before attempting to close circuit in milliseconds. | `60000` (1 min) |

### Database & Backup

| Variable | Description | Default |
| :--- | :--- | :--- |
| `DB_ENCRYPTION_ENABLED` | Enable (`true`) or disable (`false`) database encryption. | `false` |
| `DB_BACKUP_ENABLED` | Enable (`true`) or disable (`false`) automatic backups. | `true` |
| `DB_BACKUP_INTERVAL` | Backup interval in milliseconds. | `86400000` (24 hours) |

### Claude Authentication

| Variable | Description | Default |
| :--- | :--- | :--- |
| `CLAUDE_CONFIG_DIR` | Path to Claude's configuration directory. | `/home/dev/.claude` |
| `CLAUDE_CODE_OAUTH_TOKEN` | OAuth token for long-lived Claude sessions. | (optional) |

### MCP Service Control

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MCP_TCP_AUTOSTART` | Auto-start TCP server on container startup. | `true` |
| `MCP_ENABLE_TCP` | Enable TCP server. | `true` |
| `MCP_ENABLE_UNIX` | Enable Unix socket support. | `false` |

## Usage Example

Create a `.env` file in your project root:

```env
# Core Configuration
HOST_UID=1001
HOST_GID=1001
DOCKER_MEMORY=16g
DOCKER_CPUS=8

# Security - CHANGE THESE!
WS_AUTH_TOKEN=my-secure-websocket-token-abc123
TCP_AUTH_TOKEN=my-secure-tcp-token-xyz789
JWT_SECRET=my-super-secret-jwt-key-that-is-at-least-32-characters-long

# Production Settings
NODE_ENV=production
DEBUG_MODE=false
SSL_ENABLED=true
SSL_CERT_PATH=/app/certs/mycert.crt
SSL_KEY_PATH=/app/certs/mykey.key

# Custom Ports
MCP_TCP_PORT=9600
MCP_BRIDGE_PORT=3003
```

Remember to never commit your `.env` file to version control!