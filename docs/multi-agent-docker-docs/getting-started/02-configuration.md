# Configuration Guide

The Multi-Agent Docker Environment is highly configurable through a single `.env` file. This guide explains the key settings and how to customize them for your needs.

### The `.env` File

All configuration is managed through an environment file. To get started, copy the example file:

```bash
cp .env.example .env
```

Now, you can edit the `.env` file to change the environment's behavior. **Never commit your `.env` file to version control.**

### Core Configuration

These settings control file permissions and resource allocation.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `EXTERNAL_DIR` | The path on your host machine to mount into the container at `/workspace/ext`. | `./.agent-mount/ext` |
| `HOST_UID` | Your host user ID. The container's `dev` user will match this to avoid file permission issues. | `1000` |
| `HOST_GID` | Your host group ID. The container's `dev` group will match this. | `1000` |
| `DOCKER_CPUS` | The number of CPU cores to allocate to the container. Leave empty for auto-detection. | (auto) |
| `DOCKER_MEMORY` | The amount of RAM to allocate (e.g., `8g`, `16g`). Leave empty for auto-detection. | (auto) |

### Application & MCP Ports

These variables configure the network ports for various services.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `BLENDER_PORT` | The TCP port for the Blender MCP server in the `gui-tools-container`. | `9876` |
| `QGIS_PORT` | The TCP port for the QGIS MCP server in the `gui-tools-container`. | `9877` |
| `MCP_TCP_PORT` | The main TCP port for MCP communication in the `multi-agent-container`. | `9500` |
| `MCP_BRIDGE_PORT` | The WebSocket bridge port for MCP communication. | `3002` |
| `MCP_LOG_LEVEL` | The logging level for MCP services (`debug`, `info`, `warn`, `error`). | `info` |

### Security Configuration

**WARNING:** It is critical to change the default security tokens before deploying this environment in a production or publicly accessible setting.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `WS_AUTH_ENABLED` | Enable (`true`) or disable (`false`) WebSocket authentication. | `true` |
| `WS_AUTH_TOKEN` | **CHANGE THIS!** The secret bearer token for WebSocket connections. | `your-secure-websocket-token-change-me` |
| `TCP_AUTH_TOKEN` | **CHANGE THIS!** The secret token for TCP connections. | `your-secure-tcp-token-change-me` |
| `JWT_SECRET` | **CHANGE THIS!** A secret key (min 32 chars) for signing JSON Web Tokens. | `your-super-secret-jwt-key-minimum-32-chars` |
| `RATE_LIMIT_ENABLED`| Enable (`true`) or disable (`false`) rate limiting to prevent abuse. | `true` |
| `RATE_LIMIT_MAX_REQUESTS` | The maximum number of requests allowed per `RATE_LIMIT_WINDOW_MS`. | `100` |
| `SSL_ENABLED` | Enable (`true`) or disable (`false`) SSL/TLS encryption for production. | `false` |
| `SSL_CERT_PATH` | Path inside the container to your SSL certificate file. | `/app/certs/server.crt` |
| `SSL_KEY_PATH` | Path inside the container to your SSL private key file. | `/app/certs/server.key` |

### Claude Authentication

The environment supports mounting your host Claude credentials for authenticated access:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `CLAUDE_CONFIG_DIR` | Path to Claude's configuration directory | `/home/dev/.claude` |
| `CLAUDE_CODE_OAUTH_TOKEN` | OAuth token for long-lived Claude sessions | (optional) |

The container automatically mounts:
- `~/.claude` → `/home/dev/.claude` (for credentials and config)
- `~/.claude.json` → `/home/dev/.claude.json` (for main config)

To authenticate Claude:
1. On your host: `claude login`
2. Complete the OAuth flow
3. Start the container - it will inherit your authentication

For a complete list of all available settings, including advanced options for CORS, rate limiting, and session management, please see the **[Environment Variables Reference](../reference/01-environment-variables.md)**.