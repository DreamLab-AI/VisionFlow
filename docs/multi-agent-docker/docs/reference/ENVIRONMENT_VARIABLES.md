# Environment Variables Reference

Complete reference for all environment variables used in the Agentic Flow Docker environment.

## Quick Start

Copy `.env.example` to `.env` and configure required variables:

```bash
cp .env.example .env
```

At minimum, configure one AI provider API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_GEMINI_API_KEY).

## Variable Index

- [API Keys](#api-keys)
- [Management API](#management-api)
- [Model Router](#model-router)
- [GPU Configuration](#gpu-configuration)
- [Optional Services](#optional-services)
- [System Configuration](#system-configuration)
- [Common Configurations](#common-configurations)
- [Security Considerations](#security-considerations)

---

## API Keys

Authentication credentials for AI model providers and external services.

### ANTHROPIC_API_KEY

**Description:** API key for Anthropic Claude models
**Type:** String
**Default:** None
**Required:** Optional (but one provider required)
**Valid Values:** sk-ant-* format key from https://console.anthropic.com/

```bash
ANTHROPIC_API_KEY=sk-ant-api03-xxx
```

### ANTHROPIC_BASE_URL

**Description:** Base URL for Anthropic API requests (enables Z.AI proxy)
**Type:** URL
**Default:** https://api.anthropic.com (implicit)
**Required:** Required for Z.AI integration
**Valid Values:** Valid HTTPS URL

```bash
ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
```

**Note:** Set this to Z.AI endpoint to enable extended context and caching features.

### BRAVE_API_KEY

**Description:** API key for Brave Search integration
**Type:** String
**Default:** None
**Required:** Optional
**Valid Values:** Valid Brave API key from https://brave.com/search/api/

```bash
BRAVE_API_KEY=BSA...
```

### CONTEXT7_API_KEY

**Description:** API key for Context7 service
**Type:** String
**Default:** None
**Required:** Optional
**Valid Values:** Valid Context7 API key

```bash
CONTEXT7_API_KEY=ctx7_...
```

### GITHUB_TOKEN

**Description:** GitHub personal access token for repository integration
**Type:** String
**Default:** None
**Required:** Optional
**Valid Values:** GitHub PAT with appropriate scopes

```bash
GITHUB_TOKEN=ghp_...
```

**Required Scopes:** `repo`, `read:org` (minimum)

### GOOGLE_GEMINI_API_KEY

**Description:** API key for Google Gemini models
**Type:** String
**Default:** None
**Required:** Optional (but one provider required)
**Valid Values:** Valid API key from https://makersuite.google.com/

```bash
GOOGLE_GEMINI_API_KEY=AIza...
```

### OPENAI_API_KEY

**Description:** API key for OpenAI models
**Type:** String
**Default:** None
**Required:** Optional (but one provider required)
**Valid Values:** sk-* format key from https://platform.openai.com/

```bash
OPENAI_API_KEY=sk-proj-...
```

### OPENROUTER_API_KEY

**Description:** API key for OpenRouter multi-model access
**Type:** String
**Default:** None
**Required:** Optional
**Valid Values:** Valid OpenRouter API key from https://openrouter.ai/

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Management API

Configuration for the internal management API that controls workstation lifecycle.

### MANAGEMENT_API_KEY

**Description:** Authentication key for management API endpoints
**Type:** String
**Default:** change-this-secret-key
**Required:** Yes
**Valid Values:** Any secure string (min 16 characters recommended)

```bash
MANAGEMENT_API_KEY=your-secure-random-key-here
```

**Security:** MUST be changed from default before production use.

### MANAGEMENT_API_PORT

**Description:** Port for management API service
**Type:** Integer
**Default:** 9090
**Required:** No
**Valid Values:** 1024-65535

```bash
MANAGEMENT_API_PORT=9090
```

### MANAGEMENT_API_HOST

**Description:** Host binding for management API service
**Type:** String
**Default:** 0.0.0.0
**Required:** No
**Valid Values:** IP address or hostname

```bash
MANAGEMENT_API_HOST=0.0.0.0
```

**Options:**
- `0.0.0.0` - Listen on all interfaces
- `127.0.0.1` - Localhost only (more secure)
- Specific IP - Bind to specific interface

---

## Model Router

Configuration for intelligent model routing and fallback behavior.

### ROUTER_MODE

**Description:** Optimization strategy for model selection
**Type:** Enum
**Default:** performance
**Required:** No
**Valid Values:** `performance`, `cost`, `balanced`

```bash
ROUTER_MODE=performance
```

**Modes:**
- `performance` - Prioritize speed and capability
- `cost` - Minimize API costs
- `balanced` - Balance cost and performance

### PRIMARY_PROVIDER

**Description:** Default AI provider for routing
**Type:** Enum
**Default:** gemini
**Required:** No
**Valid Values:** `gemini`, `openai`, `claude`, `openrouter`

```bash
PRIMARY_PROVIDER=gemini
```

**Note:** Ensure corresponding API key is configured.

### FALLBACK_CHAIN

**Description:** Ordered list of providers to try if primary fails
**Type:** Comma-separated list
**Default:** gemini,openai,claude,openrouter
**Required:** No
**Valid Values:** Any combination of: `gemini`, `openai`, `claude`, `openrouter`

```bash
FALLBACK_CHAIN=gemini,openai,claude,openrouter
```

**Behavior:** Router attempts providers left-to-right until success.

---

## GPU Configuration

Hardware acceleration settings for compute-intensive workloads.

### GPU_ACCELERATION

**Description:** Enable GPU acceleration for supported workloads
**Type:** Boolean
**Default:** true
**Required:** No
**Valid Values:** `true`, `false`

```bash
GPU_ACCELERATION=true
```

**Requirements:** NVIDIA GPU with CUDA support, nvidia-docker runtime.

### CUDA_VISIBLE_DEVICES

**Description:** GPU device selection for CUDA workloads
**Type:** String
**Default:** all
**Required:** No
**Valid Values:** `all`, device IDs (e.g., `0`, `0,1`, `1,2,3`)

```bash
CUDA_VISIBLE_DEVICES=all
```

**Examples:**
- `all` - Use all available GPUs
- `0` - Use first GPU only
- `0,2` - Use first and third GPU
- `-1` - Disable GPU (equivalent to GPU_ACCELERATION=false)

---

## Optional Services

Configuration for additional integrated services.

### ENABLE_DESKTOP

**Description:** Enable VNC/noVNC desktop environment
**Type:** Boolean
**Default:** false
**Required:** No
**Valid Values:** `true`, `false`

```bash
ENABLE_DESKTOP=false
```

**Ports:** Exposes VNC on 5900 and noVNC on 6080 when enabled.

### ENABLE_CODE_SERVER

**Description:** Enable VS Code Server (code-server)
**Type:** Boolean
**Default:** false
**Required:** No
**Valid Values:** `true`, `false`

```bash
ENABLE_CODE_SERVER=false
```

**Port:** Exposes code-server on 8080 when enabled.

---

## System Configuration

General system and logging configuration.

### LOG_LEVEL

**Description:** Logging verbosity level
**Type:** Enum
**Default:** info
**Required:** No
**Valid Values:** `debug`, `info`, `warn`, `error`

```bash
LOG_LEVEL=info
```

**Levels:**
- `debug` - Verbose debugging output
- `info` - Standard informational messages
- `warn` - Warnings only
- `error` - Errors only

### NODE_ENV

**Description:** Node.js environment mode
**Type:** Enum
**Default:** production
**Required:** No
**Valid Values:** `production`, `development`, `test`

```bash
NODE_ENV=production
```

**Impact:** Affects error reporting, optimization, and debug features.

### CLAUDE_WORKER_POOL_SIZE

**Description:** Number of concurrent Claude worker processes
**Type:** Integer
**Default:** 4
**Required:** No
**Valid Values:** 1-32 (depends on system resources)

```bash
CLAUDE_WORKER_POOL_SIZE=4
```

**Recommendation:** Set to number of CPU cores for optimal performance.

### CLAUDE_MAX_QUEUE_SIZE

**Description:** Maximum queued requests for Claude workers
**Type:** Integer
**Default:** 50
**Required:** No
**Valid Values:** 1-1000

```bash
CLAUDE_MAX_QUEUE_SIZE=50
```

**Behavior:** Requests exceeding this limit are rejected with 503 error.

---

## Common Configurations

### Development Setup

```bash
# API Keys
ANTHROPIC_API_KEY=sk-ant-api03-xxx
ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic

# Management
MANAGEMENT_API_KEY=dev-key-not-for-production
MANAGEMENT_API_PORT=9090

# Router
ROUTER_MODE=performance
PRIMARY_PROVIDER=claude
FALLBACK_CHAIN=claude,gemini,openai

# GPU
GPU_ACCELERATION=true
CUDA_VISIBLE_DEVICES=0

# Services
ENABLE_DESKTOP=true
ENABLE_CODE_SERVER=true

# System
LOG_LEVEL=debug
NODE_ENV=development
CLAUDE_WORKER_POOL_SIZE=2
```

### Production Setup

```bash
# API Keys (multiple providers for redundancy)
ANTHROPIC_API_KEY=sk-ant-api03-xxx
OPENAI_API_KEY=sk-proj-xxx
GOOGLE_GEMINI_API_KEY=AIza...

# Management (secure)
MANAGEMENT_API_KEY=generated-secure-key-32-chars-min
MANAGEMENT_API_PORT=9090
MANAGEMENT_API_HOST=127.0.0.1

# Router (with fallbacks)
ROUTER_MODE=balanced
PRIMARY_PROVIDER=gemini
FALLBACK_CHAIN=gemini,claude,openai

# GPU
GPU_ACCELERATION=true
CUDA_VISIBLE_DEVICES=all

# Services (minimal)
ENABLE_DESKTOP=false
ENABLE_CODE_SERVER=false

# System
LOG_LEVEL=info
NODE_ENV=production
CLAUDE_WORKER_POOL_SIZE=8
CLAUDE_MAX_QUEUE_SIZE=100
```

### Cost-Optimized Setup

```bash
# API Keys (focus on cost-effective providers)
GOOGLE_GEMINI_API_KEY=AIza...
OPENROUTER_API_KEY=sk-or-v1-...

# Router (minimize costs)
ROUTER_MODE=cost
PRIMARY_PROVIDER=gemini
FALLBACK_CHAIN=gemini,openrouter

# GPU (disabled to save resources)
GPU_ACCELERATION=false

# Services (disabled)
ENABLE_DESKTOP=false
ENABLE_CODE_SERVER=false

# System
LOG_LEVEL=warn
NODE_ENV=production
CLAUDE_WORKER_POOL_SIZE=2
CLAUDE_MAX_QUEUE_SIZE=25
```

### Z.AI Extended Context Setup

```bash
# Z.AI Configuration
ANTHROPIC_API_KEY=sk-ant-api03-xxx
ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic

# Router (Claude primary for Z.AI)
PRIMARY_PROVIDER=claude
FALLBACK_CHAIN=claude,gemini,openai

# Workers (optimize for long context)
CLAUDE_WORKER_POOL_SIZE=8
CLAUDE_MAX_QUEUE_SIZE=100
```

---

## Security Considerations

### Secrets Management

**Critical Variables:**
- `MANAGEMENT_API_KEY` - MUST be changed from default
- All `*_API_KEY` variables - Never commit to version control
- `GITHUB_TOKEN` - Limit scopes to minimum required

**Best Practices:**

1. **Use environment-specific .env files:**
   ```bash
   .env.development
   .env.production
   .env.local (git-ignored)
   ```

2. **Rotate keys regularly:**
   - API keys: Every 90 days
   - MANAGEMENT_API_KEY: Every 30 days

3. **Secure storage:**
   - Use secrets management tools (Vault, AWS Secrets Manager)
   - Encrypt .env files at rest
   - Restrict file permissions: `chmod 600 .env`

4. **Access control:**
   - Bind MANAGEMENT_API_HOST to 127.0.0.1 in production
   - Use firewall rules to restrict API access
   - Enable authentication on all exposed services

### Network Security

**MANAGEMENT_API_HOST Settings:**

- `0.0.0.0` - Exposes API to network (use with firewall)
- `127.0.0.1` - Localhost only (recommended for production)
- Docker network IP - Container-to-container only

**Desktop/Code Server:**

When `ENABLE_DESKTOP=true` or `ENABLE_CODE_SERVER=true`:
- Use strong passwords
- Enable TLS/SSL
- Restrict access by IP
- Use VPN for remote access

### API Key Scopes

**GitHub Token Minimum Scopes:**
- `repo` - Repository access (if needed)
- `read:org` - Organization metadata (if needed)

**Avoid:**
- `admin:*` scopes unless absolutely required
- `delete:*` scopes
- Full `repo` scope if read-only access sufficient

### Validation

Environment variables are validated on startup. Missing required variables or invalid values will prevent service startup.

**Required Variables:**
- At least one provider API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_GEMINI_API_KEY)
- MANAGEMENT_API_KEY must be changed from default

**Validation Errors:**

```bash
# Check validation
docker-compose logs management-api | grep -i error

# Common issues
ERROR: MANAGEMENT_API_KEY is set to default value
ERROR: No provider API keys configured
ERROR: Invalid ROUTER_MODE value
```

### Audit Logging

Sensitive operations are logged when `LOG_LEVEL=info` or higher:

- API key validation (keys are redacted)
- Management API authentication attempts
- Provider fallback events
- Worker pool scaling events

**Log Monitoring:**

```bash
# Monitor authentication failures
docker-compose logs -f | grep "auth failed"

# Monitor API usage
docker-compose logs -f management-api | grep "provider:"
```

---

## Environment Loading Order

Variables are loaded with the following precedence (highest to lowest):

1. Shell environment variables
2. `.env.local` (git-ignored)
3. `.env.${NODE_ENV}` (e.g., `.env.production`)
4. `.env`
5. Default values in code

**Example:**

```bash
# Override specific variables
MANAGEMENT_API_PORT=9091 docker-compose up
```

---

## Troubleshooting

### Variable Not Taking Effect

1. Restart services: `docker-compose restart`
2. Rebuild if needed: `docker-compose up --build`
3. Check loading: `docker-compose config` shows resolved values
4. Verify syntax: No spaces around `=`

### API Key Issues

```bash
# Test provider connectivity
curl -H "x-api-key: $MANAGEMENT_API_KEY" \
     http://localhost:9090/api/health

# Check configured providers
docker-compose logs management-api | grep "provider configured"
```

### GPU Not Detected

```bash
# Verify GPU visibility
docker-compose exec workstation nvidia-smi

# Check CUDA devices
docker-compose exec workstation env | grep CUDA
```

---

## Related Documentation

- [Getting Started](../GETTING_STARTED.md) - Initial setup guide
- [Architecture](../architecture/ARCHITECTURE.md) - System architecture
- [Deployment](../DEPLOYMENT.md) - Production deployment
- [Security](../guides/SECURITY.md) - Security best practices

---

**Last Updated:** 2025-10-12
**Version:** 1.0.0
