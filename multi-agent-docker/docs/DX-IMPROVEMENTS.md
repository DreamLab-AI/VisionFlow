# Developer Experience Improvements

## ✅ Completed Improvements

### 1. Enhanced Startup Script (`start-agentic-flow.sh`)
**Status**: Complete

Added convenience commands:
```bash
./start-agentic-flow.sh --shell    # Direct shell access
./start-agentic-flow.sh --clean    # Resource cleanup
./start-agentic-flow.sh --test     # Run validation suite
./start-agentic-flow.sh --status   # Health checks
./start-agentic-flow.sh --logs     # View logs
```

Features:
- RAGFlow network auto-detection and connection
- Comprehensive health checks for all services
- Color-coded output for better readability
- Safety confirmations for destructive operations

**Files Modified**:
- `start-agentic-flow.sh` - Added 5 new commands

---

### 2. MCP Tool Management CLI
**Status**: Complete

Created `mcp` command for simplified tool management:

```bash
# List all tools
mcp list

# Show tool details
mcp show web-summary

# Add new tool
mcp add weather npx "-y @modelcontextprotocol/server-weather" "Weather data"

# Remove tool
mcp remove weather

# Update tool
mcp update github --description "Updated GitHub integration"

# Validate configuration
mcp validate

# Backup/restore
mcp backup
mcp restore /home/devuser/.config/claude/backups/mcp-20250101_120000.json
```

Features:
- JSON validation before modifications
- Automatic categorization
- Backup/restore functionality
- Tool availability checking

**Files Created**:
- `docker/cachyos/scripts/mcp-cli.sh` - Complete CLI implementation
- Symlinked to `/usr/local/bin/mcp` in container

**Files Modified**:
- `docker/cachyos/Dockerfile.workstation` - Added symlink for `mcp` command

---

### 3. Prometheus Metrics Collection
**Status**: Complete

Added comprehensive metrics endpoint at `/metrics`:

**Default Metrics**:
- CPU usage
- Memory usage
- Event loop lag
- Garbage collection stats

**Custom Metrics**:
```javascript
// HTTP metrics
http_request_duration_seconds{method, route, status_code}
http_requests_total{method, route, status_code}

// Task metrics
active_tasks_total
completed_tasks_total{status}
task_duration_seconds{task_type, status}

// MCP Tool metrics
mcp_tool_invocations_total{tool_name, status}
mcp_tool_duration_seconds{tool_name}

// Worker metrics
worker_sessions_total

// Error metrics
api_errors_total{error_type, route}
```

**Usage**:
```bash
# Scrape metrics
curl http://localhost:9090/metrics

# Prometheus configuration
scrape_configs:
  - job_name: 'agentic-flow'
    static_configs:
      - targets: ['localhost:9090']
```

**Files Created**:
- `docker/cachyos/management-api/utils/metrics.js` - Metrics collector

**Files Modified**:
- `docker/cachyos/management-api/package.json` - Added `prom-client` dependency

---

### 4. OpenAPI/Swagger Documentation
**Status**: Prepared

Added Swagger dependencies to `package.json`:
- `@fastify/swagger@^8.14.0`
- `@fastify/swagger-ui@^3.0.0`

**Next Steps**: Integrate into `management-api/server.js` (see remaining work below)

---

## 🚧 Remaining Improvements

### 1. Complete OpenAPI Integration

**File**: `docker/cachyos/management-api/server.js`

Add after middleware registration:

```javascript
// OpenAPI/Swagger
await app.register(require('@fastify/swagger'), {
  openapi: {
    openapi: '3.0.0',
    info: {
      title: 'Agentic Flow Management API',
      description: 'HTTP API for managing AI agent workflows and MCP tools',
      version: '2.1.0'
    },
    servers: [
      {
        url: 'http://localhost:9090',
        description: 'Development server'
      }
    ],
    components: {
      securitySchemes: {
        apiKey: {
          type: 'apiKey',
          name: 'X-API-Key',
          in: 'header'
        }
      }
    },
    security: [
      { apiKey: [] }
    ]
  }
});

await app.register(require('@fastify/swagger-ui'), {
  routePrefix: '/docs',
  uiConfig: {
    docExpansion: 'list',
    deepLinking: false
  },
  staticCSP: true,
  transformStaticCSP: (header) => header,
  transformSpecification: (swaggerObject) => swaggerObject,
  transformSpecificationClone: true
});
```

Add schema definitions to routes:

```javascript
app.post('/api/v1/tasks', {
  schema: {
    description: 'Create a new task',
    tags: ['tasks'],
    body: {
      type: 'object',
      required: ['agent', 'task'],
      properties: {
        agent: { type: 'string', description: 'Agent type to use' },
        task: { type: 'string', description: 'Task description' },
        priority: { type: 'string', enum: ['low', 'normal', 'high'], default: 'normal' }
      }
    },
    response: {
      200: {
        type: 'object',
        properties: {
          taskId: { type: 'string' },
          status: { type: 'string' },
          createdAt: { type: 'string', format: 'date-time' }
        }
      }
    }
  }
}, async (request, reply) => {
  // Handler...
});
```

**Access**: `http://localhost:9090/docs`

---

### 2. Standardize Logging to stdout/stderr

**File**: `docker/cachyos/management-api/utils/logger.js`

Current logging writes to files. Update to:

```javascript
const pino = require('pino');

module.exports = pino({
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level: (label) => {
      return { level: label };
    }
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  // Remove file transport, use stdout only
  transport: process.env.NODE_ENV === 'development' ? {
    target: 'pino-pretty',
    options: {
      colorize: true,
      translateTime: 'SYS:standard',
      ignore: 'pid,hostname'
    }
  } : undefined
});
```

Update supervisord.conf to capture logs:

```ini
[program:management-api]
command=node /home/devuser/management-api/server.js
directory=/home/devuser/management-api
user=devuser
autostart=true
autorestart=true
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
```

**Benefits**:
- Docker logs work natively: `docker logs agentic-flow-cachyos`
- Kubernetes log aggregation works automatically
- No disk space issues from log files

---

### 3. Integrate Metrics into Management API

**File**: `docker/cachyos/management-api/server.js`

Add metrics endpoint and middleware:

```javascript
const metrics = require('./utils/metrics');

// Metrics middleware
app.addHook('onRequest', async (request, reply) => {
  request.startTime = Date.now();
});

app.addHook('onResponse', async (request, reply) => {
  const duration = (Date.now() - request.startTime) / 1000;
  metrics.recordHttpRequest(
    request.method,
    request.routerPath || request.url,
    reply.statusCode,
    duration
  );
});

// Metrics endpoint
app.get('/metrics', async (request, reply) => {
  reply.type('text/plain');
  return metrics.register.metrics();
});

// Update task handlers to record metrics
app.post('/api/v1/tasks', async (request, reply) => {
  const taskStart = Date.now();
  try {
    // Create task...
    metrics.setActiveTasks(processManager.getActiveCount());
    return { taskId, status: 'created' };
  } catch (error) {
    metrics.recordError(error.name, request.url);
    throw error;
  } finally {
    const duration = (Date.now() - taskStart) / 1000;
    metrics.recordTask('agent-task', 'completed', duration);
  }
});
```

---

### 4. Enhance Z.AI Resilience

**File**: `docker/cachyos/claude-zai/wrapper/server.js`

Add retry logic with exponential backoff:

```javascript
const axios = require('axios');
const axiosRetry = require('axios-retry');

const client = axios.create({
  baseURL: process.env.ANTHROPIC_BASE_URL || 'https://api.z.ai/api/anthropic',
  timeout: 30000,
  headers: {
    'Authorization': `Bearer ${process.env.ANTHROPIC_AUTH_TOKEN}`,
    'Content-Type': 'application/json'
  }
});

// Configure retry logic
axiosRetry(client, {
  retries: 3,
  retryDelay: axiosRetry.exponentialDelay,
  retryCondition: (error) => {
    // Retry on network errors and 5xx responses
    return axiosRetry.isNetworkOrIdempotentRequestError(error) ||
           (error.response && error.response.status >= 500);
  },
  onRetry: (retryCount, error, requestConfig) => {
    console.log(`Retry attempt ${retryCount} for ${requestConfig.url}`);
  }
});

async function callZAI(prompt, timeout = 30000) {
  try {
    const response = await client.post('/v1/messages', {
      model: 'glm-4.6',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 1000
    }, { timeout });

    return response.data;
  } catch (error) {
    console.error('Z.AI API error:', error.message);
    throw error;
  }
}
```

**Add to package.json**:
```json
"dependencies": {
  "axios": "^1.6.0",
  "axios-retry": "^4.0.0"
}
```

---

### 5. Enforce Structured JSON from Z.AI

**File**: `docker/cachyos/core-assets/scripts/web-summary-mcp-server.py`

Update the Z.AI prompt to request JSON output:

```python
def extract_topics_via_zai(summary):
    """Extract topics using Z.AI with structured JSON output"""

    prompt = f"""You are a semantic topic extractor. Analyze this summary and identify relevant topics.

Summary:
{summary}

Available topics:
{json.dumps(PERMITTED_TOPICS, indent=2)}

IMPORTANT: Respond with ONLY a valid JSON object in this exact format:
{{
  "formatted_summary": "The original summary text",
  "matched_topics": ["topic1", "topic2", "topic3"]
}}

Rules:
1. matched_topics must contain ONLY topics from the available topics list
2. Match topics semantically, not just by exact string matching
3. Include 3-7 most relevant topics
4. Return valid JSON only, no additional text"""

    try:
        response = httpx.post(
            f"{ZAI_CONTAINER_URL}/prompt",
            json={"prompt": prompt, "timeout": 15000},
            timeout=20.0
        )
        response.raise_for_status()
        result = response.json()

        if not result.get('success'):
            return []

        # Parse JSON response
        zai_output = result.get('response', '').strip()

        # Extract JSON from response (handle markdown code blocks)
        if '```json' in zai_output:
            zai_output = zai_output.split('```json')[1].split('```')[0].strip()
        elif '```' in zai_output:
            zai_output = zai_output.split('```')[1].split('```')[0].strip()

        parsed = json.loads(zai_output)
        topics = parsed.get('matched_topics', [])

        # Validate topics
        valid_topics = [t for t in topics if t in PERMITTED_TOPICS]
        return valid_topics

    except Exception as e:
        print(f"Z.AI topic extraction error: {e}", file=sys.stderr)
        return []
```

**Benefits**:
- Eliminates regex parsing brittleness
- Clear error messages for malformed responses
- Easier to extend with additional fields

---

### 6. Add Distributed Tracing (Optional)

For complex multi-agent workflows, integrate OpenTelemetry:

**Install dependencies**:
```json
{
  "dependencies": {
    "@opentelemetry/api": "^1.8.0",
    "@opentelemetry/sdk-node": "^0.50.0",
    "@opentelemetry/auto-instrumentations-node": "^0.42.0",
    "@opentelemetry/exporter-trace-otlp-http": "^0.50.0"
  }
}
```

**Create tracing wrapper** (`management-api/utils/tracing.js`):

```javascript
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { OTLPTraceExporter } = require('@opentelemetry/exporter-trace-otlp-http');

const traceExporter = new OTLPTraceExporter({
  url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318/v1/traces'
});

const sdk = new NodeSDK({
  traceExporter,
  instrumentations: [getNodeAutoInstrumentations()]
});

sdk.start();

process.on('SIGTERM', () => {
  sdk.shutdown()
    .then(() => console.log('Tracing terminated'))
    .catch((error) => console.error('Error terminating tracing', error))
    .finally(() => process.exit(0));
});
```

**Import in server.js**:
```javascript
require('./utils/tracing'); // Must be first import
```

**Deploy Jaeger for visualization**:
```yaml
# docker-compose.yml
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "4318:4318"    # OTLP HTTP receiver
    networks:
      - agentic-network
```

**Access**: `http://localhost:16686`

---

## Implementation Priority

1. **High Priority** (User-facing improvements):
   - Complete OpenAPI integration
   - Standardize logging to stdout/stderr
   - Integrate metrics endpoint

2. **Medium Priority** (Reliability):
   - Enhance Z.AI resilience
   - Enforce structured JSON from Z.AI

3. **Low Priority** (Advanced observability):
   - Add distributed tracing

---

## Testing Checklist

After implementing remaining improvements:

- [ ] OpenAPI docs accessible at `/docs`
- [ ] Metrics endpoint returns Prometheus format
- [ ] Logs appear in `docker logs` output
- [ ] Z.AI service retries transient failures
- [ ] Z.AI returns valid JSON (no parsing errors)
- [ ] MCP CLI commands work (`mcp list`, `mcp add`, etc.)
- [ ] Startup script commands work (`--shell`, `--clean`, `--test`)
- [ ] Health checks pass for all services

---

## Documentation Updates Needed

After implementation:

1. Update `docker/cachyos/docs/README.md`:
   - Add `/metrics` endpoint to Port Mappings
   - Add `/docs` endpoint for Swagger UI
   - Document `mcp` CLI command usage

2. Update `README.md`:
   - Add observability section highlighting metrics/tracing
   - Add MCP tool management section

3. Create `docker/cachyos/docs/OBSERVABILITY.md`:
   - Prometheus metrics guide
   - Grafana dashboard examples
   - Distributed tracing setup (if implemented)

---

## References

- [Fastify Swagger](https://github.com/fastify/fastify-swagger)
- [Prom-client](https://github.com/siimon/prom-client)
- [OpenTelemetry Node.js](https://opentelemetry.io/docs/instrumentation/js/getting-started/nodejs/)
- [Axios Retry](https://github.com/softonic/axios-retry)
