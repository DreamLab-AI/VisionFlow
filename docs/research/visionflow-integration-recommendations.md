# VisionFlow Integration Recommendations

**Analysis Date:** 2025-10-23
**System:** VisionFlow Container Architecture
**Priority:** Critical issues identified for integration stability

---

## Executive Summary

VisionFlow is a sophisticated GPU-accelerated WebXR knowledge graph visualization system with a well-designed three-tier architecture (Nginx → Rust Backend → React Frontend). The system is production-capable with minor critical fixes needed.

**Overall Assessment:**
- ✅ **Architecture:** Well-designed, clean separation of concerns
- ✅ **Performance:** GPU acceleration, efficient proxy layer
- ✅ **Development:** Hot reload, good developer experience
- ⚠️ **Monitoring:** Needs health checks and better logging
- ⚠️ **Production:** Requires optimization pass before deployment

---

## Critical Issues (Must Fix)

### 1. Backend Health Checks Missing ⚠️ CRITICAL

**Problem:**
- `/health` and `/api/health` endpoints timeout
- No way to verify backend is operational
- Container appears "healthy" even if backend crashes

**Impact:**
- Cannot implement load balancing
- Difficult to diagnose startup failures
- No automated health monitoring

**Recommendation:**
```rust
// src/handlers/health.rs
use actix_web::{get, HttpResponse, Responder};

#[get("/health")]
async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now(),
        "version": env!("CARGO_PKG_VERSION")
    }))
}

#[get("/health/ready")]
async fn readiness_check(gpu_actor: web::Data<Addr<GpuActor>>) -> impl Responder {
    // Check GPU availability
    let gpu_ready = gpu_actor.send(CheckHealth).await.is_ok();

    // Check MCP connection
    let mcp_ready = check_mcp_connection().await;

    if gpu_ready && mcp_ready {
        HttpResponse::Ok().json(serde_json::json!({
            "status": "ready",
            "gpu": true,
            "mcp": true
        }))
    } else {
        HttpResponse::ServiceUnavailable().json(serde_json::json!({
            "status": "not_ready",
            "gpu": gpu_ready,
            "mcp": mcp_ready
        }))
    }
}

#[get("/health/live")]
async fn liveness_check() -> impl Responder {
    HttpResponse::Ok().body("alive")
}
```

**Implementation:**
1. Add routes to `src/main.rs`
2. Test endpoints: `curl http://localhost:3001/health`
3. Update Nginx to use for health monitoring
4. Add Docker healthcheck to compose file

**Priority:** HIGH - Required for production deployment

---

### 2. Rust Backend Startup Logging Missing ⚠️ CRITICAL

**Problem:**
- Backend restarts show only wrapper script logs
- No application-level logging visible
- Impossible to diagnose initialization failures
- Last logged entry: "Starting Rust backend from /app/target/debug/webxr..."
- Then silence - no confirmation of successful startup

**Impact:**
- Cannot diagnose GPU initialization failures
- Cannot verify MCP connection status
- Cannot track startup time
- Difficult to debug configuration issues

**Recommendation:**
```rust
// src/main.rs
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logging first
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .format_timestamp_millis()
        .init();

    log::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    log::info!("🚀 VisionFlow WebXR Server Starting");
    log::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    log::info!("Version: {}", env!("CARGO_PKG_VERSION"));
    log::info!("Build: {} mode", if cfg!(debug_assertions) { "debug" } else { "release" });

    // Load configuration
    let config = load_config()?;
    log::info!("✓ Configuration loaded from {}", config.path());

    // Initialize GPU
    log::info!("Initializing CUDA GPU support...");
    let gpu_actor = match GpuActor::new() {
        Ok(actor) => {
            log::info!("✓ GPU initialized successfully");
            log::info!("  Device: {}", actor.device_name());
            log::info!("  Compute capability: {}", actor.compute_capability());
            log::info!("  Memory: {} GB", actor.memory_gb());
            actor.start()
        },
        Err(e) => {
            log::error!("✗ GPU initialization failed: {}", e);
            log::warn!("Continuing without GPU acceleration");
            return Err(e.into());
        }
    };

    // Initialize MCP connection
    log::info!("Connecting to MCP server at {}:{}...",
        std::env::var("MCP_HOST").unwrap_or_else(|_| "localhost".to_string()),
        std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string())
    );

    match initialize_mcp().await {
        Ok(_) => log::info!("✓ MCP connection established"),
        Err(e) => {
            log::warn!("⚠ MCP connection failed: {}", e);
            log::warn!("Falling back to mock MCP");
        }
    }

    // Start HTTP server
    let bind_addr = format!("0.0.0.0:{}",
        std::env::var("SYSTEM_NETWORK_PORT").unwrap_or_else(|_| "4000".to_string())
    );

    log::info!("Starting HTTP server...");
    log::info!("Binding to {}", bind_addr);

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .wrap(Cors::permissive())
            .app_data(web::Data::new(gpu_actor.clone()))
            .configure(configure_routes)
    })
    .bind(&bind_addr)?
    .run()
    .await?;

    log::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    log::info!("✓ Server running at http://{}", bind_addr);
    log::info!("✓ Health check: http://{}/health", bind_addr);
    log::info!("✓ API endpoints: http://{}/api/*", bind_addr);
    log::info!("✓ WebSocket: ws://{}/wss", bind_addr);
    log::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    log::info!("Ready to accept connections");

    Ok(())
}
```

**Additional Logging:**
```rust
// Throughout codebase
log::debug!("Processing graph with {} nodes", node_count);
log::info!("Graph layout calculated in {:.2}ms", elapsed);
log::warn!("High GPU memory usage: {}%", usage);
log::error!("Failed to process node {}: {}", node_id, error);
```

**Implementation:**
1. Add structured logging throughout codebase
2. Use appropriate log levels (debug, info, warn, error)
3. Include context (timing, resource usage, IDs)
4. Test by tailing logs: `docker exec visionflow_container tail -f /app/logs/rust.log`

**Priority:** HIGH - Critical for operations

---

### 3. Remove Redundant Vite Proxy Configuration ⚠️ MEDIUM

**Problem:**
- Vite configured to proxy API requests to backend
- Nginx already handles all routing
- Unnecessary complexity and potential confusion
- Could cause double-proxying or routing conflicts

**Current Vite Config:**
```typescript
proxy: {
  '/api': {
    target: 'http://visionflow_container:4000',
    changeOrigin: true,
  }
}
```

**Issue:**
- Vite dev server at :5173 proxies to backend :4000
- But users access through Nginx at :3001
- Nginx proxies `/api/*` to :4000
- Result: Two proxy layers, unnecessary complexity

**Recommendation:**
```typescript
// vite.config.ts - Remove proxy section entirely
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    hmr: {
      clientPort: 3001,
      path: '/vite-hmr',
    },
    // Remove proxy section - Nginx handles all routing
  }
})
```

**Update API Client:**
```typescript
// client/src/api/client.ts
// Use relative URLs - Nginx will handle routing
const API_BASE = '/api';  // Not 'http://localhost:4000/api'

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});
```

**Benefits:**
- Simpler architecture
- Single source of truth for routing (Nginx)
- Easier to debug
- Consistent behavior in dev and production

**Priority:** MEDIUM - Clean up technical debt

---

### 4. Docker Compose Missing from Version Control 📦 MEDIUM

**Problem:**
- No `docker-compose.yml` found in repository
- Container orchestration not documented
- Difficult to reproduce deployment
- Manual container management required

**Recommendation:**
Create comprehensive Docker Compose configuration:

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  visionflow:
    build:
      context: .
      dockerfile: Dockerfile.dev
      args:
        BUILDKIT_INLINE_CACHE: 1

    container_name: visionflow_container
    hostname: webxr

    ports:
      - "3001:3001"  # Main entry point

    environment:
      # Rust Backend
      - RUST_LOG=debug
      - RUST_BACKTRACE=1
      - RUST_LOG_REDIRECT=true
      - SYSTEM_NETWORK_PORT=4000
      - DOCKER_ENV=1

      # MCP Integration
      - MCP_TRANSPORT=tcp
      - MCP_TCP_PORT=9500
      - MCP_HOST=agentic-workstation
      - CLAUDE_FLOW_HOST=agentic-workstation
      - MCP_LOG_LEVEL=debug

      # Vite Frontend
      - NODE_ENV=development
      - VITE_DEV_SERVER_PORT=5173
      - VITE_HMR_PORT=24678

    volumes:
      # Source code (bind mounts for live reload)
      - ./src:/app/src:ro
      - ./client:/app/client:rw
      - ./Cargo.toml:/app/Cargo.toml:ro
      - ./Cargo.lock:/app/Cargo.lock:ro

      # Configuration
      - ./nginx.dev.conf:/etc/nginx/nginx.conf:ro
      - ./supervisord.dev.conf:/app/supervisord.dev.conf:ro
      - ./data/settings.yaml:/app/settings.yaml:rw

      # Data directories
      - ./data/markdown:/app/data/markdown:rw
      - ./data/metadata:/app/data/metadata:rw
      - ./data/user_settings:/app/user_settings:rw

      # Logs (persist on host)
      - ./logs:/app/logs:rw
      - ./logs/nginx:/var/log/nginx:rw

      # Build caches (named volumes for speed)
      - cargo-cache:/root/.cargo/registry
      - cargo-git:/root/.cargo/git
      - cargo-target:/app/target
      - npm-cache:/root/.npm

      # Docker socket (for dev-exec and container management)
      - /var/run/docker.sock:/var/run/docker.sock

    runtime: nvidia  # NVIDIA GPU support

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu, compute, utility]

    networks:
      - docker_ragflow

    restart: unless-stopped

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3001/health"]
      interval: 30s
      timeout: 3s
      start_period: 60s
      retries: 3

networks:
  docker_ragflow:
    external: true

volumes:
  cargo-cache:
    driver: local
  cargo-git:
    driver: local
  cargo-target:
    driver: local
  npm-cache:
    driver: local
```

**Production Compose:**
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
      - frontend
    networks:
      - visionflow
    restart: always

  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
      args:
        BUILD_MODE: release

    deploy:
      replicas: 2
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    environment:
      - RUST_LOG=info
      - SYSTEM_NETWORK_PORT=4000

    networks:
      - visionflow
      - docker_ragflow

    restart: always

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:4000/health"]
      interval: 10s
      timeout: 3s
      retries: 3

  frontend:
    build:
      context: ./client
      dockerfile: Dockerfile.frontend
      args:
        NODE_ENV: production

    deploy:
      replicas: 2

    networks:
      - visionflow

    restart: always

  redis:
    image: redis:alpine
    networks:
      - visionflow
    restart: always

networks:
  visionflow:
  docker_ragflow:
    external: true
```

**Benefits:**
- Reproducible deployments
- Easy scaling
- Environment management
- Health checks automated
- Network configuration documented

**Priority:** MEDIUM - Operational improvement

---

## Medium Priority Improvements

### 5. Add Comprehensive Logging 📊

**Implementation:**
```rust
// Use tracing for structured logging
use tracing::{info, debug, warn, error, instrument};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[instrument(skip(data))]
async fn process_graph_update(data: GraphData) -> Result<()> {
    debug!("Processing graph update with {} nodes", data.nodes.len());

    let start = Instant::now();
    // ... processing ...
    let elapsed = start.elapsed();

    info!(
        nodes = data.nodes.len(),
        edges = data.edges.len(),
        duration_ms = elapsed.as_millis(),
        "Graph update processed"
    );

    Ok(())
}
```

**Benefits:**
- Better debugging
- Performance tracking
- Audit trail
- Correlation IDs

---

### 6. Implement Prometheus Metrics 📈

**Recommendation:**
```rust
// Cargo.toml
[dependencies]
actix-web-prom = "0.8"

// src/main.rs
use actix_web_prom::PrometheusMetricsBuilder;

let prometheus = PrometheusMetricsBuilder::new("webxr")
    .endpoint("/metrics")
    .build()
    .unwrap();

HttpServer::new(move || {
    App::new()
        .wrap(prometheus.clone())
        .configure(configure_routes)
})
```

**Metrics to Track:**
- Request rate and latency
- WebSocket connections
- GPU memory usage
- Graph node/edge count
- MCP message rate
- Error rate

**Grafana Dashboard:**
```json
{
  "dashboard": {
    "title": "VisionFlow Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "gpu_memory_used_bytes / gpu_memory_total_bytes"
          }
        ]
      }
    ]
  }
}
```

---

### 7. Production Build Optimization 🚀

**Backend (Cargo.toml):**
```toml
[profile.release]
opt-level = 3           # Maximum optimization
lto = "fat"             # Link-time optimization
codegen-units = 1       # Better optimization, slower compile
strip = true            # Remove debug symbols
panic = "abort"         # Smaller binary

[profile.release.package."*"]
opt-level = 3
```

**Frontend Build:**
```typescript
// vite.config.ts
export default defineConfig({
  build: {
    target: 'es2020',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,  // Remove console.log in production
        drop_debugger: true,
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'babylon': ['@babylonjs/core', '@babylonjs/gui'],
          'vendor': ['react', 'react-dom'],
          'ui': ['@radix-ui/react-dialog', '@radix-ui/react-select'],
        }
      }
    },
    chunkSizeWarningLimit: 1000,
  }
})
```

**Benefits:**
- 30-50% smaller binaries
- 10-20% faster execution
- Better caching with chunking

---

### 8. Add Rate Limiting ⚡

**Nginx Configuration:**
```nginx
# nginx.prod.conf
http {
    # Rate limiting zones
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=ws:10m rate=5r/s;

    # Connection limiting
    limit_conn_zone $binary_remote_addr zone=addr:10m;

    server {
        # API endpoints - 10 req/s with burst of 20
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            limit_conn addr 10;
            # ... rest of config
        }

        # WebSocket - 5 req/s
        location /wss {
            limit_req zone=ws burst=5 nodelay;
            # ... rest of config
        }
    }
}
```

**Rust Backend:**
```rust
// Use actix-web-ratelimit
use actix_web_ratelimit::{RateLimiter, MemoryStore};

let store = MemoryStore::new();
App::new()
    .wrap(
        RateLimiter::new(store.clone())
            .with_interval(Duration::from_secs(60))
            .with_max_requests(100)
    )
```

---

### 9. Implement Caching Strategy 💾

**Nginx Static Asset Caching:**
```nginx
# Cache zone
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=static_cache:10m
                 max_size=1g inactive=60m use_temp_path=off;

# Static assets
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
    proxy_cache static_cache;
    proxy_cache_valid 200 1d;
    proxy_cache_valid 404 1m;
    proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
    proxy_cache_background_update on;
    proxy_cache_lock on;

    add_header X-Cache-Status $upstream_cache_status;
    add_header Cache-Control "public, max-age=86400";

    proxy_pass http://vite_frontend;
}
```

**Backend Redis Caching:**
```rust
// Add Redis for graph data caching
use redis::AsyncCommands;

async fn get_graph(
    graph_id: &str,
    redis: web::Data<redis::Client>
) -> Result<Graph> {
    let mut conn = redis.get_async_connection().await?;

    // Try cache first
    if let Ok(cached) = conn.get::<_, String>(&format!("graph:{}", graph_id)).await {
        return Ok(serde_json::from_str(&cached)?);
    }

    // Cache miss - load from storage
    let graph = load_graph_from_storage(graph_id).await?;

    // Cache for 5 minutes
    let _: () = conn.set_ex(
        &format!("graph:{}", graph_id),
        serde_json::to_string(&graph)?,
        300
    ).await?;

    Ok(graph)
}
```

---

## Low Priority Enhancements

### 10. HTTPS/SSL Support 🔒

**Let's Encrypt Integration:**
```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d visionflow.info -d www.visionflow.info

# Auto-renewal
certbot renew --dry-run
```

**Nginx SSL Configuration:**
```nginx
server {
    listen 443 ssl http2;
    server_name visionflow.info;

    ssl_certificate /etc/letsencrypt/live/visionflow.info/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/visionflow.info/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name visionflow.info;
    return 301 https://$server_name$request_uri;
}
```

---

### 11. Monitoring and Alerting 🔔

**Prometheus Alerts:**
```yaml
# prometheus/alerts.yml
groups:
  - name: visionflow
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: GPUMemoryHigh
        expr: gpu_memory_usage_percent > 90
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage above 90%"

      - alert: ServiceDown
        expr: up{job="visionflow"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "VisionFlow service is down"
```

---

### 12. CI/CD Pipeline 🔄

**GitHub Actions Workflow:**
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run backend tests
        run: cargo test --features gpu

      - name: Run clippy
        run: cargo clippy -- -D warnings

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install frontend deps
        run: cd client && npm ci

      - name: Run frontend tests
        run: cd client && npm test

      - name: Build frontend
        run: cd client && npm run build

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker images
        run: |
          docker build -t visionflow-backend:${{ github.sha }} -f Dockerfile.backend .
          docker build -t visionflow-frontend:${{ github.sha }} -f Dockerfile.frontend ./client

      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push visionflow-backend:${{ github.sha }}
          docker push visionflow-frontend:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          # Deploy via SSH, Kubernetes, or your deployment method
          ssh deploy@server "cd /app && docker-compose pull && docker-compose up -d"
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
**Must complete before production use**

1. ✅ Add health check endpoints
   - `/health` - liveness
   - `/health/ready` - readiness
   - `/health/live` - simple check
   - Test thoroughly

2. ✅ Implement startup logging
   - GPU initialization logs
   - MCP connection logs
   - Service readiness logs
   - Error handling logs

3. ✅ Remove Vite proxy config
   - Update vite.config.ts
   - Update API client
   - Test all API calls
   - Verify WebSocket connections

4. ✅ Create Docker Compose files
   - Development compose
   - Production compose
   - Document usage
   - Test deployments

**Acceptance Criteria:**
- [ ] Health endpoints return 200 OK
- [ ] Startup logs show all initialization steps
- [ ] All API calls work through Nginx
- [ ] `docker-compose up` works without manual steps

---

### Phase 2: Medium Priority (Week 2-3)
**Improve observability and reliability**

1. Comprehensive logging
2. Prometheus metrics
3. Production build optimization
4. Rate limiting
5. Caching strategy

**Acceptance Criteria:**
- [ ] Structured logs in production
- [ ] Metrics dashboard in Grafana
- [ ] Production build 30% faster
- [ ] Rate limiting prevents abuse
- [ ] Cache hit rate > 70% for static assets

---

### Phase 3: Low Priority (Week 4+)
**Polish and automation**

1. HTTPS/SSL setup
2. Monitoring and alerting
3. CI/CD pipeline
4. Documentation updates
5. Load testing

**Acceptance Criteria:**
- [ ] HTTPS enabled with valid certificate
- [ ] Alerts configured in PagerDuty/Slack
- [ ] Automated deployments on push
- [ ] Load test passes 1000 concurrent users
- [ ] Documentation updated

---

## Testing Checklist

### Manual Testing

**Backend Health:**
```bash
# Health check
curl http://localhost:3001/health
# Expected: {"status":"healthy","timestamp":"...","version":"0.1.0"}

# Readiness check
curl http://localhost:3001/health/ready
# Expected: {"status":"ready","gpu":true,"mcp":true}

# Liveness check
curl http://localhost:3001/health/live
# Expected: alive
```

**API Endpoints:**
```bash
# Graph API
curl http://localhost:3001/api/graph/list
# Expected: JSON array of graphs

# Settings API
curl http://localhost:3001/api/settings
# Expected: JSON settings object
```

**WebSocket:**
```bash
# Use wscat for testing
npm install -g wscat

# Graph WebSocket
wscat -c ws://localhost:3001/wss
# Expected: Connection established, receive graph updates
```

**Frontend:**
```bash
# Access in browser
open http://localhost:3001/

# Check HMR
# 1. Edit client/src/App.tsx
# 2. Save file
# 3. Browser should auto-reload
```

---

### Automated Testing

**Backend Tests:**
```rust
// tests/health_checks.rs
#[actix_rt::test]
async fn health_check_works() {
    let app = test::init_service(App::new().configure(configure_routes)).await;
    let req = test::TestRequest::get().uri("/health").to_request();
    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());
}

#[actix_rt::test]
async fn readiness_check_reports_dependencies() {
    let app = test::init_service(App::new().configure(configure_routes)).await;
    let req = test::TestRequest::get().uri("/health/ready").to_request();
    let resp = test::call_service(&app, req).await;
    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body["gpu"].as_bool().is_some());
    assert!(body["mcp"].as_bool().is_some());
}
```

**Frontend Tests:**
```typescript
// client/src/api/client.test.ts
import { apiClient } from './client';
import axios from 'axios';

jest.mock('axios');

describe('API Client', () => {
  it('uses relative URLs for API calls', () => {
    expect(apiClient.defaults.baseURL).toBe('/api');
  });

  it('handles API errors gracefully', async () => {
    (axios.get as jest.Mock).mockRejectedValue(new Error('Network error'));

    await expect(apiClient.get('/graph/list')).rejects.toThrow();
  });
});
```

**Integration Tests:**
```bash
#!/bin/bash
# tests/integration/test_full_flow.sh

echo "Testing full VisionFlow integration..."

# 1. Check health
echo "1. Checking health..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/health)
if [ "$STATUS" != "200" ]; then
    echo "FAIL: Health check returned $STATUS"
    exit 1
fi
echo "PASS: Health check OK"

# 2. Check frontend
echo "2. Checking frontend..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/)
if [ "$STATUS" != "200" ]; then
    echo "FAIL: Frontend returned $STATUS"
    exit 1
fi
echo "PASS: Frontend OK"

# 3. Check API
echo "3. Checking API..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/api/health)
if [ "$STATUS" != "200" ]; then
    echo "FAIL: API returned $STATUS"
    exit 1
fi
echo "PASS: API OK"

# 4. Check WebSocket
echo "4. Checking WebSocket..."
# Use websocat or similar
WSTEST=$(timeout 5s websocat ws://localhost:3001/wss || echo "failed")
if [ "$WSTEST" == "failed" ]; then
    echo "FAIL: WebSocket connection failed"
    exit 1
fi
echo "PASS: WebSocket OK"

echo ""
echo "✓ All integration tests passed"
```

---

## Performance Benchmarks

**Load Testing with k6:**
```javascript
// tests/load/basic_load.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 50 },   // Ramp up to 50 users
    { duration: '3m', target: 50 },   // Stay at 50 users
    { duration: '1m', target: 100 },  // Ramp up to 100 users
    { duration: '3m', target: 100 },  // Stay at 100 users
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests < 500ms
    http_req_failed: ['rate<0.01'],    // Error rate < 1%
  },
};

export default function() {
  // Test health endpoint
  let res = http.get('http://localhost:3001/health');
  check(res, {
    'health check status 200': (r) => r.status === 200,
    'health check response time < 100ms': (r) => r.timings.duration < 100,
  });

  // Test API endpoint
  res = http.get('http://localhost:3001/api/graph/list');
  check(res, {
    'API status 200': (r) => r.status === 200,
    'API response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);
}
```

**Run Load Test:**
```bash
k6 run tests/load/basic_load.js

# Expected Results:
# ✓ health check status 200
# ✓ health check response time < 100ms
# ✓ API status 200
# ✓ API response time < 500ms
#
# checks.........................: 100.00% ✓ 48000  ✗ 0
# http_req_duration..............: avg=45ms    min=10ms  med=40ms  max=150ms  p(95)=85ms
# http_reqs......................: 24000   133.33/s
```

---

## Monitoring Dashboard

**Grafana Dashboard JSON:**
```json
{
  "dashboard": {
    "title": "VisionFlow Production Monitoring",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(http_requests_total[5m])"
        }],
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8}
      },
      {
        "id": 2,
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(http_requests_total{status=~\"5..\"}[5m])"
        }],
        "alert": {
          "threshold": 0.01,
          "message": "Error rate above 1%"
        },
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8}
      },
      {
        "id": 3,
        "title": "Response Time (p95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
        }],
        "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8}
      },
      {
        "id": 4,
        "title": "GPU Memory Usage",
        "targets": [{
          "expr": "gpu_memory_used_bytes / gpu_memory_total_bytes * 100"
        }],
        "alert": {
          "threshold": 90,
          "message": "GPU memory above 90%"
        },
        "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8}
      },
      {
        "id": 5,
        "title": "Active WebSocket Connections",
        "targets": [{
          "expr": "websocket_connections_active"
        }],
        "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8}
      },
      {
        "id": 6,
        "title": "MCP Message Rate",
        "targets": [{
          "expr": "rate(mcp_messages_total[5m])"
        }],
        "gridPos": {"x": 12, "y": 16, "w": 12, "h": 8}
      }
    ]
  }
}
```

---

## Success Criteria

### Development Environment
- ✅ Container starts in < 2 minutes
- ✅ Hot reload works for frontend (< 1s)
- ✅ Backend rebuild works automatically
- ✅ All logs visible and well-formatted
- ✅ Health checks pass consistently

### Production Environment
- ✅ Health checks pass 99.9% uptime
- ✅ API response time p95 < 500ms
- ✅ WebSocket latency < 100ms
- ✅ Error rate < 0.1%
- ✅ GPU memory usage < 80%
- ✅ Zero-downtime deployments
- ✅ Automatic failover works
- ✅ Monitoring alerts functional

---

## Conclusion

VisionFlow has a solid architectural foundation with clean separation of concerns and efficient GPU acceleration. The critical improvements focus on:

1. **Observability:** Health checks and logging
2. **Simplicity:** Remove redundant configuration
3. **Automation:** Docker Compose and CI/CD
4. **Performance:** Production optimizations
5. **Reliability:** Monitoring and alerting

With these improvements, VisionFlow will be production-ready with excellent operational characteristics.

---

## References

**Documentation Created:**
1. `/home/devuser/docs/visionflow-architecture-analysis.md` - Complete system analysis
2. `/home/devuser/docs/visionflow-architecture-diagrams.md` - Visual architecture diagrams
3. `/home/devuser/docs/visionflow-integration-recommendations.md` - This document

**Key Files to Review:**
- `/etc/nginx/nginx.conf` - Nginx proxy configuration
- `/app/supervisord.dev.conf` - Process management
- `/app/client/vite.config.ts` - Frontend build config
- `/app/Cargo.toml` - Backend dependencies
- `/app/settings.yaml` - Application configuration

**External Resources:**
- Actix-web: https://actix.rs/
- Vite: https://vitejs.dev/
- Babylon.js: https://www.babylonjs.com/
- CUDA: https://docs.nvidia.com/cuda/
- MCP Protocol: (agentic-workstation documentation)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-23
**Author:** System Architecture Analysis Team
