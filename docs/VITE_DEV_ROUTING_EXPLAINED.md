# Vite Development Routing Configuration Explained

## Architecture Overview

The VisionFlow development environment uses a sophisticated routing setup to handle API requests, WebSocket connections, and hot module replacement (HMR). This document explains how the routing works and common issues.

## Port Configuration

### Docker Development Mode
When running in Docker (`docker-compose.dev.yml`):

- **Port 3001** (exposed to host): Nginx reverse proxy - main entry point
- **Port 4000** (internal): Rust backend API server
- **Port 5173** (internal): Vite development server
- **Port 24678** (internal): Vite HMR WebSocket

### Local Development Mode
When running locally (without Docker):

- **Port 4000**: Rust backend API server (run directly)
- **Port 5173**: Vite development server (with proxy to 4000)

## Request Flow in Docker

```
Browser (localhost:3001)
    ↓
Nginx (port 3001)
    ├── /api/* → Rust Backend (port 4000)
    ├── /ws, /wss → Rust Backend WebSocket (port 4000)
    ├── /__vite_hmr → Vite HMR (port 5173)
    └── /* → Vite Dev Server (port 5173)
```

## Key Configuration Files

### 1. nginx.dev.conf
- Routes `/api/*` requests to Rust backend on port 4000
- Routes WebSocket connections to appropriate services
- Proxies frontend requests to Vite on port 5173
- Handles HMR WebSocket for hot reload

### 2. vite.config.ts
- When `DOCKER_ENV=true`: Disables Vite proxy (Nginx handles it)
- When `DOCKER_ENV` not set: Proxies `/api/*` to localhost:4000
- Configures HMR path to work with Nginx

### 3. docker-compose.dev.yml
- Sets `DOCKER_ENV=true` to disable Vite proxy
- Maps port 3001 to host (Nginx entry point)
- Configures internal service ports

## Common Issues and Solutions

### Issue 1: 400 Bad Request on /api/settings/batch

**Symptoms:**
- Browser shows 400 error for batch API calls
- Individual API calls may work

**Causes:**
1. Request format mismatch (server expects `{updates: [...]}`)
2. Validation failures on server side
3. Serialization issues (camelCase vs snake_case)

**Solution:**
- Server models use `#[serde(rename_all = "camelCase")]`
- Client sends camelCase, server handles camelCase
- Check server logs for detailed validation errors

### Issue 2: API calls going to wrong port

**Symptoms:**
- API calls fail with connection refused
- Requests going to port 3001 instead of 4000

**Causes:**
1. Vite proxy misconfigured
2. Missing DOCKER_ENV variable
3. Client using absolute URLs instead of relative

**Solution:**
- Ensure `DOCKER_ENV=true` in docker-compose.dev.yml
- Use relative URLs in client (`/api/...` not `http://localhost:3001/api/...`)
- Vite proxy should be disabled in Docker mode

### Issue 3: WebSocket connections failing

**Symptoms:**
- Real-time updates not working
- WebSocket upgrade fails

**Solution:**
- Check Nginx configuration for WebSocket headers
- Ensure `proxy_set_header Upgrade $http_upgrade`
- Verify connection upgrade mapping in nginx.conf

## Testing the Configuration

### 1. Test API endpoints:
```bash
# From host machine (through Nginx)
curl http://localhost:3001/api/settings

# Inside container (direct to Rust)
docker exec visionflow_container curl http://localhost:4000/api/settings
```

### 2. Test batch endpoints:
```bash
# Test batch GET
curl -X POST http://localhost:3001/api/settings/batch \
  -H "Content-Type: application/json" \
  -d '{"paths": ["physics.springK"]}'

# Test batch UPDATE
curl -X PUT http://localhost:3001/api/settings/batch \
  -H "Content-Type: application/json" \
  -d '{"updates": [{"path": "physics.springK", "value": 0.005}]}'
```

### 3. Check routing:
```bash
# See which service handles the request
docker logs visionflow_container -f | grep -E "proxy|route"
```

## Development Workflow

1. **Start services:**
   ```bash
   docker-compose -f docker-compose.dev.yml up
   ```

2. **Access application:**
   - Open http://localhost:3001 (NOT port 5173 or 4000)

3. **Monitor logs:**
   ```bash
   # Nginx logs
   docker exec visionflow_container tail -f /var/log/nginx/error.log
   
   # Rust backend logs
   docker logs visionflow_container -f | grep "settings"
   ```

4. **Rebuild after changes:**
   - Frontend (React): Automatic via HMR
   - Backend (Rust): Run `./scripts/dev-rebuild-rust.sh`

## Best Practices

1. **Always use relative URLs** in the client code (`/api/...`)
2. **Check server logs** for detailed error messages
3. **Verify environment variables** are set correctly
4. **Use the test script** to validate endpoints work
5. **Monitor Nginx logs** for routing issues

## Summary

The development routing is designed to provide a unified entry point (port 3001) while maintaining separation of concerns:
- Nginx handles all routing and proxying
- Vite serves the frontend with HMR
- Rust backend handles API and WebSocket
- Docker environment variables control behaviour

When issues arise, check:
1. Is `DOCKER_ENV=true` set?
2. Are you accessing via port 3001?
3. Are API paths relative?
4. What do the server logs say?