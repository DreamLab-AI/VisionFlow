# 502 Bad Gateway - Nginx Can't Reach Backend

## Status
- ✅ Rust backend compiles successfully (specta linker issue fixed)
- ✅ Backend process starts and runs
- ✅ Backend listening on `0.0.0.0:4000` (confirmed in logs)
- ❌ Nginx returns 502 when proxying to backend
- ❌ Client gets HTTP 502 errors from settings API

## Evidence from Logs
```
[2025-10-17T14:59:18Z INFO  webxr] Starting HTTP server on 0.0.0.0:4000
[2025-10-17T14:59:18Z INFO  actix_server::server] Actix runtime found; starting in Actix runtime
```

Backend is definitely listening.

## Root Cause
Nginx proxy configuration likely pointing to wrong address for backend. Common issues:
1. Nginx config uses `localhost:4000` instead of `127.0.0.1:4000` (DNS issue)
2. Nginx config uses old/wrong port
3. Docker network configuration issue

## Fixes to Try (Outside Container)

### 1. Check Nginx Config
Find your nginx config file (likely in `/etc/nginx` or mounted config):
```bash
# From host machine
docker exec <container_name> cat /etc/nginx/nginx.conf
# or
docker exec <container_name> find /etc/nginx -name "*.conf"
```

Look for lines like:
```nginx
location /api {
    proxy_pass http://localhost:4000;  # ← might need to be 127.0.0.1:4000
}
```

### 2. Fix Option A: Change localhost to 127.0.0.1
If config uses `localhost`, change to explicit IP:
```nginx
location /api {
    proxy_pass http://127.0.0.1:4000;  # ← explicit IP
}

location /wss {
    proxy_pass http://127.0.0.1:4000;
}
```

### 3. Fix Option B: Verify Port
Ensure all proxy_pass directives use port 4000 (not 8000 or other):
```nginx
proxy_pass http://127.0.0.1:4000;  # ← must match backend port
```

### 4. Reload Nginx
After fixing config:
```bash
docker exec <container_name> nginx -s reload
# or restart container
docker restart <container_name>
```

### 5. Test from Inside Container
```bash
docker exec <container_name> curl http://localhost:4000/api/health
docker exec <container_name> curl http://127.0.0.1:4000/api/health
```

One of these should work if backend is healthy.

## Quick Test Command
```bash
# From host
docker exec <container> sh -c "curl -s http://127.0.0.1:4000/api/health && echo 'Backend OK' || echo 'Backend unreachable'"
```

## Expected Result
After fix, client should successfully fetch settings and you'll see:
```
[SettingsStore] Settings initialized successfully
```

Instead of:
```
[SettingsStore] Failed to initialize: Error: HTTP 502: Bad Gateway
```

## If Still Failing
Check supervisor logs to ensure backend isn't crashing:
```bash
tail -f logs/supervisord.log
```

Should show:
```
INFO success: rust-backend entered RUNNING state
```

Not:
```
INFO exited: rust-backend (exit status 1; not expected)
```
