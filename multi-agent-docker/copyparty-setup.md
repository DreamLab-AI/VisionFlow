# Copyparty File Browser Setup

## ✅ COMPLETE - Fully Integrated!

Copyparty is now fully integrated into the multi-agent-container and auto-starts with the container.

## Access URL
**http://192.168.0.51:3001/browser/**

## Architecture
```
Client (Browser)
    ↓
nginx (visionflow_container:3001) @ 172.18.0.9
    ↓  (proxied via /browser/)
copyparty (multi-agent-container:3923) @ 172.18.0.4
    ↓
/workspace filesystem (read-only)
```

## Configuration Files

### 1. Dockerfile
**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/Dockerfile`

```dockerfile
# Install copyparty file server
RUN /opt/venv312/bin/pip install --no-cache-dir copyparty
```

### 2. Supervisor Configuration
**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/supervisord.conf`

Copyparty runs as a supervised service that auto-starts and auto-restarts on failure.

### 3. Nginx Reverse Proxy
**Location:** `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/nginx.dev.conf`

Routes `/browser/` requests to the copyparty service with proper headers and timeouts.

## Service Management

### Check Status
```bash
docker exec multi-agent-container supervisorctl status copyparty
```

### View Logs
```bash
docker exec multi-agent-container tail -f /workspace/logs/copyparty.log
```

### Restart Service
```bash
docker exec multi-agent-container supervisorctl restart copyparty
```

### Stop/Start Service
```bash
docker exec multi-agent-container supervisorctl stop copyparty
docker exec multi-agent-container supervisorctl start copyparty
```

## Features Available

### File Browsing
- Navigate through `/workspace` directory
- View file metadata
- Search files (indexing enabled with `-e2dsa`)
- Thumbnail previews (requires FFmpeg/Pillow)
- Terminal-style colored file listing

### Downloads
- Download individual files
- Download folders as ZIP/TAR
- Resume interrupted downloads
- HTTP range requests supported

### Read-Only Mode
Currently configured for read-only access to prevent unauthorized modifications.

To enable uploads, modify the supervisord.conf:
- Change `-v /workspace::r` to `-v /workspace::rw`
- Or use `:A` for full admin access

## Network Details
- **Docker Network:** `docker_ragflow` (172.18.0.0/16)
- **multi-agent-container IP:** 172.18.0.4
- **visionflow_container IP:** 172.18.0.9
- **External Access:** 192.168.0.51:3001

## URL Routing

After rebuild, the following routes are available:
- `http://192.168.0.51:3001/` → VisionFlow app
- `http://192.168.0.51:3001/browser/` → Copyparty file browser ✨ **NEW**
- `http://192.168.0.51:3001/api/` → Rust backend

## Troubleshooting

### Service Not Running
Check supervisor status:
```bash
docker exec multi-agent-container supervisorctl status copyparty
```

If stopped, start it:
```bash
docker exec multi-agent-container supervisorctl start copyparty
```

### 502 Bad Gateway
Copyparty service is not running or not listening on port 3923:
```bash
# Check if process is running
docker exec multi-agent-container ps aux | grep copyparty

# Check logs
docker exec multi-agent-container tail -50 /workspace/logs/copyparty.log
```

### Database Permissions
If you see "readonly database" errors:
```bash
docker exec multi-agent-container bash -c "
  rm -rf /workspace/.hist
  mkdir -p /workspace/.hist
  chown -R dev:dev /workspace/.hist
  supervisorctl restart copyparty
"
```

### Cannot Access Files
Verify volume permissions:
```bash
docker exec multi-agent-container ls -la /workspace
```

### Nginx Issues
Check nginx status and logs:
```bash
docker exec visionflow_container nginx -t
docker logs visionflow_container | tail -50
```

## Security Notes
1. ✅ Read-only by default to prevent unauthorized modifications
2. ✅ Runs as `dev` user (not root)
3. ✅ Behind nginx reverse proxy for additional control
4. ✅ Client IPs properly logged via `X-Forwarded-For`
5. ✅ Auto-managed by supervisor with restart policies

## Upgrade Path

To change copyparty settings:

1. **Edit supervisord.conf**
   ```bash
   vim /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/supervisord.conf
   ```

2. **Rebuild container**
   ```bash
   cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph
   docker-compose -f docker-compose.dev.yml up -d --build webxr
   ```

3. **Or reload supervisor config without rebuild**
   ```bash
   docker exec multi-agent-container supervisorctl reread
   docker exec multi-agent-container supervisorctl update
   ```

## Future Enhancements
- [ ] Add authentication (--auth)
- [ ] Enable write access for specific users
- [ ] Configure file upload rules and quotas
- [ ] Enable thumbnail generation for more file types
- [ ] Add custom volume mounts for additional directories
- [ ] Configure bandwidth throttling if needed

## Success! 🎉

Copyparty is now a permanent part of the multi-agent-docker infrastructure, providing instant web-based file access to the entire `/workspace` directory through a clean, terminal-style interface accessible at `http://192.168.0.51:3001/browser/`
